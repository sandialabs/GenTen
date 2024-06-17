//@HEADER
// ************************************************************************
//     Genten: Software for Generalized Tensor Decompositions
//     by Sandia National Laboratories
//
// Sandia National Laboratories is a multimission laboratory managed
// and operated by National Technology and Engineering Solutions of Sandia,
// LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
// U.S. Department of Energy's National Nuclear Security Administration under
// contract DE-NA0003525.
//
// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// ************************************************************************
//@HEADER

#include "Genten_HigherMoments.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Kokkos.hpp"
#include "Genten_Util.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBatched_Gemm_TeamVector_Impl.hpp"
#include <cmath>
#include <cstddef>
#include <limits>

namespace Genten {

namespace impl {

// Here we assume that X has 4 column blocks stacked into a 3d tensor, each
// frontal slice of X is a column block.
template <class MemberType, class MatrixViewType, class ScratchMatrixViewType>
KOKKOS_INLINE_FUNCTION
void khatri_rao_product(
  const MemberType &teamMember, int blockIndex1, int blockIndex2,
  int block2Size, int stdBlockSize, int currentTileSize, int baseRowIndex,
  MatrixViewType X, ScratchMatrixViewType result
) {
  const int resultNCol = result.extent_int(1);
  Kokkos::parallel_for(
    Kokkos::TeamThreadRange(teamMember, resultNCol), [&](int i) {
      const int block1ColPos = i / block2Size;
      const int block2ColPos = i % block2Size;
      const int block1BaseColInd = blockIndex1 * stdBlockSize;
      const int block2BaseColInd = blockIndex2 * stdBlockSize;
      Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(teamMember, currentTileSize), [&](int j) {
          result(j, i) =
            X(baseRowIndex + j, block1BaseColInd + block1ColPos) *
            X(baseRowIndex + j, block2BaseColInd + block2ColPos);
        }
      );
    }
  );
}

// We want to create a matrix where each row contains the 4d index of each of
// the unique blocks in the moment tensor. This matrix is stored row-major in a
// 1D array and variable a will point to this array.
template <class IndexViewType>
KOKKOS_INLINE_FUNCTION
void getIndexArray(
  const int nblocks, IndexViewType h_indexArray
) {
  int x = 0;
  for (int i = 0; i < nblocks; i++) {
    for (int j = i; j < nblocks; j++) {
      for (int k = j; k < nblocks; k++) {
        for (int l = k; l < nblocks; l++) {
          h_indexArray(x) = i;
          h_indexArray(x + 1) = j;
          h_indexArray(x + 2) = k;
          h_indexArray(x + 3) = l;
          x += 4;
        }
      }
    }
  }
}

template <class IndexViewType>
KOKKOS_INLINE_FUNCTION
void rankToBlockIndex(
  const int rank, int* blockIndex, IndexViewType indexArray
) {
  const int startPos = rank * 4;
  blockIndex[0] = indexArray(startPos);
  blockIndex[1] = indexArray(startPos+1);
  blockIndex[2] = indexArray(startPos+2);
  blockIndex[3] = indexArray(startPos+3);
}

template<typename ViewT>
TensorT<Kokkos::DefaultHostExecutionSpace> cokurtosis_impl(
  ViewT dataMatrix, int stdBlockSize, int tileSize, int teamSize
) {
  using exec_space = typename ViewT::execution_space;
  using scratch_memory_space = typename exec_space::scratch_memory_space;
  using policy_type = Kokkos::TeamPolicy<exec_space>;
  using member_type = typename policy_type::member_type;

  using flat_index_arr_type = Kokkos::View<int*, exec_space>;
  using flat_results_view_type = Kokkos::View<ttb_real***, exec_space>;
  using scratch_matrix_view = Kokkos::View<ttb_real**, scratch_memory_space>;

  const auto XnRow = dataMatrix.extent(0);
  const auto XnCol = dataMatrix.extent(1);

  const int nBlocks = (XnCol + stdBlockSize - 1) / stdBlockSize;
  const int oddBlockSize = XnCol - stdBlockSize * (nBlocks - 1);
  //# of unique blocks = (n+4-1) choose 4 or n choose 4 with repetition
  const int nUniqueBlocks = ((nBlocks+3)*(nBlocks+2)*(nBlocks+1)*nBlocks)/(4*3*2*1);
  std::cout<<"nBlocks: "<<nBlocks<<std::endl;
  std::cout<<"nUniqueBlocks: "<<nUniqueBlocks<<std::endl;

  const int nTile = XnRow%tileSize == 0 ? XnRow/tileSize : XnRow/tileSize + 1;

  flat_index_arr_type flatIndexArr("flatIndexArr", nUniqueBlocks*4);
  auto h_flatIndexArr = Kokkos::create_mirror_view(flatIndexArr);
  getIndexArray(nBlocks, h_flatIndexArr);
  Kokkos::deep_copy(flatIndexArr, h_flatIndexArr);

  flat_results_view_type flatResults(
    "flatResults", std::pow(stdBlockSize, 2),
    std::pow(stdBlockSize, 2), nUniqueBlocks
  );

  policy_type policy(teamSize, Kokkos::AUTO);

  const int stdNBlock = nUniqueBlocks/teamSize;
  const int firstNTeam = nUniqueBlocks%teamSize;
  const int scratch_size =
    2*scratch_matrix_view::shmem_size(tileSize, std::pow(stdBlockSize, 2));

  Kokkos::parallel_for(
    "BlocksLoop", policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
    KOKKOS_LAMBDA(const member_type &teamMember) {
      //const int tRank = teamMember.team_rank();
      const int lRank = teamMember.league_rank();

      int variableTileSize = tileSize;
      int localNBlocks;
      int startingBlockIndex;

      if (firstNTeam > 0) {
        if (lRank < firstNTeam) {
          localNBlocks = stdNBlock + 1;
          startingBlockIndex = lRank * (stdNBlock + 1);
        } else {
          localNBlocks = stdNBlock;
          startingBlockIndex = firstNTeam + lRank * stdNBlock;
        }
      } else {
        localNBlocks = stdNBlock;
        startingBlockIndex = lRank * stdNBlock;
      }

      scratch_matrix_view krp1_team(
        teamMember.team_scratch(0), tileSize, stdBlockSize*stdBlockSize
      );

      scratch_matrix_view krp2_team(
        teamMember.team_scratch(0), tileSize, stdBlockSize*stdBlockSize
      );

      for (
        int blockLinInd = startingBlockIndex;
        blockLinInd < startingBlockIndex + localNBlocks;
        blockLinInd++
      ) {
        // 4 int that represent the index of the block that this team is
        // supposed to compute
        int blockIndex [4];
        rankToBlockIndex(blockLinInd, blockIndex, flatIndexArr);

        // printf(
        //   "======indices========blockLinInd: %d, blockIndex[0]: %d, blockIndex[1]: %d, blockIndex[2]: %d, blockIndex[3]: %d\n",
        //   blockLinInd, blockIndex[0], blockIndex[1], blockIndex[2], blockIndex[3]
        // );

        // the number of columns in each of the column blocks of the input matrix
        int blockSizes [4];
        for (int i = 0; i < 4; i++) {
          blockSizes[i] =

            ((XnCol % stdBlockSize == 0) || blockIndex[i] != (nBlocks-1))
            ? stdBlockSize
            : oddBlockSize;
        }

        // printf(
        //   "-----sizes-----------blockSizes[0]: %d, blockSizes[1]: %d, blockSizes[2]: %d, blockSizes[3]: %d\n",
        //   blockSizes[0], blockSizes[1], blockSizes[2], blockSizes[3]
        // );

        auto krp1 = Kokkos::subview(
          krp1_team, Kokkos::ALL(),
          Kokkos::make_pair(0, blockSizes[0] * blockSizes[1])
        );
        auto krp2 = Kokkos::subview(
          krp2_team, Kokkos::ALL(),
          Kokkos::make_pair(0, blockSizes[2] * blockSizes[3])
        );

        auto gemmResult = Kokkos::subview(
          flatResults, Kokkos::ALL(), Kokkos::ALL(), blockLinInd
        );

        for (int i = 0; i < nTile; i++) {
          const int baseRowInd = i * tileSize;
          if ((i == nTile - 1) && (XnRow % tileSize != 0)) {
            variableTileSize = XnRow % tileSize;
          }

          khatri_rao_product(
            teamMember, blockIndex[1], blockIndex[0],
            blockSizes[0], stdBlockSize, variableTileSize,
            baseRowInd, dataMatrix, krp1
          );

          khatri_rao_product(
            teamMember, blockIndex[3], blockIndex[2],
            blockSizes[2], stdBlockSize, variableTileSize,
            baseRowInd, dataMatrix, krp2
          );

          teamMember.team_barrier();
          const ttb_real alpha = 1.0 / XnRow;

          if ((i == nTile - 1) && (XnRow % tileSize != 0)) {
            auto krp1SubView = Kokkos::subview(
              krp1, Kokkos::make_pair(0, variableTileSize), Kokkos::ALL()
            );

            auto krp2SubView = Kokkos::subview(
              krp2, Kokkos::make_pair(0, variableTileSize), Kokkos::ALL()
            );

            KokkosBatched::TeamGemm<
              member_type, KokkosBatched::Trans::Transpose,
              KokkosBatched::Trans::NoTranspose, KokkosBatched::Algo::Gemm::Unblocked
            >::invoke(teamMember, 1.0, krp1SubView, krp2SubView, 1.0, gemmResult);
          } else {
            KokkosBatched::TeamGemm<
              member_type, KokkosBatched::Trans::Transpose,
              KokkosBatched::Trans::NoTranspose, KokkosBatched::Algo::Gemm::Unblocked
            >::invoke(teamMember, alpha, krp1, krp2, 1.0, gemmResult);
          }
        }
      }
    }
  );

  Kokkos::fence();

  // Reconstruct tensor
  TensorT<Kokkos::DefaultHostExecutionSpace> moment(
    IndxArrayT<Kokkos::DefaultHostExecutionSpace>{
      XnCol, XnCol, XnCol, XnCol
    }
  );

  const auto flatResults_host = Kokkos::create_mirror_view_and_copy(
    Kokkos::DefaultHostExecutionSpace(), flatResults
  );

  const auto flatIndexArr_host = Kokkos::create_mirror_view_and_copy(
    Kokkos::DefaultHostExecutionSpace(), flatIndexArr
  );

  for (std::size_t ki = 0; ki < flatResults_host.extent(2); ++ki) {
    const auto blockView =
      Kokkos::subview(flatResults_host, Kokkos::ALL(), Kokkos::ALL(), ki);

    int blockIndex [4];
    rankToBlockIndex(ki, blockIndex, flatIndexArr_host);

	auto s = stdBlockSize;
	int blockSizes_h [4];
    for (int i = 0; i < 4; i++) {
		blockSizes_h[i] = ((XnCol % stdBlockSize == 0) || blockIndex[i] != (nBlocks-1))
            ? stdBlockSize
            : oddBlockSize;
			std::cout<<"blockSizes[i]"<<blockSizes_h[i]<<std::endl;
	}

    const auto size_i = blockSizes_h[0];
	const auto size_j = blockSizes_h[1];
	const auto size_k = blockSizes_h[2];
	const auto size_l = blockSizes_h[3];

    const auto iShift = blockIndex[0] * s;
    const auto jShift = blockIndex[1] * s;
    const auto kShift = blockIndex[2] * s;
    const auto lShift = blockIndex[3] * s;

	const auto n_elements_in_block = size_i*size_j*size_k*size_l;

    std::cout<<"ji: "<<blockView.extent(1)<<std::endl;
	std::cout<<"ii: "<<blockView.extent(0)<<std::endl;
	std::cout<<"shifts, i,j,k,l: "<<iShift<<" "<<jShift<<" "<<kShift<<" "<<lShift<<" "<<std::endl;
	std::cout<<"Number of Elements in Block "<<n_elements_in_block<<std::endl;
	const std::size_t upper_limit_ji = size_k*size_l;
	const std::size_t upper_limit_ii = size_i*size_j;
	std::cout<<"upper_limit_ji: "<<upper_limit_ji<<std::endl;
	std::cout<<"upper_limit_ii: "<<upper_limit_ii<<std::endl;

    for (std::size_t ji = 0; ji < upper_limit_ji; ++ji) {
      for (std::size_t ii = 0; ii < upper_limit_ii; ++ii) {
        const auto linIdx = ii + ji*(size_i*size_j);
        gt_assert(static_cast<int>(linIdx)<n_elements_in_block);

        const auto lIdx = linIdx / (size_i*size_j*size_k) + lShift;
        const auto b = linIdx % (size_i*size_j*size_k);
        const auto kIdx = b / (size_i*size_j) + kShift;
        const auto c = b % (size_i*size_j);
        const auto jIdx = c / size_i + jShift;
        const auto iIdx = c % size_i + iShift;

        //std::cout<<"linIdx, i,j,k,l: "<<linIdx<<", "<<iIdx<<", "<<jIdx<<", "<<kIdx<<", "<<lIdx<<std::endl;
		gt_assert(iIdx<XnCol);
		gt_assert(jIdx<XnCol);
		gt_assert(kIdx<XnCol);
		gt_assert(lIdx<XnCol);

        std::vector<long unsigned int> indices{iIdx, jIdx, kIdx, lIdx};
        std::sort(indices.begin(), indices.end());
		//This works because moment is a symmetric tensor
        const auto m = blockView(ii, ji);
        do {
          moment(indices[0], indices[1], indices[2], indices[3]) = m;
        } while (std::next_permutation(indices.begin(), indices.end()));
	  }

    }//end ji
  } //end ki

//   printf("\n# ncokurtosis_impl()\n\n");
//   printf("nUniqueBlocks = %d\n", nUniqueBlocks);
//   printf("| moment | k | i | j | count |\n");
//   printf("| ------ | - | - | - | ----- |\n");
//   std::size_t count = 0;
//   for (std::size_t k = 0; k < flatResults.extent(2); ++k) {
//     auto blockView =
//         Kokkos::subview(flatResults, Kokkos::ALL(), Kokkos::ALL(), k);
//     for (std::size_t j = 0; j < blockView.extent(1); ++j) {
//       for (std::size_t i = 0; i < blockView.extent(0); ++i) {
//         printf(
//           "| %f | %zu | %zu | %zu | %zu |\n",
//           blockView(i, j), k, i, j, count
//         );

//         ++count;
//       }
//     }
//   }

  return moment;
}

} // namespace impl

template <typename ExecSpace>
TensorT<Kokkos::DefaultHostExecutionSpace> create_and_compute_moment_tensor(
  Kokkos::View<ttb_real**, Kokkos::LayoutLeft, ExecSpace> dataMatrix,
  const int blockSize, const int teamSize
) {
  // these need to be moved
  int tileSize = 1;
  const auto refactoredAlgoRes =
    impl::cokurtosis_impl(dataMatrix, blockSize, tileSize, teamSize);

  return refactoredAlgoRes;
}

#ifdef KOKKOS_ENABLE_CUDA
template TensorT<Kokkos::DefaultHostExecutionSpace>
create_and_compute_moment_tensor<Kokkos::Cuda>(
  Kokkos::View<ttb_real**, Kokkos::LayoutLeft, Kokkos::Cuda> x,
  const int blockSize, const int teamSize
);
#endif

#ifdef KOKKOS_ENABLE_HIP
template TensorT<Kokkos::DefaultHostExecutionSpace>
create_and_compute_moment_tensor<Kokkos::Experimental::HIP>(
  Kokkos::View<ttb_real**, Kokkos::LayoutLeft, Kokkos::Experimental::HIP> x,
  const int blockSize, const int teamSize
);
#endif

#ifdef KOKKOS_ENABLE_OPENMP
template TensorT<Kokkos::DefaultHostExecutionSpace>
create_and_compute_moment_tensor<Kokkos::OpenMP>(
  Kokkos::View<ttb_real**, Kokkos::LayoutLeft, Kokkos::OpenMP> x,
  const int blockSize, const int teamSize
);
#endif

#ifdef KOKKOS_ENABLE_THREADS
template TensorT<Kokkos::DefaultHostExecutionSpace>
create_and_compute_moment_tensor<Kokkos::Threads>(
  Kokkos::View<ttb_real**, Kokkos::LayoutLeft, Kokkos::Threads> x,
  const int blockSize, const int teamSize
);
#endif

#ifdef KOKKOS_ENABLE_SERIAL
template TensorT<Kokkos::DefaultHostExecutionSpace>
create_and_compute_moment_tensor<Kokkos::Serial>(
  Kokkos::View<ttb_real**, Kokkos::LayoutLeft, Kokkos::Serial> x,
  const int blockSize, const int teamSize
);
#endif

} // namespace Genten
