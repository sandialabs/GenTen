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

#include "Genten_Kokkos.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_HigherMoments.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBatched_Gemm_TeamVector_Impl.hpp"

namespace Genten {
namespace impl{

// Here we assume that X has 4 column blocks stacked
// into a 3d tensor, each frontal slice of X is a column block.
template<class MemberType, class MatrixViewType, class ScratchMatrixViewType>
KOKKOS_INLINE_FUNCTION
void khatri_rao_product(const MemberType & teamMember,
			int blockIndex1,
			int blockIndex2,
			int block2Size,
			int stdBlockSize,
			int currentTileSize,
			int baseRowIndex,
			MatrixViewType X,
			ScratchMatrixViewType result)
{

  const int resultNCol = result.extent_int(1);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, resultNCol),
		       [&](int i){
			 const int block1ColPos = i / block2Size;
			 const int block2ColPos = i % block2Size;
			 const int block1BaseColInd = blockIndex1*stdBlockSize;
			 const int block2BaseColInd = blockIndex2*stdBlockSize;
			 Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember, currentTileSize),
					      [&](int j){
						result(j, i) = X(baseRowIndex+j, block1BaseColInd+block1ColPos) *
						  X(baseRowIndex+j, block2BaseColInd+block2ColPos);
					      });
		       });
}

//We want to create a matrix where each row contains the 4d index of each of the unique
//blocks in the moment tensor. This matrix is stored row-major in a 1D array and variable
// a will point to this array.
template<class IndexViewType>
KOKKOS_INLINE_FUNCTION
void getIndexArray(int nblocks, IndexViewType h_indexArray)
{
  int x = 0;
  for(int i=0; i<nblocks; i++){
    // printf("%d, \n", indexArray(x));
    for(int j=i; j<nblocks; j++){
      for(int k=j; k<nblocks; k++){
        for(int l=k; l<nblocks; l++){
          h_indexArray(x) = i;
          h_indexArray(x+1) = j;
          h_indexArray(x+2) = k;
          h_indexArray(x+3) = l;
          x = x + 4;
        }
      }
    }
  }
}

template<class IndexViewType>
KOKKOS_INLINE_FUNCTION
void rankToBlockIndex(int rank, int* blockIndex, IndexViewType indexArray)
{
  int startPos = rank*4;
  blockIndex[0] = indexArray(startPos);
  blockIndex[1] = indexArray(startPos+1);
  blockIndex[2] = indexArray(startPos+2);
  blockIndex[3] = indexArray(startPos+3);
}

template<class ... Properties>
void cokurtosis_impl(Kokkos::View<ttb_real**, Properties...> dataMatrix,
		     int stdBlockSize, int tileSize, int teamSize)
{

  const int XnRow = dataMatrix.extent(0);
  const int XnCol = dataMatrix.extent(1);

  int nBlocks = std::ceil(XnCol/stdBlockSize);
  int oddBlockSize = XnCol - stdBlockSize * (nBlocks - 1);
  //# of unique blocks = (n+4-1) choose 4 or n choose 4 with repetition
  int nUniqueBlocks = ((nBlocks+3)*(nBlocks+2)*(nBlocks+1)*nBlocks)/(4*3*2*1);
  std::cout << "nUniqueBlocks = " << nUniqueBlocks << '\n';

  int nTile = XnRow%tileSize ==0 ? XnRow/tileSize : XnRow/tileSize + 1;

  using flat_index_arr_type = Kokkos::View<int*>;
  using flat_results_view_type = Kokkos::View<ttb_real***>;
  using ScratchVectorView = Kokkos::View<ttb_real*,Kokkos::DefaultExecutionSpace::scratch_memory_space>;
  using ScratchMatrixView = Kokkos::View<ttb_real**,Kokkos::DefaultExecutionSpace::scratch_memory_space>;
  using ScratchTensorView = Kokkos::View<ttb_real***,Kokkos::DefaultExecutionSpace::scratch_memory_space>;


  flat_index_arr_type flatIndexArr("flatIndexArr", nUniqueBlocks*4);
  auto h_flatIndexArr = Kokkos::create_mirror_view(flatIndexArr);
  getIndexArray(nBlocks, h_flatIndexArr);
  Kokkos::deep_copy(flatIndexArr, h_flatIndexArr);

  flat_results_view_type flatResults("flatResults",
				     std::pow(stdBlockSize,2), std::pow(stdBlockSize,2), nUniqueBlocks);
  using policy_type = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
  policy_type policy(teamSize, Kokkos::AUTO);
  using member_type = typename policy_type::member_type;

  int stdNBlock = nUniqueBlocks/teamSize;
  int firstNTeam = nUniqueBlocks%teamSize;
  const int scratch_size = 2*ScratchMatrixView::shmem_size(tileSize, std::pow(stdBlockSize,2));
  Kokkos::parallel_for("BlocksLoop",
		       policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
		       KOKKOS_LAMBDA(const member_type &teamMember)
		       {
			 const int tRank = teamMember.team_rank();
			 const int lRank = teamMember.league_rank();

			 int variableTileSize = tileSize;
			 int localNBlocks;
			 int startingBlockIndex;
			 if(firstNTeam > 0){
			   if(lRank < firstNTeam){
			     localNBlocks = stdNBlock+1;
			     startingBlockIndex = lRank*(stdNBlock+1);
			   }
			   else{
			     localNBlocks = stdNBlock;
			     startingBlockIndex = firstNTeam + lRank*stdNBlock;
			   }
			 }
			 else{
			   localNBlocks = stdNBlock;
			   startingBlockIndex = lRank*stdNBlock;
			 }

			 ScratchMatrixView krp1_team(teamMember.team_scratch(0),
						     tileSize, stdBlockSize*stdBlockSize);
			 ScratchMatrixView krp2_team(teamMember.team_scratch(0),
						     tileSize, stdBlockSize*stdBlockSize);

			 // 4 int that represent the index of the block that this team is supposed to compute
			 int blockIndex [4];
			 // the number of columns in each of the column blocks of the input matrix
			 int blockSizes [4];
			 ttb_real ONE = 1.0;
			 for(int blockLinInd=startingBlockIndex; blockLinInd<startingBlockIndex+localNBlocks; blockLinInd++)
			   {
			     rankToBlockIndex(blockLinInd, blockIndex, flatIndexArr);
			     for(int i=0; i<4; i++){
			       blockSizes[i] = (XnCol % stdBlockSize == 0 || blockIndex[i] != nBlocks) ? stdBlockSize : oddBlockSize;
			     }
			     auto krp1 = Kokkos::subview(krp1_team,
							 Kokkos::ALL(),
							 Kokkos::make_pair(0, blockSizes[0]*blockSizes[1]) );
			     auto krp2 = Kokkos::subview(krp2_team,
							 Kokkos::ALL(),
							 Kokkos::make_pair(0, blockSizes[2]*blockSizes[3]) );
			     auto gemmResult = Kokkos::subview(flatResults,
							       Kokkos::ALL(),
							       Kokkos::ALL(),
							       blockLinInd);

			     for(int i=0; i<nTile; i++){
			       int baseRowInd = i*tileSize;
			       if((i == nTile-1) && (XnRow%tileSize!=0)){
				 variableTileSize = XnRow%tileSize;
			       }

			       khatri_rao_product(teamMember, blockIndex[1],
						blockIndex[0], blockSizes[0],
						stdBlockSize, variableTileSize,
						baseRowInd, dataMatrix, krp1);
			       khatri_rao_product(teamMember, blockIndex[3],
						blockIndex[2], blockSizes[2],
						stdBlockSize, variableTileSize,
						baseRowInd, dataMatrix, krp2);

			       teamMember.team_barrier();
			       const ttb_real alpha = 1.0/XnRow;

			       if ( (i == nTile-1) && (XnRow%tileSize!=0) )
			       {
				 auto krp1SubView = Kokkos::subview(krp1,
								    Kokkos::make_pair(0,variableTileSize),
								    Kokkos::ALL());
				 auto krp2SubView = Kokkos::subview(krp2,
								    Kokkos::make_pair(0,variableTileSize),
								    Kokkos::ALL());

				 KokkosBatched::TeamGemm<
				   member_type, KokkosBatched::Trans::Transpose,
				   KokkosBatched::Trans::NoTranspose, KokkosBatched::Algo::Gemm::Unblocked
				   >::invoke(teamMember, ONE, krp1SubView, krp2SubView, ONE, gemmResult);
			       }

			       else{
				 KokkosBatched::TeamGemm<
				   member_type, KokkosBatched::Trans::Transpose,
				   KokkosBatched::Trans::NoTranspose, KokkosBatched::Algo::Gemm::Unblocked
				   >::invoke(teamMember, alpha, krp1, krp2, ONE, gemmResult);
			       }
			     }
			   }
		       });
  Kokkos::fence();

  int count=0;
  for (std::size_t k=0; k<flatResults.extent(2); ++k){
    auto blockView = Kokkos::subview(flatResults, Kokkos::ALL(), Kokkos::ALL(), k);
    for (std::size_t j=0; j<blockView.extent(1); ++j){
      for (std::size_t i=0; i<blockView.extent(0); ++i){
	std::cout << count++ << ' '
		  << k << ' '
		  << i << ' '
		  << j << ' '
		  << blockView(i,j)
		  << '\n';
      }
    }
  }
}

// template<class ... Properties>
// auto moment_tensor_impl(Kokkos::View<ttb_real**, Properties...> dataMatrix,
// 			int order)
// {
//   using data_matrix_type = decltype(dataMatrix);
//   using exe_space = typename data_matrix_type::execution_space;
//   using default_space_host = Genten::DefaultHostExecutionSpace;

//   using index_arr_type      = Genten::IndxArrayT<exe_space>;
//   using index_arr_type_host = Genten::IndxArrayT<default_space_host>;
//   using moment_tensor_type  = Genten::TensorT<exe_space>;

//   //moment tensor is size nvars^d, where d is order of moment, i.e. nvars*nvars*.... (d times)
//   index_arr_type momentTensorSize(order, dataMatrix.extent(1));

//   moment_tensor_type X(momentTensorSize, 0.0);
//   cokurtosis_impl(dataMatrix, X);

//   // if (order == 4){
//   //   cokurtosis_impl(dataMatrix, X);
//   // }
//   // else{
//   //   throw std::runtime_error("Missing impl for oder != 4");
//   // }

//   return X;
// }
}// namespace impl

ttb_real * create_and_compute_raw_moment_tensor(ttb_real *rawDataPtr,
						int nsamples,
						int nvars,
						const int order)
{

  using exe_space = Genten::DefaultExecutionSpace;
  using exe_space_host = Genten::DefaultHostExecutionSpace;

  //Example: https://github.com/kokkos/kokkos-fortran-interop/blob/master/src/flcl-cxx.hpp/#L157
  //raw data is "viewed" as a 2D-array in layoutleft order
  using mem_unmanged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  using data_matrix_type_host = Kokkos::View<ttb_real**, Kokkos::LayoutLeft, exe_space_host, mem_unmanged>;
  data_matrix_type_host dataMatrixHost(rawDataPtr, nsamples, nvars);

  auto dataMatrix = Kokkos::create_mirror_view(exe_space(), dataMatrixHost);
  Kokkos::deep_copy(dataMatrix, dataMatrixHost);

  // these need to be moved
  int targetBlockSize = 1;
  int tileSize = 1;
  int teamSize = 1;
  impl::cokurtosis_impl(dataMatrix, targetBlockSize, tileSize, teamSize);

  //auto momentTensor = impl::moment_tensor_impl(dataMatrix, order);
  // auto momentTensorHost = create_mirror_view(momentTensor);
  // deep_copy(momentTensorHost, momentTensor);

  return nullptr; //momentTensorHost.getValues().ptr();
}

}// namespace Genten
