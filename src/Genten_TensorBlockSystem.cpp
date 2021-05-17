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

#include "Genten_TensorBlockSystem.hpp"
#include <string>

namespace Genten {
namespace detail {

bool fileFormatIsBinary(std::string const &file_name) {
  std::ifstream tensor_file(file_name, std::ios::binary);
  std::string header;
  header.resize(4);
  try {
    tensor_file.read(&header[0], 4);
  } catch (...) {
    return false;
  }

  if (header == "sptn") {
    return true;
  }

  return false;
}

small_vector<int> singleDimMediumGrainBlocking(int ModeLength,
                                               int ProcsInMode) {
  small_vector<int> Range{0};
  const auto FibersPerBlock = ModeLength / ProcsInMode;
  auto Remainder = ModeLength % ProcsInMode;

  // We ended up with more processors than rows in the fiber :O Just return all
  // fibers in the same block. It seems easier to handle this here than to try
  // to make the while loop logic do something smart
  if (FibersPerBlock == 0) {
    Range.push_back(ModeLength);
  }

  while (Range.back() < ModeLength) {
    const auto back = Range.back();
    // This branch makes our blocks 1 bigger to eat the Remainder fibers
    if (Remainder > 0) {
      Range.push_back(back + FibersPerBlock + 1);
      --Remainder;
    } else {
      Range.push_back(back + FibersPerBlock);
    }
  }

  // Sanity check that we ended with the correct number of blocks and fibers
  assert(Range.size() == ProcsInMode + 1);
  assert(Range.back() == ModeLength);

  return Range;
}

std::vector<small_vector<int>>
generateMediumGrainBlocking(std::vector<int> ModeLengths,
                            small_vector<int> const &ProcGridSizes) {
  const auto Ndims = ModeLengths.size();
  std::vector<small_vector<int>> blocking;
  blocking.reserve(Ndims);

  for (auto i = 0; i < Ndims; ++i) {
    blocking.emplace_back(
        singleDimMediumGrainBlocking(ModeLengths[i], ProcGridSizes[i]));
  }

  return blocking;
}

std::vector<TDatatype> distributeTensorToVectors(std::ifstream &ifs,
                                                 uint64_t nnz, int indexbase,
                                                 MPI_Comm comm, int rank,
                                                 int nprocs) {
  constexpr auto dt_size = sizeof(detail::TDatatype);
  std::vector<detail::TDatatype> Tvec;
  small_vector<int> who_gets_what =
      detail::singleDimMediumGrainBlocking(nnz, nprocs);

  if (rank == 0) {
    { // Write tensor to form we can MPI_Send more easily.
      typename SptensorT<Kokkos::Serial>::HostMirror sp_tensor_host;
      import_sptensor(ifs, sp_tensor_host, indexbase, false);

      if (sp_tensor_host.ndims() > 12) {
        throw std::logic_error(
            "Distributed tensors with more than 12 dimensions "
            "can't be read by the ascii based parsers.");
      }

      Tvec.resize(sp_tensor_host.nnz());
      for (auto i = 0ull; i < sp_tensor_host.nnz(); ++i) {
        auto &dt = Tvec[i];
        for (auto j = 0; j < sp_tensor_host.ndims(); ++j) {
          dt.coo[j] = sp_tensor_host.subscript(i, j);
        }
        dt.val = sp_tensor_host.value(i);
      }
    }

    std::vector<MPI_Request> requests(nprocs - 1);
    std::vector<MPI_Status> statuses(nprocs - 1);
    auto total_sent = 0;
    for (auto i = 1; i < nprocs; ++i) {
      // Size to sent to rank i
      const auto nelements = who_gets_what[i + 1] - who_gets_what[i];
      const auto nbytes = nelements * dt_size;
      total_sent += nelements;

      const auto index_of_first_element = who_gets_what[i];
      if(std::uint64_t(nbytes) >= std::numeric_limits<int>::max()){
        std::cout << "OOPs, tried to send to many bytes :/." << std::endl;
      }
      MPI_Isend(Tvec.data() + index_of_first_element, nbytes, MPI_BYTE, i, i,
                comm, &requests[i - 1]);
    }
    MPI_Waitall(requests.size(), requests.data(), statuses.data());
    auto total_before = Tvec.size();
    auto begin = Tvec.begin();
    std::advance(begin, who_gets_what[1]); // wgw[0] == 0 always
    Tvec.erase(begin, Tvec.end());
    Tvec.shrink_to_fit(); // Yay now I only have rank 0 data

    auto total_after = Tvec.size() + total_sent;
    if (total_after != total_before) {
      throw std::logic_error(
          "The number of elements after sending and shrinking did not match "
          "the input number of elements.");
    }
  } else {
    const auto nelements = who_gets_what[rank + 1] - who_gets_what[rank];
    Tvec.resize(nelements);
    const auto nbytes = nelements * dt_size;
    if(std::uint64_t(nbytes) >= std::numeric_limits<int>::max()){
      std::cout << "OOPs, tried to recieve to many bytes :/." << std::endl;
    }
    MPI_Recv(Tvec.data(), nbytes, MPI_BYTE, 0, rank, comm, MPI_STATUS_IGNORE);
  }

  return Tvec;
}

namespace {
int blockInThatDim(int element, small_vector<int> const &range) {
  const auto nblocks = range.size();
  assert(element < range.back()); // This would mean the element is too large
  assert(range.size() >= 2);      // Range always has at least 2 elements

  // We could binary search, which could be faster for large ranges, but I
  // suspect this is fine. Because range.back() is always 1 more than the
  // largest possible element and range.size() >= 2 we don't have to worry
  // about block_guess + 1 going past the end.
  auto block_guess = 0;
  while (element >= range[block_guess + 1]) {
    ++block_guess;
  }

  return block_guess;
}

// The MPI_Comm must be the one that represents the grid for this to work
int rankInGridThatOwns(int const *COO, MPI_Comm grid_comm,
                       std::vector<small_vector<int>> const &ElementRanges) {
  const auto ndims = ElementRanges.size();
  small_vector<int> GridPos(ndims);
  for (auto i = 0; i < ndims; ++i) {
    GridPos[i] = blockInThatDim(COO[i], ElementRanges[i]);
  }

  int rank;
  MPI_Cart_rank(grid_comm, GridPos.data(), &rank);

  return rank;
}
} // namespace

std::vector<TDatatype> redistributeTensor(
    std::vector<TDatatype> const &Tvec, std::vector<int> const &TDims,
    std::vector<small_vector<int>> const &blocking, ProcessorMap const &pmap) {

  const auto nprocs = pmap.gridSize();
  const auto rank = pmap.gridRank();
  MPI_Comm grid_comm = pmap.gridComm();

  std::vector<std::vector<TDatatype>> elems_to_write(nprocs);
  for (auto const &elem : Tvec) {
    auto elem_owner_rank = rankInGridThatOwns(elem.coo, grid_comm, blocking);
    elems_to_write[elem_owner_rank].push_back(elem);
  }

  small_vector<int> amount_to_write(nprocs);
  for (auto i = 0; i < nprocs; ++i) {
    amount_to_write[i] = elems_to_write[i].size();
  }

  small_vector<int> offset_to_write_at(nprocs);
  MPI_Exscan(amount_to_write.data(), offset_to_write_at.data(), nprocs, MPI_INT,
             MPI_SUM, grid_comm);

  int amount_to_allocate_for_window = 0;
  MPI_Reduce_scatter_block(amount_to_write.data(),
                           &amount_to_allocate_for_window, 1, MPI_INT, MPI_SUM,
                           grid_comm);

  if (amount_to_allocate_for_window == 0) {
    const auto my_rank = pmap.gridRank();
    std::stringstream ss;
    ss << "WARNING Node(" << my_rank
       << "), recieved zero nnz in the current blocking\n";
    std::cout << ss.str() << std::flush;
    // TODO Handle this better than just aborting, but I don't have another
    // good solution for now.
    if (pmap.gridSize() > 1) {
      MPI_Abort(pmap.gridComm(), MPI_ERR_UNKNOWN);
    } else {
      std::cout << "Zero tensor on a single node? Something probably went "
                   "really wrong."
                << std::endl;
      std::abort();
    }
  }

  TDatatype *data;
  MPI_Win window;
  constexpr auto DataElemSize = sizeof(TDatatype);
  MPI_Win_allocate(amount_to_allocate_for_window * DataElemSize,
                   /*displacement = */ DataElemSize, MPI_INFO_NULL, grid_comm,
                   &data, &window);

  // Jonathan L. told me for AllToAll Fences are probably better than locking if
  // communication don't conflict
  MPI_Win_fence(0, window);
  for (auto i = 0; i < nprocs; ++i) {
    const auto bytes_to_write = DataElemSize * amount_to_write[i];
    MPI_Put(
        /* Origin ptr */ elems_to_write[i].data(),
        /* Origin num bytes */ bytes_to_write,
        /* Datatype for put */ MPI_BYTE,
        /* Target */ i,
        /* Displacement at target (not in bytes) */ offset_to_write_at[i],
        /* Target num bytes */ bytes_to_write,
        /* Origin data type */ MPI_BYTE, window);
  }
  MPI_Win_fence(0, window);

  // Copy data to the output vector
  std::vector<TDatatype> redistributedData(amount_to_allocate_for_window);
  std::copy(data, data + amount_to_allocate_for_window,
            redistributedData.data());

  // Free the MPI window and the buffer that it was allocated in
  MPI_Win_free(&window);
  return redistributedData;
}

} // namespace detail
} // namespace Genten
