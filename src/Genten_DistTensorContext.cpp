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

#include <string>

#include "Genten_DistTensorContext.hpp"

#ifdef HAVE_DIST

namespace Genten {
namespace detail {

void printGrids(const ProcessorMap& pmap) {
  if (DistContext::isDebug()) {
    if (pmap.gridRank() == 0) {
      std::cout << "Pmap initalization complete with grid: ";
      for (auto p : pmap.gridDims()) {
        std::cout << p << " ";
      }
      std::cout << std::endl;
    }
    pmap.gridBarrier();
  }
}

void printBlocking(const ProcessorMap& pmap,
                   const std::vector<small_vector<int>>& blocking) {
  if (DistContext::isDebug()) {
    if (pmap.gridRank() == 0) {
      std::cout << "With blocking:\n";
      auto dim = 0;
      for (auto const &inner : blocking) {
        std::cout << "\tdim(" << dim << "): ";
        ++dim;
        for (auto i : inner) {
          std::cout << i << " ";
        }
        std::cout << "\n";
      }
      std::cout << std::endl;
    }
    pmap.gridBarrier();
  }
}

void fileFormatIsBinary(const std::string& file_name,
                        bool& is_binary_sparse,
                        bool& is_binary_dense) {
  is_binary_sparse = false;
  is_binary_dense = false;
  std::ifstream tensor_file(file_name, std::ios::binary);
  std::string header;
  header.resize(4);

  try {
    tensor_file.read(&header[0], 4);
  } catch (...) {
    return;
  }

  if (header == "sptn") {
    is_binary_sparse = true;
  }
  if (header == "dntn") {
    is_binary_dense = true;
  }
}

small_vector<int> singleDimUniformBlocking(int ModeLength, int ProcsInMode) {
  small_vector<int> Range{0};
  const auto FibersPerBlock = ModeLength / ProcsInMode;
  auto Remainder = ModeLength % ProcsInMode;

  // Divide ModeLength fibers evenly across ProcsInMode processors
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

  // If ProcsInMode > FibersPerBlock, FibersPerBlock == 0 and
  // Remainder == ModeLength.  In this case, Range will be an array of 1's
  // of length ModeLength.  Expand it to the needed size of ProcsInMode+1 by
  // repeating the last entry, which will mean those proc's have 0 entries
  if (Range.size() < ProcsInMode+1)
    Range.resize(ProcsInMode+1,Range.back());

  // Sanity check that we ended with the correct number of blocks and fibers
  assert(Range.size() == ProcsInMode + 1);
  assert(Range.back() == ModeLength);

  return Range;
}

std::vector<small_vector<int>>
generateUniformBlocking(const std::vector<std::uint32_t>& ModeLengths,
                        const small_vector<int>& ProcGridSizes) {
  const auto Ndims = ModeLengths.size();
  std::vector<small_vector<int>> blocking;
  blocking.reserve(Ndims);

  for (auto i = 0; i < Ndims; ++i) {
    blocking.emplace_back(
        singleDimUniformBlocking(ModeLengths[i], ProcGridSizes[i]));
  }

  return blocking;
}

std::vector<G_MPI_IO::TDatatype<ttb_real>>
distributeTensorToVectors(const Sptensor& sp_tensor_host, uint64_t nnz,
                          MPI_Comm comm, int rank, int nprocs) {
  constexpr auto dt_size = sizeof(G_MPI_IO::TDatatype<ttb_real>);
  std::vector<G_MPI_IO::TDatatype<ttb_real>> Tvec;
  small_vector<int> who_gets_what =
      detail::singleDimUniformBlocking(nnz, nprocs);

  if (rank == 0) {
    { // Write tensor to form we can MPI_Send more easily.
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
    MPI_Recv(Tvec.data(), nbytes, MPI_BYTE, 0, rank, comm, MPI_STATUS_IGNORE);
  }

  return Tvec;
}

std::vector<ttb_real>
distributeTensorToVectors(const Tensor& dn_tensor_host, uint64_t nnz,
                          MPI_Comm comm, int rank, int nprocs,
                          ttb_indx& offset) {
  constexpr auto dt_size = sizeof(ttb_real);
  std::vector<ttb_real> Tvec;
  small_vector<int> who_gets_what =
      detail::singleDimUniformBlocking(nnz, nprocs);
  offset = who_gets_what[rank];

  if (rank == 0) {
    // Write tensor to form we can MPI_Send more easily.
    Tvec.resize(dn_tensor_host.numel());
    for (auto i = 0ull; i < dn_tensor_host.numel(); ++i)
      Tvec[i] = dn_tensor_host[i];

    std::vector<MPI_Request> requests(nprocs - 1);
    std::vector<MPI_Status> statuses(nprocs - 1);
    auto total_sent = 0;
    for (auto i = 1; i < nprocs; ++i) {
      // Size to sent to rank i
      const auto nelements = who_gets_what[i + 1] - who_gets_what[i];
      const auto nbytes = nelements * dt_size;
      total_sent += nelements;

      const auto index_of_first_element = who_gets_what[i];
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
    MPI_Recv(Tvec.data(), nbytes, MPI_BYTE, 0, rank, comm, MPI_STATUS_IGNORE);
  }

  return Tvec;
}

namespace {
int blockInThatDim(int element, const small_vector<int>& range) {
  // const auto nblocks = range.size();
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
int rankInGridThatOwns(std::uint32_t const *COO, MPI_Comm grid_comm,
                       const std::vector<small_vector<int>>& ElementRanges) {
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

std::vector<G_MPI_IO::TDatatype<ttb_real>>
redistributeTensor(const std::vector<G_MPI_IO::TDatatype<ttb_real>>& Tvec,
                   const std::vector<std::uint32_t>& TDims,
                   const std::vector<small_vector<int>>& blocking,
                   const ProcessorMap& pmap) {

  const auto nprocs = pmap.gridSize();
  const auto rank = pmap.gridRank();
  MPI_Comm grid_comm = pmap.gridComm();

  std::vector<std::vector<G_MPI_IO::TDatatype<ttb_real>>> elems_to_write(nprocs);
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
    const auto ndims = blocking.size();
    std::stringstream ss;
    ss << "WARNING MPI rank(" << my_rank
       << "), received zero nnz in the current blocking.\n\tTensor block: [ ";
    for (auto i=0; i<ndims; i++)
      ss << pmap.gridCoord(i) << " ";
    ss << "],  range: [ ";
    for (auto i=0; i<ndims; i++)
      ss << blocking[i][pmap.gridCoord(i)] << " ";
    ss << "] to [ ";
    for (auto i=0; i<ndims; i++)
      ss << blocking[i][pmap.gridCoord(i)+1] << " ";
    ss << "]\n";
    std::cout << ss.str() << std::flush;
    // TODO Handle this better than just aborting, but I don't have another
    // good solution for now.
    // NOTE (ETP, 9/19/22):  Having an empty proc does not appear to hurt
    // anything so commenting this out for now.
    /*
    if (pmap.gridSize() > 1) {
      MPI_Abort(pmap.gridComm(), MPI_ERR_UNKNOWN);
    } else {
      std::cout << "Zero tensor on a single node? Something probably went "
                   "really wrong."
                << std::endl;
      std::abort();
    }
    */
  }

  // Let's leave this onesided because IMO it makes life easier. This is self
  // contained so won't impact TBS
  G_MPI_IO::TDatatype<ttb_real> *data;
  MPI_Win window;
  constexpr auto DataElemSize = sizeof(G_MPI_IO::TDatatype<ttb_real>);
  MPI_Win_allocate(amount_to_allocate_for_window * DataElemSize,
                   /*displacement = */ DataElemSize, MPI_INFO_NULL, grid_comm,
                   &data, &window);

  MPI_Datatype element_type;
  MPI_Type_contiguous(DataElemSize, MPI_BYTE, &element_type);
  MPI_Type_commit(&element_type);

  // Jonathan L. told me for AllToAll Fences are probably better than locking
  // if communications don't conflict
  MPI_Win_fence(0, window);
  for (auto i = 0; i < nprocs; ++i) {
    MPI_Put(
        /* Origin ptr */ elems_to_write[i].data(),
        /* Origin num elements */ amount_to_write[i],
        /* Datatype for put */ element_type,
        /* Target */ i,
        /* Displacement at target (not in bytes) */ offset_to_write_at[i],
        /* Target num elements */ amount_to_write[i],
        /* Origin data type */ element_type, window);
  }
  MPI_Win_fence(0, window);

  // Copy data to the output vector
  std::vector<G_MPI_IO::TDatatype<ttb_real>> redistributedData(
      data, data + amount_to_allocate_for_window);

  // Free the MPI window and the buffer that it was allocated in
  MPI_Win_free(&window);
  MPI_Type_free(&element_type);
  return redistributedData;
}

std::vector<ttb_real>
redistributeTensor(const std::vector<ttb_real>& Tvec,
                   const ttb_indx global_nnz, const ttb_indx global_offset,
                   const std::vector<std::uint32_t>& TDims,
                   const std::vector<small_vector<int>>& blocking,
                   const ProcessorMap& pmap) {

  const auto nprocs = pmap.gridSize();
  const auto rank = pmap.gridRank();
  MPI_Comm grid_comm = pmap.gridComm();

  std::vector<std::vector<ttb_real>> elems_to_write(nprocs);
  const ttb_indx local_nnz = Tvec.size();
  const ttb_indx ndims = TDims.size();
  std::vector<std::uint32_t> sub(ndims);
  IndxArray siz(ndims);
  for (auto dim=0; dim<ndims; ++dim)
    siz[dim] = TDims[dim];
  for (auto i=0; i<local_nnz; ++i) {
    Impl::ind2sub(sub, siz, global_nnz, i+global_offset);
    auto elem_owner_rank = rankInGridThatOwns(sub.data(), grid_comm, blocking);
    elems_to_write[elem_owner_rank].push_back(Tvec[i]);
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
    const auto ndims = blocking.size();
    std::stringstream ss;
    ss << "WARNING MPI rank(" << my_rank
       << "), received zero nnz in the current blocking.\n\tTensor block: [ ";
    for (auto i=0; i<ndims; i++)
      ss << pmap.gridCoord(i) << " ";
    ss << "],  range: [ ";
    for (auto i=0; i<ndims; i++)
      ss << blocking[i][pmap.gridCoord(i)] << " ";
    ss << "] to [ ";
    for (auto i=0; i<ndims; i++)
      ss << blocking[i][pmap.gridCoord(i)+1] << " ";
    ss << "]\n";
    std::cout << ss.str() << std::flush;
    // TODO Handle this better than just aborting, but I don't have another
    // good solution for now.
    // NOTE (ETP, 9/19/22):  Having an empty proc does not appear to hurt
    // anything so commenting this out for now.
    /*
    if (pmap.gridSize() > 1) {
      MPI_Abort(pmap.gridComm(), MPI_ERR_UNKNOWN);
    } else {
      std::cout << "Zero tensor on a single node? Something probably went "
                   "really wrong."
                << std::endl;
      std::abort();
    }
    */
  }

  // Let's leave this onesided because IMO it makes life easier. This is self
  // contained so won't impact TBS
  ttb_real *data;
  MPI_Win window;
  constexpr auto DataElemSize = sizeof(ttb_real);
  MPI_Win_allocate(amount_to_allocate_for_window * DataElemSize,
                   /*displacement = */ DataElemSize, MPI_INFO_NULL, grid_comm,
                   &data, &window);

  MPI_Datatype element_type;
  MPI_Type_contiguous(DataElemSize, MPI_BYTE, &element_type);
  MPI_Type_commit(&element_type);

  // Jonathan L. told me for AllToAll Fences are probably better than locking
  // if communications don't conflict
  MPI_Win_fence(0, window);
  for (auto i = 0; i < nprocs; ++i) {
    MPI_Put(
        /* Origin ptr */ elems_to_write[i].data(),
        /* Origin num elements */ amount_to_write[i],
        /* Datatype for put */ element_type,
        /* Target */ i,
        /* Displacement at target (not in bytes) */ offset_to_write_at[i],
        /* Target num elements */ amount_to_write[i],
        /* Origin data type */ element_type, window);
  }
  MPI_Win_fence(0, window);

  // Copy data to the output vector
  std::vector<ttb_real> redistributedData(
      data, data + amount_to_allocate_for_window);

  // Free the MPI window and the buffer that it was allocated in
  MPI_Win_free(&window);
  MPI_Type_free(&element_type);
  return redistributedData;
}

} // namespace detail

} // namespace Genten

#endif
