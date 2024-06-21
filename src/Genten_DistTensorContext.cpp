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
#include "Genten_TensorIO.hpp"
#include "Genten_IOtext.hpp"

#include "CMakeInclude.h"
#if defined(HAVE_DIST)
#include "Kokkos_UnorderedMap.hpp"
#endif

namespace Genten {

#ifdef HAVE_DIST

namespace detail {

struct RangePair {
  ttb_indx lower;
  ttb_indx upper;
};

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
                   const std::vector<small_vector<ttb_indx>>& blocking) {
  if (DistContext::isDebug()) {
    if (pmap.gridRank() == 0) {
      std::cout << "With blocking:\n";
      ttb_indx dim = 0;
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

small_vector<ttb_indx> singleDimUniformBlocking(ttb_indx ModeLength, ttb_indx ProcsInMode) {
  small_vector<ttb_indx> Range{0};
  const ttb_indx FibersPerBlock = ModeLength / ProcsInMode;
  ttb_indx Remainder = ModeLength % ProcsInMode;

  // Divide ModeLength fibers evenly across ProcsInMode processors
  while (Range.back() < ModeLength) {
    const ttb_indx back = Range.back();
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
  if (ttb_indx(Range.size()) < ProcsInMode+1)
    Range.resize(ProcsInMode+1,Range.back());

  // Sanity check that we ended with the correct number of blocks and fibers
  gt_assert(ttb_indx(Range.size()) == ProcsInMode + 1);
  gt_assert(Range.back() == ModeLength);

  return Range;
}

std::vector<small_vector<ttb_indx>>
generateUniformBlocking(const std::vector<ttb_indx>& ModeLengths,
                        const small_vector<ttb_indx>& ProcGridSizes) {
  const ttb_indx Ndims = ModeLengths.size();
  std::vector<small_vector<ttb_indx>> blocking;
  blocking.reserve(Ndims);

  for (ttb_indx i = 0; i < Ndims; ++i) {
    blocking.emplace_back(
        singleDimUniformBlocking(ModeLengths[i], ProcGridSizes[i]));
  }

  return blocking;
}

template <typename ExecSpace>
auto
rangesToIndexArray(const small_vector<RangePair>& ranges)
{
  IndxArrayT<ExecSpace> outArray(ranges.size());
  auto mirrorArray = create_mirror_view(outArray);

  ttb_indx i = 0;
  for (auto const &rp : ranges) {
    const ttb_indx size = rp.upper - rp.lower;
    mirrorArray[i] = size;
    ++i;
  }

  deep_copy(outArray, mirrorArray);
  return outArray;
}

std::vector<SpDataType>
distributeTensorToVectorsSparse(const Sptensor& sp_tensor_host, ttb_indx nnz,
                                MPI_Comm comm, ttb_indx rank, ttb_indx nprocs) {
  constexpr ttb_indx dt_size = sizeof(SpDataType);
  std::vector<SpDataType> Tvec;
  small_vector<ttb_indx> who_gets_what =
      detail::singleDimUniformBlocking(nnz, nprocs);

  if (rank == 0) {
    { // Write tensor to form we can MPI_Send more easily.
      if (sp_tensor_host.ndims() > 12) {
        throw std::logic_error(
            "Distributed tensors with more than 12 dimensions "
            "can't be read by the ascii based parsers.");
      }

      Tvec.resize(sp_tensor_host.nnz());
      for (ttb_indx i = 0; i < sp_tensor_host.nnz(); ++i) {
        auto &dt = Tvec[i];
        for (ttb_indx j = 0; j < sp_tensor_host.ndims(); ++j) {
          dt.coo[j] = sp_tensor_host.subscript(i, j);
        }
        dt.val = sp_tensor_host.value(i);
      }
    }

    std::vector<MPI_Request> requests(nprocs - 1);
    std::vector<MPI_Status> statuses(nprocs - 1);
    ttb_indx total_sent = 0;
    for (ttb_indx i = 1; i < nprocs; ++i) {
      // Size to sent to rank i
      const ttb_indx nelements = who_gets_what[i + 1] - who_gets_what[i];
      const ttb_indx nbytes = nelements * dt_size;
      total_sent += nelements;

      const ttb_indx index_of_first_element = who_gets_what[i];
      MPI_Isend(Tvec.data() + index_of_first_element, nbytes, MPI_BYTE, i, i,
                comm, &requests[i - 1]);
    }
    MPI_Waitall(requests.size(), requests.data(), statuses.data());
    ttb_indx total_before = Tvec.size();
    auto begin = Tvec.begin();
    std::advance(begin, who_gets_what[1]); // wgw[0] == 0 always
    Tvec.erase(begin, Tvec.end());
    Tvec.shrink_to_fit(); // Yay now I only have rank 0 data

    ttb_indx total_after = Tvec.size() + total_sent;
    if (total_after != total_before) {
      throw std::logic_error(
        "Genten::distributeTensorToVectorsSparse():  "
        "The number of elements after sending and shrinking (" +
        std::to_string(total_after) +
        ") did not match the input number of elements (" +
        std::to_string(total_before) + ").");
    }
  } else {
    const ttb_indx nelements = who_gets_what[rank + 1] - who_gets_what[rank];
    Tvec.resize(nelements);
    const ttb_indx nbytes = nelements * dt_size;
    MPI_Recv(Tvec.data(), nbytes, MPI_BYTE, 0, rank, comm, MPI_STATUS_IGNORE);
  }

  return Tvec;
}

std::vector<ttb_real>
distributeTensorToVectorsDense(const Tensor& dn_tensor_host, ttb_indx nnz,
                               MPI_Comm comm, ttb_indx rank, ttb_indx nprocs,
                               ttb_indx& offset) {
  constexpr ttb_indx dt_size = sizeof(ttb_real);
  std::vector<ttb_real> Tvec;
  small_vector<ttb_indx> who_gets_what =
      detail::singleDimUniformBlocking(nnz, nprocs);
  offset = who_gets_what[rank];

  if (rank == 0) {
    // Write tensor to form we can MPI_Send more easily.
    Tvec.resize(dn_tensor_host.numel());
    for (ttb_indx i = 0; i < dn_tensor_host.numel(); ++i)
      Tvec[i] = dn_tensor_host[i];

    std::vector<MPI_Request> requests(nprocs - 1);
    std::vector<MPI_Status> statuses(nprocs - 1);
    ttb_indx total_sent = 0;
    for (ttb_indx i = 1; i < nprocs; ++i) {
      // Size to sent to rank i
      const ttb_indx nelements = who_gets_what[i + 1] - who_gets_what[i];
      const ttb_indx nbytes = nelements * dt_size;
      total_sent += nelements;

      const ttb_indx index_of_first_element = who_gets_what[i];
      MPI_Isend(Tvec.data() + index_of_first_element, nbytes, MPI_BYTE, i, i,
                comm, &requests[i - 1]);
    }
    MPI_Waitall(requests.size(), requests.data(), statuses.data());
    ttb_indx total_before = Tvec.size();
    auto begin = Tvec.begin();
    std::advance(begin, who_gets_what[1]); // wgw[0] == 0 always
    Tvec.erase(begin, Tvec.end());
    Tvec.shrink_to_fit(); // Yay now I only have rank 0 data

    ttb_indx total_after = Tvec.size() + total_sent;
    if (total_after != total_before) {
      throw std::logic_error(
        "Genten::distributeTensorToVectorsDense():  "
        "The number of elements after sending and shrinking (" +
        std::to_string(total_after) +
        ") did not match the input number of elements (" +
        std::to_string(total_before) + ").");
    }
  } else {
    const ttb_indx nelements = who_gets_what[rank + 1] - who_gets_what[rank];
    Tvec.resize(nelements);
    const ttb_indx nbytes = nelements * dt_size;
    MPI_Recv(Tvec.data(), nbytes, MPI_BYTE, 0, rank, comm, MPI_STATUS_IGNORE);
  }

  return Tvec;
}

namespace {
ttb_indx blockInThatDim(ttb_indx element, const small_vector<ttb_indx>& range) {
  // const ttb_indx nblocks = range.size();
  if (element >= range.back())
    Genten::error("Tensor nonzero exceeds expected range.  Is your index-base possibly set incorrectly?");
  gt_assert(range.size() >= 2);      // Range always has at least 2 elements

  // We could binary search, which could be faster for large ranges, but I
  // suspect this is fine. Because range.back() is always 1 more than the
  // largest possible element and range.size() >= 2 we don't have to worry
  // about block_guess + 1 going past the end.
  ttb_indx block_guess = 0;
  while (element >= range[block_guess + 1]) {
    ++block_guess;
  }

  return block_guess;
}

// The MPI_Comm must be the one that represents the grid for this to work
template <typename IntType>
ttb_indx rankInGridThatOwns(IntType const *COO, MPI_Comm grid_comm,
                       const std::vector<small_vector<ttb_indx>>& ElementRanges) {
  const ttb_indx ndims = ElementRanges.size();
  small_vector<int> GridPos(ndims);
  for (ttb_indx i = 0; i < ndims; ++i) {
    GridPos[i] = blockInThatDim(COO[i], ElementRanges[i]);
  }

  int rank;
  MPI_Cart_rank(grid_comm, GridPos.data(), &rank);

  return rank;
}
} // namespace

std::vector<SpDataType>
redistributeTensor(const std::vector<SpDataType>& Tvec,
                   const std::vector<ttb_indx>& TDims,
                   const std::vector<small_vector<ttb_indx>>& blocking,
                   const ProcessorMap& pmap) {

  const ttb_indx nprocs = pmap.gridSize();
  MPI_Comm grid_comm = pmap.gridComm();

  std::vector<std::vector<SpDataType>> elems_to_write(nprocs);
  for (auto const &elem : Tvec) {
    ttb_indx elem_owner_rank = rankInGridThatOwns(elem.coo, grid_comm, blocking);
    elems_to_write[elem_owner_rank].push_back(elem);
  }

  small_vector<ttb_indx> amount_to_write(nprocs);
  for (ttb_indx i = 0; i < nprocs; ++i) {
    amount_to_write[i] = elems_to_write[i].size();
  }

  small_vector<ttb_indx> offset_to_write_at(nprocs);
  MPI_Exscan(amount_to_write.data(), offset_to_write_at.data(), nprocs,
             DistContext::toMpiType<ttb_indx>(),
             MPI_SUM, grid_comm);

  ttb_indx amount_to_allocate_for_window = 0;
  MPI_Reduce_scatter_block(amount_to_write.data(),
                           &amount_to_allocate_for_window, 1,
                           DistContext::toMpiType<ttb_indx>(), MPI_SUM,
                           grid_comm);

  if (amount_to_allocate_for_window == 0) {
    const ttb_indx my_rank = pmap.gridRank();
    const ttb_indx ndims = blocking.size();
    std::stringstream ss;
    ss << "WARNING MPI rank(" << my_rank
       << "), received zero nnz in the current blocking.\n\tTensor block: [ ";
    for (ttb_indx i=0; i<ndims; i++)
      ss << pmap.gridCoord(i) << " ";
    ss << "],  range: [ ";
    for (ttb_indx i=0; i<ndims; i++)
      ss << blocking[i][pmap.gridCoord(i)] << " ";
    ss << "] to [ ";
    for (ttb_indx i=0; i<ndims; i++)
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
  SpDataType *data;
  MPI_Win window;
  constexpr ttb_indx DataElemSize = sizeof(SpDataType);
  MPI_Win_allocate(amount_to_allocate_for_window * DataElemSize,
                   /*displacement = */ DataElemSize, MPI_INFO_NULL, grid_comm,
                   &data, &window);

  MPI_Datatype element_type;
  MPI_Type_contiguous(DataElemSize, MPI_BYTE, &element_type);
  MPI_Type_commit(&element_type);

  // Jonathan L. told me for AllToAll Fences are probably better than locking
  // if communications don't conflict
  MPI_Win_fence(0, window);
  for (ttb_indx i = 0; i < nprocs; ++i) {
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
  std::vector<SpDataType> redistributedData(
      data, data + amount_to_allocate_for_window);

  // Free the MPI window and the buffer that it was allocated in
  MPI_Win_free(&window);
  MPI_Type_free(&element_type);
  return redistributedData;
}

std::vector<ttb_real>
redistributeTensor(const std::vector<ttb_real>& Tvec,
                   const ttb_indx global_nnz, const ttb_indx global_offset,
                   const std::vector<ttb_indx>& TDims,
                   const std::vector<small_vector<ttb_indx>>& blocking,
                   const TensorLayout layout,
                   const ProcessorMap& pmap)
{
  const ttb_indx nprocs = pmap.gridSize();
  MPI_Comm grid_comm = pmap.gridComm();

  std::vector<std::vector<ttb_real>> elems_to_write(nprocs);
  const ttb_indx local_nnz = Tvec.size();
  const ttb_indx ndims = TDims.size();
  IndxArray sub(ndims);
  IndxArray siz(ndims);
  for (ttb_indx dim=0; dim<ndims; ++dim)
    siz[dim] = TDims[dim];
  for (ttb_indx i=0; i<local_nnz; ++i) {
    if (layout == TensorLayout::Left)
      Impl::TensorLayoutLeft::ind2sub(sub, siz, global_nnz, i+global_offset);
    else
      Impl::TensorLayoutRight::ind2sub(sub, siz, global_nnz, i+global_offset);
    ttb_indx elem_owner_rank =
      rankInGridThatOwns(sub.values().data(), grid_comm, blocking);
    elems_to_write[elem_owner_rank].push_back(Tvec[i]);
  }

  small_vector<ttb_indx> amount_to_write(nprocs);
  for (ttb_indx i = 0; i < nprocs; ++i) {
    amount_to_write[i] = elems_to_write[i].size();
  }

  small_vector<ttb_indx> offset_to_write_at(nprocs);
  MPI_Exscan(amount_to_write.data(), offset_to_write_at.data(), nprocs,
             DistContext::toMpiType<ttb_indx>(),
             MPI_SUM, grid_comm);

  ttb_indx amount_to_allocate_for_window = 0;
  MPI_Reduce_scatter_block(amount_to_write.data(),
                           &amount_to_allocate_for_window, 1,
                           DistContext::toMpiType<ttb_indx>(), MPI_SUM,
                           grid_comm);

  if (amount_to_allocate_for_window == 0) {
    const ttb_indx my_rank = pmap.gridRank();
    const ttb_indx ndims = blocking.size();
    std::stringstream ss;
    ss << "WARNING MPI rank(" << my_rank
       << "), received zero nnz in the current blocking.\n\tTensor block: [ ";
    for (ttb_indx i=0; i<ndims; i++)
      ss << pmap.gridCoord(i) << " ";
    ss << "],  range: [ ";
    for (ttb_indx i=0; i<ndims; i++)
      ss << blocking[i][pmap.gridCoord(i)] << " ";
    ss << "] to [ ";
    for (ttb_indx i=0; i<ndims; i++)
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
  constexpr ttb_indx DataElemSize = sizeof(ttb_real);
  MPI_Win_allocate(amount_to_allocate_for_window * DataElemSize,
                   /*displacement = */ DataElemSize, MPI_INFO_NULL, grid_comm,
                   &data, &window);

  MPI_Datatype element_type;
  MPI_Type_contiguous(DataElemSize, MPI_BYTE, &element_type);
  MPI_Type_commit(&element_type);

  // Jonathan L. told me for AllToAll Fences are probably better than locking
  // if communications don't conflict
  MPI_Win_fence(0, window);
  for (ttb_indx i = 0; i < nprocs; ++i) {
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

template <typename ExecSpace>
SptensorT<ExecSpace>
DistTensorContext<ExecSpace>::
distributeTensorImpl(const Sptensor& X, const AlgParams& algParams)
{
  // Check tensor has consistent number of dimensions on all processors,
  // or is non-empty only on processor 0
  ttb_indx ndims = X.ndims();
#ifdef HAVE_DIST
  ttb_indx max_ndims = ndims;
  MPI_Allreduce(MPI_IN_PLACE, &max_ndims, 1, DistContext::toMpiType<ttb_indx>(),
                MPI_MAX, DistContext::commWorld());
  if (max_ndims != ndims && ndims != 0)
    Genten::error("Number of tensor dimensions is not consistent across processors!");
  if (ndims == 0 && DistContext::rank() == 0)
    Genten::error("Tensor cannot be empty on rank 0!");
  ndims = max_ndims;
#endif

  // Check if we have already distributed a tensor, in which case this one
  // needs to be of the same size
  if (global_dims_.size() > 0) {
    if (global_dims_.size() != ndims)
      Genten::error("distributeTensor() called twice with different number of dimensions!");
    if (X.ndims() > 0) {
      for (ttb_indx i=0; i<ndims; ++i)
        if (global_dims_[i] != X.size(i))
          Genten::error("distributeTensor() called twice with different sized tensors!");
    }
  }
  else {
    global_dims_.resize(ndims);
    for (ttb_indx i=0; i<ndims; ++i)
      global_dims_[i] = X.ndims() > 0 ? X.size(i) : 0;

#ifdef HAVE_DIST
    std::vector<ttb_indx> max_global_dims = global_dims_;
    MPI_Allreduce(MPI_IN_PLACE, max_global_dims.data(), ndims,
                  DistContext::toMpiType<ttb_indx>(), MPI_MAX,
                  DistContext::commWorld());
    if (X.ndims() > 0 && max_global_dims != global_dims_)
      Genten::error("Tensor dimensions are not consistent across processors!");
    global_dims_ = max_global_dims;
#endif

    if (algParams.proc_grid.size() > 0) {
      gt_assert(algParams.proc_grid.size() == ndims);
      small_vector<ttb_indx> grid(ndims);
      for (ttb_indx i=0; i<ndims; ++i)
        grid[i] = algParams.proc_grid[i];
      pmap_ = std::shared_ptr<ProcessorMap>(new ProcessorMap(global_dims_,
                                                             grid,
                                                             dist_method));
    }
    else
      pmap_ = std::shared_ptr<ProcessorMap>(new ProcessorMap(global_dims_,
                                                             dist_method));
    detail::printGrids(*pmap_);

    global_blocking_ =
      detail::generateUniformBlocking(global_dims_, pmap_->gridDims());

    detail::printBlocking(*pmap_, global_blocking_);
    DistContext::Barrier();
  }

  ttb_indx nnz = pmap_->gridAllReduce(X.nnz(), ProcessorMap::Max);
  auto Tvec = detail::distributeTensorToVectorsSparse(
    X, nnz, pmap_->gridComm(), pmap_->gridRank(), pmap_->gridSize());

  return distributeTensorData(Tvec, global_dims_, global_blocking_, *pmap_,
                              algParams);
}

template <typename ExecSpace>
TensorT<ExecSpace>
DistTensorContext<ExecSpace>::
distributeTensorImpl(const Tensor& X, const AlgParams& algParams)
{
  ttb_indx ndims = X.ndims();
  auto layout = X.getLayout();
#ifdef HAVE_DIST
  // Check layout is consistent across processors (unless empty) by comparing
  // to proc 0's layout (which must be non-empty).  Use proc 0's layout for
  // if tensor is empty on our proc since the code will branch later on layout.
  int int_layout = static_cast<int>(layout);
  DistContext::Bcast(int_layout, 0);
  layout = static_cast<Genten::TensorLayout>(int_layout);
  if (layout != X.getLayout() && ndims > 0)
    Genten::error("Tensor layout is not consistent across processors for non-empty tensors!");

  // Check tensor has consistent number of dimensions on all processors,
  // or is non-empty only on processor 0
  ttb_indx max_ndims = ndims;
  MPI_Allreduce(MPI_IN_PLACE, &max_ndims, 1, DistContext::toMpiType<ttb_indx>(),
                MPI_MAX, DistContext::commWorld());
  if (max_ndims != ndims && ndims != 0)
    Genten::error("Number of tensor dimensions is not consistent across processors!");
  if (ndims == 0 && DistContext::rank() == 0)
    Genten::error("Tensor cannot be empty on rank 0!");
  ndims = max_ndims;
#endif

  // Check if we have already distributed a tensor, in which case this one
  // needs to be of the same size
  if (global_dims_.size() > 0) {
    if (global_dims_.size() != ndims)
      Genten::error("distributeTensor() called twice with different number of dimensions!");
    if (X.ndims() > 0) {
      for (ttb_indx i=0; i<ndims; ++i)
        if (global_dims_[i] != X.size(i))
          Genten::error("distributeTensor() called twice with different sized tensors!");
    }
  }
  else {
    global_dims_.resize(ndims);
    for (ttb_indx i=0; i<ndims; ++i)
      global_dims_[i] = X.ndims() > 0 ? X.size(i) : 0;

#ifdef HAVE_DIST
    std::vector<ttb_indx> max_global_dims = global_dims_;
    MPI_Allreduce(MPI_IN_PLACE, max_global_dims.data(), ndims,
                  DistContext::toMpiType<ttb_indx>(), MPI_MAX,
                  DistContext::commWorld());
    if (X.ndims() > 0 && max_global_dims != global_dims_)
      Genten::error("Tensor dimensions are not consistent across processors!");
    global_dims_ = max_global_dims;
#endif

    if (algParams.proc_grid.size() > 0) {
      gt_assert(algParams.proc_grid.size() == ndims);
      small_vector<ttb_indx> grid(ndims);
      for (ttb_indx i=0; i<ndims; ++i)
        grid[i] = algParams.proc_grid[i];
      pmap_ = std::shared_ptr<ProcessorMap>(new ProcessorMap(global_dims_,
                                                             grid,
                                                             dist_method));
    }
    else
      pmap_ = std::shared_ptr<ProcessorMap>(new ProcessorMap(global_dims_,
                                                             dist_method));

    detail::printGrids(*pmap_);

    global_blocking_ =
      detail::generateUniformBlocking(global_dims_, pmap_->gridDims());

    detail::printBlocking(*pmap_, global_blocking_);
    DistContext::Barrier();
  }

  ttb_indx nnz = pmap_->gridAllReduce(X.nnz(), ProcessorMap::Max);
  ttb_indx offset = 0;
  auto Tvec = detail::distributeTensorToVectorsDense(
    X, nnz, pmap_->gridComm(), pmap_->gridRank(), pmap_->gridSize(), offset);

  return distributeTensorData(Tvec, nnz, offset, global_dims_, global_blocking_,
                              layout, *pmap_, algParams);
}

template <typename ExecSpace>
ttb_real
DistTensorContext<ExecSpace>::
globalNorm(const SptensorT<ExecSpace>& X) const
{
  const auto& values = X.getValArray();
  ttb_real norm2 = values.dot(values);
  norm2 = pmap_->gridAllReduce(norm2);
  return std::sqrt(norm2);
}

template <typename ExecSpace>
ttb_indx
DistTensorContext<ExecSpace>::
globalNNZ(const SptensorT<ExecSpace>& X) const
{
  return pmap_->gridAllReduce(X.nnz());
}

template <typename ExecSpace>
ttb_real
DistTensorContext<ExecSpace>::
globalNumelFloat(const SptensorT<ExecSpace>& X) const
{
  return pmap_->gridAllReduce(X.numel_float());
}

template <typename ExecSpace>
ttb_real
DistTensorContext<ExecSpace>::
globalNorm(const TensorT<ExecSpace>& X) const
{
  const auto& values = X.getValues();
  ttb_real norm2 = values.dot(values);
  norm2 = pmap_->gridAllReduce(norm2);
  return std::sqrt(norm2);
}

template <typename ExecSpace>
ttb_indx
DistTensorContext<ExecSpace>::
globalNNZ(const TensorT<ExecSpace>& X) const
{
  return pmap_->gridAllReduce(X.nnz());
}

template <typename ExecSpace>
ttb_real
DistTensorContext<ExecSpace>::
globalNumelFloat(const TensorT<ExecSpace>& X) const
{
  return pmap_->gridAllReduce(X.numel_float());
}

template <typename ExecSpace>
ttb_real
DistTensorContext<ExecSpace>::
globalNorm(const KtensorT<ExecSpace>& u) const
{
  return std::sqrt(u.normFsq());
}

template <typename ExecSpace>
void
DistTensorContext<ExecSpace>::
allReduce(KtensorT<ExecSpace>& u, const bool divide_by_grid_size) const
{
  const ttb_indx nd = u.ndims();
  gt_assert(ttb_indx(global_dims_.size()) == nd);

  for (ttb_indx n=0; n<nd; ++n)
    pmap_->subGridAllReduce(
      n, u[n].view().data(), u[n].view().span());

  if (divide_by_grid_size) {
    auto const &gridSizes = pmap_->subCommSizes();
    for (ttb_indx n=0; n<nd; ++n) {
      const ttb_real scale = ttb_real(1.0 / gridSizes[n]);
      u[n].times(scale);
    }
  }
}

template <typename ExecSpace>
std::tuple< SptensorT<ExecSpace>, TensorT<ExecSpace> >
DistTensorContext<ExecSpace>::
distributeTensor(const std::string& file, const ttb_indx index_base,
                 const bool compressed, const ptree& tree,
                 const AlgParams& algParams)
{
  dist_method = algParams.dist_update_method;

  SptensorT<ExecSpace> X_sparse;
  TensorT<ExecSpace> X_dense;
  TensorReader<Genten::DefaultHostExecutionSpace> reader(
    file, index_base, compressed, tree);

  std::vector<SpDataType> Tvec_sparse;
  std::vector<double> Tvec_dense;
  std::vector<ttb_indx> global_dims;
  ttb_indx nnz, offset;

  if (DistContext::rank() == 0)
    std::cout << "Reading tensor from file " << file << std::endl;

  DistContext::Barrier();
  auto t2 = MPI_Wtime();
  bool parallel_read = tree.get<bool>("parallel-read", true);
  if (parallel_read && reader.isBinary()) {
    if (reader.isSparse())
      Tvec_sparse = reader.parallelReadBinarySparse(global_dims, nnz);
    else
      Tvec_dense = reader.parallelReadBinaryDense(global_dims, nnz, offset);
  }
  else if (parallel_read && reader.isExodus()) {
#if defined(HAVE_TPETRA) && defined(HAVE_SEACAS)
      Tvec_dense = reader.parallelReadExodusDense(global_dims, nnz, offset);
#else
      Genten::error("parallel exodus read available only with tpetra enabled");
#endif
  }
  else {
    // For non-binary, read on rank 0 and broadcast dimensions.
    // We do this instead of reading the header because we want to support
    // headerless files
    Tensor X_dense_host;
    Sptensor X_sparse_host;
    std::size_t ndims;
    small_vector<ttb_indx> dims;

    if (gridRank() == 0) {
      reader.read();
      if (reader.isSparse()) {
        X_sparse_host = reader.getSparseTensor();
        nnz = X_sparse_host.nnz();
        ndims = X_sparse_host.ndims();
        dims = small_vector<ttb_indx>(ndims);
        for (std::size_t i=0; i<ndims; ++i)
          dims[i] = X_sparse_host.size(i);
      }
      else {
        X_dense_host = reader.getDenseTensor();
        nnz = X_dense_host.nnz();
        ndims = X_dense_host.ndims();
        dims = small_vector<ttb_indx>(ndims);
        for (std::size_t i=0; i<ndims; ++i)
          dims[i] = X_dense_host.size(i);
      }
    }
    DistContext::Bcast(nnz, 0);
    DistContext::Bcast(ndims, 0);
    if (gridRank() != 0)
      dims = small_vector<ttb_indx>(ndims);
    DistContext::Bcast(dims, 0);

    global_dims = std::vector<ttb_indx>(ndims);
    for (std::size_t i=0; i<ndims; ++i)
      global_dims[i] = dims[i];

    if (reader.isDense())
      Tvec_dense = detail::distributeTensorToVectorsDense(
        X_dense_host, nnz, DistContext::commWorld(), DistContext::rank(),
        DistContext::nranks(), offset);
    else
      Tvec_sparse = detail::distributeTensorToVectorsSparse(
        X_sparse_host, nnz, DistContext::commWorld(), DistContext::rank(),
        DistContext::nranks());
  }
  DistContext::Barrier();
  auto t3 = MPI_Wtime();
  if (gridRank() == 0) {
    std::cout << "  Read file in: " << t3 - t2 << "s" << std::endl;
  }

  // Check if we have already distributed a tensor, in which case this one
  // needs to be of the same size
  const ttb_indx ndims = global_dims.size();
  if (global_dims_.size() > 0) {
    if (global_dims_.size() != ndims)
      Genten::error("distributeTensor() called twice with different number of dimensions!");
    for (ttb_indx i=0; i<ndims; ++i)
      if (global_dims_[i] != global_dims[i])
          Genten::error("distributeTensor() called twice with different sized tensors!");
  }
  else {
    global_dims_ = global_dims;

    if (algParams.proc_grid.size() > 0) {
      gt_assert(algParams.proc_grid.size() == ndims);
      small_vector<ttb_indx> grid(ndims);
      for (ttb_indx i=0; i<ndims; ++i)
        grid[i] = algParams.proc_grid[i];
      pmap_ = std::shared_ptr<ProcessorMap>(new ProcessorMap(global_dims_,
                                                             grid,
                                                             dist_method));
    }
    else
      pmap_ = std::shared_ptr<ProcessorMap>(new ProcessorMap(global_dims_,
                                                             dist_method));
    detail::printGrids(*pmap_);

    global_blocking_ =
      detail::generateUniformBlocking(global_dims_, pmap_->gridDims());

    detail::printBlocking(*pmap_, global_blocking_);
  }

  if (reader.isDense())
    X_dense = distributeTensorData(Tvec_dense, nnz, offset,
                                   global_dims_, global_blocking_,
                                   TensorLayout::Left,
                                   *pmap_, algParams,
				   !reader.isExodus());
  else
    X_sparse = distributeTensorData(Tvec_sparse, global_dims_, global_blocking_,
                                    *pmap_, algParams);
  return std::make_tuple(X_sparse, X_dense);
}

template <typename ExecSpace>
SptensorT<ExecSpace>
DistTensorContext<ExecSpace>::
distributeTensorData(const std::vector<SpDataType>& Tvec,
                     const std::vector<ttb_indx>& TensorDims,
                     const std::vector<small_vector<ttb_indx>>& blocking,
                     const ProcessorMap& pmap, const AlgParams& algParams)
{
  const bool use_tpetra =
    algParams.dist_update_method == Genten::Dist_Update_Method::Tpetra;

  DistContext::Barrier();
  auto t4 = MPI_Wtime();

  // Now redistribute to final format
  auto distributedData =
    detail::redistributeTensor(Tvec, global_dims_, global_blocking_, *pmap_);

  DistContext::Barrier();
  auto t5 = MPI_Wtime();

  if (algParams.timings && gridRank() == 0) {
    std::cout << "  Redistributed tensor in: " << t5 - t4 << "s" << std::endl;
  }

  std::vector<detail::RangePair> range;
  ttb_indx ndims = TensorDims.size();
  for (ttb_indx i = 0; i < ndims; ++i) {
    ttb_indx coord = pmap_->gridCoord(i);
    range.push_back({global_blocking_[i][coord],
                      global_blocking_[i][coord + 1]});
  }

  std::vector<ttb_indx> indices(ndims);
  local_dims_.resize(ndims);
  for (ttb_indx i = 0; i < ndims; ++i) {
    auto const &rpair = range[i];
    indices[i] = rpair.upper - rpair.lower;
    local_dims_[i] = indices[i];
  }

  const ttb_indx local_nnz = distributedData.size();
  std::vector<ttb_real> values(local_nnz);
  std::vector<std::vector<ttb_indx>> subs(local_nnz);
  for (ttb_indx i = 0; i < local_nnz; ++i) {
    auto data = distributedData[i];
    values[i] = data.val;
    subs[i] = std::vector<ttb_indx>(data.coo, data.coo + ndims);

    // Do not subtract off the lower bound of the bounding box for Tpetra
    // since it will map GIDs to LIDs below
    if (!use_tpetra)
      for (ttb_indx j = 0; j < ndims; ++j)
        subs[i][j] -= range[j].lower;
  }

  SptensorT<ExecSpace> sptensor;
  if (!use_tpetra) {
    Sptensor sptensor_host(indices, values, subs);
    sptensor = create_mirror_view(ExecSpace(), sptensor_host);
    deep_copy(sptensor, sptensor_host);
  }

#ifdef HAVE_TPETRA
  // Setup Tpetra parallel maps
  if (use_tpetra) {
    const tpetra_go_type indexBase = tpetra_go_type(0);
    const Tpetra::global_size_t invalid =
      Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    tpetra_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(pmap_->gridComm()));

    // Distribute each factor matrix uniformly across all processors
    // ToDo:  consider possibly not doing this when the number of rows is
    // small.  It might be better to replicate rows instead
    factorMap.resize(ndims);
    for (ttb_indx dim=0; dim<ndims; ++dim) {
      const tpetra_go_type numGlobalElements = global_dims_[dim];
      factorMap[dim] = Teuchos::rcp(new tpetra_map_type<ExecSpace>(numGlobalElements, indexBase, tpetra_comm));
    }

    // Build hash maps of tensor nonzeros in each dimension for:
    //   1.  Mapping tensor GIDs to LIDS
    //   2.  Constructing overlapping Tpetra map for MTTKRP
    using unordered_map_type = Kokkos::UnorderedMap<tpetra_go_type,tpetra_lo_type,DefaultHostExecutionSpace>;
    std::vector<unordered_map_type> map(ndims);
    std::vector<tpetra_lo_type> cnt(ndims, 0);
    for (ttb_indx dim=0; dim<ndims; ++dim)
      map[dim].rehash(local_dims_[dim]);
    for (ttb_indx i=0; i<local_nnz; ++i) {
      for (ttb_indx dim=0; dim<ndims; ++dim) {
        ttb_indx gid = subs[i][dim];
        ttb_indx idx = map[dim].find(gid);
        if (!map[dim].valid_at(idx)) {
          tpetra_lo_type lid = cnt[dim]++;
          if (map[dim].insert(gid,lid).failed())
            Genten::error("Insertion of GID failed, something is wrong!");
        }
      }
    }
    for (ttb_indx dim=0; dim<ndims; ++dim)
      gt_assert(cnt[dim] == tpetra_lo_type(map[dim].size()));

    // Map tensor GIDs to LIDs.  We use the hash-map for this instead of just
    // subtracting off the lower bound because there may be empty slices
    // in our block (and LIDs must be contiguous)
    std::vector<std::vector<ttb_indx>> subs_gids(local_nnz);
    for (ttb_indx i=0; i<local_nnz; ++i) {
      subs_gids[i].resize(ndims);
      for (ttb_indx dim=0; dim<ndims; ++dim) {
        const ttb_indx gid = subs[i][dim];
        const ttb_indx idx = map[dim].find(gid);
        const ttb_indx lid = map[dim].value_at(idx);
        subs[i][dim] = lid;
        subs_gids[i][dim] = gid;
      }
    }

    // Construct overlap maps for each dimension
    overlapFactorMap.resize(ndims);
    for (ttb_indx dim=0; dim<ndims; ++dim) {
      Kokkos::View<tpetra_go_type*,ExecSpace> gids("gids", cnt[dim]);
      auto gids_host = create_mirror_view(gids);
      const ttb_indx sz = map[dim].capacity();
      for (ttb_indx idx=0; idx<sz; ++idx) {
        if (map[dim].valid_at(idx)) {
          const ttb_indx gid = map[dim].key_at(idx);
          const ttb_indx lid = map[dim].value_at(idx);
          gids_host[lid] = gid;
        }
      }
      deep_copy(gids, gids_host);
      overlapFactorMap[dim] =
        Teuchos::rcp(new tpetra_map_type<ExecSpace>(invalid, gids, indexBase,
                                                    tpetra_comm));
      indices[dim] = overlapFactorMap[dim]->getLocalNumElements();

      if (algParams.optimize_maps) {
        bool err = false;
        overlapFactorMap[dim] = Tpetra::Details::makeOptimizedColMap(
          std::cerr, err, *factorMap[dim], *overlapFactorMap[dim]);
        if (err)
          Genten::error("Tpetra::Details::makeOptimizedColMap failed!");
        for (ttb_indx i=0; i<local_nnz; ++i)
          subs[i][dim] =
            overlapFactorMap[dim]->getLocalElement(subs_gids[i][dim]);
      }
    }

    // Build sparse tensor
    std::vector<ttb_indx> lower(ndims), upper(ndims);
    for (ttb_indx dim=0; dim<ndims; ++dim) {
      lower[dim] = range[dim].lower;
      upper[dim] = range[dim].upper;
    }
    Sptensor sptensor_host(indices, values, subs, subs_gids, lower, upper);
    sptensor = create_mirror_view(ExecSpace(), sptensor_host);
    deep_copy(sptensor, sptensor_host);
    for (ttb_indx dim=0; dim<ndims; ++dim) {
      sptensor.factorMap(dim) = factorMap[dim];
      sptensor.tensorMap(dim) = overlapFactorMap[dim];
      if (!overlapFactorMap[dim]->isSameAs(*factorMap[dim]))
        sptensor.importer(dim) =
          Teuchos::rcp(new tpetra_import_type<ExecSpace>(
                         factorMap[dim], overlapFactorMap[dim]));
    }
  }
#else
  if (use_tpetra)
    Genten::error("Cannot use tpetra distribution approach without enabling Tpetra!");
#endif

  // Compute local dimensions of compatible factor matrices
  ktensor_local_dims_.resize(ndims);
  ktensor_local_offsets_.resize(ndims);
  for (ttb_indx n=0; n<ndims; ++n) {
    if (algParams.dist_update_method == Dist_Update_Method::AllReduce)
    {
      ktensor_local_dims_[n] = local_dims_[n];
      const ttb_indx coord = pmap_->gridCoord(n);
      ktensor_local_offsets_[n] = global_blocking_[n][coord];
    }
    else if (
      algParams.dist_update_method == Dist_Update_Method::AllGatherReduce ||
      algParams.dist_update_method == Dist_Update_Method::OneSided ||
      algParams.dist_update_method == Dist_Update_Method::TwoSided)
    {
      // Distributed ktensor in blocks across subgrid layers, then
      // distribute each block uniformly across procs in the layer
      const ttb_indx procs_in_layer = pmap_->subCommSize(n);
      const ttb_indx my_proc = pmap_->subCommRank(n);
      const ttb_indx rows_in_layer = local_dims_[n];
      ttb_indx num_my_rows = rows_in_layer / procs_in_layer;
      const ttb_indx rem = rows_in_layer - num_my_rows*procs_in_layer;

      // Distribute remainder across the first rem procs
      if (my_proc < rem)
        ++num_my_rows;

      // Compute local offset
      std::vector<ttb_indx> local_sizes(procs_in_layer);
      local_sizes[my_proc] = num_my_rows;
      pmap_->subGridAllGather(n, local_sizes.data(), 1);
      ttb_indx my_offset = 0;
      for (unsigned proc=0; proc<my_proc; ++proc)
        my_offset += local_sizes[proc];

      // Add offset from other layers
      const ttb_indx coord = pmap_->gridCoord(n);
      my_offset += global_blocking_[n][coord];

      ktensor_local_dims_[n] = num_my_rows;
      ktensor_local_offsets_[n] = my_offset;
    }
#ifdef HAVE_TPETRA
    else if (algParams.dist_update_method == Genten::Dist_Update_Method::Tpetra)
    {
      ktensor_local_dims_[n] = factorMap[n]->getLocalNumElements();
      ktensor_local_offsets_[n] = factorMap[n]->getGlobalElement(0);
    }
#endif
    else
      Genten::error(std::string("Unknown distributed-guess method: ") +
                    Dist_Update_Method::names[algParams.dist_update_method]);
  }

  if (DistContext::isDebug()) {
    if (gridRank() == 0) {
      std::cout << "MPI Ranks in each dimension: ";
      for (auto p : pmap_->subCommSizes()) {
        std::cout << p << " ";
      }
      std::cout << std::endl;
    }
    DistContext::Barrier();
  }

  sptensor.setProcessorMap(&pmap);
  return sptensor;
}

template <typename ExecSpace>
TensorT<ExecSpace>
DistTensorContext<ExecSpace>::
distributeTensorData(const std::vector<ttb_real>& Tvec,
                     const ttb_indx global_nnz, const ttb_indx global_offset,
                     const std::vector<ttb_indx>& TensorDims,
                     const std::vector<small_vector<ttb_indx>>& blocking,
                     const TensorLayout layout,
                     const ProcessorMap& pmap, const AlgParams& algParams,
		     bool redistribute_needed)
{
  const bool use_tpetra =
    algParams.dist_update_method == Genten::Dist_Update_Method::Tpetra;

  DistContext::Barrier();
  auto t4 = MPI_Wtime();

  // Now redistribute to final format
  // exodus reads already follow a uniform nodal distribution
  std::vector<ttb_real> values;
  if(redistribute_needed)
    values =
      detail::redistributeTensor(Tvec, global_nnz, global_offset,
                                 global_dims_, global_blocking_, layout, *pmap_);
  else
    values = Tvec;

  DistContext::Barrier();
  auto t5 = MPI_Wtime();

  if (algParams.timings && gridRank() == 0) {
    std::cout << "  Redistributed tensor in: " << t5 - t4 << "s" << std::endl;
  }

  std::vector<detail::RangePair> range;
  ttb_indx ndims = TensorDims.size();
  for (ttb_indx i = 0; i < ndims; ++i) {
    ttb_indx coord = pmap_->gridCoord(i);
    range.push_back({global_blocking_[i][coord],
                      global_blocking_[i][coord + 1]});
  }

  std::vector<ttb_indx> indices(ndims);
  local_dims_.resize(ndims);
  for (ttb_indx i = 0; i < ndims; ++i) {
    auto const &rpair = range[i];
    indices[i] = rpair.upper - rpair.lower;
    local_dims_[i] = indices[i];
  }

  const ttb_indx local_nnz = values.size();

  TensorT<ExecSpace> tensor;
  if (!use_tpetra) {
    Tensor tensor_host(IndxArray(ndims, indices.data()),
                       Array(local_nnz, values.data(), false),
                       layout);
    tensor = create_mirror_view(ExecSpace(), tensor_host);
    deep_copy(tensor, tensor_host);
  }

#ifdef HAVE_TPETRA
  // Setup Tpetra parallel maps
  if (use_tpetra) {
    const tpetra_go_type indexBase = tpetra_go_type(0);
    const Tpetra::global_size_t invalid =
      Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    tpetra_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(pmap_->gridComm()));

    // Distribute each factor matrix uniformly across all processors
    // ToDo:  consider possibly not doing this when the number of rows is
    // small.  It might be better to replicate rows instead
    factorMap.resize(ndims);
    for (ttb_indx dim=0; dim<ndims; ++dim) {
      const tpetra_go_type numGlobalElements = global_dims_[dim];
      factorMap[dim] = Teuchos::rcp(new tpetra_map_type<ExecSpace>(numGlobalElements, indexBase, tpetra_comm));
    }

    // Build tensor maps based on slices owned by each processor
    overlapFactorMap.resize(ndims);
    for (ttb_indx dim=0; dim<ndims; ++dim) {
      const ttb_indx sz = local_dims_[dim];
      Kokkos::View<tpetra_go_type*,ExecSpace> gids("gids", sz);
      auto gids_host = create_mirror_view(gids);
      const ttb_indx l = range[dim].lower;
      for (ttb_indx idx=0; idx<sz; ++idx)
        gids_host[idx] = l + idx;
      deep_copy(gids, gids_host);
      overlapFactorMap[dim] =
        Teuchos::rcp(new tpetra_map_type<ExecSpace>(invalid, gids, indexBase,
                                                    tpetra_comm));
    }

    // Build dense tensor
    Tensor tensor_host(IndxArray(ndims, indices.data()),
                       Array(local_nnz, values.data(), false),
                       layout);
    IndxArray lower = tensor_host.getLowerBounds();
    IndxArray upper = tensor_host.getUpperBounds();
    for (ttb_indx dim=0; dim<ndims; ++dim) {
      lower[dim] = range[dim].lower;
      upper[dim] = range[dim].upper;
    }
    tensor = create_mirror_view(ExecSpace(), tensor_host);
    deep_copy(tensor, tensor_host);
    for (ttb_indx dim=0; dim<ndims; ++dim) {
      tensor.factorMap(dim) = factorMap[dim];
      tensor.tensorMap(dim) = overlapFactorMap[dim];
      if (!overlapFactorMap[dim]->isSameAs(*factorMap[dim]))
        tensor.importer(dim) =
          Teuchos::rcp(new tpetra_import_type<ExecSpace>(
                         factorMap[dim], overlapFactorMap[dim]));
    }
  }
#else
  if (use_tpetra)
    Genten::error("Cannot use tpetra distribution approach without enabling Tpetra!");
#endif

  // Compute local dimensions of compatible factor matrices
  ktensor_local_dims_.resize(ndims);
  ktensor_local_offsets_.resize(ndims);
  for (ttb_indx n=0; n<ndims; ++n) {
    if (algParams.dist_update_method == Dist_Update_Method::AllReduce)
    {
      ktensor_local_dims_[n] = local_dims_[n];
      const ttb_indx coord = pmap_->gridCoord(n);
      ktensor_local_offsets_[n] = global_blocking_[n][coord];
    }
    else if (
      algParams.dist_update_method == Dist_Update_Method::AllGatherReduce ||
      algParams.dist_update_method == Dist_Update_Method::OneSided ||
      algParams.dist_update_method == Dist_Update_Method::TwoSided)
    {
      const ttb_indx procs_in_layer = pmap_->subCommSize(n);
      const ttb_indx my_proc = pmap_->subCommRank(n);
      const ttb_indx rows_in_layer = local_dims_[n];
      ttb_indx num_my_rows = rows_in_layer / procs_in_layer;
      const ttb_indx rem = rows_in_layer - num_my_rows*procs_in_layer;

      // Distribute remainder across the first rem procs
      if (my_proc < rem)
        ++num_my_rows;

      // Compute local offset
      std::vector<ttb_indx> local_sizes(procs_in_layer);
      local_sizes[my_proc] = num_my_rows;
      pmap_->subGridAllGather(n, local_sizes.data(), 1);
      ttb_indx my_offset = 0;
      for (unsigned proc=0; proc<my_proc; ++proc)
        my_offset += local_sizes[proc];

      // Add offset from other layers
      const ttb_indx coord = pmap_->gridCoord(n);
      my_offset += global_blocking_[n][coord];

      ktensor_local_dims_[n] = num_my_rows;
      ktensor_local_offsets_[n] = my_offset;
    }
#ifdef HAVE_TPETRA
    else if (algParams.dist_update_method == Genten::Dist_Update_Method::Tpetra)
    {
      ktensor_local_dims_[n] = factorMap[n]->getLocalNumElements();
      ktensor_local_offsets_[n] = factorMap[n]->getGlobalElement(0);
    }
#endif
    else
      Genten::error(std::string("Unknown distributed ktensor method: ") +
                    Dist_Update_Method::names[algParams.dist_update_method]);
  }

  if (DistContext::isDebug()) {
    if (gridRank() == 0) {
      std::cout << "MPI Ranks in each dimension: ";
      for (auto p : pmap_->subCommSizes()) {
        std::cout << p << " ";
      }
      std::cout << std::endl;
    }
    DistContext::Barrier();
  }

  tensor.setProcessorMap(&pmap);
  return tensor;
}

#else

template <typename ExecSpace>
std::tuple< SptensorT<ExecSpace>, TensorT<ExecSpace> >
DistTensorContext<ExecSpace>::
distributeTensor(const std::string& file,
                 const ttb_indx index_base,
                 const bool compressed,
                 const ptree& tree,
                 const AlgParams& algParams)
{
  dist_method = algParams.dist_update_method;

  SptensorT<ExecSpace> X_sparse;
  TensorT<ExecSpace> X_dense;
  Genten::TensorReader<ExecSpace> reader(file, index_base, compressed, tree);
  reader.read();
  if (reader.isSparse()) {
    X_sparse = reader.getSparseTensor();
    const ttb_indx nd = X_sparse.ndims();
    global_dims_.resize(nd);
    ktensor_local_dims_.resize(nd);
    for (ttb_indx i=0; i<nd; ++i) {
      global_dims_[i] = X_sparse.size(i);
      ktensor_local_dims_[i] = X_sparse.size(i);
    }
  }
  else if (reader.isDense()) {
    X_dense = reader.getDenseTensor();
    const ttb_indx nd = X_dense.ndims();
    global_dims_.resize(nd);
    ktensor_local_dims_.resize(nd);
    for (ttb_indx i=0; i<nd; ++i) {
      global_dims_[i] = X_dense.size(i);
      ktensor_local_dims_[i] = X_dense.size(i);
    }
  }
  else
    Genten::error("Tensor is neither sparse nor dense, something is wrong!");
  return std::make_tuple(X_sparse, X_dense);
}

#endif

template <typename ExecSpace>
std::tuple< SptensorT<ExecSpace>, TensorT<ExecSpace> >
DistTensorContext<ExecSpace>::
distributeTensor(const ptree& tree,
                 const AlgParams& algParams)
{
  std::string inputfilename = "";
  ttb_indx index_base = 0;
  ttb_bool gz = false;
  auto tensor_input_o = tree.get_child_optional("tensor");
  if (tensor_input_o) {
    auto& tensor_input = *tensor_input_o;
    Genten::parse_ptree_value(tensor_input, "input-file", inputfilename);
    Genten::parse_ptree_value(tensor_input, "index-base", index_base, 0, INT_MAX);
    Genten::parse_ptree_value(tensor_input, "compressed", gz);
  }
  return distributeTensor(inputfilename, index_base, gz, tensor_input_o,
                          algParams);
}

template <typename ExecSpace>
void
DistTensorContext<ExecSpace>::
exportToFile(const KtensorT<ExecSpace>& u, const std::string& file_name) const
{
  auto out = importToRoot<Genten::DefaultHostExecutionSpace>(u);
  if (pmap_->gridRank() == 0) {
    // Normalize Ktensor u before writing out
    out.normalize(Genten::NormTwo);
    out.arrange();

    std::cout << "Saving final Ktensor to " << file_name << std::endl;
    Genten::export_ktensor(file_name, out);
  }
}

template <typename ExecSpace>
KtensorT<ExecSpace>
DistTensorContext<ExecSpace>::
readInitialGuess(const std::string& file_name) const
{
  KtensorT<DefaultHostExecutionSpace> u_host;
  import_ktensor(file_name, u_host);
  KtensorT<ExecSpace> u = create_mirror_view(ExecSpace(), u_host);
  deep_copy(u, u_host);
  return exportFromRoot(u);
}

template <typename ExecSpace>
KtensorT<ExecSpace>
DistTensorContext<ExecSpace>::
randomInitialGuess(const SptensorT<ExecSpace>& X,
                   const ttb_indx rank,
                   const ttb_indx seed,
                   const bool prng,
                   const bool scale_guess_by_norm_x,
                   const std::string& dist_guess_method) const
{
  const ttb_indx nd = X.ndims();
  const ttb_real norm_x = globalNorm(X);
  RandomMT cRMT(seed);

  Genten::KtensorT<ExecSpace> u;

  if (dist_guess_method == "serial") {
    // Compute random ktensor on rank 0 and broadcast to all proc's
    IndxArrayT<ExecSpace> sz(nd);
    auto hsz = create_mirror_view(sz);
    for (ttb_indx i=0; i<nd; ++i)
      hsz[i] = global_dims_[i];
    deep_copy(sz,hsz);
    Genten::KtensorT<ExecSpace> u0(rank, nd, sz);
    if (pmap_->gridRank() == 0) {
      u0.setWeights(1.0);
      u0.setMatricesScatter(false, prng, cRMT);
    }
    u = exportFromRoot(u0);
  }
  else if (dist_guess_method == "parallel" ||
           dist_guess_method == "parallel-drew") {
    const ttb_indx nd = X.ndims();
    IndxArrayT<ExecSpace> sz(nd);
    auto hsz = create_mirror_view(sz);
    for (ttb_indx i=0; i<nd; ++i)
      hsz[i] = ktensor_local_dims_[i];
    deep_copy(sz,hsz);
    u = KtensorT<ExecSpace>(rank, nd, sz);
    u.setWeights(1.0);
    u.setMatricesScatter(false, prng, cRMT);
    u.setProcessorMap(&pmap());
    if (dist_method == Dist_Update_Method::AllReduce) {
      allReduce(u, true); // make replicated proc's consistent
    }
  }
  else
    Genten::error("Unknown distributed-guess method: " + dist_guess_method);

  if (dist_guess_method == "parallel-drew")
    u.weights().times(1.0 / norm_x); // don't understand this
  else {
    const ttb_real norm_u = globalNorm(u);
    const ttb_real scale =
      scale_guess_by_norm_x ? norm_x / norm_u : ttb_real(1.0) / norm_u;
    u.weights().times(scale);
  }
  u.distribute(); // distribute weights across factor matrices
  return u;
}

template <typename ExecSpace>
KtensorT<ExecSpace>
DistTensorContext<ExecSpace>::
randomInitialGuess(const TensorT<ExecSpace>& X,
                   const ttb_indx rank,
                   const ttb_indx seed,
                   const bool prng,
                   const bool scale_guess_by_norm_x,
                   const std::string& dist_guess_method) const
{
  const ttb_indx nd = X.ndims();
  const ttb_real norm_x = globalNorm(X);
  RandomMT cRMT(seed);

  Genten::KtensorT<ExecSpace> u;

  if (dist_guess_method == "serial") {
    // Compute random ktensor on rank 0 and broadcast to all proc's
    IndxArrayT<ExecSpace> sz(nd);
    auto hsz = create_mirror_view(sz);
    for (ttb_indx i=0; i<nd; ++i)
      hsz[i] = global_dims_[i];
    deep_copy(sz,hsz);
    Genten::KtensorT<ExecSpace> u0(rank, nd, sz);
    if (pmap_->gridRank() == 0) {
      u0.setWeights(1.0);
      u0.setMatricesScatter(false, prng, cRMT);
    }
    u = exportFromRoot(u0);
  }
  else if (dist_guess_method == "parallel" ||
           dist_guess_method == "parallel-drew") {
    const ttb_indx nd = X.ndims();
    IndxArrayT<ExecSpace> sz(nd);
    auto hsz = create_mirror_view(sz);
    for (ttb_indx i=0; i<nd; ++i)
      hsz[i] = ktensor_local_dims_[i];
    deep_copy(sz,hsz);
    u = KtensorT<ExecSpace>(rank, nd, sz);
    u.setWeights(1.0);
    u.setMatricesScatter(false, prng, cRMT);
    u.setProcessorMap(&pmap());
    if (dist_method == Dist_Update_Method::AllReduce) {
      allReduce(u, true); // make replicated proc's consistent
    }
  }
  else
    Genten::error("Unknown distributed-guess method: " + dist_guess_method);

  if (dist_guess_method == "parallel-drew")
    u.weights().times(1.0 / norm_x); // don't understand this
  else {
    const ttb_real norm_u = globalNorm(u);
    const ttb_real scale =
      scale_guess_by_norm_x ? norm_x / norm_u : ttb_real(1.0) / norm_u;
    u.weights().times(scale);
  }
  u.distribute(); // distribute weights across factor matrices
  return u;
}

template <typename ExecSpace>
KtensorT<ExecSpace>
DistTensorContext<ExecSpace>::
computeInitialGuess(const SptensorT<ExecSpace>& X, const ptree& input) const
{
  KtensorT<ExecSpace> u;

  auto kt_input = input.get_child("k-tensor");
  std::string init_method = kt_input.get<std::string>("initial-guess", "rand");
  if (init_method == "file") {
    std::string file_name = kt_input.get<std::string>("initial-file");
    u = readInitialGuess(file_name);
  }
  else if (init_method == "rand") {
    const ttb_indx seed = kt_input.get<int>("seed",std::random_device{}());
    const bool prng = kt_input.get<bool>("prng",true);
    const bool scale_by_x = kt_input.get<bool>("scale-guess-by-norm-x", false);
    const ttb_indx nc = kt_input.get<int>("rank");
    const std::string dist_method =
      kt_input.get<std::string>("distributed-guess", "serial");
    u = randomInitialGuess(X, nc, seed, prng, scale_by_x, dist_method);
  }
  else
    Genten::error("Unknown initial-guess method: " + init_method);

  return u;
}

} // namespace Genten

#define INST_MACRO(SPACE) \
  template class Genten::DistTensorContext<SPACE>;

GENTEN_INST(INST_MACRO)
