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

#pragma once

#include <random>

#include "Genten_Ptree.hpp"
#include "Genten_SmallVector.hpp"
#include "Genten_DistContext.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Pmap.hpp"
#include "Genten_Sptensor.hpp"

#include "CMakeInclude.h"
#if defined(HAVE_DIST)
#include <cmath>
#include <fstream>
#include <memory>
#include "Genten_MPI_IO.hpp"
#include "Genten_SpTn_Util.hpp"
#endif

namespace Genten {

#if defined(HAVE_DIST)

namespace detail {
void printGrids(const ProcessorMap& pmap);
void printBlocking(const ProcessorMap& pmap,
                   const std::vector<small_vector<int>>& blocking);

std::vector<small_vector<int>>
generateUniformBlocking(const std::vector<std::uint32_t>& ModeLengths,
                        const small_vector<int>& ProcGridSizes);
} // namespace detail

// Class to describe and manipulate tensor data in a distributed context
class DistTensorContext {
public:
  DistTensorContext() = default;
  ~DistTensorContext() = default;

  DistTensorContext(DistTensorContext&&) = default;
  DistTensorContext(const DistTensorContext&) = default;
  DistTensorContext& operator=(DistTensorContext&&) = default;
  DistTensorContext& operator=(const DistTensorContext&) = default;

  template <typename ExecSpace>
  SptensorT<ExecSpace> distributeTensor(const ptree& tree);
  template <typename ExecSpace>
  SptensorT<ExecSpace> distributeTensor(const std::string& file,
                                        const ttb_indx index_base,
                                        const bool compressed);
  template <typename ExecSpaceDst, typename ExecSpaceSrc>
  SptensorT<ExecSpaceDst> distributeTensor(const SptensorT<ExecSpaceSrc>& X);

  // Parallel info
  std::int32_t ndims() const { return global_dims_.size(); }
  const std::vector<std::uint32_t>& dims() const { return global_dims_; }
  static std::int64_t nprocs() { return DistContext::nranks(); }
  static std::int64_t gridRank() { return DistContext::rank(); }
  const std::vector<small_vector<int>>& blocking() const { return global_blocking_; }

  // Processor map for communication
  const ProcessorMap& pmap() const { return *pmap_; }
  std::shared_ptr<const ProcessorMap> pmap_ptr() const { return pmap_; }

  // Sptensor operations
  template <typename ExecSpace>
  ttb_real globalNorm(const SptensorT<ExecSpace>& X) const;
  template <typename ExecSpace>
  std::uint64_t globalNNZ(const SptensorT<ExecSpace>& X) const;
  template <typename ExecSpace>
  ttb_real globalNumelFloat(const SptensorT<ExecSpace>& X) const;

  // Ktensor operations
  template <typename ExecSpace>
  ttb_real globalNorm(const KtensorT<ExecSpace>& X) const;
  template <typename ExecSpaceDst, typename ExecSpaceSrc>
  KtensorT<ExecSpaceDst> exportFromRoot(const KtensorT<ExecSpaceSrc>& u) const;
  template <typename ExecSpaceDst, typename ExecSpaceSrc>
  KtensorT<ExecSpaceDst> importToRoot(const KtensorT<ExecSpaceSrc>& u) const;
  template <typename ExecSpace>
  void allReduce(KtensorT<ExecSpace>& u,
                 const bool divide_by_grid_size = false) const;
  template <typename ExecSpace>
  void exportToFile(const KtensorT<ExecSpace>& u,
                    const std::string& file_name) const;

  // Factor matrix operations
  template <typename ExecSpaceDst, typename ExecSpaceSrc>
  FacMatrixT<ExecSpaceDst> exportFromRoot(const int dim, const FacMatrixT<ExecSpaceSrc>& u) const;
  template <typename ExecSpaceDst, typename ExecSpaceSrc>
  FacMatrixT<ExecSpaceDst> importToRoot(const int dim, const FacMatrixT<ExecSpaceSrc>& u) const;

  // Initial guess computations
  template <typename ExecSpace>
  KtensorT<ExecSpace> readInitialGuess(const std::string& file_name) const;
  template <typename ExecSpace>
  KtensorT<ExecSpace> randomInitialGuess(const SptensorT<ExecSpace>& X,
                                         const int rank,
                                         const int seed,
                                         const bool prng,
                                         const std::string& dist_method) const;
  template <typename ExecSpace>
  KtensorT<ExecSpace> computeInitialGuess(const SptensorT<ExecSpace>& X,
                                          const ptree& input) const;

private:
  template <typename ExecSpace>
  SptensorT<ExecSpace> distributeTensorData(
    const std::vector<G_MPI_IO::TDatatype<ttb_real>>& Tvec,
    const std::vector<std::uint32_t>& TensorDims,
    const std::vector<small_vector<int>>& blocking,
    const ProcessorMap& pmap);

  std::pair<G_MPI_IO::SptnFileHeader, MPI_File>
  readBinaryHeader(const std::string& file_name, int indexbase,
                   std::vector<std::uint32_t>& dims, std::uint64_t& nnz);

  std::vector<std::uint32_t> local_dims_;
  std::vector<std::uint32_t> global_dims_;
  std::shared_ptr<ProcessorMap> pmap_;
  std::vector<small_vector<int>> global_blocking_;

  MPI_Datatype mpiElemType_ = DistContext::toMpiType<ttb_real>();
};

// Helper declerations
namespace detail {

struct RangePair {
  int64_t lower;
  int64_t upper;
};

bool fileFormatIsBinary(const std::string& file_name);

template <typename ExecSpace>
auto rangesToIndexArray(const small_vector<RangePair>& ranges);
small_vector<int> singleDimUniformBlocking(int ModeLength, int ProcsInMode);

std::vector<G_MPI_IO::TDatatype<ttb_real>>
distributeTensorToVectors(const Sptensor& sp_tensor_host, uint64_t nnz,
                          MPI_Comm comm, int rank, int nprocs);

std::vector<G_MPI_IO::TDatatype<ttb_real>>
redistributeTensor(const std::vector<G_MPI_IO::TDatatype<ttb_real>>& Tvec,
                   const std::vector<std::uint32_t>& TensorDims,
                   const std::vector<small_vector<int>>& blocking,
                   const ProcessorMap& pmap);

template <typename ExecSpace>
void printRandomElements(const SptensorT<ExecSpace>& tensor,
                         int num_elements_per_rank,
                         const ProcessorMap& pmap,
                         const small_vector<RangePair>& ranges);

} // namespace detail

template <typename ExecSpace>
ttb_real
DistTensorContext::
globalNorm(const SptensorT<ExecSpace>& X) const
{
  const auto& values = X.getValArray();
  ttb_real norm2 = values.dot(values);
  norm2 = pmap_->gridAllReduce(norm2);
  return std::sqrt(norm2);
}

template <typename ExecSpace>
std::uint64_t
DistTensorContext::
globalNNZ(const SptensorT<ExecSpace>& X) const
{
  return pmap_->gridAllReduce(X.nnz());
}

template <typename ExecSpace>
ttb_real
DistTensorContext::
globalNumelFloat(const SptensorT<ExecSpace>& X) const
{
  return pmap_->gridAllReduce(X.numel_float());
}

template <typename ExecSpace>
ttb_real
DistTensorContext::
globalNorm(const KtensorT<ExecSpace>& u) const
{
  return std::sqrt(u.normFsq());
}

template <typename ExecSpaceDst, typename ExecSpaceSrc>
KtensorT<ExecSpaceDst>
DistTensorContext::
exportFromRoot(const KtensorT<ExecSpaceSrc>& u) const
{
  // Broadcast ktensor values from 0 to all procs
  const int nd = u.ndims();
  const int nc = u.ncomponents();
  assert(global_dims_.size() == nd);

  for (int i=0; i<nd; ++i)
    pmap_->gridBcast(u[i].view().data(), u[i].view().span(), 0);
  pmap_->gridBcast(u.weights().values().data(), u.weights().values().span(), 0);
  pmap_->gridBarrier();

  // Copy our portion from u into ktensor_
  IndxArrayT<ExecSpaceDst> sz(nd);
  auto hsz = create_mirror_view(sz);
  for (int i=0; i<nd; ++i)
    hsz[i] = local_dims_[i];
  deep_copy(sz,hsz);
  Genten::KtensorT<ExecSpaceDst> exp(nc, nd, sz);
  exp.setMatrices(0.0);
  deep_copy(exp.weights(), u.weights());
  for (int i=0; i<nd; ++i) {
    auto coord = pmap_->gridCoord(i);
    auto rng = std::make_pair(global_blocking_[i][coord],
                              global_blocking_[i][coord + 1]);
    auto sub = Kokkos::subview(u[i].view(), rng, Kokkos::ALL);
    deep_copy(exp[i].view(), sub);
  }
  return exp;
}

namespace {

template <typename T> bool isValueSame(T x) {
  T p[]{-x, x};
  MPI_Allreduce(MPI_IN_PLACE, p, 2, DistContext::toMpiType<T>(), MPI_MIN,
                MPI_COMM_WORLD);
  return p[0] == -p[1];
}

template <typename ExecSpace>
bool weightsAreSame(const KtensorT<ExecSpace> &u) {
#ifdef NDEBUG
  return true;
#else
  const auto wspan = u.weights().values().span();
  if (!isValueSame(wspan)) {
    return false;
  }

  for (std::size_t i = 0; i < wspan; ++i) {
    if (!isValueSame(u.weights(i))) {
      return false;
    }
  }

  return true;
#endif
}

} // namespace

template <typename ExecSpaceDst, typename ExecSpaceSrc>
KtensorT<ExecSpaceDst>
DistTensorContext::
importToRoot(const KtensorT<ExecSpaceSrc>& u) const
{
  const bool print =
    DistContext::isDebug() && (pmap_->gridRank() == 0);

  const int nd = u.ndims();
  const int nc = u.ncomponents();
  assert(global_dims_.size() == nd);

  KtensorT<ExecSpaceDst> out;
  IndxArrayT<ExecSpaceDst> sizes_idx(nd);
  auto sizes_idx_host = create_mirror_view(sizes_idx);
  for (int i=0; i<nd; ++i) {
    sizes_idx_host[i] = global_dims_[i];
  }
  deep_copy(sizes_idx, sizes_idx_host);
  out = KtensorT<ExecSpaceDst>(nc, nd, sizes_idx);

  if (!weightsAreSame(u)) {
    throw std::string(
        "Ktensor weights are expected to be the same on all ranks");
  }

  deep_copy(out.weights(), u.weights());

  if (print)
    std::cout << "Blocking:\n";

  small_vector<int> grid_pos(nd, 0);
  for (int d=0; d<nd; ++d) {
    std::vector<int> recvcounts(pmap_->gridSize(), 0);
    std::vector<int> displs(pmap_->gridSize(), 0);
    const auto nblocks = global_blocking_[d].size() - 1;
    if (print)
      std::cout << "\tDim(" << d << ")\n";
    for (auto b = 0; b < nblocks; ++b) {
      if (print)
        std::cout << "\t\t{" << global_blocking_[d][b]
                  << ", " << global_blocking_[d][b + 1]
                  << "} owned by ";
      grid_pos[d] = b;
      int owner = 0;
      MPI_Cart_rank(pmap_->gridComm(), grid_pos.data(), &owner);
      if (print)
        std::cout << owner << "\n";
      recvcounts[owner] =
        u[d].view().stride(0)*(global_blocking_[d][b+1]-global_blocking_[d][b]);
      displs[owner] = out[d].view().stride(0)*global_blocking_[d][b];
      grid_pos[d] = 0;
    }

    const bool is_sub_root = pmap_->subCommRank(d) == 0;
    std::size_t send_size = is_sub_root ? u[d].view().span() : 0;
    MPI_Gatherv(u[d].view().data(), send_size,
                DistContext::toMpiType<ttb_real>(),
                out[d].view().data(), recvcounts.data(), displs.data(),
                DistContext::toMpiType<ttb_real>(), 0,
                pmap_->gridComm());
    pmap_->gridBarrier();
  }

  if (print) {
    std::cout << std::endl;
    std::cout << "Subcomm sizes: ";
    for (auto s : pmap_->subCommSizes()) {
      std::cout << s << " ";
    }
    std::cout << std::endl;
  }

  return out;
}

template <typename ExecSpace>
void
DistTensorContext::
allReduce(KtensorT<ExecSpace>& u, const bool divide_by_grid_size) const
{
  const int nd = u.ndims();
  assert(global_dims_.size() == nd);

  for (int n=0; n<nd; ++n)
    pmap_->subGridAllReduce(
      n, u[n].view().data(), u[n].view().span());

  if (divide_by_grid_size) {
    auto const &gridSizes = pmap_->subCommSizes();
    for (int n=0; n<nd; ++n) {
      const ttb_real scale = ttb_real(1.0 / gridSizes[n]);
      u[n].times(scale);
    }
  }
}

template <typename ExecSpace>
void
DistTensorContext::
exportToFile(const KtensorT<ExecSpace>& u, const std::string& file_name) const
{
  Ktensor out = importToRoot<Genten::DefaultHostExecutionSpace>(u);
  if (pmap_->gridRank() == 0) {
    // Normalize Ktensor u before writing out
    out.normalize(Genten::NormTwo);
    out.arrange();

    std::cout << "Saving final Ktensor to " << file_name << std::endl;
    Genten::export_ktensor(file_name, out);
  }
}

template <typename ExecSpaceDst, typename ExecSpaceSrc>
FacMatrixT<ExecSpaceDst>
DistTensorContext::
exportFromRoot(const int dim, const FacMatrixT<ExecSpaceSrc>& u) const
{
  // Broadcast factor matrix values from 0 to all procs
  pmap_->gridBcast(u.view().data(), u.view().span(), 0);
  pmap_->gridBarrier();

  // Copy our portion
  FacMatrixT<ExecSpaceDst> exp(u.nRows(), u.nCols());
  auto coord = pmap_->gridCoord(dim);
  auto rng = std::make_pair(global_blocking_[dim][coord],
                            global_blocking_[dim][coord + 1]);
  auto sub = Kokkos::subview(u.view(), rng, Kokkos::ALL);
  deep_copy(exp.view(), sub);
  return exp;
}

template <typename ExecSpaceDst, typename ExecSpaceSrc>
FacMatrixT<ExecSpaceDst>
DistTensorContext::
importToRoot(const int dim, const FacMatrixT<ExecSpaceSrc>& u) const
{
  FacMatrixT<ExecSpaceDst> out(global_dims_[dim], u.nCols());

  small_vector<int> grid_pos(global_dims_.size(), 0);
  std::vector<int> recvcounts(pmap_->gridSize(), 0);
  std::vector<int> displs(pmap_->gridSize(), 0);
  const auto nblocks = global_blocking_[dim].size() - 1;
  for (auto b = 0; b < nblocks; ++b) {
    grid_pos[dim] = b;
    int owner = 0;
    MPI_Cart_rank(pmap_->gridComm(), grid_pos.data(), &owner);
    recvcounts[owner] =
      u.view().stride(0)*(global_blocking_[dim][b+1]-global_blocking_[dim][b]);
    displs[owner] = out.view().stride(0)*global_blocking_[dim][b];
  }

  const bool is_sub_root = pmap_->subCommRank(dim) == 0;
  std::size_t send_size = is_sub_root ? u.view().span() : 0;
  MPI_Gatherv(u.view().data(), send_size,
              DistContext::toMpiType<ttb_real>(),
              out.view().data(), recvcounts.data(), displs.data(),
              DistContext::toMpiType<ttb_real>(), 0,
              pmap_->gridComm());
  pmap_->gridBarrier();

  return out;
}

template <typename ExecSpace>
KtensorT<ExecSpace>
DistTensorContext::
readInitialGuess(const std::string& file_name) const
{
  KtensorT<DefaultHostExecutionSpace> u_host;
  import_ktensor(file_name, u_host);
  KtensorT<ExecSpace> u = create_mirror_view(ExecSpace(), u_host);
  deep_copy(u, u_host);
  return exportFromRoot<ExecSpace>(u);
}

template <typename ExecSpace>
KtensorT<ExecSpace>
DistTensorContext::
randomInitialGuess(const SptensorT<ExecSpace>& X,
                   const int rank,
                   const int seed,
                   const bool prng,
                   const std::string& dist_method) const
{
  const ttb_indx nd = X.ndims();
  const ttb_real norm_x = globalNorm(X);
  RandomMT cRMT(seed);

  if (dist_method == "serial") {
    // Compute random ktensor on rank 0 and broadcast to all proc's
    IndxArrayT<ExecSpace> sz(nd);
    auto hsz = create_mirror_view(sz);
    for (int i=0; i<nd; ++i)
      hsz[i] = global_dims_[i];
    deep_copy(sz,hsz);
    Genten::KtensorT<ExecSpace> u(rank, nd, sz);
    if (pmap_->gridRank() == 0) {
      u.setWeights(1.0);
      u.setMatricesScatter(false, prng, cRMT);
      const auto norm_k = std::sqrt(u.normFsq());
      u.weights().times(norm_x / norm_k);
      u.distribute();
    }
    return exportFromRoot<ExecSpace>(u);
  }
  else if (dist_method == "parallel") {
    // Compute random ktensor on each node
    KtensorT<ExecSpace> u(rank, nd, X.size());
    u.setWeights(1.0);
    u.setMatricesScatter(false, prng, cRMT);
    const ttb_real norm_k = globalNorm(u);
    u.weights().times(norm_x / norm_k);
    u.distribute();
    return u;
  }
  else if (dist_method == "parallel-drew") {
    // Drew's funky random ktensor that I don't understand
    KtensorT<ExecSpace> u(rank, nd, X.size());
    u.setWeights(1.0);
    u.setMatricesScatter(false, prng, cRMT);
    u.weights().times(1.0 / norm_x);
    u.distribute();
    allReduce(u, true);
    return u;
  }
  else
    Genten::error("Unknown distributed-guess method: " + dist_method);

  return Genten::KtensorT<ExecSpace>();
}

template <typename ExecSpace>
SptensorT<ExecSpace>
DistTensorContext::
distributeTensor(const std::string& file, const ttb_indx index_base,
                 const bool compressed)
{
  const bool is_binary = detail::fileFormatIsBinary(file);
  if (is_binary && index_base != 0)
    Genten::error("The binary format only supports zero based indexing\n");
  if (is_binary && compressed)
    Genten::error("The binary format does not support compression\n");

  std::vector<G_MPI_IO::TDatatype<ttb_real>> Tvec;

  if (DistContext::rank() == 0)
    std::cout << "Reading tensor from file " << file << std::endl;

  DistContext::Barrier();
  auto t2 = MPI_Wtime();
  std::vector<std::uint32_t> global_dims;
  if (is_binary) {
    // For binary file, do a parallel read
    std::uint64_t nnz = 0;
    auto binary_header = readBinaryHeader(file, index_base, global_dims, nnz);
    Tvec = G_MPI_IO::parallelReadElements(DistContext::commWorld(),
                                          binary_header.second,
                                          binary_header.first);
  }
  else {
    // For non-binary, read on rank 0 and broadcast dimensions.
    // We do this instead of reading the header because we want to support
    // headerless files
    Sptensor X_host;
    if (gridRank() == 0)
      import_sptensor(file, X_host, index_base, compressed);

    std::size_t nnz = X_host.nnz();
    DistContext::Bcast(nnz, 0);

    std::size_t ndims = X_host.ndims();
    DistContext::Bcast(ndims, 0);

    small_vector<int> dims(ndims);
    if (gridRank() == 0) {
      for (std::size_t i=0; i<ndims; ++i)
        dims[i] = X_host.size(i);
    }
    DistContext::Bcast(dims, 0);

    global_dims = std::vector<std::uint32_t>(ndims);
    for (std::size_t i=0; i<ndims; ++i)
      global_dims[i] = dims[i];

    Tvec = detail::distributeTensorToVectors(
      X_host, nnz, DistContext::commWorld(), DistContext::rank(),
      DistContext::nranks());
  }
  DistContext::Barrier();
  auto t3 = MPI_Wtime();
  if (gridRank() == 0) {
    std::cout << "  Read file in: " << t3 - t2 << "s" << std::endl;
  }

  // Check if we have already distributed a tensor, in which case this one
  // needs to be of the same size
  const int ndims = global_dims.size();
  if (global_dims_.size() > 0) {
    if (global_dims_.size() != ndims)
      Genten::error("distributeTensor() called twice with different number of dimensions!");
    for (int i=0; i<ndims; ++i)
      if (global_dims_[i] != global_dims[i])
          Genten::error("distributeTensor() called twice with different sized tensors!");
  }
  else {
    global_dims_ = global_dims;

    pmap_ = std::shared_ptr<ProcessorMap>(new ProcessorMap(global_dims_));
    detail::printGrids(*pmap_);

    global_blocking_ =
      detail::generateUniformBlocking(global_dims_, pmap_->gridDims());
    detail::printBlocking(*pmap_, global_blocking_);
  }

  return distributeTensorData<ExecSpace>(Tvec, global_dims_, global_blocking_,
                                         *pmap_);
}

template <typename ExecSpaceDst, typename ExecSpaceSrc>
SptensorT<ExecSpaceDst>
DistTensorContext::
distributeTensor(const SptensorT<ExecSpaceSrc>& X)
{
  // Check if we have already distributed a tensor, in which case this one
  // needs to be of the same size
  const int ndims = X.ndims();
  if (global_dims_.size() > 0) {
    if (global_dims_.size() != ndims)
      Genten::error("distributeTensor() called twice with different number of dimensions!");
    for (int i=0; i<ndims; ++i)
      if (global_dims_[i] != X.size(i))
          Genten::error("distributeTensor() called twice with different sized tensors!");
  }
  else {
    global_dims_.resize(ndims);
    for (int i=0; i<ndims; ++i)
      global_dims_[i] = X.size(i);

    pmap_ = std::shared_ptr<ProcessorMap>(new ProcessorMap(global_dims_));
    detail::printGrids(*pmap_);

    global_blocking_ =
      detail::generateUniformBlocking(global_dims_, pmap_->gridDims());

    detail::printBlocking(*pmap_, global_blocking_);
    DistContext::Barrier();
  }

  auto X_host = create_mirror_view(X);
  deep_copy(X_host, X);
  auto Tvec = detail::distributeTensorToVectors(
    X_host, X.nnz(), pmap_->gridComm(), pmap_->gridRank(),
    pmap_->gridSize());

  return distributeTensorData<ExecSpaceDst>(
    Tvec, global_dims_, global_blocking_, *pmap_);
}

template <typename ExecSpace>
SptensorT<ExecSpace>
DistTensorContext::
distributeTensorData(const std::vector<G_MPI_IO::TDatatype<ttb_real>>& Tvec,
                     const std::vector<std::uint32_t>& TensorDims,
                     const std::vector<small_vector<int>>& blocking,
                     const ProcessorMap& pmap)
{
  DistContext::Barrier();
  auto t4 = MPI_Wtime();

  // Now redistribute to final format
  auto distributedData =
    detail::redistributeTensor(Tvec, global_dims_, global_blocking_, *pmap_);

  DistContext::Barrier();
  auto t5 = MPI_Wtime();

  if (gridRank() == 0) {
    std::cout << "  Redistributied file in: " << t5 - t4 << "s" << std::endl;
  }

  DistContext::Barrier();
  auto t6 = MPI_Wtime();

  std::vector<detail::RangePair> range;
  auto ndims = TensorDims.size();
  for (auto i = 0; i < ndims; ++i) {
    auto coord = pmap_->gridCoord(i);
    range.push_back({global_blocking_[i][coord],
                      global_blocking_[i][coord + 1]});
  }

  std::vector<ttb_indx> indices(ndims);
  local_dims_.resize(ndims);
  for (auto i = 0; i < ndims; ++i) {
    auto const &rpair = range[i];
    indices[i] = rpair.upper - rpair.lower;
    local_dims_[i] = indices[i];
  }

  const auto local_nnz = distributedData.size();
  std::vector<ttb_real> values(local_nnz);
  std::vector<std::vector<ttb_indx>> subs(local_nnz);
  for (auto i = 0; i < local_nnz; ++i) {
    auto data = distributedData[i];
    values[i] = data.val;
    subs[i] = std::vector<ttb_indx>(data.coo, data.coo + ndims);
    for (auto j = 0; j < ndims; ++j) {
      subs[i][j] -= range[j].lower;
    }
  }

  Sptensor sptensor_host(indices, values, subs);
  SptensorT<ExecSpace> sptensor = create_mirror_view(ExecSpace(), sptensor_host);
  deep_copy(sptensor, sptensor_host);

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

  DistContext::Barrier();
  auto t7 = MPI_Wtime();

  // if (gridRank() == 0) {
  //   std::cout << "Copied to data struct in: " << t7 - t6 << "s" << std::endl;
  // }

  return sptensor;
}

namespace detail {

template <typename ExecSpace>
void
printRandomElements(const SptensorT<ExecSpace>& tensor,
                    int num_elements_per_rank, const ProcessorMap& pmap,
                    const small_vector<RangePair>& ranges)
{
  static_assert(
    std::is_same<ExecSpace, Kokkos::DefaultHostExecutionSpace>::value,
    "To print random elements we want a host tensor.");

  const auto size = pmap.gridSize();
  const auto rank = pmap.gridRank();
  auto gComm = pmap.gridComm();

  const auto nnz = tensor.nnz();
  std::uniform_int_distribution<> dist(0, nnz - 1);
  std::mt19937_64 gen(std::random_device{}());

  for (auto i = 0; i < size; ++i) {
    if (rank != i) {
      continue;
    }
    std::cout << "Rank: " << rank << " ranges:[";
    for (auto j = 0; j < ranges.size(); ++j) {
      std::cout << "{" << ranges[j].lower << ", " << ranges[j].upper << "}";
      if (j < ranges.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]\n";
    if (nnz >= num_elements_per_rank) {
      for (auto i = 0; i < num_elements_per_rank; ++i) {
        auto rand_idx = dist(gen);
        auto indices = tensor.getSubscripts(rand_idx);
        auto value = tensor.value(rand_idx);

        std::cout << "\t";
        for (auto j = 0; j < tensor.ndims(); ++j) {
          std::cout << indices[j] + ranges[j].lower << " ";
        }
        std::cout << value << "\n";
      }
    } else {
      std::cout << "Rank: " << pmap.gridRank() << " had 0 nnz\n";
    }
    std::cout << std::endl;
    MPI_Barrier(gComm);
    //sleep(1);
  }
}

template <typename ExecSpace>
auto
rangesToIndexArray(const small_vector<RangePair>& ranges)
{
  IndxArrayT<ExecSpace> outArray(ranges.size());
  auto mirrorArray = create_mirror_view(outArray);

  auto i = 0;
  for (auto const &rp : ranges) {
    const auto size = rp.upper - rp.lower;
    mirrorArray[i] = size;
    ++i;
  }

  deep_copy(outArray, mirrorArray);
  return outArray;
}

} // namespace detail

#else

class DistTensorContext {
public:
  DistTensorContext() = default;
  ~DistTensorContext() = default;

  DistTensorContext(DistTensorContext&&) = default;
  DistTensorContext(const DistTensorContext&) = default;
  DistTensorContext& operator=(DistTensorContext&&) = default;
  DistTensorContext& operator=(const DistTensorContext&) = default;

  template <typename ExecSpace>
  SptensorT<ExecSpace> distributeTensor(const ptree& tree);
  template <typename ExecSpace>
  SptensorT<ExecSpace> distributeTensor(const std::string& file,
                                        const ttb_indx index_base,
                                        const bool compressed)
  {
    Sptensor x_host;
    Genten::import_sptensor(file, x_host, index_base, compressed, true);
    SptensorT<ExecSpace> x = create_mirror_view( ExecSpace(), x_host );
    deep_copy( x, x_host );

    auto sz = x_host.size();
    const int nd = x_host.ndims();
    global_dims_.resize(nd);
    for (int i=0; i<nd; ++i)
      global_dims_[i] = sz[i];

    return x;
  }
  template <typename ExecSpaceDst, typename ExecSpaceSrc>
  SptensorT<ExecSpaceDst> distributeTensor(const SptensorT<ExecSpaceSrc>& X)
  {
    SptensorT<ExecSpaceDst> X_dst = create_mirror_view(ExecSpaceDst(), X);
    deep_copy(X_dst, X);
    return X_dst;
  }

  // Parallel info
  std::int32_t ndims() const { return global_dims_.size(); }
  const std::vector<std::uint32_t>& dims() const { return global_dims_; }
  std::int64_t nprocs() const { return 1; }
  std::int64_t gridRank() const { return 0; }

  // Processor map for communication
  const ProcessorMap& pmap() const { return *pmap_; }
  std::shared_ptr<const ProcessorMap> pmap_ptr() const { return pmap_; }

  // Sptensor operations
  template <typename ExecSpace>
  ttb_real globalNorm(const SptensorT<ExecSpace>& X) const { return X.norm(); }
  template <typename ExecSpace>
  std::uint64_t globalNNZ(const SptensorT<ExecSpace>& X) const { return X.nnz(); }
  template <typename ExecSpace>
  ttb_real globalNumelFloat(const SptensorT<ExecSpace>& X) const { return X.numel_float(); }

  // Ktensor operations
  template <typename ExecSpace>
  ttb_real globalNorm(const KtensorT<ExecSpace>& u) const { return std::sqrt(u.normFsq()); }
  template <typename ExecSpaceDst, typename ExecSpaceSrc>
  KtensorT<ExecSpaceDst> exportFromRoot(const KtensorT<ExecSpaceSrc>& u) const {
    KtensorT<ExecSpaceDst> v = create_mirror_view(ExecSpaceDst(), u);
    deep_copy(v,u);
    return v;
  }
  template <typename ExecSpaceDst, typename ExecSpaceSrc>
  KtensorT<ExecSpaceDst> importToRoot(const KtensorT<ExecSpaceSrc>& u) const {
    KtensorT<ExecSpaceDst> v = create_mirror_view(ExecSpaceDst(), u);
    deep_copy(v,u);
    return v;
  }
  template <typename ExecSpace>
  void allReduce(KtensorT<ExecSpace>& u,
                 const bool divide_by_grid_size = false) const {}
  template <typename ExecSpace>
  void exportToFile(const KtensorT<ExecSpace>& out,
                    const std::string& file_name) const {
    out.normalize(Genten::NormTwo);
    out.arrange();

    std::cout << "Saving final Ktensor to " << file_name << std::endl;
    auto out_h = create_mirror_view(out);
    deep_copy(out_h, out);
    Genten::export_ktensor(file_name, out_h);
  }
  template <typename ExecSpaceDst, typename ExecSpaceSrc>
  FacMatrixT<ExecSpaceDst> exportFromRoot(const int dim, const FacMatrixT<ExecSpaceSrc>& u) const {
    FacMatrixT<ExecSpaceDst> v = create_mirror_view(ExecSpaceDst(), u);
    deep_copy(v,u);
    return v;
  }
  template <typename ExecSpaceDst, typename ExecSpaceSrc>
  FacMatrixT<ExecSpaceDst> importToRoot(const int dim, const FacMatrixT<ExecSpaceSrc>& u) const {
    FacMatrixT<ExecSpaceDst> v = create_mirror_view(ExecSpaceDst(), u);
    deep_copy(v,u);
    return v;
  }

  template <typename ExecSpace>
  KtensorT<ExecSpace> readInitialGuess(const std::string& file_name) const {
    KtensorT<DefaultHostExecutionSpace> u_host;
    import_ktensor(file_name, u_host);
    KtensorT<ExecSpace> u = create_mirror_view(ExecSpace(), u_host);
    deep_copy(u, u_host);
    return u;
  }
  template <typename ExecSpace>
  KtensorT<ExecSpace> randomInitialGuess(const SptensorT<ExecSpace>& X,
                                         const int rank,
                                         const int seed,
                                         const bool prng,
                                         const std::string& dist_method) const {
    const ttb_indx nd = X.ndims();
    const ttb_real norm_x = globalNorm(X);
    RandomMT cRMT(seed);

    if (dist_method == "serial") {
      // Compute random ktensor on rank 0 and broadcast to all proc's
      IndxArrayT<ExecSpace> sz(nd);
      auto hsz = create_mirror_view(sz);
      for (int i=0; i<nd; ++i)
        hsz[i] = global_dims_[i];
      deep_copy(sz,hsz);
      Genten::KtensorT<ExecSpace> u(rank, nd, sz);
      if (pmap_->gridRank() == 0) {
        u.setWeights(1.0);
        u.setMatricesScatter(false, prng, cRMT);
        const auto norm_k = std::sqrt(u.normFsq());
        u.weights().times(norm_x / norm_k);
        u.distribute();
      }
      return u;
    }
    else if (dist_method == "parallel") {
      // Compute random ktensor on each node
      KtensorT<ExecSpace> u(rank, nd, X.size());
      u.setWeights(1.0);
      u.setMatricesScatter(false, prng, cRMT);
      const ttb_real norm_k = globalNorm(u);
      u.weights().times(norm_x / norm_k);
      u.distribute();
      return u;
    }
    else if (dist_method == "parallel-drew") {
      // Drew's funky random ktensor that I don't understand
      KtensorT<ExecSpace> u(rank, nd, X.size());
      u.setWeights(1.0);
      u.setMatricesScatter(false, prng, cRMT);
      u.weights().times(1.0 / norm_x);
      u.distribute();
      allReduce(u, true);
      return u;
    }
    else
      Genten::error("Unknown distributed-guess method: " + dist_method);
    return Genten::KtensorT<ExecSpace>();
  }
  template <typename ExecSpace>
  KtensorT<ExecSpace> computeInitialGuess(const SptensorT<ExecSpace>& X,
                                          const ptree& input) const;

private:
  std::vector<std::uint32_t> global_dims_;
  std::shared_ptr<ProcessorMap> pmap_;
};

#endif

template <typename ExecSpace>
SptensorT<ExecSpace>
DistTensorContext::
distributeTensor(const ptree& tree)
{
  auto t_tree = tree.get_child("tensor");
  const std::string file_name = t_tree.get<std::string>("input-file");
  const ttb_indx index_base = t_tree.get<int>("index-base", 0);
  const bool compressed = t_tree.get<bool>("compressed", false);
  return distributeTensor<ExecSpace>(file_name, index_base, compressed);
}

template <typename ExecSpace>
KtensorT<ExecSpace>
DistTensorContext::
computeInitialGuess(const SptensorT<ExecSpace>& X, const ptree& input) const
{
  KtensorT<ExecSpace> u;

  auto kt_input = input.get_child("k-tensor");
  std::string init_method = kt_input.get<std::string>("initial-guess", "rand");
  if (init_method == "file") {
    std::string file_name = kt_input.get<std::string>("initial-file");
    u = readInitialGuess<ExecSpace>(file_name);
  }
  else if (init_method == "rand") {
    const int seed = kt_input.get<int>("seed",std::random_device{}());
    const bool prng = kt_input.get<bool>("prng",true);
    const int nc = kt_input.get<int>("rank");
    const std::string dist_method =
      kt_input.get<std::string>("distributed-guess", "serial");
    u = randomInitialGuess(X, nc, seed, prng, dist_method);
  }
  else
    Genten::error("Unknown initial-guess method: " + init_method);

  return u;
}

} // namespace Genten
