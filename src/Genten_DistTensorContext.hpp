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
// DATA, OR PROFITS; OR BUSINESS TTB_INDXERRUPTION) HOWEVER CAUSED AND ON ANY
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
#include "Genten_Pmap.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_AlgParams.hpp"

#include "CMakeInclude.h"
#if defined(HAVE_DIST)
#include <cmath>
#include <memory>
#include "Genten_Tpetra.hpp"
#include "Genten_DistFacMatrix.hpp"
#endif

namespace Genten {

#if defined(HAVE_DIST)

// Forward declarations
struct SpDataType;

// Class to describe and manipulate tensor data in a distributed context
template <typename ExecSpace>
class DistTensorContext {
public:
  DistTensorContext() = default;
  ~DistTensorContext() = default;

  DistTensorContext(DistTensorContext&&) = default;
  DistTensorContext(const DistTensorContext&) = default;
  DistTensorContext& operator=(DistTensorContext&&) = default;
  DistTensorContext& operator=(const DistTensorContext&) = default;

  template <typename ExecSpaceSrc>
  DistTensorContext(const DistTensorContext<ExecSpaceSrc>& src);

  std::tuple< SptensorT<ExecSpace>, TensorT<ExecSpace> >
  distributeTensor(const ptree& tree,
                   const AlgParams& algParams);
  std::tuple< SptensorT<ExecSpace>, TensorT<ExecSpace> >
  distributeTensor(const std::string& file,
                   const ttb_indx index_base,
                   const bool compressed,
                   const ptree& tree,
                   const AlgParams& algParams);
  template <typename ExecSpaceSrc>
  SptensorT<ExecSpace> distributeTensor(const SptensorT<ExecSpaceSrc>& X,
                                        const AlgParams& algParams = AlgParams());
  template <typename ExecSpaceSrc>
  TensorT<ExecSpace> distributeTensor(const TensorT<ExecSpaceSrc>& X,
                                      const AlgParams& algParams = AlgParams());

  // Parallel info
  ttb_indx ndims() const { return global_dims_.size(); }
  const std::vector<ttb_indx>& dims() const { return global_dims_; }
  static ttb_indx nprocs() { return DistContext::nranks(); }
  static ttb_indx gridRank() { return DistContext::rank(); }
  const std::vector<small_vector<ttb_indx>>& blocking() const { return global_blocking_; }

  // Processor map for communication
  const ProcessorMap& pmap() const { return *pmap_; }
  std::shared_ptr<const ProcessorMap> pmap_ptr() const { return pmap_; }

#ifdef HAVE_TPETRA
  Teuchos::RCP<const tpetra_map_type<ExecSpace> >
  getFactorMap(unsigned n) const {
    return factorMap[n];
  }
  Teuchos::RCP<const tpetra_map_type<ExecSpace> >
  getOverlapFactorMap(unsigned n) const {
    return overlapFactorMap[n];
  }
#endif

  // Sptensor operations
  ttb_real globalNorm(const SptensorT<ExecSpace>& X) const;
  ttb_indx globalNNZ(const SptensorT<ExecSpace>& X) const;
  ttb_real globalNumelFloat(const SptensorT<ExecSpace>& X) const;

  // Tensor operations
  ttb_real globalNorm(const TensorT<ExecSpace>& X) const;
  ttb_indx globalNNZ(const TensorT<ExecSpace>& X) const;
  ttb_real globalNumelFloat(const TensorT<ExecSpace>& X) const;

  // Ktensor operations
  ttb_real globalNorm(const KtensorT<ExecSpace>& X) const;
  template <typename ExecSpaceSrc>
  KtensorT<ExecSpace> exportFromRoot(const KtensorT<ExecSpaceSrc>& u) const;
  template <typename ExecSpaceDst>
  KtensorT<ExecSpaceDst> importToRoot(const KtensorT<ExecSpace>& u) const;
  template <typename ExecSpaceDst>
  KtensorT<ExecSpaceDst> importToAll(const KtensorT<ExecSpace>& u) const;
  void allReduce(KtensorT<ExecSpace>& u,
                 const bool divide_by_grid_size = false) const;
  void exportToFile(const KtensorT<ExecSpace>& u,
                    const std::string& file_name) const;

  // Factor matrix operations
  template <typename ExecSpaceSrc>
  FacMatrixT<ExecSpace> exportFromRoot(const ttb_indx dim, const FacMatrixT<ExecSpaceSrc>& u) const;
  template <typename ExecSpaceDst>
  FacMatrixT<ExecSpaceDst> importToRoot(const ttb_indx dim, const FacMatrixT<ExecSpace>& u) const;

  // Initial guess computations
  KtensorT<ExecSpace> readInitialGuess(const std::string& file_name) const;
  KtensorT<ExecSpace> randomInitialGuess(const SptensorT<ExecSpace>& X,
                                         const ttb_indx rank,
                                         const ttb_indx seed,
                                         const bool prng,
                                         const bool scale_guess_by_norm_x,
                                         const std::string& dist_method) const;
  KtensorT<ExecSpace> randomInitialGuess(const TensorT<ExecSpace>& X,
                                         const ttb_indx rank,
                                         const ttb_indx seed,
                                         const bool prng,
                                         const bool scale_guess_by_norm_x,
                                         const std::string& dist_method) const;
  KtensorT<ExecSpace> computeInitialGuess(const SptensorT<ExecSpace>& X,
                                          const ptree& input) const;

private:
  SptensorT<ExecSpace>
  distributeTensorImpl(const Sptensor& X, const AlgParams& algParams);
  TensorT<ExecSpace>
  distributeTensorImpl(const Tensor& X, const AlgParams& algParams);

  SptensorT<ExecSpace> distributeTensorData(
    const std::vector<SpDataType>& Tvec,
    const std::vector<ttb_indx>& TensorDims,
    const std::vector<small_vector<ttb_indx>>& blocking,
    const ProcessorMap& pmap,
    const AlgParams& algParams);
  TensorT<ExecSpace> distributeTensorData(
    const std::vector<double>& Tvec,
    const ttb_indx global_nnz, const ttb_indx global_offset,
    const std::vector<ttb_indx>& TensorDims,
    const std::vector<small_vector<ttb_indx>>& blocking,
    const TensorLayout layout,
    const ProcessorMap& pmap,
    const AlgParams& algParams,
    bool redistribute_needed = true);

  std::vector<ttb_indx> local_dims_;
  std::vector<ttb_indx> global_dims_;
  std::vector<ttb_indx> ktensor_local_dims_;
  std::vector<ttb_indx> ktensor_local_offsets_;
  std::shared_ptr<ProcessorMap> pmap_;
  std::vector<small_vector<ttb_indx>> global_blocking_;

  Dist_Update_Method::type dist_method;

#ifdef HAVE_TPETRA
  Teuchos::RCP<const Teuchos::Comm<int> > tpetra_comm;
  std::vector< Teuchos::RCP<const tpetra_map_type<ExecSpace> > > factorMap;
  std::vector< Teuchos::RCP<const tpetra_map_type<ExecSpace> > > overlapFactorMap;
#endif

  MPI_Datatype mpiElemType_ = DistContext::toMpiType<ttb_real>();

  template <typename ExecSpaceSrc> friend class DistTensorContext;
};

template <typename ExecSpace>
template <typename ExecSpaceSrc>
DistTensorContext<ExecSpace>::
DistTensorContext(const DistTensorContext<ExecSpaceSrc>& src) :
  local_dims_(src.local_dims_),
  global_dims_(src.global_dims_),
  ktensor_local_dims_(src.ktensor_local_dims_),
  ktensor_local_offsets_(src.ktensor_local_offsets_),
  pmap_(src.pmap_),
  global_blocking_(src.global_blocking_)
{
#ifdef HAVE_TPETRA
  tpetra_comm = src.tpetra_comm;
  const ttb_indx ndims = src.factorMap.size();
  factorMap.resize(ndims);
  overlapFactorMap.resize(ndims);
  auto create_and_copy_map = [](const tpetra_map_type<ExecSpaceSrc>& map) {
    auto gids_src = map.getMyGlobalIndices();
    Kokkos::View<tpetra_go_type*,ExecSpace> gids("gids", gids_src.extent(0));
    deep_copy(gids, gids_src);
    return Teuchos::rcp(new tpetra_map_type<ExecSpace>(
        map.getGlobalNumElements(), gids, map.getIndexBase(), map.getComm()));
  };
  for (ttb_indx dim=0; dim<ndims; ++dim) {
    factorMap[dim]          = create_and_copy_map(*src.factorMap[dim]);
    overlapFactorMap[dim]   = create_and_copy_map(*src.overlapFactorMap[dim]);
  }
#endif
}

template <typename ExecSpace>
template <typename ExecSpaceSrc>
SptensorT<ExecSpace>
DistTensorContext<ExecSpace>::
distributeTensor(const SptensorT<ExecSpaceSrc>& X, const AlgParams& algParams)
{
  dist_method = algParams.dist_update_method;
  Sptensor X_host = create_mirror_view(X);
  deep_copy(X_host, X);
  return distributeTensorImpl(X_host, algParams);
}

template <typename ExecSpace>
template <typename ExecSpaceSrc>
TensorT<ExecSpace>
DistTensorContext<ExecSpace>::
distributeTensor(const TensorT<ExecSpaceSrc>& X, const AlgParams& algParams)
{
  dist_method = algParams.dist_update_method;
  Tensor X_host = create_mirror_view(X);
  deep_copy(X_host, X);
  return distributeTensorImpl(X_host, algParams);
}

template <typename ExecSpace>
template <typename ExecSpaceSrc>
KtensorT<ExecSpace>
DistTensorContext<ExecSpace>::
exportFromRoot(const KtensorT<ExecSpaceSrc>& u) const
{
  const ttb_indx nd = u.ndims();
  const ttb_indx nc = u.ncomponents();
  gt_assert(ttb_indx(global_dims_.size()) == nd);

  Genten::KtensorT<ExecSpace> exp = Genten::KtensorT<ExecSpace>(nc, nd);
  deep_copy(exp.weights(), u.weights());

  // Broadcast ktensor values from 0 to all procs
  for (ttb_indx i=0; i<nd; ++i)
    pmap_->gridBcast(u[i].view().data(), u[i].view().span(), 0);
  pmap_->gridBcast(
    u.weights().values().data(), u.weights().values().span(),0);
  pmap_->gridBarrier();

  for (ttb_indx n=0; n<nd; ++n) {
    const ttb_indx num_my_rows = ktensor_local_dims_[n];
    const ttb_indx my_offset = ktensor_local_offsets_[n];

    // Copy our portion of u into exp
    FacMatrixT<ExecSpace> mat(num_my_rows, nc);
    auto rng = std::make_pair(my_offset, my_offset+num_my_rows);
    auto sub = Kokkos::subview(u[n].view(), rng, Kokkos::ALL);
    Kokkos::deep_copy(mat.view(), sub);
    exp.set_factor(n, mat);
  }

  exp.setProcessorMap(&pmap());
  return exp;
}

namespace {

template <typename T> bool isValueSame(T x) {
  T p[]{-x, x};
  MPI_Allreduce(MPI_IN_PLACE, p, 2, DistContext::toMpiType<T>(), MPI_MIN,
                DistContext::commWorld());
  return p[0] == -p[1];
}

// This check requires parallel communication so turn it off in optimized builds
#ifdef NDEBUG
template <typename ExecSpace>
bool weightsAreSame(const KtensorT<ExecSpace>&) {
  return true;
}
#else
template <typename ExecSpace>
bool weightsAreSame(const KtensorT<ExecSpace> &u) {
  auto w = u.weights();
  const ttb_indx wspan = w.values().span();
  if (!isValueSame(wspan)) {
    return false;
  }
  auto w_h = create_mirror_view(w);
  deep_copy(w_h, w);
  for (std::size_t i = 0; i < wspan; ++i) {
    if (!isValueSame(w_h[i])) {
      return false;
    }
  }
  return true;
}
#endif

} // namespace

template <typename ExecSpace>
template <typename ExecSpaceDst>
KtensorT<ExecSpaceDst>
DistTensorContext<ExecSpace>::
importToRoot(const KtensorT<ExecSpace>& u) const
{
  const ttb_indx nd = u.ndims();
  const ttb_indx nc = u.ncomponents();
  gt_assert(ttb_indx(global_dims_.size()) == nd);

  IndxArrayT<ExecSpaceDst> sizes_idx(nd);
  auto sizes_idx_host = create_mirror_view(sizes_idx);
  for (ttb_indx i=0; i<nd; ++i) {
    sizes_idx_host[i] = global_dims_[i];
  }
  deep_copy(sizes_idx, sizes_idx_host);

  KtensorT<ExecSpaceDst> out(nc, nd, sizes_idx);
  out.setMatrices(0.0);

  if (!weightsAreSame(u)) {
    throw std::string(
        "Ktensor weights are expected to be the same on all ranks");
  }
  deep_copy(out.weights(), u.weights());

  std::vector<int> recvcounts(pmap_->gridSize(), 0);
  std::vector<int> displs(pmap_->gridSize(), 0);
  for (ttb_indx n=0; n<nd; ++n) {
    const ttb_indx num_my_rows = ktensor_local_dims_[n];
    const ttb_indx my_offset = ktensor_local_offsets_[n];
    gt_assert(num_my_rows == u[n].nRows());

    // Send sizes and offsets to proc 0
    int my_send_size = num_my_rows*u[n].view().stride(0);
    int my_send_offset = my_offset*u[n].view().stride(0);
    MPI_Gather(&my_send_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0,
               pmap_->gridComm());
    MPI_Gather(&my_send_offset, 1, MPI_INT, displs.data(), 1, MPI_INT, 0,
               pmap_->gridComm());

    // Now send the data
    MPI_Gatherv(u[n].view().data(), my_send_size,
                DistContext::toMpiType<ttb_real>(),
                out[n].view().data(), recvcounts.data(), displs.data(),
                DistContext::toMpiType<ttb_real>(), 0,
                pmap_->gridComm());
    pmap_->gridBarrier();

  }

  return out;
}

template <typename ExecSpace>
template <typename ExecSpaceDst>
KtensorT<ExecSpaceDst>
DistTensorContext<ExecSpace>::
importToAll(const KtensorT<ExecSpace>& u) const
{
  const ttb_indx nd = u.ndims();
  const ttb_indx nc = u.ncomponents();
  gt_assert(ttb_indx(global_dims_.size()) == nd);

  IndxArrayT<ExecSpaceDst> sizes_idx(nd);
  auto sizes_idx_host = create_mirror_view(sizes_idx);
  for (ttb_indx i=0; i<nd; ++i) {
    sizes_idx_host[i] = global_dims_[i];
  }
  deep_copy(sizes_idx, sizes_idx_host);

  KtensorT<ExecSpaceDst> out(nc, nd, sizes_idx);
  out.setMatrices(0.0);

  if (!weightsAreSame(u)) {
    throw std::string(
        "Ktensor weights are expected to be the same on all ranks");
  }
  deep_copy(out.weights(), u.weights());


  std::vector<int> recvcounts(pmap_->gridSize(), 0);
  std::vector<int> displs(pmap_->gridSize(), 0);
  for (ttb_indx n=0; n<nd; ++n) {
    const ttb_indx num_my_rows = ktensor_local_dims_[n];
    const ttb_indx my_offset = ktensor_local_offsets_[n];
    gt_assert(num_my_rows == u[n].nRows());

    // Send sizes and offsets to all procs
    int my_send_size = num_my_rows*u[n].view().stride(0);
    int my_send_offset = my_offset*u[n].view().stride(0);
    MPI_Allgather(&my_send_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT,
                  pmap_->gridComm());
    MPI_Allgather(&my_send_offset, 1, MPI_INT, displs.data(), 1, MPI_INT,
                  pmap_->gridComm());

    // Now send the data
    MPI_Allgatherv(u[n].view().data(), my_send_size,
                   DistContext::toMpiType<ttb_real>(),
                   out[n].view().data(), recvcounts.data(), displs.data(),
                   DistContext::toMpiType<ttb_real>(),
                   pmap_->gridComm());
    pmap_->gridBarrier();
  }

  return out;
}

template <typename ExecSpace>
template <typename ExecSpaceSrc>
FacMatrixT<ExecSpace>
DistTensorContext<ExecSpace>::
exportFromRoot(const ttb_indx dim, const FacMatrixT<ExecSpaceSrc>& u) const
{
  // Broadcast factor matrix values from 0 to all procs
  pmap_->gridBcast(u.view().data(), u.view().span(), 0);
  pmap_->gridBarrier();

  // Copy our portion of u into exp
  const ttb_indx num_my_rows = ktensor_local_dims_[dim];
  const ttb_indx my_offset = ktensor_local_offsets_[dim];
  FacMatrixT<ExecSpace> exp = FacMatrixT<ExecSpace>(num_my_rows, u.nCols());
  auto rng = std::make_pair(my_offset, my_offset+num_my_rows);
  auto sub = Kokkos::subview(u.view(), rng, Kokkos::ALL);
  Kokkos::deep_copy(exp.view(), sub);

  return exp;
}

template <typename ExecSpace>
template <typename ExecSpaceDst>
FacMatrixT<ExecSpaceDst>
DistTensorContext<ExecSpace>::
importToRoot(const ttb_indx dim, const FacMatrixT<ExecSpace>& u) const
{
  FacMatrixT<ExecSpaceDst> out(global_dims_[dim], u.nCols());

  const ttb_indx num_my_rows = ktensor_local_dims_[dim];
  const ttb_indx my_offset = ktensor_local_offsets_[dim];
  gt_assert(num_my_rows == u.nRows());

  // Send sizes and offsets to proc 0
  std::vector<int> recvcounts(pmap_->gridSize(), 0);
  std::vector<int> displs(pmap_->gridSize(), 0);
  int my_send_size = num_my_rows*u.view().stride(0);
  int my_send_offset = my_offset*u.view().stride(0);
  MPI_Gather(&my_send_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0,
             pmap_->gridComm());
  MPI_Gather(&my_send_offset, 1, MPI_INT, displs.data(), 1, MPI_INT, 0,
             pmap_->gridComm());

  // Now send the data
  MPI_Gatherv(u.view().data(), my_send_size,
              DistContext::toMpiType<ttb_real>(),
              out.view().data(), recvcounts.data(), displs.data(),
              DistContext::toMpiType<ttb_real>(), 0,
              pmap_->gridComm());
  pmap_->gridBarrier();

  return out;
}

#else

template <typename ExecSpace>
class DistTensorContext {
public:
  DistTensorContext() = default;
  ~DistTensorContext() = default;

  DistTensorContext(DistTensorContext&&) = default;
  DistTensorContext(const DistTensorContext&) = default;
  DistTensorContext& operator=(DistTensorContext&&) = default;
  DistTensorContext& operator=(const DistTensorContext&) = default;

  template <typename ExecSpaceSrc>
  DistTensorContext(const DistTensorContext<ExecSpaceSrc>& src) :
    global_dims_(src.global_dims_),
    pmap_(src.pmap_),
    ktensor_local_dims_(src.ktensor_local_dims_) {}

  std::tuple< SptensorT<ExecSpace>, TensorT<ExecSpace> >
  distributeTensor(const ptree& tree,
                   const AlgParams& algParams);
  std::tuple< SptensorT<ExecSpace>, TensorT<ExecSpace> >
  distributeTensor(const std::string& file,
                   const ttb_indx index_base,
                   const bool compressed,
                   const ptree& tree,
                   const AlgParams& algParams);
  template <typename ExecSpaceSrc>
  SptensorT<ExecSpace> distributeTensor(const SptensorT<ExecSpaceSrc>& X,
                                        const AlgParams& algParams = AlgParams())
  {
    dist_method = algParams.dist_update_method;
    const ttb_indx ndims = X.ndims();
    global_dims_.resize(ndims);
    ktensor_local_dims_.resize(ndims);
    for (ttb_indx i=0; i<ndims; ++i) {
      global_dims_[i] = X.size(i);
      ktensor_local_dims_[i] = X.size(i);
    }
    SptensorT<ExecSpace> X_dst = create_mirror_view(ExecSpace(), X);
    deep_copy(X_dst, X);
    return X_dst;
  }
  template <typename ExecSpaceSrc>
  TensorT<ExecSpace> distributeTensor(const TensorT<ExecSpaceSrc>& X,
                                      const AlgParams& algParams = AlgParams())
  {
    dist_method = algParams.dist_update_method;
    const ttb_indx ndims = X.ndims();
    global_dims_.resize(ndims);
    ktensor_local_dims_.resize(ndims);
    for (ttb_indx i=0; i<ndims; ++i) {
      global_dims_[i] = X.size(i);
      ktensor_local_dims_[i] = X.size(i);
    }
    TensorT<ExecSpace> X_dst = create_mirror_view(ExecSpace(), X);
    deep_copy(X_dst, X);
    return X_dst;
  }

  // Parallel info
  ttb_indx ndims() const { return global_dims_.size(); }
  const std::vector<ttb_indx>& dims() const { return global_dims_; }
  ttb_indx nprocs() const { return 1; }
  ttb_indx gridRank() const { return 0; }

  // Processor map for communication
  const ProcessorMap& pmap() const { return *pmap_; }
  std::shared_ptr<const ProcessorMap> pmap_ptr() const { return pmap_; }

  // Sptensor operations
  ttb_real globalNorm(const SptensorT<ExecSpace>& X) const { return X.norm(); }
  ttb_indx globalNNZ(const SptensorT<ExecSpace>& X) const { return X.nnz(); }
  ttb_real globalNumelFloat(const SptensorT<ExecSpace>& X) const { return X.numel_float(); }

  // Tensor operations
  ttb_real globalNorm(const TensorT<ExecSpace>& X) const { return X.norm(); }
  ttb_indx globalNNZ(const TensorT<ExecSpace>& X) const { return X.nnz(); }
  ttb_real globalNumelFloat(const TensorT<ExecSpace>& X) const { return X.numel_float(); }

  // Ktensor operations
  ttb_real globalNorm(const KtensorT<ExecSpace>& u) const { return std::sqrt(u.normFsq()); }
  template <typename ExecSpaceSrc>
  KtensorT<ExecSpace> exportFromRoot(const KtensorT<ExecSpaceSrc>& u) const {
    KtensorT<ExecSpace> v = create_mirror_view(ExecSpace(), u);
    deep_copy(v, u);
    return v;
  }
  template <typename ExecSpaceDst>
  KtensorT<ExecSpaceDst> importToRoot(const KtensorT<ExecSpace>& u) const {
    KtensorT<ExecSpaceDst> v = create_mirror_view(ExecSpaceDst(), u);
    deep_copy(v, u);
    return v;
  }
  template <typename ExecSpaceDst>
  KtensorT<ExecSpaceDst> importToAll(const KtensorT<ExecSpace>& u) const {
    KtensorT<ExecSpaceDst> v = create_mirror_view(ExecSpaceDst(), u);
    deep_copy(v, u);
    return v;
  }
  void allReduce(KtensorT<ExecSpace>& u,
                 const bool divide_by_grid_size = false) const {}
  void exportToFile(const KtensorT<ExecSpace>& out,
                    const std::string& file_name) const;

  // Factor matrix operations
  template <typename ExecSpaceSrc>
  FacMatrixT<ExecSpace> exportFromRoot(const ttb_indx dim, const FacMatrixT<ExecSpaceSrc>& u) const {
    FacMatrixT<ExecSpace> v = create_mirror_view(ExecSpace(), u);
    deep_copy(v, u);
    return v;
  }
  template <typename ExecSpaceDst>
  FacMatrixT<ExecSpaceDst> importToRoot(const ttb_indx dim, const FacMatrixT<ExecSpace>& u) const {
    FacMatrixT<ExecSpaceDst> v = create_mirror_view(ExecSpaceDst(), u);
    deep_copy(v, u);
    return v;
  }

  KtensorT<ExecSpace> readInitialGuess(const std::string& file_name) const;
  KtensorT<ExecSpace> randomInitialGuess(const SptensorT<ExecSpace>& X,
                                         const ttb_indx rank,
                                         const ttb_indx seed,
                                         const bool prng,
                                         const bool scale_guess_by_norm_x,
                                         const std::string& dist_method) const;
  KtensorT<ExecSpace> randomInitialGuess(const TensorT<ExecSpace>& X,
                                         const ttb_indx rank,
                                         const ttb_indx seed,
                                         const bool prng,
                                         const bool scale_guess_by_norm_x,
                                         const std::string& dist_method) const;
  KtensorT<ExecSpace> computeInitialGuess(const SptensorT<ExecSpace>& X,
                                          const ptree& input) const;

private:
  std::vector<ttb_indx> global_dims_;
  std::shared_ptr<ProcessorMap> pmap_;
  std::vector<ttb_indx> ktensor_local_dims_;

  Dist_Update_Method::type dist_method;

  template <typename ExecSpaceSrc> friend class DistTensorContext;
};

#endif

} // namespace Genten
