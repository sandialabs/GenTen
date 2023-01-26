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

  void distributeTensor(const std::string& file,
                        const ttb_indx index_base,
                        const bool compressed,
                        const ptree& tree,
                        const AlgParams& algParams,
                        SptensorT<ExecSpace>& X_sparse,
                        TensorT<ExecSpace>& X_dense);
  template <typename ExecSpaceSrc>
  SptensorT<ExecSpace> distributeTensor(const SptensorT<ExecSpaceSrc>& X,
                                        const AlgParams& algParams = AlgParams());
  template <typename ExecSpaceSrc>
  TensorT<ExecSpace> distributeTensor(const TensorT<ExecSpaceSrc>& X,
                                      const AlgParams& algParams = AlgParams());

  // Parallel info
  std::int32_t ndims() const { return global_dims_.size(); }
  const std::vector<ttb_indx>& dims() const { return global_dims_; }
  static std::int64_t nprocs() { return DistContext::nranks(); }
  static std::int64_t gridRank() { return DistContext::rank(); }
  const std::vector<small_vector<int>>& blocking() const { return global_blocking_; }

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
  std::uint64_t globalNNZ(const SptensorT<ExecSpace>& X) const;
  ttb_real globalNumelFloat(const SptensorT<ExecSpace>& X) const;

  // Tensor operations
  ttb_real globalNorm(const TensorT<ExecSpace>& X) const;
  std::uint64_t globalNNZ(const TensorT<ExecSpace>& X) const;
  ttb_real globalNumelFloat(const TensorT<ExecSpace>& X) const;

  // Ktensor operations
  ttb_real globalNorm(const KtensorT<ExecSpace>& X) const;
  template <typename ExecSpaceSrc>
  KtensorT<ExecSpace> exportFromRoot(const KtensorT<ExecSpaceSrc>& u) const;
  template <typename ExecSpaceDst>
  KtensorT<ExecSpaceDst> importToRoot(const KtensorT<ExecSpace>& u) const;
  void allReduce(KtensorT<ExecSpace>& u,
                 const bool divide_by_grid_size = false) const;
  void exportToFile(const KtensorT<ExecSpace>& u,
                    const std::string& file_name) const;

  // Factor matrix operations
  template <typename ExecSpaceSrc>
  FacMatrixT<ExecSpace> exportFromRoot(const int dim, const FacMatrixT<ExecSpaceSrc>& u) const;
  template <typename ExecSpaceDst>
  FacMatrixT<ExecSpaceDst> importToRoot(const int dim, const FacMatrixT<ExecSpace>& u) const;

  // Initial guess computations
  KtensorT<ExecSpace> readInitialGuess(const std::string& file_name) const;
  KtensorT<ExecSpace> randomInitialGuess(const SptensorT<ExecSpace>& X,
                                         const int rank,
                                         const int seed,
                                         const bool prng,
                                         const bool scale_guess_by_norm_x,
                                         const std::string& dist_method) const;
  KtensorT<ExecSpace> randomInitialGuess(const TensorT<ExecSpace>& X,
                                         const int rank,
                                         const int seed,
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
    const std::vector<small_vector<int>>& blocking,
    const ProcessorMap& pmap,
    const AlgParams& algParams);
  TensorT<ExecSpace> distributeTensorData(
    const std::vector<double>& Tvec,
    const ttb_indx global_nnz, const ttb_indx global_offset,
    const std::vector<ttb_indx>& TensorDims,
    const std::vector<small_vector<int>>& blocking,
    const ProcessorMap& pmap,
    const AlgParams& algParams);

  std::vector<ttb_indx> local_dims_;
  std::vector<ttb_indx> global_dims_;
  std::vector<ttb_indx> ktensor_local_dims_;
  std::shared_ptr<ProcessorMap> pmap_;
  std::vector<small_vector<int>> global_blocking_;

#ifdef HAVE_TPETRA
  Teuchos::RCP<const Teuchos::Comm<int> > tpetra_comm;
  std::vector< Teuchos::RCP<const tpetra_map_type<ExecSpace> > > factorMap;
  std::vector< Teuchos::RCP<const tpetra_map_type<ExecSpace> > > overlapFactorMap;
  std::vector< Teuchos::RCP<const tpetra_map_type<ExecSpace> > > rootMap;
  std::vector< Teuchos::RCP<const tpetra_import_type<ExecSpace> > > rootImporter;
#endif

  MPI_Datatype mpiElemType_ = DistContext::toMpiType<ttb_real>();
};

template <typename ExecSpace>
template <typename ExecSpaceSrc>
SptensorT<ExecSpace>
DistTensorContext<ExecSpace>::
distributeTensor(const SptensorT<ExecSpaceSrc>& X, const AlgParams& algParams)
{
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
  const int nd = u.ndims();
  const int nc = u.ncomponents();
  assert(int(global_dims_.size()) == nd);

  Genten::KtensorT<ExecSpace> exp;
  IndxArrayT<ExecSpace> sz(nd);
  auto hsz = create_mirror_view(sz);

#ifdef HAVE_TPETRA
  if (tpetra_comm != Teuchos::null) {
    for (int i=0; i<nd; ++i)
      hsz[i] = factorMap[i]->getLocalNumElements();
    deep_copy(sz,hsz);
    exp = Genten::KtensorT<ExecSpace>(nc, nd, sz);
    exp.setMatrices(0.0);
    deep_copy(exp.weights(), u.weights());
    for (int i=0; i<nd; ++i) {
      if (rootImporter[i] != Teuchos::null) {
        FacMatrixT<ExecSpace> ui = create_mirror_view(ExecSpace(), u[i]);
        deep_copy(ui, u[i]);
        DistFacMatrix<ExecSpace> dist_u(ui, rootMap[i]);
        DistFacMatrix<ExecSpace> dist_exp(exp[i], factorMap[i]);
        dist_exp.doExport(dist_u, *(rootImporter[i]), Tpetra::INSERT);
      }
      else
        deep_copy(exp[i], u[i]);
    }
  }
  else
#endif
  {
    // Broadcast ktensor values from 0 to all procs
    for (int i=0; i<nd; ++i)
      pmap_->gridBcast(u[i].view().data(), u[i].view().span(), 0);
    pmap_->gridBcast(
      u.weights().values().data(), u.weights().values().span(),0);
    pmap_->gridBarrier();

    // Copy our portion from u into ktensor_
    for (int i=0; i<nd; ++i)
      hsz[i] = local_dims_[i];
    deep_copy(sz,hsz);
    exp = Genten::KtensorT<ExecSpace>(nc, nd, sz);
    exp.setMatrices(0.0);
    deep_copy(exp.weights(), u.weights());
    for (int i=0; i<nd; ++i) {
      auto coord = pmap_->gridCoord(i);
      auto rng = std::make_pair(global_blocking_[i][coord],
                                global_blocking_[i][coord + 1]);
      auto sub = Kokkos::subview(u[i].view(), rng, Kokkos::ALL);
      deep_copy(exp[i].view(), sub);
    }
  }

  exp.setProcessorMap(&pmap());
  return exp;
}

namespace {

template <typename T> bool isValueSame(T x) {
  T p[]{-x, x};
  MPI_Allreduce(MPI_IN_PLACE, p, 2, DistContext::toMpiType<T>(), MPI_MIN,
                MPI_COMM_WORLD);
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
}
#endif

} // namespace

template <typename ExecSpace>
template <typename ExecSpaceDst>
KtensorT<ExecSpaceDst>
DistTensorContext<ExecSpace>::
importToRoot(const KtensorT<ExecSpace>& u) const
{
  const bool print =
    DistContext::isDebug() && (pmap_->gridRank() == 0);

  const int nd = u.ndims();
  const int nc = u.ncomponents();
  assert(int(global_dims_.size()) == nd);

  IndxArrayT<ExecSpaceDst> sizes_idx(nd);
  auto sizes_idx_host = create_mirror_view(sizes_idx);
  for (int i=0; i<nd; ++i) {
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

#ifdef HAVE_TPETRA
  if (tpetra_comm != Teuchos::null) {
    for (int i=0; i<nd; ++i) {
      if (rootImporter[i] != Teuchos::null) {
        FacMatrixT<ExecSpace> oi = create_mirror_view(ExecSpace(), out[i]);
        DistFacMatrix<ExecSpace> dist_u(u[i], factorMap[i]);
        DistFacMatrix<ExecSpace> dist_out(oi, rootMap[i]);
        dist_out.doImport(dist_u, *(rootImporter[i]), Tpetra::INSERT);
        deep_copy(out[i], oi);
      }
      else
        deep_copy(out[i], u[i]);
    }
  }
  else
#endif
  {
    if (print)
      std::cout << "Blocking:\n";

    small_vector<int> grid_pos(nd, 0);
    for (int d=0; d<nd; ++d) {
      std::vector<int> recvcounts(pmap_->gridSize(), 0);
      std::vector<int> displs(pmap_->gridSize(), 0);
      const auto nblocks = global_blocking_[d].size() - 1;
      if (print)
        std::cout << "\tDim(" << d << ")\n";
      for (auto b = 0u; b < nblocks; ++b) {
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
  }

  return out;
}

template <typename ExecSpace>
template <typename ExecSpaceSrc>
FacMatrixT<ExecSpace>
DistTensorContext<ExecSpace>::
exportFromRoot(const int dim, const FacMatrixT<ExecSpaceSrc>& u) const
{
  FacMatrixT<ExecSpace> exp;

#ifdef HAVE_TPETRA
  if (tpetra_comm != Teuchos::null) {
     exp = FacMatrixT<ExecSpace>(factorMap[dim]->getLocalNumElements(),
                                 u.nCols());
     if (rootImporter[dim] != Teuchos::null) {
       FacMatrixT<ExecSpace> v = create_mirror_view(ExecSpace(), u);
       deep_copy(v, u);
       DistFacMatrix<ExecSpace> dist_u(v, rootMap[dim]);
       DistFacMatrix<ExecSpace> dist_exp(exp, factorMap[dim]);
       dist_exp.doExport(dist_u, *(rootImporter[dim]), Tpetra::INSERT);
     }
     else
       deep_copy(exp, u);
  }
  else
#endif
  {
    // Broadcast factor matrix values from 0 to all procs
    pmap_->gridBcast(u.view().data(), u.view().span(), 0);
    pmap_->gridBarrier();

    // Copy our portion
    exp = FacMatrixT<ExecSpace>(local_dims_[dim], u.nCols());
    auto coord = pmap_->gridCoord(dim);
    auto rng = std::make_pair(global_blocking_[dim][coord],
                              global_blocking_[dim][coord + 1]);
    auto sub = Kokkos::subview(u.view(), rng, Kokkos::ALL);
    deep_copy(exp.view(), sub);
  }

  return exp;
}

template <typename ExecSpace>
template <typename ExecSpaceDst>
FacMatrixT<ExecSpaceDst>
DistTensorContext<ExecSpace>::
importToRoot(const int dim, const FacMatrixT<ExecSpace>& u) const
{
  FacMatrixT<ExecSpaceDst> out(global_dims_[dim], u.nCols());

#ifdef HAVE_TPETRA
  if (tpetra_comm != Teuchos::null) {
    if (rootImporter[dim] != Teuchos::null) {
      FacMatrixT<ExecSpace> o = create_mirror_view(ExecSpace(), out);
      DistFacMatrix<ExecSpace> dist_u(u, factorMap[dim]);
      DistFacMatrix<ExecSpace> dist_o(o, rootMap[dim]);
      dist_o.doImport(dist_u, *(rootImporter[dim]), Tpetra::INSERT);
      deep_copy(out, o);
    }
    else
      deep_copy(out, u);
  }
  else
#endif
  {
    small_vector<int> grid_pos(global_dims_.size(), 0);
    std::vector<int> recvcounts(pmap_->gridSize(), 0);
    std::vector<int> displs(pmap_->gridSize(), 0);
    const auto nblocks = global_blocking_[dim].size() - 1;
    for (auto b = 0u; b < nblocks; ++b) {
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
  }

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

  void distributeTensor(const std::string& file,
                        const ttb_indx index_base,
                        const bool compressed,
                        const ptree& tree,
                        const AlgParams& algParams,
                        SptensorT<ExecSpace>& X_sparse,
                        TensorT<ExecSpace>& X_dense);
  template <typename ExecSpaceSrc>
  SptensorT<ExecSpace> distributeTensor(const SptensorT<ExecSpaceSrc>& X,
                                        const AlgParams& algParams = AlgParams())
  {
    SptensorT<ExecSpace> X_dst = create_mirror_view(ExecSpace(), X);
    deep_copy(X_dst, X);
    return X_dst;
  }
  template <typename ExecSpaceSrc>
  TensorT<ExecSpace> distributeTensor(const TensorT<ExecSpaceSrc>& X,
                                      const AlgParams& algParams = AlgParams())
  {
    TensorT<ExecSpace> X_dst = create_mirror_view(ExecSpace(), X);
    deep_copy(X_dst, X);
    return X_dst;
  }

  // Parallel info
  std::int32_t ndims() const { return global_dims_.size(); }
  const std::vector<ttb_indx>& dims() const { return global_dims_; }
  std::int64_t nprocs() const { return 1; }
  std::int64_t gridRank() const { return 0; }

  // Processor map for communication
  const ProcessorMap& pmap() const { return *pmap_; }
  std::shared_ptr<const ProcessorMap> pmap_ptr() const { return pmap_; }

  // Sptensor operations
  ttb_real globalNorm(const SptensorT<ExecSpace>& X) const { return X.norm(); }
  std::uint64_t globalNNZ(const SptensorT<ExecSpace>& X) const { return X.nnz(); }
  ttb_real globalNumelFloat(const SptensorT<ExecSpace>& X) const { return X.numel_float(); }

  // Tensor operations
  ttb_real globalNorm(const TensorT<ExecSpace>& X) const { return X.norm(); }
  std::uint64_t globalNNZ(const TensorT<ExecSpace>& X) const { return X.nnz(); }
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
  void allReduce(KtensorT<ExecSpace>& u,
                 const bool divide_by_grid_size = false) const {}
  void exportToFile(const KtensorT<ExecSpace>& out,
                    const std::string& file_name) const;

  // Factor matrix operations
  template <typename ExecSpaceSrc>
  FacMatrixT<ExecSpace> exportFromRoot(const int dim, const FacMatrixT<ExecSpaceSrc>& u) const {
    FacMatrixT<ExecSpace> v = create_mirror_view(ExecSpace(), u);
    deep_copy(v, u);
    return v;
  }
  template <typename ExecSpaceDst>
  FacMatrixT<ExecSpaceDst> importToRoot(const int dim, const FacMatrixT<ExecSpace>& u) const {
    FacMatrixT<ExecSpaceDst> v = create_mirror_view(ExecSpaceDst(), u);
    deep_copy(v, u);
    return v;
  }

  KtensorT<ExecSpace> readInitialGuess(const std::string& file_name) const;
  KtensorT<ExecSpace> randomInitialGuess(const SptensorT<ExecSpace>& X,
                                         const int rank,
                                         const int seed,
                                         const bool prng,
                                         const bool scale_guess_by_norm_x,
                                         const std::string& dist_method) const;
  KtensorT<ExecSpace> randomInitialGuess(const TensorT<ExecSpace>& X,
                                         const int rank,
                                         const int seed,
                                         const bool prng,
                                         const bool scale_guess_by_norm_x,
                                         const std::string& dist_method) const;
  KtensorT<ExecSpace> computeInitialGuess(const SptensorT<ExecSpace>& X,
                                          const ptree& input) const;

private:
  std::vector<ttb_indx> global_dims_;
  std::shared_ptr<ProcessorMap> pmap_;
};

#endif

} // namespace Genten
