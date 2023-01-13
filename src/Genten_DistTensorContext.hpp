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
#include "Genten_Tensor.hpp"

#include "CMakeInclude.h"
#if defined(HAVE_DIST)
#include <cmath>
#include <fstream>
#include <memory>
#include "Genten_MPI_IO.hpp"
#include "Genten_SpTn_Util.hpp"
#include "Genten_Tpetra.hpp"
#include "Genten_DistFacMatrix.hpp"
#include "Genten_AlgParams.hpp"
#include "Kokkos_UnorderedMap.hpp"
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
  const std::vector<std::uint32_t>& dims() const { return global_dims_; }
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
  SptensorT<ExecSpace> distributeTensorData(
    const std::vector<G_MPI_IO::TDatatype<ttb_real>>& Tvec,
    const std::vector<std::uint32_t>& TensorDims,
    const std::vector<small_vector<int>>& blocking,
    const ProcessorMap& pmap,
    const AlgParams& algParams);
  TensorT<ExecSpace> distributeTensorData(
    const std::vector<double>& Tvec,
    const ttb_indx global_nnz, const ttb_indx global_offset,
    const std::vector<std::uint32_t>& TensorDims,
    const std::vector<small_vector<int>>& blocking,
    const ProcessorMap& pmap,
    const AlgParams& algParams);

  std::vector<std::uint32_t> local_dims_;
  std::vector<std::uint32_t> global_dims_;
  std::vector<std::uint32_t> ktensor_local_dims_;
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

// Helper declerations
namespace detail {

struct RangePair {
  int64_t lower;
  int64_t upper;
};

void fileFormatIsBinary(const std::string& file_name,
                        bool& is_binary_sparse,
                        bool& is_binary_dense);

template <typename ExecSpace>
auto rangesToIndexArray(const small_vector<RangePair>& ranges);
small_vector<int> singleDimUniformBlocking(int ModeLength, int ProcsInMode);

std::vector<G_MPI_IO::TDatatype<ttb_real>>
distributeTensorToVectors(const Sptensor& sp_tensor_host, uint64_t nnz,
                          MPI_Comm comm, int rank, int nprocs);

std::vector<ttb_real>
distributeTensorToVectors(const Tensor& dn_tensor_host, uint64_t nnz,
                          MPI_Comm comm, int rank, int nprocs,
                          ttb_indx& offset);

std::vector<G_MPI_IO::TDatatype<ttb_real>>
redistributeTensor(const std::vector<G_MPI_IO::TDatatype<ttb_real>>& Tvec,
                   const std::vector<std::uint32_t>& TensorDims,
                   const std::vector<small_vector<int>>& blocking,
                   const ProcessorMap& pmap);

std::vector<ttb_real>
redistributeTensor(const std::vector<ttb_real>& Tvec,
                   const ttb_indx global_nnz, const ttb_indx global_offset,
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
DistTensorContext<ExecSpace>::
globalNorm(const SptensorT<ExecSpace>& X) const
{
  const auto& values = X.getValArray();
  ttb_real norm2 = values.dot(values);
  norm2 = pmap_->gridAllReduce(norm2);
  return std::sqrt(norm2);
}

template <typename ExecSpace>
std::uint64_t
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
std::uint64_t
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
template <typename ExecSpaceSrc>
KtensorT<ExecSpace>
DistTensorContext<ExecSpace>::
exportFromRoot(const KtensorT<ExecSpaceSrc>& u) const
{
  const int nd = u.ndims();
  const int nc = u.ncomponents();
  assert(global_dims_.size() == nd);

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
  assert(global_dims_.size() == nd);

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
  }

  return out;
}

template <typename ExecSpace>
void
DistTensorContext<ExecSpace>::
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
  }

  return out;
}

template <typename ExecSpace>
void
DistTensorContext<ExecSpace>::
distributeTensor(const std::string& file, const ttb_indx index_base,
                 const bool compressed, const AlgParams& algParams,
                 SptensorT<ExecSpace>& X_sparse,
                 TensorT<ExecSpace>& X_dense)
{
  bool dense = false;
  bool is_binary_sparse = false;
  bool is_binary_dense = false;
  detail::fileFormatIsBinary(file, is_binary_sparse, is_binary_dense);
  if (is_binary_dense)
    dense = true;
  if (is_binary_sparse && index_base != 0)
    Genten::error("The binary sparse format only supports zero based indexing\n");
  if (is_binary_sparse && compressed)
    Genten::error("The binary sparse format does not support compression\n");

  std::vector<G_MPI_IO::TDatatype<ttb_real>> Tvec_sparse;
  std::vector<double> Tvec_dense;

  if (DistContext::rank() == 0)
    std::cout << "Reading tensor from file " << file << std::endl;

  DistContext::Barrier();
  auto t2 = MPI_Wtime();
  std::vector<std::uint32_t> global_dims;
  ttb_indx nnz, offset;
  if (is_binary_sparse) {
    // For binary sparse file, do a parallel read
    auto mpi_file = G_MPI_IO::openFile(DistContext::commWorld(), file);
    auto header = G_MPI_IO::readSparseHeader(DistContext::commWorld(),
                                             mpi_file);
    TensorInfo ti = header.toTensorInfo();
    global_dims = ti.dim_sizes;
    nnz = ti.nnz;
    Tvec_sparse = G_MPI_IO::parallelReadElements(DistContext::commWorld(),
                                                 mpi_file, header);
  }
  else if (is_binary_dense) {
    auto mpi_file = G_MPI_IO::openFile(DistContext::commWorld(), file);
    auto header = G_MPI_IO::readDenseHeader(DistContext::commWorld(),
                                            mpi_file);
    TensorInfo ti = header.toTensorInfo();
    global_dims = ti.dim_sizes;
    nnz = ti.nnz;
    offset = header.getLocalOffsetRange(DistContext::rank(),
                                        DistContext::nranks()).first;
    Tvec_dense = G_MPI_IO::parallelReadElements(DistContext::commWorld(),
                                                mpi_file, header);
  }
  else {
    // For non-binary, read on rank 0 and broadcast dimensions.
    // We do this instead of reading the header because we want to support
    // headerless files

    // Determine if the file is a sparse or dense tensor.  Dense tensors
    // must have a header with the first line being "tensor".  Sparse
    // tensors might not have a header
    std::ifstream f(file);
    if (!f)
      Genten::error("Cannot open input file: " + file);
    std::string line;
    std::getline(f, line);
    if (line == "tensor")
      dense = true;

    Tensor X_dense_host;
    Sptensor X_sparse_host;
    std::size_t ndims;
    small_vector<int> dims;

    if (gridRank() == 0) {
      if (dense) {
        import_tensor(file, X_dense_host);
        nnz = X_dense_host.nnz();
        ndims = X_dense_host.ndims();
        dims = small_vector<int>(ndims);
        for (std::size_t i=0; i<ndims; ++i)
          dims[i] = X_dense_host.size(i);
      }
      else {
        import_sptensor(file, X_sparse_host, index_base, compressed);
        nnz = X_sparse_host.nnz();
        ndims = X_sparse_host.ndims();
        dims = small_vector<int>(ndims);
        for (std::size_t i=0; i<ndims; ++i)
          dims[i] = X_sparse_host.size(i);
      }
    }
    DistContext::Bcast(nnz, 0);
    DistContext::Bcast(ndims, 0);
    if (gridRank() != 0)
      dims = small_vector<int>(ndims);
    DistContext::Bcast(dims, 0);

    global_dims = std::vector<std::uint32_t>(ndims);
    for (std::size_t i=0; i<ndims; ++i)
      global_dims[i] = dims[i];

    if (dense)
      Tvec_dense = detail::distributeTensorToVectors(
        X_dense_host, nnz, DistContext::commWorld(), DistContext::rank(),
        DistContext::nranks(), offset);
    else
      Tvec_sparse = detail::distributeTensorToVectors(
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

    const bool use_tpetra =
      algParams.dist_update_method == Genten::Dist_Update_Method::Tpetra;

    pmap_ = std::shared_ptr<ProcessorMap>(new ProcessorMap(global_dims_,
                                                           use_tpetra));
    detail::printGrids(*pmap_);

    global_blocking_ =
      detail::generateUniformBlocking(global_dims_, pmap_->gridDims());
    detail::printBlocking(*pmap_, global_blocking_);
  }

  if (dense)
    X_dense = distributeTensorData(Tvec_dense, nnz, offset,
                                   global_dims_, global_blocking_,
                                   *pmap_, algParams);
  else
    X_sparse = distributeTensorData(Tvec_sparse, global_dims_, global_blocking_,
                                    *pmap_, algParams);
}

template <typename ExecSpace>
template <typename ExecSpaceSrc>
SptensorT<ExecSpace>
DistTensorContext<ExecSpace>::
distributeTensor(const SptensorT<ExecSpaceSrc>& X, const AlgParams& algParams)
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

    const bool use_tpetra =
      algParams.dist_update_method == Genten::Dist_Update_Method::Tpetra;
    pmap_ = std::shared_ptr<ProcessorMap>(new ProcessorMap(global_dims_,
                                                           use_tpetra));
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

  return distributeTensorData(Tvec, global_dims_, global_blocking_, *pmap_,
                              algParams);
}

template <typename ExecSpace>
template <typename ExecSpaceSrc>
TensorT<ExecSpace>
DistTensorContext<ExecSpace>::
distributeTensor(const TensorT<ExecSpaceSrc>& X, const AlgParams& algParams)
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

    const bool use_tpetra =
      algParams.dist_update_method == Genten::Dist_Update_Method::Tpetra;
    pmap_ = std::shared_ptr<ProcessorMap>(new ProcessorMap(global_dims_,
                                                           use_tpetra));
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

  return distributeTensorData(Tvec, global_dims_, global_blocking_, *pmap_,
                              algParams);
}

template <typename ExecSpace>
SptensorT<ExecSpace>
DistTensorContext<ExecSpace>::
distributeTensorData(const std::vector<G_MPI_IO::TDatatype<ttb_real>>& Tvec,
                     const std::vector<std::uint32_t>& TensorDims,
                     const std::vector<small_vector<int>>& blocking,
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

    // Do not subtract off the lower bound of the bounding box for Tpetra
    // since it will map GIDs to LIDs below
    if (!use_tpetra)
      for (auto j = 0; j < ndims; ++j)
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
    for (auto dim=0; dim<ndims; ++dim) {
      const tpetra_go_type numGlobalElements = global_dims_[dim];
      factorMap[dim] = Teuchos::rcp(new tpetra_map_type<ExecSpace>(numGlobalElements, indexBase, tpetra_comm));
    }

    // Build hash maps of tensor nonzeros in each dimension for:
    //   1.  Mapping tensor GIDs to LIDS
    //   2.  Constructing overlapping Tpetra map for MTTKRP
    using unordered_map_type = Kokkos::UnorderedMap<tpetra_go_type,tpetra_lo_type,DefaultHostExecutionSpace>;
    std::vector<unordered_map_type> map(ndims);
    std::vector<tpetra_lo_type> cnt(ndims, 0);
    for (auto dim=0; dim<ndims; ++dim)
      map[dim].rehash(local_dims_[dim]);
    for (auto i=0; i<local_nnz; ++i) {
      for (auto dim=0; dim<ndims; ++dim) {
        auto gid = subs[i][dim];
        auto idx = map[dim].find(gid);
        if (!map[dim].valid_at(idx)) {
          tpetra_lo_type lid = cnt[dim]++;
          if (map[dim].insert(gid,lid).failed())
            Genten::error("Insertion of GID failed, something is wrong!");
        }
      }
    }
    for (auto dim=0; dim<ndims; ++dim)
      assert(cnt[dim] == map[dim].size());

    // Map tensor GIDs to LIDs.  We use the hash-map for this instead of just
    // subtracting off the lower bound because there may be empty slices
    // in our block (and LIDs must be contiguous)
    std::vector<std::vector<ttb_indx>> subs_gids(local_nnz);
    for (auto i=0; i<local_nnz; ++i) {
      subs_gids[i].resize(ndims);
      for (auto dim=0; dim<ndims; ++dim) {
        const auto gid = subs[i][dim];
        const auto idx = map[dim].find(gid);
        const auto lid = map[dim].value_at(idx);
        subs[i][dim] = lid;
        subs_gids[i][dim] = gid;
      }
    }

    // Construct overlap maps for each dimension
    overlapFactorMap.resize(ndims);
    for (auto dim=0; dim<ndims; ++dim) {
      Kokkos::View<tpetra_go_type*,ExecSpace> gids("gids", cnt[dim]);
      auto gids_host = create_mirror_view(gids);
      const auto sz = map[dim].capacity();
      for (auto idx=0; idx<sz; ++idx) {
        if (map[dim].valid_at(idx)) {
          const auto gid = map[dim].key_at(idx);
          const auto lid = map[dim].value_at(idx);
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
        for (auto i=0; i<local_nnz; ++i)
          subs[i][dim] =
            overlapFactorMap[dim]->getLocalElement(subs_gids[i][dim]);
      }
    }

    // Build sparse tensor
    std::vector<ttb_indx> lower(ndims), upper(ndims);
    for (auto dim=0; dim<ndims; ++dim) {
      lower[dim] = range[dim].lower;
      upper[dim] = range[dim].upper;
    }
    Sptensor sptensor_host(indices, values, subs, subs_gids, lower, upper);
    sptensor = create_mirror_view(ExecSpace(), sptensor_host);
    deep_copy(sptensor, sptensor_host);
    for (auto dim=0; dim<ndims; ++dim) {
      sptensor.factorMap(dim) = factorMap[dim];
      sptensor.tensorMap(dim) = overlapFactorMap[dim];
      if (!overlapFactorMap[dim]->isSameAs(*factorMap[dim]))
        sptensor.importer(dim) =
          Teuchos::rcp(new tpetra_import_type<ExecSpace>(
                         factorMap[dim], overlapFactorMap[dim]));
    }

    // Build maps and importers for importing factor matrices to/from root
    rootMap.resize(ndims);
    rootImporter.resize(ndims);
    for (auto dim=0; dim<ndims; ++dim) {
      const Tpetra::global_size_t numGlobalElements = global_dims_[dim];
      const size_t numLocalElements =
        (gridRank() == 0) ? global_dims_[dim] : 0;
      rootMap[dim] = Teuchos::rcp(new tpetra_map_type<ExecSpace>(numGlobalElements, numLocalElements, indexBase, tpetra_comm));
      rootImporter[dim] = Teuchos::rcp(new tpetra_import_type<ExecSpace>(factorMap[dim], rootMap[dim]));
    }
  }
#else
  if (use_tpetra)
    Genten::error("Cannot use tpetra distribution approach without enabling Tpetra!");
#endif

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

  sptensor.setProcessorMap(&pmap);
  return sptensor;
}

template <typename ExecSpace>
TensorT<ExecSpace>
DistTensorContext<ExecSpace>::
distributeTensorData(const std::vector<ttb_real>& Tvec,
                     const ttb_indx global_nnz, const ttb_indx global_offset,
                     const std::vector<std::uint32_t>& TensorDims,
                     const std::vector<small_vector<int>>& blocking,
                     const ProcessorMap& pmap, const AlgParams& algParams)
{
  const bool use_tpetra =
    algParams.dist_update_method == Genten::Dist_Update_Method::Tpetra;

  DistContext::Barrier();
  auto t4 = MPI_Wtime();

  // Now redistribute to final format
  auto values =
    detail::redistributeTensor(Tvec, global_nnz, global_offset,
                               global_dims_, global_blocking_, *pmap_);

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

  const auto local_nnz = values.size();

  TensorT<ExecSpace> tensor;
  if (!use_tpetra) {
    Tensor tensor_host(IndxArray(ndims, indices.data()),
                       Array(local_nnz, values.data(), false));
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
    for (auto dim=0; dim<ndims; ++dim) {
      const tpetra_go_type numGlobalElements = global_dims_[dim];
      factorMap[dim] = Teuchos::rcp(new tpetra_map_type<ExecSpace>(numGlobalElements, indexBase, tpetra_comm));
    }

    // Build tensor maps based on slices owned by each processor
    overlapFactorMap.resize(ndims);
    for (auto dim=0; dim<ndims; ++dim) {
      const auto sz = local_dims_[dim];
      Kokkos::View<tpetra_go_type*,ExecSpace> gids("gids", sz);
      auto gids_host = create_mirror_view(gids);
      const auto l = range[dim].lower;
      for (auto idx=0; idx<sz; ++idx)
        gids_host[idx] = l + idx;
      deep_copy(gids, gids_host);
      overlapFactorMap[dim] =
        Teuchos::rcp(new tpetra_map_type<ExecSpace>(invalid, gids, indexBase,
                                                    tpetra_comm));
    }

    // Build dense tensor
    Tensor tensor_host(IndxArray(ndims, indices.data()),
                       Array(local_nnz, values.data(), false));
    IndxArray lower = tensor_host.getLowerBounds();
    IndxArray upper = tensor_host.getUpperBounds();
    for (auto dim=0; dim<ndims; ++dim) {
      lower[dim] = range[dim].lower;
      upper[dim] = range[dim].upper;
    }
    tensor = create_mirror_view(ExecSpace(), tensor_host);
    deep_copy(tensor, tensor_host);
    for (auto dim=0; dim<ndims; ++dim) {
      tensor.factorMap(dim) = factorMap[dim];
      tensor.tensorMap(dim) = overlapFactorMap[dim];
      if (!overlapFactorMap[dim]->isSameAs(*factorMap[dim]))
        tensor.importer(dim) =
          Teuchos::rcp(new tpetra_import_type<ExecSpace>(
                         factorMap[dim], overlapFactorMap[dim]));
    }

    // Build maps and importers for importing factor matrices to/from root
    rootMap.resize(ndims);
    rootImporter.resize(ndims);
    for (auto dim=0; dim<ndims; ++dim) {
      const Tpetra::global_size_t numGlobalElements = global_dims_[dim];
      const size_t numLocalElements =
        (gridRank() == 0) ? global_dims_[dim] : 0;
      rootMap[dim] = Teuchos::rcp(new tpetra_map_type<ExecSpace>(numGlobalElements, numLocalElements, indexBase, tpetra_comm));
      rootImporter[dim] = Teuchos::rcp(new tpetra_import_type<ExecSpace>(factorMap[dim], rootMap[dim]));
    }
  }
#else
  if (use_tpetra)
    Genten::error("Cannot use tpetra distribution approach without enabling Tpetra!");
#endif

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

  tensor.setProcessorMap(&pmap);
  return tensor;
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

template <typename ExecSpace>
class DistTensorContext {
public:
  DistTensorContext() = default;
  ~DistTensorContext() = default;

  DistTensorContext(DistTensorContext&&) = default;
  DistTensorContext(const DistTensorContext&) = default;
  DistTensorContext& operator=(DistTensorContext&&) = default;
  DistTensorContext& operator=(const DistTensorContext&) = default;

  SptensorT<ExecSpace> distributeTensor(const std::string& file,
                                        const ttb_indx index_base,
                                        const bool compressed,
                                        const AlgParams& algParams)
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
  template <typename ExecSpaceSrc>
  SptensorT<ExecSpace> distributeTensor(const SptensorT<ExecSpaceSrc>& X,
                                        const AlgParams& algParams = AlgParams())
  {
    SptensorT<ExecSpace> X_dst = create_mirror_view(ExecSpace(), X);
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
  ttb_real globalNorm(const SptensorT<ExecSpace>& X) const { return X.norm(); }
  std::uint64_t globalNNZ(const SptensorT<ExecSpace>& X) const { return X.nnz(); }
  ttb_real globalNumelFloat(const SptensorT<ExecSpace>& X) const { return X.numel_float(); }

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
  KtensorT<ExecSpace> computeInitialGuess(const SptensorT<ExecSpace>& X,
                                          const ptree& input) const;

private:
  std::vector<std::uint32_t> global_dims_;
  std::shared_ptr<ProcessorMap> pmap_;
};

#endif

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
                   const int rank,
                   const int seed,
                   const bool prng,
                   const bool scale_guess_by_norm_x,
                   const std::string& dist_method) const
{
  const ttb_indx nd = X.ndims();
  const ttb_real norm_x = globalNorm(X);
  RandomMT cRMT(seed);

  Genten::KtensorT<ExecSpace> u;

  if (dist_method == "serial") {
    // Compute random ktensor on rank 0 and broadcast to all proc's
    IndxArrayT<ExecSpace> sz(nd);
    auto hsz = create_mirror_view(sz);
    for (int i=0; i<nd; ++i)
      hsz[i] = global_dims_[i];
    deep_copy(sz,hsz);
    Genten::KtensorT<ExecSpace> u0(rank, nd, sz);
    if (pmap_->gridRank() == 0) {
      u0.setWeights(1.0);
      u0.setMatricesScatter(false, prng, cRMT);
    }
    u = exportFromRoot(u0);
  }
  else if (dist_method == "parallel" || dist_method == "parallel-drew") {
#ifdef HAVE_TPETRA
    if (tpetra_comm != Teuchos::null) {
      const int nd = X.ndims();
      IndxArrayT<ExecSpace> sz(nd);
      auto hsz = create_mirror_view(sz);
      for (int i=0; i<nd; ++i)
        hsz[i] = factorMap[i]->getLocalNumElements();
      deep_copy(sz,hsz);
      u = KtensorT<ExecSpace>(rank, nd, sz);
      u.setWeights(1.0);
      u.setMatricesScatter(false, prng, cRMT);
      u.setProcessorMap(&pmap());
    }
    else
#endif
    {
      u = KtensorT<ExecSpace>(rank, nd, X.size());
      u.setWeights(1.0);
      u.setMatricesScatter(false, prng, cRMT);
      u.setProcessorMap(&pmap());
      allReduce(u, true); // make replicated proc's consistent
    }
  }
  else
    Genten::error("Unknown distributed-guess method: " + dist_method);

  if (dist_method == "parallel-drew")
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
                   const int rank,
                   const int seed,
                   const bool prng,
                   const bool scale_guess_by_norm_x,
                   const std::string& dist_method) const
{
  const ttb_indx nd = X.ndims();
  const ttb_real norm_x = globalNorm(X);
  RandomMT cRMT(seed);

  Genten::KtensorT<ExecSpace> u;

  if (dist_method == "serial") {
    // Compute random ktensor on rank 0 and broadcast to all proc's
    IndxArrayT<ExecSpace> sz(nd);
    auto hsz = create_mirror_view(sz);
    for (int i=0; i<nd; ++i)
      hsz[i] = global_dims_[i];
    deep_copy(sz,hsz);
    Genten::KtensorT<ExecSpace> u0(rank, nd, sz);
    if (pmap_->gridRank() == 0) {
      u0.setWeights(1.0);
      u0.setMatricesScatter(false, prng, cRMT);
    }
    u = exportFromRoot(u0);
  }
  else if (dist_method == "parallel" || dist_method == "parallel-drew") {
#ifdef HAVE_TPETRA
    if (tpetra_comm != Teuchos::null) {
      const int nd = X.ndims();
      IndxArrayT<ExecSpace> sz(nd);
      auto hsz = create_mirror_view(sz);
      for (int i=0; i<nd; ++i)
        hsz[i] = factorMap[i]->getLocalNumElements();
      deep_copy(sz,hsz);
      u = KtensorT<ExecSpace>(rank, nd, sz);
      u.setWeights(1.0);
      u.setMatricesScatter(false, prng, cRMT);
      u.setProcessorMap(&pmap());
    }
    else
#endif
    {
      u = KtensorT<ExecSpace>(rank, nd, X.size());
      u.setWeights(1.0);
      u.setMatricesScatter(false, prng, cRMT);
      u.setProcessorMap(&pmap());
      allReduce(u, true); // make replicated proc's consistent
    }
  }
  else
    Genten::error("Unknown distributed-guess method: " + dist_method);

  if (dist_method == "parallel-drew")
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
    const int seed = kt_input.get<int>("seed",std::random_device{}());
    const bool prng = kt_input.get<bool>("prng",true);
    const bool scale_by_x = kt_input.get<bool>("scale-guess-by-norm-x", false);
    const int nc = kt_input.get<int>("rank");
    const std::string dist_method =
      kt_input.get<std::string>("distributed-guess", "serial");
    u = randomInitialGuess(X, nc, seed, prng, scale_by_x, dist_method);
  }
  else
    Genten::error("Unknown initial-guess method: " + init_method);

  return u;
}

} // namespace Genten
