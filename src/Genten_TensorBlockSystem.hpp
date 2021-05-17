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

#include "Genten_Annealer.hpp"
#include "Genten_Boost.hpp"
#include "Genten_DistContext.hpp"
#include "Genten_Driver.hpp"
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_GCP_SGD_Iter.hpp"
#include "Genten_GCP_SemiStratifiedSampler.hpp"
#include "Genten_GCP_ValueKernels.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Pmap.hpp"
#include "Genten_Sptensor.hpp"

#include <cmath>
#include <fstream>
#include <memory>
#include <random>

namespace Genten {

struct RangePair {
  int64_t lower;
  int64_t upper;
};

// Class to hold a block of the tensor and the factor matrix blocks that can
// be used to generate a representation of the tensor block, if the entire
// tensor is placed into one block this is just a TensorSystem
template <typename ElementType, typename ExecSpace = DefaultExecutionSpace>
class TensorBlockSystem {
  static_assert(std::is_floating_point<ElementType>::value,
                "DistSpSystem Requires that the element type be a floating "
                "point type.");

  /// The normal initialization method, inializes in parallel
  void init_distributed(std::string const &file_name, int indexbase,
                        int tensor_rank);

  // Initializaton for creating the tensor block system all on all ranks
  void init_independent(std::string const &file_name, int indexbase, int rank);

  /// Initialize the factor matrices
  void init_factors();

  void allReduceKT(KtensorT<ExecSpace> &g, bool divide_by_grid_size = true);
  void iAllReduceKT(KtensorT<ExecSpace> &g, std::vector<MPI_Request> &Requests);

  template <typename Loss> ElementType allReduceSGD(Loss const &loss);
  template <typename Loss> ElementType allReduceADAM(Loss const &loss);
  template <typename Loss> ElementType elasticAverageSGD(Loss const &loss);
  template <typename Loss> ElementType pickMethod(Loss const &loss);

  AlgParams setAlgParams() const;

public:
  TensorBlockSystem(ptree const &tree);
  ElementType getTensorNorm() const;

  std::uint64_t globalNNZ() const { return Ti_.nnz; }
  std::int64_t localNNZ() const { return sp_tensor_.nnz(); }
  std::int32_t ndims() const { return sp_tensor_.ndims(); }
  std::vector<int> const &dims() const { return Ti_.dim_sizes; }
  std::int64_t nprocs() const { return pmap_ptr_->gridSize(); }
  std::int64_t gridRank() const { return pmap_ptr_->gridRank(); }

  KtensorT<ExecSpace> collectFactorsRank0();
  ElementType SGD();

private:
  ptree input_;
  small_vector<RangePair> range_;
  SptensorT<ExecSpace> sp_tensor_;
  KtensorT<ExecSpace> Kfac_;
  std::unique_ptr<ProcessorMap> pmap_ptr_;
  TensorInfo Ti_;
};

// Helper declerations
namespace detail {
bool fileFormatIsBinary(std::string const &file_name);

template <typename ExecSpace>
auto rangesToIndexArray(small_vector<RangePair> const &ranges);
small_vector<int> singleDimMediumGrainBlocking(int ModeLength, int ProcsInMode);

std::vector<small_vector<int>>
generateMediumGrainBlocking(std::vector<int> ModeLengths,
                            small_vector<int> const &ProcGridSizes);

struct TDatatype {
  int coo[6] = {-1, -1, -1, -1, -1, -1};
  double val;
};

std::vector<TDatatype> distributeTensorToVectors(std::ifstream &ifs,
                                                 uint64_t nnz, int indexbase,
                                                 MPI_Comm comm, int rank,
                                                 int nprocs);

std::vector<TDatatype> redistributeTensor(
    std::vector<TDatatype> const &Tvec, std::vector<int> const &TensorDims,
    std::vector<small_vector<int>> const &blocking, ProcessorMap const &pmap);

template <typename ExecSpace>
void printRandomElements(SptensorT<ExecSpace> const &tensor,
                         int num_elements_per_rank, ProcessorMap const &pmap,
                         small_vector<RangePair> const &ranges);
} // namespace detail

template <typename ElementType, typename ExecSpace>
TensorBlockSystem<ElementType, ExecSpace>::TensorBlockSystem(ptree const &tree)
    : input_(tree.get_child("tensor")) {
  const auto file_name = input_.get<std::string>("file");
  const auto indexbase = input_.get<int>("indexbase", 0);
  const auto rank = input_.get<int>("rank", 5);

  const auto init_strategy =
      input_.get<std::string>("tensor.initialization", "distributed");
  if (init_strategy == "distributed") {
    init_distributed(file_name, indexbase, rank);
  } else if (init_strategy == "replicated") {
    // TODO This should really do single then bcast
    init_independent(file_name, indexbase, rank);
  } else if (init_strategy == "single") {
    if (DistContext::rank() == 0) {
      init_independent(file_name, indexbase, rank);
    }
  } else {
    throw std::logic_error("Tensor initialization must be one of of "
                           "{distributed, replicated, or single}\n");
  }
}

template <typename ElementType, typename ExecSpace>
void TensorBlockSystem<ElementType, ExecSpace>::init_independent(
    std::string const &file_name, int indexbase, int rank) {
  if (detail::fileFormatIsBinary(file_name)) {
    throw std::logic_error(
        "I can't quite read the binary format just yet, sorry.");
  }

  typename SptensorT<ExecSpace>::HostMirror sp_tensor_host;
  import_sptensor(file_name, sp_tensor_host, indexbase, false, false);
  std::cout << "The size of the sp_tensor is: " << sp_tensor_host.size()
            << "\n";
  sp_tensor_ = create_mirror_view(ExecSpace{}, sp_tensor_host);
  deep_copy(sp_tensor_, sp_tensor_host);

  auto const &index_view = sp_tensor_host.size();
  const auto ndims = index_view.size();
  range_.reserve(ndims);
  for (auto i = 0; i < ndims; ++i) {
    range_.push_back({0ll, int64_t(index_view[i])});
  }

  Kfac_ = KtensorT<ExecSpace>(rank, ndims,
                              detail::rangesToIndexArray<ExecSpace>(range_));
  Kfac_.setMatrices(0.0);
}

// We'll put this down here to save some space
template <typename ElementType, typename ExecSpace>
void TensorBlockSystem<ElementType, ExecSpace>::init_distributed(
    std::string const &file_name, int indexbase, int tensor_rank) {
  if (detail::fileFormatIsBinary(file_name)) {
    throw std::logic_error(
        "I can't quite read the binary format just yet, sorry.");
  }

  // TODO Bcast Ti so we don't read it on every node
  std::ifstream tensor_file(file_name);
  Ti_ = read_sptensor_header(tensor_file);
  const auto ndims = Ti_.dim_sizes.size();
  pmap_ptr_ = std::unique_ptr<ProcessorMap>(
      new ProcessorMap(DistContext::input(), Ti_));
  auto &pmap_ = *pmap_ptr_;

  if (DistContext::isDebug()) {
    if (this->gridRank() == 0) {
      std::cout << "Pmap initalization complete with grid: ";
      for (auto p : pmap_ptr_->gridDims()) {
        std::cout << p << " ";
      }
      std::cout << std::endl;
    } 
    pmap_ptr_->gridBarrier();
  }

  // TODO blocking could be better
  const auto blocking =
      detail::generateMediumGrainBlocking(Ti_.dim_sizes, pmap_.gridDims());

  if (DistContext::isDebug()) {
    if (this->gridRank() == 0) {
      std::cout << "With blocking:\n";
      auto dim = 0;
      for (auto const &inner : blocking) {
        std::cout << "\tdim(" << dim << "): ";
        ++dim;
        for(auto i : inner){
          std::cout << i << " ";
        }
        std::cout << "\n";
      }
      std::cout << std::endl;
    } 
    pmap_ptr_->gridBarrier();
  }

  // Evenly distribute the tensor around the world
  auto Tvec = detail::distributeTensorToVectors(
      tensor_file, Ti_.nnz, indexbase, pmap_.gridComm(), pmap_.gridRank(),
      pmap_.gridSize());

  // Now redistribute to medium grain format
  auto distributedData =
      detail::redistributeTensor(Tvec, Ti_.dim_sizes, blocking, pmap_);
  const auto local_nnz = distributedData.size();

  for (auto i = 0; i < ndims; ++i) {
    auto coord = pmap_.gridCoord(i);
    range_.push_back({blocking[i][coord], blocking[i][coord + 1]});
  }

  std::vector<ttb_indx> indices(ndims);
  for (auto i = 0; i < ndims; ++i) {
    auto const &rpair = range_[i];
    indices[i] = rpair.upper - rpair.lower;
  }

  std::vector<ttb_real> values(local_nnz);
  std::vector<std::vector<ttb_indx>> subs(local_nnz);
  for (auto i = 0; i < local_nnz; ++i) {
    auto data = distributedData[i];
    values[i] = data.val;
    subs[i] = std::vector<ttb_indx>(data.coo, data.coo + ndims);
    for (auto j = 0; j < ndims; ++j) {
      subs[i][j] -= range_[j].lower;
    }
  }

  sp_tensor_ = SptensorT<ExecSpace>(indices, values, subs);
  if (DistContext::input().get<bool>("debug", false)) {
    if (this->gridRank() == 0) {
      std::cout << "MPI Ranks in each dimension: ";
      for (auto p : pmap_ptr_->gridDims()) {
        std::cout << p << " ";
      }
      std::cout << std::endl;
    }
    pmap_ptr_->gridBarrier();
    // detail::printRandomElements(sp_tensor_, 3, *pmap_ptr_, range_);
  }
  init_factors();
}

template <typename ElementType, typename ExecSpace>
ElementType TensorBlockSystem<ElementType, ExecSpace>::getTensorNorm() const {
  auto const &values = sp_tensor_.getValArray();
  static double norm2 = values.dot(values);
  MPI_Allreduce(MPI_IN_PLACE, &norm2, 1, MPI_DOUBLE, MPI_SUM,
                pmap_ptr_->gridComm());
  return std::sqrt(ElementType(norm2));
}

// Try something a bit silly, let each node do a really simple decomp like it
// is the only one in existence this will give us a embarrassingly parallel
// initial guess which hopefully isn't actually all that bad!
template <typename ElementType, typename ExecSpace>
void TensorBlockSystem<ElementType, ExecSpace>::init_factors() {
  const auto rank = input_.get<int>("rank");
  const auto nd = ndims();

  // Init KFac_ randomly on each node
  Kfac_ = KtensorT<ExecSpace>(rank, nd, sp_tensor_.size());
  Genten::RandomMT cRMT(std::random_device{}());
  Kfac_.setWeights(1.0); // Matlab cp_als always sets the weights to one.
  Kfac_.setMatricesScatter(false, true, cRMT);

  // Specifically don't get full norm so that all norms are local
  const auto norm_x = sp_tensor_.norm();
  auto norm_k = std::sqrt(Kfac_.normFsq());
  // Kfac_.weights().times(norm_x / norm_k);
  Kfac_.weights().times(1.0 / norm_x);
  Kfac_.distribute();

  if (pmap_ptr_->gridSize() > 1) {
    allReduceKT(Kfac_);
  }
}

template <typename ElementType, typename ExecSpace>
void TensorBlockSystem<ElementType, ExecSpace>::allReduceKT(
    KtensorT<ExecSpace> &g, bool divide_by_grid_size) {
  if (nprocs() == 1) {
    return;
  }

  // Do the AllReduce for each gradient
  const auto ndims = this->ndims();
  auto const &gridSizes = pmap_ptr_->subCommSizes();

  for (auto i = 0; i < ndims; ++i) {
    if (gridSizes[i] == 1) {
      // No need to AllReduce when one rank owns all the data
      continue;
    }

    auto subComm = pmap_ptr_->subComm(i);
    auto subRank = pmap_ptr_->subCommRank(i);

    FacMatrixT<ExecSpace> const &fac_mat = g.factors()[i];
    auto fac_ptr = fac_mat.view().data();
    const auto fac_size = fac_mat.view().span();
    MPI_Allreduce(MPI_IN_PLACE, fac_ptr, fac_size, MPI_DOUBLE, MPI_SUM,
                  subComm);
  }

  if (divide_by_grid_size) {
    for (auto d = 0; d < ndims; ++d) {
      const ttb_real scale = double(1.0 / gridSizes[d]);
      g.factors()[d].times(scale);
    }
  }
}

template <typename ElementType, typename ExecSpace>
void TensorBlockSystem<ElementType, ExecSpace>::iAllReduceKT(
    KtensorT<ExecSpace> &g, std::vector<MPI_Request> &Requests) {
  if (nprocs() == 1) {
    return;
  }

  // Do the AllReduce for each gradient
  const auto ndims = sp_tensor_.ndims();
  auto const &gridSizes = pmap_ptr_->subCommSizes();
  for (auto i = 0; i < ndims; ++i) {
    if (gridSizes[i] == 1) {
      // No need to AllReduce when one rank owns all the data
      continue;
    }

    auto subComm = pmap_ptr_->subComm(i);
    auto subRank = pmap_ptr_->subCommRank(i);

    FacMatrixT<ExecSpace> const &fac_mat = g.factors()[i];
    auto fac_ptr = fac_mat.view().data();
    const auto fac_size = fac_mat.view().span();
    MPI_Iallreduce(MPI_IN_PLACE, fac_ptr, fac_size, MPI_DOUBLE, MPI_SUM,
                   subComm, &Requests[i]);
  }
}

template <typename ElementType, typename ExecSpace>
template <typename Loss>
ElementType
TensorBlockSystem<ElementType, ExecSpace>::pickMethod(Loss const &loss) {
  auto method = input_.get<std::string>("method", "elastic");
  if (method == "elastic") {
    return elasticAverageSGD(loss);
  } else if (method == "allreduce") {
    return allReduceSGD(loss);
  } else if (method == "adam") {
    return allReduceADAM(loss);
  }
  Genten::error("Your method for distributed SGD wasn't recognized.\n");
  return -1.0;
}

template <typename ElementType, typename ExecSpace>
ElementType TensorBlockSystem<ElementType, ExecSpace>::SGD() {
  auto loss = input_.get<std::string>("loss", "gaussian");

  // MAYBE one day decouple this so it's not N^2 but what ever
  if (loss == "gaussian") {
    return pickMethod(GaussianLossFunction(1e-10));
  } else if (loss == "poisson") {
    return pickMethod(PoissonLossFunction(1e-10));
  } else if (loss == "bernoulli") {
    return pickMethod(BernoulliLossFunction(1e-10));
  }

  Genten::error("Need to add more loss functions to distributed SGD.\n");
  return -1.0;
}

template <typename ElementType, typename ExecSpace>
AlgParams TensorBlockSystem<ElementType, ExecSpace>::setAlgParams() const {
  const auto np = nprocs();
  const auto lnz = localNNZ();
  const auto gnz = globalNNZ();

  AlgParams algParams;
  algParams.maxiters = input_.get<int>("max_epochs", 1000);

  auto global_batch_size_nz = input_.get<int>("batch_size_nz", 128);
  auto global_batch_size_z =
      input_.get<int>("batch_size_zero", global_batch_size_nz);

  // If we have fewer nnz than the batch size don't over sample them
  algParams.num_samples_nonzeros_grad =
      std::min(lnz, global_batch_size_nz / np);
  algParams.num_samples_zeros_grad = global_batch_size_z / np;

  // No point in sampling more nonzeros than we actually have
  algParams.num_samples_nonzeros_value = std::min(lnz, 100000 / np);
  algParams.num_samples_zeros_value = 100000 / np;

  algParams.sampling_type = Genten::GCP_Sampling::SemiStratified;
  algParams.mttkrp_method = Genten::MTTKRP_Method::Default;
  algParams.mttkrp_all_method = Genten::MTTKRP_All_Method::Duplicated;
  algParams.fuse = true;

  // If the epoch size isnt provided we should just try to hit every non-zero
  // approximately 1 time

  // Reset the global_batch_size_nz to the value that we will actually use
  global_batch_size_nz =
      pmap_ptr_->gridAllReduce(algParams.num_samples_nonzeros_grad);

  auto epoch_size = input_.get_optional<int>("epoch_size");
  if (epoch_size) {
    algParams.epoch_iters = epoch_size.get();
  } else {
    algParams.epoch_iters = gnz / global_batch_size_nz;
  }

  algParams.fixup<ExecSpace>(std::cout);

  const auto my_rank = pmap_ptr_->gridRank();

  const int local_batch_size = algParams.num_samples_nonzeros_grad;
  const int local_batch_sizez = algParams.num_samples_zeros_grad;
  const int global_zg = pmap_ptr_->gridAllReduce(local_batch_sizez);

  const auto percent_nz_batch =
      double(global_batch_size_nz) / double(gnz) * 100.0;
  const auto percent_nz_epoch =
      double(algParams.epoch_iters * global_batch_size_nz) / double(gnz) *
      100.0;

  // Collect info from the other ranks
  if (my_rank > 0) {
    MPI_Gather(&lnz, 1, MPI_INT, nullptr, 1, MPI_INT, 0, pmap_ptr_->gridComm());
    MPI_Gather(&local_batch_size, 1, MPI_INT, nullptr, 1, MPI_INT, 0,
               pmap_ptr_->gridComm());
    MPI_Gather(&local_batch_sizez, 1, MPI_INT, nullptr, 1, MPI_INT, 0,
               pmap_ptr_->gridComm());
  } else {
    std::vector<int> lnzs(np, 0);
    std::vector<int> bs_nz(np, 0);
    std::vector<int> bs_z(np, 0);
    MPI_Gather(&lnz, 1, MPI_INT, lnzs.data(), 1, MPI_INT, 0,
               pmap_ptr_->gridComm());
    MPI_Gather(&local_batch_size, 1, MPI_INT, bs_nz.data(), 1, MPI_INT, 0,
               pmap_ptr_->gridComm());
    MPI_Gather(&local_batch_sizez, 1, MPI_INT, bs_z.data(), 1, MPI_INT, 0,
               pmap_ptr_->gridComm());

    std::cout << "Iters/epoch:           " << algParams.epoch_iters << "\n";
    std::cout << "batch size nz:         " << global_batch_size_nz << "\n";
    std::cout << "batch size zeros:      " << global_zg << "\n";
    std::cout << "NZ percent per epoch : " << percent_nz_epoch << "\n";
    std::cout << "Node Specific info: {local_nnz, local_batch_size_nz, "
                 "local_batch_size_z}\n";
    for (auto i = 0; i < np; ++i) {
      std::cout << "\tNode(" << i << "): " << lnzs[i] << ", " << bs_nz[i]
                << ", " << bs_z[i] << "\n";
    }
    std::cout << std::endl;
  }
  pmap_ptr_->gridBarrier();

  return algParams;
}

template <typename ElementType, typename ExecSpace>
template <typename Loss>
ElementType
TensorBlockSystem<ElementType, ExecSpace>::elasticAverageSGD(Loss const &loss) {
  const auto nprocs = this->nprocs();
  const auto my_rank = gridRank();

  auto algParams = setAlgParams();

  // This is a lot of copies :/ but accept it for now
  using VectorType = GCP::KokkosVector<ExecSpace>;
  auto u = VectorType(Kfac_);
  u.copyFromKtensor(Kfac_);
  KtensorT<ExecSpace> ut = u.getKtensor();

  VectorType g = u.clone(); // Gradient Ktensor
  g.zero();
  decltype(Kfac_) GFac = g.getKtensor();

  VectorType center = u.clone(); // Center Tensor for elastic avg
  center.set(u);
  KtensorT<ExecSpace> CFac = center.getKtensor();
  allReduceKT(CFac); // All must agree on what the center is

  VectorType diff = u.clone(); // Center Tensor for elastic avg
  diff.zero();
  KtensorT<ExecSpace> diffKT = diff.getKtensor();

  auto sampler = SemiStratifiedSampler<ExecSpace, Loss>(sp_tensor_, algParams);

  Impl::SGDStep<ExecSpace, Loss> stepper;

  auto seed = input_.get<std::uint64_t>("seed", std::random_device{}());
  Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(seed);
  sampler.initialize(rand_pool, std::cout);

  SptensorT<ExecSpace> X_val;
  ArrayT<ExecSpace> w_val;
  sampler.sampleTensor(false, ut, loss, X_val, w_val);

  // Stuff for the timer that we can'g avoid providing
  SystemTimer timer;
  int tnzs = 0;
  int tzs = 0;

  // Fit stuff
  auto fest = pmap_ptr_->gridAllReduce(Impl::gcp_value(X_val, ut, w_val, loss));

  auto fest_prev = fest;
  const auto maxEpochs = algParams.maxiters;
  const auto epochIters = algParams.epoch_iters;
  const auto dp_iters = input_.get<int>("downpour_iters", 4);

  // TODO make the Annealer swappable
  CosineAnnealer annealer(input_);
  std::vector<MPI_Request> elastic_requests;
  bool first_average = true;
  for (auto gs : pmap_ptr_->subCommSizes()) {
    if (gs > 1) {
      elastic_requests.push_back(MPI_REQUEST_NULL);
    }
  }

  auto do_sync_allreduce = input_.get<bool>("sync_allreduce", true);
  auto do_extra_work = input_.get<bool>("do_extra_work", false);
  double t0 = 0, t1 = 0;

  for (auto e = 0; e < maxEpochs; ++e) { // Epochs
    t0 = MPI_Wtime();
    auto epoch_lr = annealer(e);
    stepper.setStep(epoch_lr);
    if (my_rank == 0) {
      std::cout << "Epoch LR: " << epoch_lr << std::endl;
    }

    auto do_epoch_iter = [&] {
      g.zero();
      sampler.fusedGradient(ut, loss, GFac, timer, tnzs, tzs);
      stepper.eval(g, u);
    };

    auto extraIters = 0;
    auto allreduceCounter = 0;
    const auto alpha = 0.9 / nprocs;
    for (auto i = 0; i < epochIters; ++i) {
      if (nprocs == 1) { // Short circuit if on single node
        do_epoch_iter();
        continue;
      }

      if ((i + 1) % dp_iters == 0 || i == (epochIters - 1)) {
        if (do_sync_allreduce) {
          u.elastic_difference(diff, center, 1.0);
          u.plus(diff, -1.0 * alpha);
          allReduceKT(diffKT, false);
          center.plus(diff, alpha);
          ++allreduceCounter;
        } else { // iAllReduce
          if (first_average) {
            first_average = false;
          } else {
            if (do_extra_work) {
              int flag = 0;
              MPI_Testall(elastic_requests.size(), elastic_requests.data(),
                          &flag, MPI_STATUSES_IGNORE);
              while (flag == 0) {
                MPI_Testall(elastic_requests.size(), elastic_requests.data(),
                            &flag, MPI_STATUSES_IGNORE);
                do_epoch_iter();
                ++extraIters;
              }
            } else {
              MPI_Waitall(elastic_requests.size(), elastic_requests.data(),
                          MPI_STATUSES_IGNORE);
            }
            center.plus(diff, alpha);
          }

          u.elastic_difference(diff, center, 1.0);
          u.plus(diff, -1.0 * alpha); // Subtract dist from myself
          iAllReduceKT(diffKT, elastic_requests);
          ++allreduceCounter;
        }
      }

      do_epoch_iter();
    }

    fest = pmap_ptr_->gridAllReduce(Impl::gcp_value(X_val, CFac, w_val, loss));
    t1 = MPI_Wtime();

    if (my_rank == 0) {
      std::cout << "\tFit(" << e << "): " << fest << ", " << fest_prev - fest
                << " did " << allreduceCounter << " factor allReduces"
                << " in " << (t1 - t0) << " seconds.\n"
                << "\tdid " << extraIters << " extra iters." << std::endl;
    }
    fest_prev = fest;
  }

  // We have to wait on the last all reduce if we did one since I don't want to
  // figure out how to cancel it right now
  if (!do_sync_allreduce && do_extra_work) {
    MPI_Waitall(elastic_requests.size(), elastic_requests.data(),
                MPI_STATUSES_IGNORE);
  }

  return fest_prev;
}

template <typename ElementType, typename ExecSpace>
template <typename Loss>
ElementType
TensorBlockSystem<ElementType, ExecSpace>::allReduceSGD(Loss const &loss) {
  const auto nprocs = this->nprocs();
  const auto my_rank = gridRank();

  auto algParams = setAlgParams();

  // This is a lot of copies :/ but accept it for now
  using VectorType = GCP::KokkosVector<ExecSpace>;
  auto u = VectorType(Kfac_);
  u.copyFromKtensor(Kfac_);
  KtensorT<ExecSpace> ut = u.getKtensor();

  VectorType g = u.clone(); // Gradient Ktensor
  g.zero();
  decltype(Kfac_) GFac = g.getKtensor();

  VectorType nag = u.clone(); // Center Tensor for elastic avg
  nag.set(u);
  KtensorT<ExecSpace> nagKT = nag.getKtensor();

  auto sampler = SemiStratifiedSampler<ExecSpace, Loss>(sp_tensor_, algParams);

  Impl::NAGStep<ExecSpace, Loss> stepper(algParams, u);

  auto seed = input_.get<std::uint64_t>("seed", std::random_device{}());
  Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(seed);
  sampler.initialize(rand_pool, std::cout);

  SptensorT<ExecSpace> X_val;
  ArrayT<ExecSpace> w_val;
  sampler.sampleTensor(false, ut, loss, X_val, w_val);

  // Stuff for the timer that we can'g avoid providing
  SystemTimer timer;
  int tnzs = 0;
  int tzs = 0;

  // Fit stuff
  auto fest = pmap_ptr_->gridAllReduce(Impl::gcp_value(X_val, ut, w_val, loss));

  auto fest_prev = fest;
  const auto maxEpochs = algParams.maxiters;
  const auto epochIters = algParams.epoch_iters;
  const auto dp_iters = input_.get<int>("downpour_iters", 4);

  // TODO make the Annealer swappable
  CosineAnnealer annealer(input_);
  double t0 = 0, t1 = 0;
  for (auto e = 0; e < maxEpochs; ++e) { // Epochs
    t0 = MPI_Wtime();
    auto epoch_lr = annealer(e);
    stepper.setStep(epoch_lr);
    if (my_rank == 0) {
      std::cout << "Epoch LR: " << epoch_lr << std::endl;
    }

    auto do_epoch_iter = [&] {
      g.zero();
      nag.set(u);
      nag.plus(stepper.velocity(), 0.9);
      sampler.fusedGradient(nagKT, loss, GFac, timer, tnzs, tzs);
      stepper.eval(g, u);
    };

    auto allreduceCounter = 0;
    for (auto i = 0; i < epochIters; ++i) {
      if (nprocs == 1) { // Short circuit if on single node
        do_epoch_iter();
        continue;
      }

      if ((i + 1) % dp_iters == 0) {
        allReduceKT(ut);
        ++allreduceCounter;
      }

      do_epoch_iter();
    }
    if (nprocs > 1) {
      allReduceKT(ut);
      ++allreduceCounter;
    }

    fest = pmap_ptr_->gridAllReduce(Impl::gcp_value(X_val, ut, w_val, loss));
    t1 = MPI_Wtime();

    if (my_rank == 0) {
      std::cout << "\tFit(" << e << "): " << fest << ", " << fest_prev - fest
                << " did " << allreduceCounter << " factor allReduces."
                << " in " << (t1 - t0) << " seconds.\n"
                << std::flush;
    }
    fest_prev = fest;
  }

  return fest_prev;
}

template <typename ElementType, typename ExecSpace>
template <typename Loss>
ElementType
TensorBlockSystem<ElementType, ExecSpace>::allReduceADAM(Loss const &loss) {
  const auto nprocs = this->nprocs();
  const auto my_rank = gridRank();

  auto algParams = setAlgParams();

  using VectorType = GCP::KokkosVector<ExecSpace>;
  auto u = VectorType(Kfac_);
  u.copyFromKtensor(Kfac_);
  KtensorT<ExecSpace> ut = u.getKtensor();

  VectorType g = u.clone(); // Gradient Ktensor
  g.zero();
  decltype(Kfac_) GFac = g.getKtensor();

  auto sampler = SemiStratifiedSampler<ExecSpace, Loss>(sp_tensor_, algParams);

  Impl::AdamStep<ExecSpace, Loss> stepper(algParams, u);

  auto seed = input_.get<std::uint64_t>("seed", std::random_device{}());
  Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(seed);
  sampler.initialize(rand_pool, std::cout);

  SptensorT<ExecSpace> X_val;
  ArrayT<ExecSpace> w_val;
  sampler.sampleTensor(false, ut, loss, X_val, w_val);

  // Stuff for the timer that we can'g avoid providing
  SystemTimer timer;
  int tnzs = 0;
  int tzs = 0;

  // Fit stuff
  auto fest = pmap_ptr_->gridAllReduce(Impl::gcp_value(X_val, ut, w_val, loss));
  if (my_rank == 0) {
    std::cout << "Initial guess fest: " << fest << std::endl;
  }
  pmap_ptr_->gridBarrier();

  auto fest_prev = fest;
  const auto maxEpochs = algParams.maxiters;
  const auto epochIters = algParams.epoch_iters;

  // TODO make the Annealer swappable
  auto epoch_lr = 1e-3;
  double t0 = 0, t1 = 0;
  for (auto e = 0; e < maxEpochs; ++e) { // Epochs
    t0 = MPI_Wtime();
    stepper.setStep(epoch_lr);
    if (my_rank == 0) {
      std::cout << "Epoch LR: " << epoch_lr << std::endl;
    }

    auto allreduceCounter = 0;
    for (auto i = 0; i < epochIters; ++i) {
      stepper.update();
      g.zero();
      sampler.fusedGradient(ut, loss, GFac, timer, tnzs, tzs);
      allReduceKT(GFac, false /* don't average */);
      ++allreduceCounter;
      stepper.eval(g, u);
    }

    fest = pmap_ptr_->gridAllReduce(Impl::gcp_value(X_val, ut, w_val, loss));
    t1 = MPI_Wtime();

    if (my_rank == 0) {
      std::cout << "\tFit(" << e << "): " << fest << ", " << fest_prev - fest
                << " did " << allreduceCounter << " factor allReduces."
                << " in " << (t1 - t0) << " seconds.\n"
                << std::flush;
    }
    fest_prev = fest;
  }

  return fest_prev;
}

namespace detail {

template <typename ExecSpace>
void printRandomElements(SptensorT<ExecSpace> const &tensor,
                         int num_elements_per_rank, ProcessorMap const &pmap,
                         small_vector<RangePair> const &ranges) {
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
    sleep(1);
  }
}

template <typename ExecSpace>
auto rangesToIndexArray(small_vector<RangePair> const &ranges) {
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

} // namespace Genten
