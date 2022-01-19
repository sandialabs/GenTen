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
#include "Genten_MPI_IO.h"
#include "Genten_Pmap.hpp"
#include "Genten_SpTn_Util.h"
#include "Genten_Sptensor.hpp"

#include <cmath>
#include <fstream>
#include <memory>
#include <random>
#include <unordered_map>

namespace Genten {

namespace detail {
void printGrids(ProcessorMap const &pmap);
void printBlocking(ProcessorMap const &pmap,
                   std::vector<small_vector<int>> const &blocking);

std::vector<small_vector<int>>
generateUniformBlocking(std::vector<std::uint32_t> const &ModeLengths,
                        small_vector<int> const &ProcGridSizes);
} // namespace detail

struct RangePair {
  int64_t lower;
  int64_t upper;
};

// Class to hold a block of the tensor and the factor matrix blocks that can
// be used to generate a representation of the tensor block, if the entire
// tensor is placed into one block this is just a TensorSystem
template <typename ElementType, typename ExecSpace = DefaultExecutionSpace>
class TensorBlockSystem {
  static_assert(
      std::is_floating_point<ElementType>::value,
      "TensorBlockSystem Requires that the element type be a floating "
      "point type.");

  /// The normal initialization method, inializes in parallel
  void init_distributed(std::string const &file_name, int indexbase);

  /// Initialize the factor matrices
  void init_factors();

  void allReduceKT(KtensorT<ExecSpace> &g, bool divide_by_grid_size = true);

  template <typename Stepper, typename Loss>
  ElementType allReduceTrad(Loss const &loss);
  template <typename Loss> ElementType fedOpt(Loss const &loss);
  template <typename Loss> ElementType pickMethod(Loss const &loss);

  AlgParams setAlgParams() const;

public:
  TensorBlockSystem(ptree const &tree);
  ~TensorBlockSystem() = default;

  // For now let's delete these so they aren't accidently used
  // Can come back and define them later if needed
  TensorBlockSystem(TensorBlockSystem const &) = delete;
  TensorBlockSystem &operator=(TensorBlockSystem const &) = delete;

  TensorBlockSystem(TensorBlockSystem &&) = delete;
  TensorBlockSystem &operator=(TensorBlockSystem &&) = delete;

  ElementType getTensorNorm() const;

  std::uint64_t globalNNZ() const { return Ti_.nnz; }
  std::int64_t localNNZ() const { return sp_tensor_.nnz(); }
  std::int32_t ndims() const { return sp_tensor_.ndims(); }
  std::vector<std::uint32_t> const &dims() const { return Ti_.dim_sizes; }
  std::int64_t nprocs() const { return pmap_ptr_->gridSize(); }
  std::int64_t gridRank() const { return pmap_ptr_->gridRank(); }

  ElementType SGD();
  void exportKTensor(std::string const &file_name);

private:
  boost::optional<std::pair<MPI_IO::SptnFileHeader, MPI_File>>
  readHeader(std::string const &file_name, int indexbase);

  ptree input_;
  small_vector<RangePair> range_;
  SptensorT<ExecSpace> sp_tensor_;
  KtensorT<ExecSpace> Kfac_;
  std::unique_ptr<ProcessorMap> pmap_ptr_;
  TensorInfo Ti_;
  bool dump_; // I don't love keeping this flag, but it's easy
  unsigned int seed_;
  small_vector<small_vector<int>> global_blocking_;
};

// Helper declerations
namespace detail {
bool fileFormatIsBinary(std::string const &file_name);

template <typename ExecSpace>
auto rangesToIndexArray(small_vector<RangePair> const &ranges);
small_vector<int> singleDimUniformBlocking(int ModeLength, int ProcsInMode);

std::vector<MPI_IO::TDatatype<double>>
distributeTensorToVectors(std::ifstream &ifs, uint64_t nnz, int indexbase,
                          MPI_Comm comm, int rank, int nprocs);

std::vector<MPI_IO::TDatatype<double>>
redistributeTensor(std::vector<MPI_IO::TDatatype<double>> const &Tvec,
                   std::vector<std::uint32_t> const &TensorDims,
                   std::vector<small_vector<int>> const &blocking,
                   ProcessorMap const &pmap);

template <typename ExecSpace>
void printRandomElements(SptensorT<ExecSpace> const &tensor,
                         int num_elements_per_rank, ProcessorMap const &pmap,
                         small_vector<RangePair> const &ranges);
} // namespace detail

template <typename ElementType, typename ExecSpace>
TensorBlockSystem<ElementType, ExecSpace>::TensorBlockSystem(ptree const &tree)
    : input_(tree.get_child("tensor")), dump_(tree.get<bool>("dump", false)),
      seed_(input_.get<unsigned int>("seed", std::random_device{}())) {

  if (dump_) {
    if (DistContext::rank() == 0) {
      std::cout
          << "tensor:\n"
             "\tfile: The input file\n"
             "\tindexbase: Value that indices start at (defaults to 0)\n"
             "\trank: rank at which to decompose the tensor\n"
             "\tloss: Loss function options are {guassian, poisson, "
             "bernoulli}\n"
             "\tmethod: The SGD method to use (default: adam), options {adam, "
             "fedopt, sgd, sgdm, adagrad, demon, elasticAvgOneSided}\n"
             "\tmax_epochs: the number of epochs to run.\n"
             "\tbatch_size_nz: the number of non-zeros to sample per batch.\n"
             "\tbatch_size_zero: the number of zeros to sample per batch.\n"
             "\tepoch_size: the number of `epoch_iters` to run, defaults to "
             "number of "
             "non-zeros  divided by the number of non-zeros per batch.\n"
             "\tseed: Random seed default(std::random_device{}()).\n";
    }
    return;
  }
  const auto file_name = input_.get<std::string>("file");
  const auto indexbase = input_.get<int>("indexbase", 0);
  const auto rank = input_.get<int>("rank", 5);
  init_distributed(file_name, indexbase);
}

// We'll put this down here to save some space
template <typename ElementType, typename ExecSpace>
void TensorBlockSystem<ElementType, ExecSpace>::init_distributed(
    std::string const &file_name, int indexbase) {

  auto binary_header = readHeader(file_name, indexbase);

  const auto ndims = Ti_.dim_sizes.size();
  pmap_ptr_ = std::unique_ptr<ProcessorMap>(
      new ProcessorMap(DistContext::input(), Ti_.dim_sizes));
  auto &pmap_ = *pmap_ptr_;

  detail::printGrids(pmap_);

  const auto blocking =
      detail::generateUniformBlocking(Ti_.dim_sizes, pmap_.gridDims());

  detail::printBlocking(pmap_, blocking);
  DistContext::Barrier();

  auto t2 = MPI_Wtime();
  auto Tvec = [&] {
    if (binary_header) {
      return MPI_IO::parallelReadElements(DistContext::commWorld(),
                                          binary_header->second,
                                          binary_header->first);
    } else {
      auto tensor_file = std::ifstream(file_name);
      return detail::distributeTensorToVectors(
          tensor_file, Ti_.nnz, indexbase, pmap_.gridComm(), pmap_.gridRank(),
          pmap_.gridSize());
    }
  }();
  DistContext::Barrier();
  auto t3 = MPI_Wtime();

  if (gridRank() == 0) {
    std::cout << "Read in file in: " << t3 - t2 << "s" << std::endl;
  }

  DistContext::Barrier();
  auto t4 = MPI_Wtime();

  // Now redistribute to final format
  auto distributedData =
      detail::redistributeTensor(Tvec, Ti_.dim_sizes, blocking, pmap_);

  DistContext::Barrier();
  auto t5 = MPI_Wtime();

  if (gridRank() == 0) {
    std::cout << "Redistributied file in: " << t5 - t4 << "s" << std::endl;
  }

  DistContext::Barrier();
  auto t6 = MPI_Wtime();

  for (auto i = 0; i < ndims; ++i) {
    auto coord = pmap_.gridCoord(i);
    range_.push_back({blocking[i][coord], blocking[i][coord + 1]});
  }

  std::vector<ttb_indx> indices(ndims);
  for (auto i = 0; i < ndims; ++i) {
    auto const &rpair = range_[i];
    indices[i] = rpair.upper - rpair.lower;
  }

  const auto local_nnz = distributedData.size();
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
  if (DistContext::isDebug()) {
    if (gridRank() == 0) {
      std::cout << "MPI Ranks in each dimension: ";
      for (auto p : pmap_.subCommSizes()) {
        std::cout << p << " ";
      }
      std::cout << std::endl;
    }
    DistContext::Barrier();
  }

  DistContext::Barrier();
  auto t7 = MPI_Wtime();

  if (gridRank() == 0) {
    std::cout << "Copied to data struct in: " << t7 - t6 << "s" << std::endl;
  }

  init_factors();
}

template <typename ElementType, typename ExecSpace>
ElementType TensorBlockSystem<ElementType, ExecSpace>::getTensorNorm() const {
  auto const &values = sp_tensor_.getValArray();
  double norm2 = values.dot(values);
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
  Kfac_.setWeights(1.0);
  Kfac_.setMatricesScatter(false, true, cRMT);

  const auto norm_x = sp_tensor_.norm();
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

    // MPI_SUM doesn't support user defined types for AllReduce BOO
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
template <typename Loss>
ElementType
TensorBlockSystem<ElementType, ExecSpace>::pickMethod(Loss const &loss) {
  auto method = input_.get<std::string>("method", "adam");
  if (method == "fedopt") {
    return fedOpt(loss);
  } else if (method == "sgd") {
    return allReduceTrad<Impl::SGDStep<ExecSpace, Loss>>(loss);
  } else if (method == "sgdm") {
    return allReduceTrad<Impl::SGDMomentumStep<ExecSpace, Loss>>(loss);
  } else if (method == "adam") {
    return allReduceTrad<Impl::AdamStep<ExecSpace, Loss>>(loss);
  } else if (method == "adagrad") {
    return allReduceTrad<Impl::AdaGradStep<ExecSpace, Loss>>(loss);
  } else if (method == "demon") {
    return allReduceTrad<Impl::DEMON<ExecSpace, Loss>>(loss);
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
  if (ExecSpace::concurrency() > 1) {
    algParams.mttkrp_all_method = Genten::MTTKRP_All_Method::Atomic;
  } else {
    algParams.mttkrp_all_method = Genten::MTTKRP_All_Method::Single;
  }
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
    std::cout << "MTTKRP_All_Method :    ";
    if (algParams.mttkrp_all_method == MTTKRP_All_Method::Duplicated) {
      std::cout << "Duplicated\n";
    } else if (algParams.mttkrp_all_method == MTTKRP_All_Method::Single) {
      std::cout << "Single\n";
    } else if (algParams.mttkrp_all_method == MTTKRP_All_Method::Atomic) {
      std::cout << "Atomic\n";
    } else {
      std::cout << "method(" << algParams.mttkrp_all_method << ")\n";
    }
    if (np < 41) {
      std::cout << "Node Specific info: {local_nnz, local_batch_size_nz, "
                   "local_batch_size_z}\n";
      for (auto i = 0; i < np; ++i) {
        std::cout << "\tNode(" << i << "): " << lnzs[i] << ", " << bs_nz[i]
                  << ", " << bs_z[i] << "\n";
      }
    }
    std::cout << std::endl;
  }
  pmap_ptr_->gridBarrier();

  return algParams;
}

template <typename ElementType, typename ExecSpace>
template <typename Loss>
ElementType
TensorBlockSystem<ElementType, ExecSpace>::fedOpt(Loss const &loss) {
  const auto start_time = MPI_Wtime();
  const auto nprocs = this->nprocs();
  const auto my_rank = gridRank();

  auto algParams = setAlgParams();
  if (auto eps = input_.get_optional<ElementType>("eps")) {
    algParams.adam_eps = *eps;
  }

  // This is a lot of copies :/ but accept it for now
  using VectorType = GCP::KokkosVector<ExecSpace>;
  auto u = VectorType(Kfac_);
  u.copyFromKtensor(Kfac_);
  KtensorT<ExecSpace> ut = u.getKtensor();
  allReduceKT(ut);

  VectorType u_best = u.clone();
  u_best.set(u);

  VectorType g = u.clone(); // Gradient Ktensor
  g.zero();
  decltype(Kfac_) GFac = g.getKtensor();

  VectorType meta_u = u.clone();
  meta_u.set(u);
  KtensorT<ExecSpace> MUFac = meta_u.getKtensor();

  VectorType diff = meta_u.clone();
  diff.zero();
  KtensorT<ExecSpace> Dfac = diff.getKtensor();

  auto sampler = SemiStratifiedSampler<ExecSpace, Loss>(sp_tensor_, algParams);

  Impl::SGDStep<ExecSpace, Loss> stepper;
  Impl::AdamStep<ExecSpace, Loss> meta_stepper(algParams, meta_u);
  Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(seed_);

  std::stringstream ss;
  sampler.initialize(rand_pool, ss);
  if (nprocs < 41) {
    std::cout << ss.str();
  }

  SptensorT<ExecSpace> X_val;
  ArrayT<ExecSpace> w_val;
  sampler.sampleTensor(false, ut, loss, X_val, w_val);

  // Stuff for the timer that we can'g avoid providing
  SystemTimer timer;
  int tnzs = 0;
  int tzs = 0;

  // Fit stuff
  auto fest = pmap_ptr_->gridAllReduce(Impl::gcp_value(X_val, ut, w_val, loss));
  auto fest_best = fest;
  auto fest_prev = fest;

  const auto maxEpochs = algParams.maxiters;
  const auto epochIters = algParams.epoch_iters;
  const auto dp_iters = input_.get<int>("downpour_iters", 4);

  auto annealer_ptr = getAnnealer(input_);
  auto &annealer = *annealer_ptr;

  auto fedavg = input_.get<bool>("fedavg", false);
  auto meta_lr = input_.get<ElementType>("meta_lr", 1e-3);

  double t0 = 0;
  double t1 = 0;
  for (auto e = 0; e < maxEpochs; ++e) { // Epochs
    t0 = MPI_Wtime();
    auto epoch_lr = annealer(e);
    stepper.setStep(epoch_lr);

    auto do_epoch_iter = [&](double &gtime, double &etime) {
      g.zero();
      auto start = MPI_Wtime();
      sampler.fusedGradient(ut, loss, GFac, timer, tnzs, tzs);
      auto ge = MPI_Wtime();
      stepper.eval(g, u);
      auto end = MPI_Wtime();

      gtime += ge - start;
      etime += end - ge;
    };

    auto allreduceCounter = 0;
    auto gradient_time = 0.0;
    auto evaluation_time = 0.0;
    auto sync_time = 0.0;

    for (auto i = 0; i < epochIters; ++i) {
      do_epoch_iter(gradient_time, evaluation_time);
      if ((i + 1) % dp_iters == 0 || i == (epochIters - 1)) {
        auto s0 = MPI_Wtime();
        if (fedavg) {
          allReduceKT(ut, true);
        } else {
          diff.set(meta_u);
          diff.plus(u, -1.0); // Subtract u from meta_u to get Meta grad
          allReduceKT(Dfac, true);

          meta_stepper.update();
          meta_stepper.setStep(meta_lr);
          meta_stepper.eval(diff, meta_u);
          u.set(meta_u); // Everyone agrees that meta_u is the new factors
        }
        ++allreduceCounter;
        sync_time += MPI_Wtime() - s0;
      }
    }

    fest = pmap_ptr_->gridAllReduce(Impl::gcp_value(X_val, ut, w_val, loss));
    t1 = MPI_Wtime();

    const auto fest_diff = fest_prev - fest;

    std::vector<double> gradient_times;
    std::vector<double> elastic_times;
    std::vector<double> eval_times;
    if (my_rank == 0) {
      if (std::isnan(fest)) {
        std::cout << "IS NAN: Best result was: " << fest_best << std::endl;
        return fest_best;
      }
      gradient_times.resize(nprocs);
      elastic_times.resize(nprocs);
      eval_times.resize(nprocs);

      MPI_Gather(&gradient_time, 1, MPI_DOUBLE, &gradient_times[0], 1,
                 MPI_DOUBLE, 0, pmap_ptr_->gridComm());
      MPI_Gather(&evaluation_time, 1, MPI_DOUBLE, &eval_times[0], 1, MPI_DOUBLE,
                 0, pmap_ptr_->gridComm());
      MPI_Gather(&sync_time, 1, MPI_DOUBLE, &elastic_times[0], 1, MPI_DOUBLE, 0,
                 pmap_ptr_->gridComm());
    } else {
      if (std::isnan(fest)) {
        return fest_best;
      }
      MPI_Gather(&gradient_time, 1, MPI_DOUBLE, nullptr, 1, MPI_DOUBLE, 0,
                 pmap_ptr_->gridComm());
      MPI_Gather(&evaluation_time, 1, MPI_DOUBLE, nullptr, 1, MPI_DOUBLE, 0,
                 pmap_ptr_->gridComm());
      MPI_Gather(&sync_time, 1, MPI_DOUBLE, nullptr, 1, MPI_DOUBLE, 0,
                 pmap_ptr_->gridComm());
    }

    if (my_rank == 0) {
      auto min_max_gradient =
          std::minmax_element(gradient_times.begin(), gradient_times.end());
      auto grad_avg =
          std::accumulate(gradient_times.begin(), gradient_times.end(), 0.0) /
          double(nprocs);

      auto min_max_elastic =
          std::minmax_element(elastic_times.begin(), elastic_times.end());
      auto elastic_avg =
          std::accumulate(elastic_times.begin(), elastic_times.end(), 0.0) /
          double(nprocs);

      auto min_max_evals =
          std::minmax_element(eval_times.begin(), eval_times.end());
      auto eval_avg =
          std::accumulate(eval_times.begin(), eval_times.end(), 0.0) /
          double(nprocs);

      std::cout << "Fit(" << e << "): " << fest
                << "\n\tchange in fit: " << fest_diff
                << "\n\tlr:            " << epoch_lr
                << "\n\tallReduces:    " << allreduceCounter
                << "\n\tSeconds:       " << (t1 - t0)
                << "\n\tElapsed Time:  " << (t1 - start_time);
      std::cout << "\n\t\tGradient(avg, min, max):  " << grad_avg << ", "
                << *min_max_gradient.first << ", " << *min_max_gradient.second
                << "\n\t\tAllReduce(avg, min, max):   " << elastic_avg << ", "
                << *min_max_elastic.first << ", " << *min_max_elastic.second
                << "\n\t\tEval(avg, min, max):      " << eval_avg << ", "
                << *min_max_evals.first << ", " << *min_max_evals.second
                << "\n";
    }

    if (fest_diff > -0.001 * fest_best) {
      stepper.setPassed();
      meta_stepper.setPassed();
      fest_prev = fest;
      annealer.success();
      if (fest < fest_best) { // Only set best if really best
        fest_best = fest;
        u_best.set(u);
      }
    } else {
      u.set(u_best);
      annealer.failed();
      stepper.setFailed();
      meta_stepper.setFailed();
      fest = fest_best;
    }
  }

  return fest_prev;
}

template <typename ElementType, typename ExecSpace>
template <typename Stepper, typename Loss>
ElementType
TensorBlockSystem<ElementType, ExecSpace>::allReduceTrad(Loss const &loss) {
  if (dump_) {
    std::cout
        << "Methods that use AllReduce(sgd, sgdm, adam, adagrad, demon) "
           "have the following options under `tensor`:\n"
           "\tannealer: Choice of annealer default(traditional) options "
           "{traditional, cosine}\n"
           "\tlr: (object that controls the learning rate)\n"
           "\t\tstep: IFF traditional annealer, is the value of the "
           "learning rate\n"
           "\t\tmin_lr: IFF cosine annealer, is the lower value reached.\n"
           "\t\tmax_lr: IFF cosine annealer, is the higher value reached.\n"
           "\t\tTi: IFF cosine annealer, is the cycle period default(10).\n";
    return -1.0;
  }
  const auto nprocs = this->nprocs();
  const auto my_rank = gridRank();
  const auto start_time = MPI_Wtime();

  auto algParams = setAlgParams();

  using VectorType = GCP::KokkosVector<ExecSpace>;
  auto u = VectorType(Kfac_);
  u.copyFromKtensor(Kfac_);
  KtensorT<ExecSpace> ut = u.getKtensor();

  VectorType u_best = u.clone();
  u_best.set(u);

  VectorType g = u.clone(); // Gradient Ktensor
  g.zero();
  decltype(Kfac_) GFac = g.getKtensor();

  auto sampler = SemiStratifiedSampler<ExecSpace, Loss>(sp_tensor_, algParams);

  Stepper stepper(algParams, u);

  Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(seed_);
  std::stringstream ss;
  sampler.initialize(rand_pool, ss);
  if (nprocs < 41) {
    std::cout << ss.str();
  }

  SptensorT<ExecSpace> X_val;
  ArrayT<ExecSpace> w_val;
  sampler.sampleTensor(false, ut, loss, X_val, w_val);

  // Stuff for the timer that we can'g avoid providing
  SystemTimer timer;
  int tnzs = 0;
  int tzs = 0;

  // Fit stuff
  auto fest = pmap_ptr_->gridAllReduce(Impl::gcp_value(X_val, ut, w_val, loss));
  auto fest_best = fest;
  if (my_rank == 0) {
    std::cout << "Initial guess fest: " << fest << std::endl;
  }
  pmap_ptr_->gridBarrier();

  auto fest_prev = fest;
  const auto maxEpochs = algParams.maxiters;
  const auto epochIters = algParams.epoch_iters;

  auto annealer_ptr = getAnnealer(input_);
  auto &annealer = *annealer_ptr;
  // For adam with all of the all reduces I am hopeful the barriers don't
  // really matter for timeing

  for (auto e = 0; e < maxEpochs; ++e) { // Epochs
    pmap_ptr_->gridBarrier();            // Makes times more accurate
    double e_start = MPI_Wtime();
    const auto epoch_lr = annealer(e);
    stepper.setStep(epoch_lr);

    auto allreduceCounter = 0;
    double gradient_time = 0;
    double allreduce_time = 0;
    double eval_time = 0;
    for (auto i = 0; i < epochIters; ++i) {
      stepper.update();
      g.zero();
      auto ze = MPI_Wtime();
      sampler.fusedGradient(ut, loss, GFac, timer, tnzs, tzs);
      auto ge = MPI_Wtime();
      allReduceKT(GFac, false /* don't average */);
      auto are = MPI_Wtime();
      stepper.eval(g, u);
      auto ee = MPI_Wtime();
      gradient_time += ge - ze;
      allreduce_time += are - ge;
      eval_time += ee - are;
      ++allreduceCounter;
    }

    double fest_start = MPI_Wtime();
    fest = pmap_ptr_->gridAllReduce(Impl::gcp_value(X_val, ut, w_val, loss));
    pmap_ptr_->gridBarrier(); // Makes times more accurate
    double e_end = MPI_Wtime();
    auto epoch_time = e_end - e_start;
    auto fest_time = e_end - fest_start;
    const auto fest_diff = fest_prev - fest;

    std::vector<double> gradient_times;
    std::vector<double> all_reduce_times;
    std::vector<double> eval_times;
    if (my_rank == 0) {
      gradient_times.resize(nprocs);
      all_reduce_times.resize(nprocs);
      eval_times.resize(nprocs);
      MPI_Gather(&gradient_time, 1, MPI_DOUBLE, &gradient_times[0], 1,
                 MPI_DOUBLE, 0, pmap_ptr_->gridComm());
      MPI_Gather(&allreduce_time, 1, MPI_DOUBLE, &all_reduce_times[0], 1,
                 MPI_DOUBLE, 0, pmap_ptr_->gridComm());
      MPI_Gather(&eval_time, 1, MPI_DOUBLE, &eval_times[0], 1, MPI_DOUBLE, 0,
                 pmap_ptr_->gridComm());
    } else {
      MPI_Gather(&gradient_time, 1, MPI_DOUBLE, nullptr, 1, MPI_DOUBLE, 0,
                 pmap_ptr_->gridComm());
      MPI_Gather(&allreduce_time, 1, MPI_DOUBLE, nullptr, 1, MPI_DOUBLE, 0,
                 pmap_ptr_->gridComm());
      MPI_Gather(&eval_time, 1, MPI_DOUBLE, nullptr, 1, MPI_DOUBLE, 0,
                 pmap_ptr_->gridComm());
    }

    if (my_rank == 0) {
      auto min_max_gradient =
          std::minmax_element(gradient_times.begin(), gradient_times.end());
      auto grad_avg =
          std::accumulate(gradient_times.begin(), gradient_times.end(), 0.0) /
          double(nprocs);

      auto min_max_allreduce =
          std::minmax_element(all_reduce_times.begin(), all_reduce_times.end());
      auto ar_avg = std::accumulate(all_reduce_times.begin(),
                                    all_reduce_times.end(), 0.0) /
                    double(nprocs);

      auto min_max_evals =
          std::minmax_element(eval_times.begin(), eval_times.end());
      auto eval_avg =
          std::accumulate(eval_times.begin(), eval_times.end(), 0.0) /
          double(nprocs);

      std::cout << "Fit(" << e << "): " << fest
                << "\n\tchange in fit: " << fest_diff
                << "\n\tlr:            " << epoch_lr
                << "\n\tallReduces:    " << allreduceCounter
                << "\n\tSeconds:       " << (e_end - e_start)
                << "\n\tElapsed Time:  " << (e_end - start_time)
                << "\n\t\tGradient(avg, min, max):  " << grad_avg << ", "
                << *min_max_gradient.first << ", " << *min_max_gradient.second
                << "\n\t\tAllReduce(avg, min, max): " << ar_avg << ", "
                << *min_max_allreduce.first << ", " << *min_max_allreduce.second
                << "\n\t\tEval(avg, min, max):      " << eval_avg << ", "
                << *min_max_evals.first << ", " << *min_max_evals.second
                << "\n";

      std::cout << std::flush;
    }

    if (fest_diff > -0.001 * fest_best) {
      stepper.setPassed();
      fest_prev = fest;
      annealer.success();
      if (fest < fest_best) {
        u_best.set(u);
        fest_best = fest;
      }
    } else {
      u.set(u_best);
      fest = fest_prev;
      stepper.setFailed();
      annealer.failed();
    }
  }
  u.set(u_best);
  deep_copy(Kfac_, ut);

  return fest_prev;
}

template <typename ElementType, typename ExecSpace>
void TensorBlockSystem<ElementType, ExecSpace>::exportKTensor(
    std::string const &file_name) {
  const auto blocking =
      detail::generateUniformBlocking(Ti_.dim_sizes, pmap_ptr_->gridDims());

  if (this->gridRank() == 0) { // I am the chosen one
    auto const &sizes = Ti_.dim_sizes;
    IndxArrayT<ExecSpace> sizes_idx(sizes.size());
    for (auto i = 0; i < sizes.size(); ++i) {
      sizes_idx[i] = sizes[i];
    }
    KtensorT<ExecSpace> out(Kfac_.ncomponents(), Kfac_.ndims(), sizes_idx);

    std::cout << "Blocking:\n";
    const auto ndims = blocking.size();
    small_vector<int> grid_pos(ndims, 0);
    for (auto d = 0; d < ndims; ++d) {
      const auto nblocks = blocking[d].size() - 1;
      std::cout << "\tDim(" << d << ")\n";
      for (auto b = 0; b < nblocks; ++b) {
        std::cout << "\t\t{" << blocking[d][b] << ", " << blocking[d][b + 1]
                  << "} owned by ";
        grid_pos[d] = b;
        int owner = 0;
        MPI_Cart_rank(pmap_ptr_->gridComm(), grid_pos.data(), &owner);
        std::cout << owner << "\n";
        grid_pos[d] = 0;
      }
    }
    std::cout << std::endl;

    std::cout << "Subcomm sizes: ";
    for (auto s : pmap_ptr_->subCommSizes()) {
      std::cout << s << " ";
    }
    std::cout << std::endl;
  }
}

template <typename ElementType, typename ExecSpace>
boost::optional<std::pair<MPI_IO::SptnFileHeader, MPI_File>>
TensorBlockSystem<ElementType, ExecSpace>::readHeader(
    std::string const &file_name, int indexbase) {

  bool is_binary = detail::fileFormatIsBinary(file_name);
  if (is_binary && indexbase != 0) {
    throw std::logic_error(
        "The binary format only supports zero based indexing\n");
  }

  if (!is_binary) {
    std::ifstream tensor_file(file_name);
    Ti_ = read_sptensor_header(tensor_file);
    return boost::none;
  } else {
    auto mpi_fh = MPI_IO::openFile(DistContext::commWorld(), file_name);
    auto binary_header = MPI_IO::readHeader(DistContext::commWorld(), mpi_fh);
    Ti_ = binary_header.toTensorInfo();
    return std::make_pair(std::move(binary_header), mpi_fh);
  }
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
