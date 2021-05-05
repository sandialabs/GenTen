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

  void allReduceKT(KtensorT<ExecSpace> &g);

public:
  TensorBlockSystem(ptree const &tree);
  ElementType getTensorNorm() const;

  /// Runs allReduceSGD reporting back an error estimate
  ElementType allReduceSGD();

private:
  ptree input_;
  small_vector<RangePair> range_;
  SptensorT<ExecSpace> sp_tensor_;
  KtensorT<ExecSpace> Kfac_;
  std::unique_ptr<ProcessorMap> pmap_ptr_;
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
  int coo[12] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
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
  TensorInfo Ti = read_sptensor_header(tensor_file);
  const auto ndims = Ti.dim_sizes.size();
  pmap_ptr_ =
      std::unique_ptr<ProcessorMap>(new ProcessorMap(DistContext::input(), Ti));
  auto &pmap_ = *pmap_ptr_;

  const auto blocking =
      detail::generateMediumGrainBlocking(Ti.dim_sizes, pmap_.gridDims());

  // Evenly distribute the tensor around the world
  auto Tvec = detail::distributeTensorToVectors(
      tensor_file, Ti.nnz, indexbase, pmap_.gridComm(), pmap_.gridRank(),
      pmap_.gridSize());

  // Now redistribute to medium grain format
  auto distributedData =
      detail::redistributeTensor(Tvec, Ti.dim_sizes, blocking, pmap_);
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
  detail::printRandomElements(sp_tensor_, 3, *pmap_ptr_, range_);
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
// is the only one in existance this will give us a embarrassinglly parallel
// initial guess which hopefully isn't actually all that bad!
template <typename ElementType, typename ExecSpace>
void TensorBlockSystem<ElementType, ExecSpace>::init_factors() {
  Genten::AlgParams algParams;
  {
    algParams.rank = input_.get<int>("rank", 5);
    algParams.method = Solver_Method::GCP_SGD;
    algParams.maxiters = 1;
    algParams.num_samples_nonzeros_grad = 100;
    algParams.num_samples_zeros_grad = 100;
    algParams.epoch_iters = 1000;
    algParams.rate = 1e-13;
    algParams.step_type = GCP_Step::SGDMomentum;
    algParams.mttkrp_all_method = MTTKRP_All_Method::Atomic;
    algParams.fuse = true;
    algParams.sampling_type = GCP_Sampling::SemiStratified;
    algParams.num_samples_nonzeros_value = 10000;
    algParams.num_samples_zeros_value = 10000;
  }

  const auto ndims = sp_tensor_.ndims();

  // TODO AllReduce
  const auto norm_x = getTensorNorm();

  // Init KFac_ randomly on each node
  Kfac_ = KtensorT<ExecSpace>(algParams.rank, ndims, sp_tensor_.size());
  Genten::RandomMT cRMT(42);
  Kfac_.setWeights(1.0); // Matlab cp_als always sets the weights to one.
  Kfac_.setMatricesScatter(false, false, cRMT);

  // TODO AllReduce, this one specifically is pretty tricky
  auto norm_k = std::sqrt(Kfac_.normFsq());
  Kfac_.weights().times(norm_x / norm_k);
  Kfac_.distribute();

  // std::stringstream rank_data;
  // auto result = Genten::driver(sp_tensor_, Kfac_, algParams, rank_data);
  // deep_copy(Kfac_, result);
  // if (pmap_ptr_->gridSize() == 1) {
  //   std::cout << rank_data.str() << std::endl;
  // }

  if (pmap_ptr_->gridSize() > 1) {
    allReduceKT(Kfac_);
  }
}

template <typename ElementType, typename ExecSpace>
void TensorBlockSystem<ElementType, ExecSpace>::allReduceKT(
    KtensorT<ExecSpace> &g) {
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

    FacMatrixT<ExecSpace> const &fac_mat = g.factors().values()[i];
    auto fac_ptr = fac_mat.rowptr(0);
    auto fac_size = fac_mat.nRows() * fac_mat.nCols();

    MPI_Allreduce(MPI_IN_PLACE, fac_ptr, fac_size, MPI_DOUBLE, MPI_SUM,
                  subComm);
  }

  for (auto i = 0; i < ndims; ++i) {
    FacMatrixT<ExecSpace> const &fac_mat = g.factors().values()[i];
    auto fac_ptr = fac_mat.rowptr(0);
    auto fac_size = fac_mat.nRows() * fac_mat.nCols();
    for (auto j = 0; j < fac_size; ++j) {
      fac_ptr[j] /= double(gridSizes[i]);
    }
  }
}

template <typename ElementType, typename ExecSpace>
ElementType TensorBlockSystem<ElementType, ExecSpace>::allReduceSGD() {
  AlgParams algParams;
  {
    algParams.maxiters = input_.get<int>("max_epochs", 1000);
    algParams.epoch_iters = input_.get<int>("epoch_size", 1000);
    algParams.num_samples_nonzeros_grad = input_.get<int>("batch_size_nz", 128);
    algParams.num_samples_zeros_grad =
        input_.get<int>("batch_size_zero", algParams.num_samples_nonzeros_grad);
    algParams.num_samples_nonzeros_value = 100000;
    algParams.num_samples_zeros_value = 100000;
    algParams.fuse = true;
    algParams.mttkrp_all_method = MTTKRP_All_Method::Atomic;
    // Uses defaults
    // algParams.w_f_nz;
    // algParams.w_f_z;
    // algParams.w_g_nz;
    // algParams.w_g_z;
  }
  algParams.fixup<ExecSpace>(std::cout);

  // Adjust for the size of the MPI grid to keep batch size similar
  const auto nprocs = pmap_ptr_->gridSize();
  const auto my_rank = pmap_ptr_->gridRank();

  algParams.num_samples_nonzeros_grad /= nprocs;
  algParams.num_samples_zeros_grad /= nprocs;
  algParams.num_samples_nonzeros_value /= nprocs;
  algParams.num_samples_zeros_value /= nprocs;

  // TODO Reduce some of these so they print correctly when using many procs
  if (my_rank == 0) {
    std::cout << "Iters/epoch:      " << algParams.epoch_iters << "\n";
    std::cout << "batch size nz:    " << algParams.num_samples_nonzeros_grad
              << "\n";
    std::cout << "batch size zeros: " << algParams.num_samples_zeros_grad
              << "\n";
    auto total_batch =
        algParams.num_samples_nonzeros_grad + algParams.num_samples_zeros_grad;
    std::cout << "batch size total: " << total_batch << "\n";
    std::cout << "NZ Samples / epoch:  "
              << algParams.num_samples_nonzeros_grad * algParams.epoch_iters
              << "\n";
    // std::cout << "Min LR: " << min_lr << "\n";
    // std::cout << "Max LR: " << max_lr << "\n";
    // std::cout << "Ti: " << Ti << std::endl;
    std::cout << std::endl;
  }
  MPI_Barrier(pmap_ptr_->gridComm());

  auto loss = GaussianLossFunction(1e-10);

  using VectorType = GCP::KokkosVector<ExecSpace>;
  auto u = VectorType(Kfac_);
  u.copyFromKtensor(Kfac_);
  KtensorT<ExecSpace> ut = u.getKtensor();

  VectorType g = u.clone(); // Gradient Ktensor
  decltype(Kfac_) GFac = g.getKtensor();

  VectorType center = u.clone(); // Center Tensor for elastic avg
  decltype(Kfac_) CFac = center.getKtensor();

  VectorType cdiff = u.clone(); // Center Tensor for elastic avg
  cdiff.zero();
  decltype(Kfac_) CFacDiff = cdiff.getKtensor();

  auto sampler = SemiStratifiedSampler<ExecSpace, GaussianLossFunction>(
      sp_tensor_, algParams);

  // Impl::SGDStep<ExecSpace, GaussianLossFunction> stepper;
  Impl::SGDMomentumStep<ExecSpace, GaussianLossFunction> stepper(algParams, u);


  auto seed = input_.get<unsigned long>("seed", std::random_device{}());
  RandomMT rng(seed);
  Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(rng.genrnd_int32());
  sampler.initialize(rand_pool, std::cout);

  SptensorT<ExecSpace> X_val;
  ArrayT<ExecSpace> w_val;
  sampler.sampleTensor(false, ut, loss, X_val, w_val);

  // Stuff for the timer that we can'g avoid providing
  SystemTimer timer;
  int tnzs = 0;
  int tzs = 0;

  // Fit stuff
  auto fest = Impl::gcp_value(X_val, ut, w_val, loss);

  if (nprocs > 1) {
    MPI_Allreduce(MPI_IN_PLACE, &fest, 1, MPI_DOUBLE, MPI_SUM,
                  pmap_ptr_->gridComm());
  }

  auto fest_prev = fest;
  const auto maxEpochs = algParams.maxiters;
  const auto epochIters = algParams.epoch_iters;
  const auto dp_iters = input_.get<int>("downpour_iters", 4);
  CosineAnnealer annealer(input_);
  for (auto e = 0; e < maxEpochs; ++e) { // Epochs
    auto epoch_lr = annealer(e);
    stepper.setStep(epoch_lr);
    if (pmap_ptr_->gridRank() == 0) {
      std::cout << "Epoch LR: " << epoch_lr << std::endl;
    }

    auto do_epoch_iter = [&] {
      g.zero();
      sampler.fusedGradient(ut, loss, GFac, timer, tnzs, tzs);
      stepper.eval(g, u);
    };

    auto extraIters = 0;
    auto allreduceCounter = 0;
    for (auto i = 0; i < epochIters; ++i) {
      if (nprocs > 1) {
        if ((i + 1) % dp_iters == 0 || i == algParams.epoch_iters - 1) {
          MPI_Request barrier_done;
          MPI_Ibarrier(pmap_ptr_->gridComm(), &barrier_done);

          int flag = 0;
          MPI_Test(&barrier_done, &flag, MPI_STATUS_IGNORE);
          while(flag == 0){
            do_epoch_iter();
            ++extraIters;
            MPI_Test(&barrier_done, &flag, MPI_STATUS_IGNORE);
          }
          
          ++allreduceCounter;
          allReduceKT(Kfac_);
        }
      }

      do_epoch_iter();
    }

    fest = Impl::gcp_value(X_val, ut, w_val, loss);
    if (nprocs > 1) {
      MPI_Allreduce(MPI_IN_PLACE, &fest, 1, MPI_DOUBLE, MPI_SUM,
                    pmap_ptr_->gridComm());
      MPI_Allreduce(MPI_IN_PLACE, &extraIters, 1, MPI_INT, MPI_SUM,
                    pmap_ptr_->gridComm());
    }

    if (pmap_ptr_->gridRank() == 0) {
      std::cout << "\tFit(" << e << "): " << fest << ", " << fest_prev - fest
                << ", did " << allreduceCounter << " factor allReduces.\n"
                << "\tdid " << extraIters << " extra iters."
                << std::endl;
    }
    fest_prev = fest;
  }

  return -1.0;
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
    for (auto i = 0; i < num_elements_per_rank; ++i) {
      auto rand_idx = dist(gen);
      auto indices = tensor.getSubscripts(rand_idx);
      auto value = tensor.value(rand_idx);

      std::cout << "\t";
      for (auto j = 0; j < tensor.ndims(); ++j) {
        std::cout << indices[j] << " ";
      }
      std::cout << value << "\n";
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
