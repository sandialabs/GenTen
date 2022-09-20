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

#include <cmath>

#include "Genten_Driver.hpp"
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_GCP_SGD_Iter.hpp"
#include "Genten_GCP_SemiStratifiedSampler.hpp"
#include "Genten_GCP_ValueKernels.hpp"

#include "Genten_Annealer.hpp"
#include "Genten_Ptree.hpp"
#include "Genten_DistTensorContext.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_PerfHistory.hpp"

namespace Genten {

template <typename ExecSpace>
class DistGCP {
public:
  DistGCP(const DistTensorContext& dtc,
          const SptensorT<ExecSpace>& spTensor,
          const KtensorT<ExecSpace>& kTensor,
          const ptree& tree,
          PerfHistory& history);
  ~DistGCP() = default;

  // For now let's delete these so they aren't accidently used
  // Can come back and define them later if needed
  DistGCP(DistGCP const &) = delete;
  DistGCP &operator=(DistGCP const &) = delete;

  DistGCP(DistGCP &&) = delete;
  DistGCP &operator=(DistGCP &&) = delete;

  ttb_real compute();

private:
  std::int32_t ndims() const { return dtc_.ndims(); }
  ProcessorMap const &pmap() const { return dtc_.pmap(); }

  template <typename Loss> ttb_real allReduceTrad(Loss const &loss);
  template <typename Loss> ttb_real fedOpt(Loss const &loss);
  template <typename Loss> ttb_real pickMethod(Loss const &loss);
  AlgParams setAlgParams() const;

  DistTensorContext dtc_;
  SptensorT<ExecSpace> spTensor_;
  KtensorT<ExecSpace> Kfac_;
  ptree input_;
  PerfHistory& history_;

  bool dump_; // I don't love keeping this flag, but it's easy
  unsigned int seed_;
  MPI_Datatype mpiElemType_ = DistContext::toMpiType<ttb_real>();
};

template <typename ExecSpace>
DistGCP<ExecSpace>::DistGCP(const DistTensorContext& dtc,
                            const SptensorT<ExecSpace>& spTensor,
                            const KtensorT<ExecSpace>& kTensor,
                            const ptree& tree,
                            PerfHistory& history) :
  dtc_(dtc), spTensor_(spTensor), Kfac_(kTensor),
  input_(tree.get_child("gcp-sgd-dist")), history_(history),
  dump_(tree.get<bool>("dump", false)),
  seed_(input_.get<unsigned int>("seed", std::random_device{}()))
{
  if (dump_) {
    if (DistContext::rank() == 0) {
      std::cout
          << "gcp:\n"
             "\trank: rank at which to decompose the tensor\n"
             "\tloss: Loss function options are {guassian, poisson, "
             "bernoulli}\n"
             "\tmethod: The SGD method to use (default: adam), options {adam, "
             "fedopt, sgd, sgdm, adagrad, demon}\n"
             "\tmax_epochs: the number of epochs to run.\n"
             "\tbatch_size_nz: the number of non-zeros to sample per batch.\n"
             "\tbatch_size_zero: the number of zeros to sample per batch.\n"
             "\tepoch_size: the number of `epoch_iters` to run, defaults to "
             "number of "
             "non-zeros  divided by the number of non-zeros per batch.\n"
             "\tseed: Random seed default(std::random_device{}()).\n";
    }
  }
}

template <typename ExecSpace>
template <typename Loss>
ttb_real DistGCP<ExecSpace>::pickMethod(Loss const &loss) {
  auto method = input_.get<std::string>("method", "sgd");
  if (method == "fedopt" || method == "fedavg") {
    return fedOpt(loss);
  } else if (method == "sgd") {
    return allReduceTrad(loss);
  } else {
    Genten::error("Your method for distributed SGD wasn't recognized.\n");
  }
  return -1.0;
}

template <typename ExecSpace>
ttb_real DistGCP<ExecSpace>::compute() {
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

template <typename ExecSpace>
AlgParams DistGCP<ExecSpace>::setAlgParams() const {
  const std::uint64_t np = dtc_.nprocs();
  const std::uint64_t lnz = spTensor_.nnz();
  const std::uint64_t gnz = dtc_.globalNNZ(spTensor_);
  const std::uint64_t lz = spTensor_.numel_float() - lnz;

  AlgParams algParams;
  algParams.maxiters = input_.get<int>("max-epochs", 1000);
  algParams.max_fails = input_.get<int>("fails", 2);
  algParams.gcp_tol = input_.get<double>("tol", 1e-4);

  auto global_batch_size_nz = input_.get<int>("batch-size-nz", 128);
  auto global_batch_size_z =
      input_.get<int>("batch-size-zero", global_batch_size_nz);
  auto global_value_size_nz =
    input_.get<int>("value-size-nz", 100000);
  auto global_value_size_z =
      input_.get<int>("value-size-zero", global_value_size_nz);

  // If we have fewer nnz than the batch size don't over sample them
  algParams.num_samples_nonzeros_grad =
    std::min(lnz, global_batch_size_nz / np);
  algParams.num_samples_zeros_grad =
    std::min(lz, global_batch_size_z / np);

  // No point in sampling more nonzeros than we actually have
  algParams.num_samples_nonzeros_value =
    std::min(lnz, global_value_size_nz / np);
  algParams.num_samples_zeros_value =
    std::min(lz, global_value_size_z / np);

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
      pmap().gridAllReduce(algParams.num_samples_nonzeros_grad);

  if (input_.contains("epoch-size")) {
    algParams.epoch_iters = input_.get<int>("epoch-size");
  } else {
    algParams.epoch_iters = gnz / global_batch_size_nz;
  }

  algParams.fixup<ExecSpace>(std::cout);

  const auto my_rank = pmap().gridRank();

  const int local_batch_size = algParams.num_samples_nonzeros_grad;
  const int local_batch_sizez = algParams.num_samples_zeros_grad;
  const int local_value_size = algParams.num_samples_nonzeros_value;
  const int local_value_sizez = algParams.num_samples_zeros_value;
  const int global_nzg = pmap().gridAllReduce(local_batch_size);
  const int global_nzf = pmap().gridAllReduce(local_value_size);
  const int global_zg = pmap().gridAllReduce(local_batch_sizez);
  const int global_zf = pmap().gridAllReduce(local_value_sizez);

  const auto percent_nz_batch =
      double(global_batch_size_nz) / double(gnz) * 100.0;
  const auto percent_nz_epoch =
      double(algParams.epoch_iters * global_batch_size_nz) / double(gnz) *
      100.0;

  // Collect info from the other ranks
  if (my_rank > 0) {
    MPI_Gather(&lnz, 1, MPI_INT, nullptr, 1, MPI_INT, 0, pmap().gridComm());
    MPI_Gather(&local_batch_size, 1, MPI_INT, nullptr, 1, MPI_INT, 0,
               pmap().gridComm());
    MPI_Gather(&local_batch_sizez, 1, MPI_INT, nullptr, 1, MPI_INT, 0,
               pmap().gridComm());
    MPI_Gather(&local_value_size, 1, MPI_INT, nullptr, 1, MPI_INT, 0,
               pmap().gridComm());
    MPI_Gather(&local_value_sizez, 1, MPI_INT, nullptr, 1, MPI_INT, 0,
               pmap().gridComm());
  } else {
    std::vector<int> lnzs(np, 0);
    std::vector<int> bs_nz(np, 0);
    std::vector<int> bs_z(np, 0);
    std::vector<int> val_nz(np, 0);
    std::vector<int> val_z(np, 0);
    MPI_Gather(&lnz, 1, MPI_INT, lnzs.data(), 1, MPI_INT, 0, pmap().gridComm());
    MPI_Gather(&local_batch_size, 1, MPI_INT, bs_nz.data(), 1, MPI_INT, 0,
               pmap().gridComm());
    MPI_Gather(&local_batch_sizez, 1, MPI_INT, bs_z.data(), 1, MPI_INT, 0,
               pmap().gridComm());
    MPI_Gather(&local_value_size, 1, MPI_INT, val_nz.data(), 1, MPI_INT, 0,
               pmap().gridComm());
    MPI_Gather(&local_value_sizez, 1, MPI_INT, val_z.data(), 1, MPI_INT, 0,
               pmap().gridComm());

    /*
    std::cout << "Iters/epoch:           " << algParams.epoch_iters << "\n";
    std::cout << "value size nz:         " << global_nzf << "\n";
    std::cout << "value size zeros:      " << global_zf << "\n";
    std::cout << "batch size nz:         " << global_nzg << "\n";
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
    if (false && np < 41) {
      std::cout << "Node Specific info: {local_nnz, local_value_size_nz, "
                << "local_value_size_z, local_batch_size_nz, "
                << "local_batch_size_z}\n";
      for (auto i = 0; i < np; ++i) {
        std::cout << "\tNode(" << i << "): "
                  << lnzs[i] << ", "
                  << val_nz[i] << ", "
                  << val_z[i] << ", "
                  << bs_nz[i] << ", "
                  << bs_z[i] << "\n";
      }
    }
    std::cout << std::endl;
    */
  }
  pmap().gridBarrier();

  return algParams;
}

template <typename ExecSpace>
template <typename Loss>
ttb_real DistGCP<ExecSpace>::fedOpt(Loss const &loss) {
  const ProcessorMap* pmap = &dtc_.pmap();

  const auto start_time = MPI_Wtime();
  const auto nprocs = dtc_.nprocs();
  const auto my_rank = pmap->gridRank();

  std::ostream& out = my_rank == 0 ? std::cout : Genten::bhcout;

  auto algParams = setAlgParams();
  if (input_.contains("eps")) {
    algParams.adam_eps = input_.get<ttb_real>("eps");
  }

  // Distribute the initial guess to have weights of one.
  const bool normalize = input_.get<bool>("normalize", true);
  if (normalize)
    Kfac_.normalize(NormTwo);
  Kfac_.distribute();

  // This is a lot of copies :/ but accept it for now
  using VectorType = KokkosVector<ExecSpace>;
  auto u = VectorType(Kfac_);
  u.copyFromKtensor(Kfac_);
  KtensorT<ExecSpace> ut = u.getKtensor();
  dtc_.allReduce(ut, true);

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

  auto sampler = SemiStratifiedSampler<ExecSpace, Loss>(
    spTensor_, algParams, false);

  // Create steppers
  const std::string meta_step = input_.get<std::string>("meta-step","adam");
  Impl::GCP_SGD_Step<ExecSpace,Loss> *stepper =
    Impl::createStepper<ExecSpace,Loss>(algParams, u);
  Impl::GCP_SGD_Step<ExecSpace,Loss> *meta_stepper =
    Impl::createStepper<ExecSpace,Loss>(algParams, meta_u, meta_step);

  const auto maxEpochs = algParams.maxiters;
  const auto epochIters = algParams.epoch_iters;
  const auto max_fails = algParams.max_fails;
  const auto tol = algParams.gcp_tol;
  const auto dp_iters = input_.get<int>("downpour-iters", 4);

  auto annealer_ptr = getAnnealer(input_);
  auto &annealer = *annealer_ptr;

  bool fedavg = input_.get<std::string>("method") == "fedavg";
  auto meta_lr = input_.get<ttb_real>("meta-lr", 1e-3);

  if (fedavg)
    out << "\nGCP-FedAvg:\n";
  else
    out << "\nGCP-FedOpt:\n";
  out << "  Generalized function type: " << loss.name() << std::endl
      << "  Meta step method: " << meta_step << std::endl
      << "  Local step method: " << GCP_Step::names[algParams.step_type]
      << std::endl
      << "  Max epochs: " << maxEpochs << std::endl
      << "  Iterations per epoch: " << epochIters << std::endl
      << "  Downpour interations: " << dp_iters << std::endl
      << "  Max fails: " << max_fails << std::endl;
  annealer.print(out);
  sampler.print(out);
  out << "  Gradient method: Fused sampling and "
      << MTTKRP_All_Method::names[algParams.mttkrp_all_method]
      << " MTTKRP\n";
  out << std::endl;

  // Timers -- turn on fences when timing info is requested so we get
  // accurate kernel times
  int num_timers = 0;
  const int timer_sgd = num_timers++;
  const int timer_sort = num_timers++;
  const int timer_sample_f = num_timers++;
  const int timer_fest = num_timers++;
  const int timer_sample_g = num_timers++;
  const int timer_grad = num_timers++;
  const int timer_grad_nzs = num_timers++;
  const int timer_grad_zs = num_timers++;
  const int timer_grad_init = num_timers++;
  const int timer_step = num_timers++;
  const int timer_meta_step = num_timers++;
  const int timer_allreduce = num_timers++;
  SystemTimer timer(num_timers, algParams.timings, pmap);

  // Start timer for total execution time of the algorithm.
  timer.start(timer_sgd);

  timer.start(timer_sort);
  Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(seed_);
  sampler.initialize(rand_pool, out);
  timer.stop(timer_sort);

  SptensorT<ExecSpace> X_val;
  ArrayT<ExecSpace> w_val;
  timer.start(timer_sample_f);
  sampler.sampleTensor(false, ut, loss, X_val, w_val);
  timer.stop(timer_sample_f);

  // Fit stuff
  timer.start(timer_fest);
  auto fest = pmap->gridAllReduce(Impl::gcp_value(X_val, ut, w_val, loss));
  auto fest_best = fest;
  auto fest_prev = fest;
  auto fest_init = fest;
  out << "Initial f-est: "
      << std::setw(13) << std::setprecision(6) << std::scientific
      << fest << std::endl;
  timer.stop(timer_fest);

  {
    history_.addEmpty();
    auto& p = history_.lastEntry();
    p.iteration = 0;
    p.residual = fest;
    p.cum_time = MPI_Wtime()-start_time;
  }

  double t0 = 0;
  double t1 = 0;
  int nfails = 0;
  for (auto e = 0; e < maxEpochs; ++e) { // Epochs
    t0 = MPI_Wtime();
    auto epoch_lr = annealer(e);
    stepper->setStep(epoch_lr);

    auto do_epoch_iter = [&](double &gtime, double &etime) {
      timer.start(timer_grad);
      timer.start(timer_grad_init);
      g.zero();
      timer.stop(timer_grad_init);
      auto start = MPI_Wtime();
      sampler.fusedGradient(ut, loss, GFac, timer, timer_grad_nzs,
                            timer_grad_zs);
      timer.stop(timer_grad);
      auto ge = MPI_Wtime();
      timer.start(timer_step);
      stepper->update();
      stepper->eval(g, u);
      timer.stop(timer_step);
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
          timer.start(timer_allreduce);
          dtc_.allReduce(ut, true);
          timer.stop(timer_allreduce);
        } else {
          timer.start(timer_meta_step);
          diff.axpby(1.0, meta_u, -1.0, u); // Subtract u from meta_u to get Meta grad
          timer.stop(timer_meta_step);

          timer.start(timer_allreduce);
          dtc_.allReduce(Dfac, true);
          timer.stop(timer_allreduce);

          timer.start(timer_meta_step);
          meta_stepper->update();
          meta_stepper->setStep(meta_lr);
          meta_stepper->eval(diff, meta_u);
          u.set(meta_u); // Everyone agrees that meta_u is the new factors
          timer.stop(timer_meta_step);
        }
        ++allreduceCounter;
        sync_time += MPI_Wtime() - s0;
      }
    }

    timer.start(timer_fest);
    fest = pmap->gridAllReduce(Impl::gcp_value(X_val, ut, w_val, loss));
    timer.stop(timer_fest);
    t1 = MPI_Wtime();

    const auto fest_diff = fest_prev - fest;
    bool passed_epoch = fest_diff > -0.001 * fest_best && !std::isnan(fest);

    out << "Epoch " << std::setw(3) << e + 1 << ": f-est = "
        << std::setw(13) << std::setprecision(6) << std::scientific
        << fest;
    out << ", meta step = "
        << std::setw(8) << std::setprecision(1) << std::scientific
        << meta_stepper->getStep();
    out << ", local step = "
        << std::setw(8) << std::setprecision(1) << std::scientific
        << stepper->getStep();
    out << ", time = "
        << std::setw(8) << std::setprecision(2) << std::scientific
        << timer.getTotalTime(timer_sgd) << " sec";
    if (!passed_epoch)
      out << ", nfails = " << nfails;
    out << std::endl;

    /*
    std::vector<double> gradient_times;
    std::vector<double> elastic_times;
    std::vector<double> eval_times;
    if (my_rank == 0) {
      gradient_times.resize(nprocs);
      elastic_times.resize(nprocs);
      eval_times.resize(nprocs);

      MPI_Gather(&gradient_time, 1, mpiElemType_, &gradient_times[0], 1,
                 mpiElemType_, 0, pmap->gridComm());
      MPI_Gather(&evaluation_time, 1, mpiElemType_, &eval_times[0], 1, mpiElemType_,
                 0, pmap->gridComm());
      MPI_Gather(&sync_time, 1, mpiElemType_, &elastic_times[0], 1, mpiElemType_, 0,
                 pmap->gridComm());
    } else {
      MPI_Gather(&gradient_time, 1, mpiElemType_, nullptr, 1, mpiElemType_, 0,
                 pmap->gridComm());
      MPI_Gather(&evaluation_time, 1, mpiElemType_, nullptr, 1, mpiElemType_, 0,
                 pmap->gridComm());
      MPI_Gather(&sync_time, 1, mpiElemType_, nullptr, 1, mpiElemType_, 0,
                 pmap->gridComm());
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

      out << "Fit(" << e << "): " << fest
          << "\n\tchange in fit: " << fest_diff
          << "\n\tlr:            " << epoch_lr
          << "\n\tallReduces:    " << allreduceCounter
          << "\n\tSeconds:       " << (t1 - t0)
          << "\n\tElapsed Time:  " << (t1 - start_time);
      out << "\n\t\tGradient(avg, min, max):  " << grad_avg << ", "
          << *min_max_gradient.first << ", " << *min_max_gradient.second
          << "\n\t\tAllReduce(avg, min, max):   " << elastic_avg << ", "
          << *min_max_elastic.first << ", " << *min_max_elastic.second
          << "\n\t\tEval(avg, min, max):      " << eval_avg << ", "
          << *min_max_evals.first << ", " << *min_max_evals.second
          << "\n";
    }
    */

    {
      history_.addEmpty();
      auto& p = history_.lastEntry();
      p.iteration = e+1;
      p.residual = fest;
      p.cum_time = t1 - start_time;
    }

    if (passed_epoch) {
      stepper->setPassed();
      meta_stepper->setPassed();
      fest_prev = fest;
      annealer.success();
      if (fest < fest_best) { // Only set best if really best
        fest_best = fest;
        u_best.set(u);
      }
    } else {
      u.set(u_best);
      annealer.failed();
      stepper->setFailed();
      meta_stepper->setFailed();
      fest = fest_best;
      ++nfails;
    }

    if (nfails > max_fails || fest < tol*fest_init)
      break;
  }

  timer.stop(timer_sgd);
  out << "Final f-est: "
      << std::setw(13) << std::setprecision(6) << std::scientific
      << fest << std::endl << std::endl;
  out << "GCP-";
  if (fedavg)
    out << "FedAvg";
  else
    out << "FedOpt";
  out << " completed in "
      << std::setw(8) << std::setprecision(2) << std::scientific
      << timer.getTotalTime(timer_sgd) << " seconds\n"
      << "\tsort/hash: " << timer.getTotalTime(timer_sort) << " seconds\n"
      << "\tsample-f:  " << timer.getTotalTime(timer_sample_f) << " seconds\n"
      << "\tf-est:     " << timer.getTotalTime(timer_fest) << " seconds\n"
      << "\tgradient:  " << timer.getTotalTime(timer_grad) << " seconds\n"
      << "\t\tinit:    " << timer.getTotalTime(timer_grad_init) << " seconds\n"
      << "\t\tnzs:     " << timer.getTotalTime(timer_grad_nzs) << " seconds\n"
      << "\t\tzs:      " << timer.getTotalTime(timer_grad_zs) << " seconds\n"
      << "\tstep/clip: " << timer.getTotalTime(timer_step) << " seconds\n"
      << "\tmeta step: " << timer.getTotalTime(timer_meta_step) << " seconds\n"
      << "\tallreduce: " << timer.getTotalTime(timer_allreduce) << " seconds\n"
      << std::endl;

  delete stepper;
  delete meta_stepper;

  return fest_prev;
}

template <typename ExecSpace>
template <typename Loss>
ttb_real DistGCP<ExecSpace>::allReduceTrad(Loss const &loss) {
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
  const auto nprocs = dtc_.nprocs();
  const auto my_rank = pmap().gridRank();
  const auto start_time = MPI_Wtime();

  auto algParams = setAlgParams();

  // Distribute the initial guess to have weights of one.
  const bool normalize = input_.get<bool>("normalize", true);
  if (normalize)
    Kfac_.normalize(NormTwo);
  Kfac_.distribute();

  using VectorType = KokkosVector<ExecSpace>;
  auto u = VectorType(Kfac_);
  u.copyFromKtensor(Kfac_);
  KtensorT<ExecSpace> ut = u.getKtensor();

  VectorType u_best = u.clone();
  u_best.set(u);

  VectorType g = u.clone(); // Gradient Ktensor
  g.zero();
  decltype(Kfac_) GFac = g.getKtensor();

  auto sampler = SemiStratifiedSampler<ExecSpace, Loss>(
    spTensor_, algParams, false);

  // Create stepper
  Impl::GCP_SGD_Step<ExecSpace,Loss> *stepper =
    Impl::createStepper<ExecSpace,Loss>(algParams, u);

  Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(seed_);
  std::stringstream ss;
  sampler.initialize(rand_pool, ss);
  if (my_rank == 0) {
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
  auto fest = pmap().gridAllReduce(Impl::gcp_value(X_val, ut, w_val, loss));
  auto fest_best = fest;
  if (my_rank == 0) {
    std::cout << "Initial guess fest: " << fest << std::endl;
  }
  pmap().gridBarrier();

  {
    history_.addEmpty();
    auto& p = history_.lastEntry();
    p.iteration = 0;
    p.residual = fest;
    p.cum_time = MPI_Wtime()-start_time;
  }

  auto fest_prev = fest;
  const auto maxEpochs = algParams.maxiters;
  const auto epochIters = algParams.epoch_iters;

  auto annealer_ptr = getAnnealer(input_);
  auto &annealer = *annealer_ptr;
  // For adam with all of the all reduces I am hopeful the barriers don't
  // really matter for timeing

  for (auto e = 0; e < maxEpochs; ++e) { // Epochs
    pmap().gridBarrier();                // Makes times more accurate
    double e_start = MPI_Wtime();
    const auto epoch_lr = annealer(e);
    stepper->setStep(epoch_lr);

    auto allreduceCounter = 0;
    double gradient_time = 0;
    double allreduce_time = 0;
    double eval_time = 0;
    for (auto i = 0; i < epochIters; ++i) {
      stepper->update();
      g.zero();
      auto ze = MPI_Wtime();
      sampler.fusedGradient(ut, loss, GFac, timer, tnzs, tzs);
      auto ge = MPI_Wtime();
      dtc_.allReduce(GFac, false /* don't average */);
      auto are = MPI_Wtime();
      stepper->eval(g, u);
      auto ee = MPI_Wtime();
      gradient_time += ge - ze;
      allreduce_time += are - ge;
      eval_time += ee - are;
      ++allreduceCounter;
    }

    double fest_start = MPI_Wtime();
    fest = pmap().gridAllReduce(Impl::gcp_value(X_val, ut, w_val, loss));
    pmap().gridBarrier(); // Makes times more accurate
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
      MPI_Gather(&gradient_time, 1, mpiElemType_, &gradient_times[0], 1,
                 mpiElemType_, 0, pmap().gridComm());
      MPI_Gather(&allreduce_time, 1, mpiElemType_, &all_reduce_times[0], 1,
                 mpiElemType_, 0, pmap().gridComm());
      MPI_Gather(&eval_time, 1, mpiElemType_, &eval_times[0], 1, mpiElemType_, 0,
                 pmap().gridComm());
    } else {
      MPI_Gather(&gradient_time, 1, mpiElemType_, nullptr, 1, mpiElemType_, 0,
                 pmap().gridComm());
      MPI_Gather(&allreduce_time, 1, mpiElemType_, nullptr, 1, mpiElemType_, 0,
                 pmap().gridComm());
      MPI_Gather(&eval_time, 1, mpiElemType_, nullptr, 1, mpiElemType_, 0,
                 pmap().gridComm());
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

    {
      history_.addEmpty();
      auto& p = history_.lastEntry();
      p.iteration = e+1;
      p.residual = fest;
      p.cum_time = e_end - start_time;
    }

    if (fest_diff > -0.001 * fest_best) {
      stepper->setPassed();
      fest_prev = fest;
      annealer.success();
      if (fest < fest_best) {
        u_best.set(u);
        fest_best = fest;
      }
    } else {
      u.set(u_best);
      fest = fest_prev;
      stepper->setFailed();
      annealer.failed();
    }
  }
  u.set(u_best);
  deep_copy(Kfac_, ut);

  const auto end_time = MPI_Wtime();
  if (my_rank == 0)
    std::cout << std::endl << "GCP-SGD-Dist completed in "
              << std::setw(8) << std::setprecision(2) << std::scientific
              << end_time-start_time << " seconds"
              << std::endl << std::endl;

  delete stepper;

  return fest_prev;
}

} // namespace Genten
