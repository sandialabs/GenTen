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
#include "Genten_DistTensorContext.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_PerfHistory.hpp"

// To do:
//  * use normal sampling setup
//  * use normal dist-update methods
//  * add MPI one-sided back in

namespace Genten {

template <typename ExecSpace>
class GCP_FedOpt {
public:
  GCP_FedOpt(const DistTensorContext<ExecSpace>& dtc,
             const SptensorT<ExecSpace>& spTensor,
             const KtensorT<ExecSpace>& kTensor,
             const AlgParams& algParams,
             PerfHistory& history);
  ~GCP_FedOpt() = default;

  // For now let's delete these so they aren't accidently used
  // Can come back and define them later if needed
  GCP_FedOpt(GCP_FedOpt const &) = delete;
  GCP_FedOpt &operator=(GCP_FedOpt const &) = delete;

  GCP_FedOpt(GCP_FedOpt &&) = delete;
  GCP_FedOpt &operator=(GCP_FedOpt &&) = delete;

  template <typename Loss>
  void operator() (const Loss& loss_func);

private:
  std::int32_t ndims() const { return dtc_.ndims(); }
  ProcessorMap const &pmap() const { return dtc_.pmap(); }

  template <typename Loss> ttb_real fedOpt(Loss const &loss);

  DistTensorContext<ExecSpace> dtc_;
  SptensorT<ExecSpace> spTensor_;
  KtensorT<ExecSpace> Kfac_;
  AlgParams algParams_;
  PerfHistory& history_;

  unsigned int seed_;
  MPI_Datatype mpiElemType_ = DistContext::toMpiType<ttb_real>();

  int global_nzg;
  int global_nzf;
  int global_zg;
  int global_zf;
  int percent_nz_epoch;
};

template <typename ExecSpace>
GCP_FedOpt<ExecSpace>::
GCP_FedOpt(const DistTensorContext<ExecSpace>& dtc,
           const SptensorT<ExecSpace>& spTensor,
           const KtensorT<ExecSpace>& kTensor,
           const AlgParams& algParams,
           PerfHistory& history) :
  dtc_(dtc), spTensor_(spTensor), Kfac_(kTensor), algParams_(algParams),
  history_(history)
{
  seed_ = algParams_.gcp_seed > 0 ? algParams_.gcp_seed : std::random_device{}();

  // GCP_FedOpt currently uses its own approach for dividing samples across
  // processors
  const std::uint64_t np = dtc_.nprocs();
  const std::uint64_t lnz = spTensor_.nnz();
  const std::uint64_t gnz = dtc_.globalNNZ(spTensor_);
  const std::uint64_t lz = spTensor_.numel_float() - lnz;

  std::uint64_t global_batch_size_nz = algParams_.num_samples_nonzeros_grad;
  std::uint64_t global_batch_size_z = algParams_.num_samples_zeros_grad;
  std::uint64_t global_value_size_nz = algParams_.num_samples_nonzeros_value;
  std::uint64_t global_value_size_z = algParams_.num_samples_zeros_value;

  // If we have fewer nnz than the batch size don't over sample them
  algParams_.num_samples_nonzeros_grad =
    std::min(lnz, global_batch_size_nz / np);
  algParams_.num_samples_zeros_grad =
    std::min(lz, global_batch_size_z / np);

  // No point in sampling more nonzeros than we actually have
  algParams_.num_samples_nonzeros_value =
    std::min(lnz, global_value_size_nz / np);
  algParams_.num_samples_zeros_value =
    std::min(lz, global_value_size_z / np);

  // Currently we require fused, semi-stratified sampling
  gt_assert(algParams_.sampling_type == Genten::GCP_Sampling::SemiStratified);
  gt_assert(algParams_.fuse == true);

  // Reset the global_batch_size_nz to the value that we will actually use
  global_batch_size_nz =
      pmap().gridAllReduce(algParams_.num_samples_nonzeros_grad);

  const auto my_rank = pmap().gridRank();

  const int local_batch_size = algParams_.num_samples_nonzeros_grad;
  const int local_batch_sizez = algParams_.num_samples_zeros_grad;
  const int local_value_size = algParams_.num_samples_nonzeros_value;
  const int local_value_sizez = algParams_.num_samples_zeros_value;
  global_nzg = pmap().gridAllReduce(local_batch_size);
  global_nzf = pmap().gridAllReduce(local_value_size);
  global_zg = pmap().gridAllReduce(local_batch_sizez);
  global_zf = pmap().gridAllReduce(local_value_sizez);

  percent_nz_epoch =
      double(algParams_.epoch_iters * global_batch_size_nz) / double(gnz) *
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
  }
  pmap().gridBarrier();
}

template <typename ExecSpace>
template <typename Loss>
void
GCP_FedOpt<ExecSpace>::
operator() (const Loss& loss) {
  const ProcessorMap* pmap = &dtc_.pmap();

  const auto start_time = MPI_Wtime();
  const auto my_rank = pmap->gridRank();

  std::ostream& out = my_rank == 0 ? std::cout : Genten::bhcout;

  // Distribute the initial guess to have weights of one.
  if (algParams_.normalize)
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
    spTensor_, ut, algParams_, false);
  sampler.prepareGradient(GFac);

  // Create steppers
  Impl::GCP_SGD_Step<ExecSpace,Loss> *stepper =
    Impl::createStepper<ExecSpace,Loss>(algParams_, u, algParams_.step_type);
  Impl::GCP_SGD_Step<ExecSpace,Loss> *meta_stepper =
    Impl::createStepper<ExecSpace,Loss>(algParams_, meta_u, algParams_.meta_step_type);

  const auto maxEpochs = algParams_.maxiters;
  const auto epochIters = algParams_.epoch_iters;
  const auto max_fails = algParams_.max_fails;
  const auto tol = algParams_.gcp_tol;
  const auto dp_iters = algParams_.downpour_iters;

  auto annealer_ptr = getAnnealer(algParams_);
  auto &annealer = *annealer_ptr;

  bool fedavg = algParams_.fed_method == GCP_FedMethod::FedAvg;
  auto meta_lr = algParams_.meta_rate;

  if (fedavg)
    out << "\nGCP-FedAvg:\n";
  else
    out << "\nGCP-FedOpt:\n";
  out << "Generalized function type: " << loss.name() << std::endl
      << "Meta step method: " << GCP_Step::names[algParams_.meta_step_type]
      << std::endl
      << "Local step method: " << GCP_Step::names[algParams_.step_type]
      << std::endl
      << "Max epochs: " << maxEpochs << std::endl
      << "Iterations per epoch: " << epochIters << std::endl
      << "Downpour interations: " << dp_iters << std::endl
      << "Max fails: " << max_fails << std::endl;
  annealer.print(out);
  // NOTE (ETP 9/20/22:  sampler output is not correct because pmap is not
  // passed through the tensor, so it only prints the local number of samples.
  // This will be fixed when this code is refactored.
  //sampler.print(out);
  out << "  Function sampler:  stratified with "
      << global_nzf
      << " nonzero and " << global_zf
      << " zero samples\n"
      << "  Gradient sampler:  semi-stratified with "
      << global_nzg
      << " nonzero and " << global_zg
      << " zero samples\n"
      << "  Gradient nonzero samples per epoch: "
      << global_zg*epochIters
      << " (" << std::setprecision(1) << std::fixed << percent_nz_epoch << "%)"
      << std::endl;
  out << "  Gradient method: Fused sampling and "
      << MTTKRP_All_Method::names[algParams_.mttkrp_all_method]
      << " MTTKRP\n";
  out << std::endl;

  // Timers -- turn on fences when timing info is requested so we get
  // accurate kernel times
  int num_timers = 0;
  const int timer_sgd = num_timers++;
  const int timer_sort = num_timers++;
  const int timer_sample_f = num_timers++;
  const int timer_fest = num_timers++;
  //const int timer_sample_g = num_timers++;
  const int timer_grad = num_timers++;
  const int timer_grad_nzs = num_timers++;
  const int timer_grad_zs = num_timers++;
  const int timer_grad_init = num_timers++;
  const int timer_grad_mttkrp = num_timers++;
  const int timer_grad_comm = num_timers++;
  const int timer_grad_update = num_timers++;
  const int timer_step = num_timers++;
  const int timer_meta_step = num_timers++;
  const int timer_allreduce = num_timers++;
  SystemTimer timer(num_timers, algParams_.timings, pmap);

  ttb_indx nd = ut.ndims();

  // Start timer for total execution time of the algorithm.
  timer.start(timer_sgd);

  timer.start(timer_sort);
  Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(seed_);
  sampler.initialize(rand_pool, true, out);
  timer.stop(timer_sort);

  timer.start(timer_sample_f);
  sampler.sampleTensorF(ut, loss);
  timer.stop(timer_sample_f);

  // Fit stuff
  timer.start(timer_fest);
  ttb_real fest, ften;
  StreamingHistory<ExecSpace> hist;
  ttb_real penalty = 0.0;
  sampler.value(ut, hist, penalty, loss, fest, ften);
  fest = pmap->gridAllReduce(fest);
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

  //double t0 = 0;
  double t1 = 0;
  ttb_indx nfails = 0;
  for (ttb_indx e = 0; e < maxEpochs; ++e) { // Epochs
    //t0 = MPI_Wtime();
    auto epoch_lr = annealer(e);
    stepper->setStep(epoch_lr);

    auto do_epoch_iter = [&](double &gtime, double &etime) {
      timer.start(timer_grad);
      timer.start(timer_grad_init);
      g.zero();
      timer.stop(timer_grad_init);
      auto start = MPI_Wtime();
      sampler.gradient(ut, hist, penalty,
                       loss, g, GFac, 0, nd,
                       timer, timer_grad_init, timer_grad_nzs,
                       timer_grad_zs, timer_grad_mttkrp, timer_grad_comm,
                       timer_grad_update);
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

    for (ttb_indx i = 0; i < epochIters; ++i) {
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
    sampler.value(ut, hist, penalty, loss, fest, ften);
    fest = pmap->gridAllReduce(fest);
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
      out << ", nfails = " << nfails+1;
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
      meta_u.set(u_best);
      annealer.failed();
      stepper->setFailed();
      meta_stepper->setFailed();
      fest = fest_best;
      ++nfails;
    }

    if (nfails > max_fails || fest < tol*fest_init)
      break;
  }
  u.set(u_best);
  deep_copy(Kfac_, ut);

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

  //return fest_prev;
}

template <typename ExecSpace>
void
gcp_fed(const DistTensorContext<ExecSpace>& dtc,
        const SptensorT<ExecSpace>& spTensor,
        const KtensorT<ExecSpace>& kTensor,
        const AlgParams& algParams,
        PerfHistory& history)
{
  GENTEN_TIME_MONITOR("GCP-Fed-Opt");
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::gcp_fed_opt");
#endif

  // Dispatch implementation based on loss function type
  GCP_FedOpt<ExecSpace> f(dtc, spTensor, kTensor, algParams, history);
  dispatch_loss(algParams, f);
}

} // namespace Genten
