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

#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>
#include <type_traits>

#include "Genten_GCP_SGD.hpp"
#include "Genten_GCP_SamplerFactory.hpp"
#include "Genten_GCP_ValueKernels.hpp"
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_KokkosVector.hpp"
#include "Genten_GCP_SGD_Step.hpp"
#include "Genten_GCP_SGD_IterFactory.hpp"

#include "Genten_Annealer.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_DistKtensorUpdate.hpp"

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

namespace Genten {

  template <typename TensorType, typename LossFunction>
  GCPSGD<TensorType,LossFunction>::
  GCPSGD(const KtensorT<exec_space>& u,
         const LossFunction& loss_func_,
         const ttb_indx mode_beg_,
         const ttb_indx mode_end_,
         const AlgParams& algParams_) :
    loss_func(loss_func_), mode_beg(mode_beg_), mode_end(mode_end_),
    algParams(algParams_), stepper(nullptr)
  {
    // Check for valid option combinations
    if (algParams.async &&
        algParams.sampling_type != GCP_Sampling::SemiStratified)
      Genten::error("Must use semi-stratified sampling with asynchronous solver!");
    if (algParams.async &&
        algParams.dist_update_method != Dist_Update_Method::AllReduce)
      Genten::error("Asynchronous GCP-SGD requires AllReduce distributed parallelism");
    if (algParams.fuse &&
        algParams.dist_update_method != Dist_Update_Method::AllReduce &&
        algParams.dist_update_method != Dist_Update_Method::OneSided)
      Genten::error("Fused sampling requies AllReduce or OneSided distributed parallelism");

    // Create stepper
    // Note:  uv is not a view of u, so this involves an allocation.
    // The steppers just need u for dimensions, so there should be a more
    // efficient way to do this
    KokkosVector<exec_space> uv(u);
    KokkosVector<exec_space> us = uv.subview(mode_beg, mode_end);
    if (algParams.step_type == GCP_Step::ADAM)
      stepper = new Impl::AdamStep<exec_space,LossFunction>(algParams, us);
    else if (algParams.step_type == GCP_Step::AdaGrad)
      stepper = new Impl::AdaGradStep<exec_space,LossFunction>(algParams, us);
    else if (algParams.step_type == GCP_Step::AMSGrad)
      stepper = new Impl::AMSGradStep<exec_space,LossFunction>(algParams, us);
    else
      stepper = new Impl::SGDStep<exec_space,LossFunction>();
  }

  template <typename TensorType, typename LossFunction>
  GCPSGD<TensorType,LossFunction>::
  GCPSGD(const KtensorT<exec_space>& u,
         const LossFunction& loss_func_,
         const AlgParams& algParams_) :
    GCPSGD(u,loss_func_,0,u.ndims(),algParams_) {}

  template <typename TensorType, typename LossFunction>
  GCPSGD<TensorType,LossFunction>::
  ~GCPSGD()
  {
    delete stepper;
  }

  template <typename TensorType, typename LossFunction>
  void
  GCPSGD<TensorType,LossFunction>::
  reset()
  {
    stepper->reset();
  }

  template <typename TensorType, typename LossFunction>
  void
  GCPSGD<TensorType,LossFunction>::
  solve(TensorType& X,
        KtensorT<exec_space>& u0,
        const ttb_real penalty,
        ttb_indx& numEpochs,
        ttb_real& fest,
        PerfHistory& perfInfo,
        std::ostream& out,
        const bool print_hdr,
        const bool print_ftr,
        const bool print_itn) const
  {
    ttb_real ften = 0.0;
    solve(X, u0, StreamingHistory<exec_space>(),
          penalty, numEpochs, fest, ften, perfInfo, out,
          print_hdr, print_ftr, print_itn);
  }

  template <typename TensorType, typename LossFunction>
  void
  GCPSGD<TensorType,LossFunction>::
  solve(TensorType& X,
        KtensorT<exec_space>& u0,
        const StreamingHistory<exec_space>& hist,
        const ttb_real penalty,
        ttb_indx& numEpochs,
        ttb_real& fest,
        ttb_real& ften,
        PerfHistory& perfInfo,
        std::ostream& out,
        const bool print_hdr,
        const bool print_ftr,
        const bool print_itn) const
  {
    typedef KokkosVector<exec_space> VectorType;
    using std::sqrt;
    using std::pow;

    const ProcessorMap* pmap = u0.getProcessorMap();

    // Constants for the algorithm
    const ttb_real tol = algParams.gcp_tol;
    const ttb_indx max_fails = algParams.max_fails;
    const ttb_indx epoch_iters = algParams.epoch_iters;
    const ttb_indx seed = algParams.gcp_seed > 0 ? algParams.gcp_seed : std::random_device{}();
    const ttb_indx maxEpochs = algParams.maxiters;
    const ttb_indx printIter = print_itn ? algParams.printitn : 0;
    const bool compute_fit = algParams.compute_fit;

    // Create sampler
    Sampler<TensorType,LossFunction> *sampler =
      createSampler<LossFunction>(X, u0, algParams);

    // Create annealer
    auto annealer = getAnnealer(algParams);

    if (print_hdr) {
      out << "\nGCP-SGD (Generalized CP Tensor Decomposition):\n"
          << "Generalized function type: " << loss_func.name() << std::endl
          << "Optimization method: " << GCP_Step::names[algParams.step_type]
          << std::endl
          << "Max iterations (epochs): " << maxEpochs << std::endl
          << "Iterations per epoch: " << epoch_iters << std::endl;
      annealer->print(out);
      sampler->print(out);
      out << "Gradient method: ";
      if (algParams.async)
        out << "Fused asynchronous sampling and atomic MTTKRP\n";
      else if (algParams.fuse)
        out << "Fused sampling and "
            << MTTKRP_All_Method::names[algParams.mttkrp_all_method]
            << " MTTKRP\n";
      else {
        out << MTTKRP_All_Method::names[algParams.mttkrp_all_method];
        if (algParams.mttkrp_all_method == MTTKRP_All_Method::Iterated)
          out << " (" << MTTKRP_Method::names[algParams.mttkrp_method] << ")";
        out << " MTTKRP\n";
      }
      out << std::endl;
    }

    // Timers -- turn on fences when timing info is requested so we get
    // accurate kernel times
    int num_timers = 0;
    const int timer_sgd = num_timers++;
    const int timer_sort = num_timers++;
    const int timer_sample_f = num_timers++;
    const int timer_fest = num_timers++;
    const int timer_comm = num_timers++;
    SystemTimer timer(num_timers, algParams.timings, pmap);

    // Start timer for total execution time of the algorithm.
    timer.start(timer_sgd);

    // Create iterator
    Impl::GCP_SGD_Iter<TensorType,LossFunction> *itp =
      Impl::createIter<LossFunction>(
        X, u0, hist, penalty, mode_beg, mode_end, algParams);
    Impl::GCP_SGD_Iter<TensorType,LossFunction>& it = *itp;

    // Get vector/Ktensor for current solution (this is a view of the data)
    VectorType u = it.getSolution();
    KtensorT<exec_space> ut = u.getKtensor();
    ut.setProcessorMap(pmap);

    // Copy Ktensor for restoring previous solution
    VectorType u_prev = u.clone();
    u_prev.set(u);

    // Initialize sampler (sorting, hashing, ...)
    timer.start(timer_sort);
    Kokkos::Random_XorShift64_Pool<exec_space> rand_pool(seed);
    sampler->initialize(rand_pool, print_itn, out);
    timer.stop(timer_sort);

    // Sample X for f-estimate
    GENTEN_START_TIMER("sample objective");
    timer.start(timer_sample_f);
    sampler->sampleTensorF(ut, loss_func);
    timer.stop(timer_sample_f);
    GENTEN_STOP_TIMER("sample objective");

    // Objective estimates
    ttb_real fit = 0.0;
    const ttb_real x_norm = X.global_norm();
    DistKtensorUpdate<exec_space> *dku_fit = nullptr;
    KtensorT<exec_space> ut_overlap_fit;
    if (compute_fit) {
      dku_fit = createKtensorUpdate(X, ut, algParams);
      ut_overlap_fit = dku_fit->createOverlapKtensor(ut);
    }
    GENTEN_START_TIMER("objective function");
    timer.start(timer_fest);
    sampler->value(ut, hist, penalty, loss_func, fest, ften);
    auto fitter = [&]{
      const auto x_norm2 = x_norm * x_norm;
      const auto u_norm2 = ut.normFsq();
      dku_fit->doImport(ut_overlap_fit, ut, timer, timer_comm);
      const auto dot = innerprod(X, ut_overlap_fit);
      const auto numerator = sqrt(x_norm2 + u_norm2 - 2.0 * dot);
      const auto denom = x_norm;
      return 1.0 - numerator/denom;
    };
    if (compute_fit) {
      fit = fitter();
    }
    timer.stop(timer_fest);
    ttb_real fest_prev = fest;
    ttb_real fit_prev = fit;
    GENTEN_STOP_TIMER("objective function");

    if (print_hdr || print_itn) {
      out << "Initial f-est: "
          << std::setw(13) << std::setprecision(6) << std::scientific
          << fest;
      if (compute_fit)
        out << ", fit: "
            << std::setw(10) << std::setprecision(3) << std::scientific
            << fit;
      out << ", tensor norm: "
            << std::setw(10) << std::setprecision(3) << std::scientific
            << x_norm;
      out << std::endl;
    }

    {
      perfInfo.addEmpty();
      auto& p = perfInfo.lastEntry();
      p.iteration = 0;
      p.residual = fest;
      if (compute_fit)
        p.fit = fit;
      p.cum_time = timer.getTotalTime(timer_sgd);
    }

    // SGD epoch loop
    ttb_indx nfails = 0;
    ttb_indx total_iters = 0;

    for (numEpochs=0; numEpochs<maxEpochs; ++numEpochs) {

      // Gradient step size
      auto epoch_lr = (*annealer)(numEpochs);
      stepper->setStep(epoch_lr);

      // Epoch iterations
      it.run(X, loss_func, *sampler, *stepper, total_iters);

      // compute objective estimate
      GENTEN_START_TIMER("objective function");
      timer.start(timer_fest);
      sampler->value(ut, hist, penalty, loss_func, fest, ften);
      if (compute_fit) {
        fit = fitter();
      }
      timer.stop(timer_fest);
      GENTEN_STOP_TIMER("objective function");

      // check convergence
      bool failed_epoch = fest > fest_prev || std::isnan(fest);

      if (failed_epoch)
        ++nfails;

      // Print progress of the current iteration.
      if ((printIter > 0) && (((numEpochs + 1) % printIter) == 0)) {
        out << "Epoch " << std::setw(3) << numEpochs + 1 << ": f-est = "
            << std::setw(13) << std::setprecision(6) << std::scientific
            << fest;
        if (compute_fit)
          out << ", fit = "
              << std::setw(10) << std::setprecision(3) << std::scientific
              << fit;
        out << ", step = "
            << std::setw(8) << std::setprecision(1) << std::scientific
            << stepper->getStep();
        out << ", time = "
            << std::setw(8) << std::setprecision(2) << std::scientific
            << timer.getTotalTime(timer_sgd) << " sec";
        if (failed_epoch)
          out << ", nfails = " << nfails
              << " (resetting to solution from last epoch)";
        out << std::endl;
      }

      {
        perfInfo.addEmpty();
        auto& p = perfInfo.lastEntry();
        p.iteration = numEpochs+1;
        p.residual = fest;
        if (compute_fit)
          p.fit = fit;
        p.cum_time = timer.getTotalTime(timer_sgd);
      }

      if (failed_epoch) {
        // restart from last epoch
        u.set(u_prev);
        fest = fest_prev;
        fit = fit_prev;
        stepper->setFailed();
        annealer->failed();
      }
      else {
        // update previous data
        u_prev.set(u);
        fest_prev = fest;
        fit_prev = fit;
        stepper->setPassed();
        annealer->success();
      }

      if (nfails > max_fails || fest < tol)
        break;
    }
    timer.stop(timer_sgd);

    if (print_ftr) {
      out << "Final f-est: "
          << std::setw(13) << std::setprecision(6) << std::scientific
          << fest;
      if (compute_fit)
        out << ", fit: "
            << std::setw(10) << std::setprecision(3) << std::scientific
            << fit;
      out << std::endl
          << "GCP-SGD completed " << total_iters << " iterations in "
          << std::setw(8) << std::setprecision(2) << std::scientific
          << timer.getTotalTime(timer_sgd) << " seconds" << std::endl;
      if (algParams.timings) {
        out << "\tsort/hash: " << timer.getTotalTime(timer_sort)
            << " seconds\n"
            << "\tsample-f:  " << timer.getTotalTime(timer_sample_f)
            << " seconds\n"
            << "\tf-est:     " << timer.getTotalTime(timer_fest)
            << " seconds\n";
        it.printTimers(out);
      }
    }

    u.copyToKtensor(u0);
    ften = fest;

    delete sampler;
    delete itp;
    if (dku_fit != nullptr)
      delete dku_fit;
  }

  template<typename TensorType>
  void gcp_sgd(TensorType& x,
               KtensorT<typename TensorType::exec_space>& u,
               const AlgParams& algParams,
               ttb_indx& numIters,
               ttb_real& resNorm,
               PerfHistory& perfInfo,
               std::ostream& out)
  {
    GENTEN_TIME_MONITOR("GCP-SGD");
#ifdef HAVE_CALIPER
    cali::Function cali_func("Genten::gcp_sgd");
#endif

    // Check size compatibility of the arguments.
    if (u.isConsistent() == false)
      Genten::error("Genten::gcp_sgd - ktensor u is not consistent");
    if (x.ndims() != u.ndims())
      Genten::error("Genten::gcp_sgd - u and x have different num dims");

    // Distribute the initial guess to have weights of one.
    u.normalize(Genten::NormTwo);
    u.distribute();

    // Dispatch implementation based on loss function type
    dispatch_loss(algParams, [&](const auto& loss)
    {
      using LossType =
        std::remove_cv_t< std::remove_reference_t<decltype(loss)> >;
      GCPSGD<TensorType,LossType> gcpsgd(u, loss, algParams);
      gcpsgd.solve(x, u, ttb_real(0.0), numIters, resNorm, perfInfo, out,
                   true, true, algParams.printitn);
    });

    // Normalize Ktensor u
    u.normalize(Genten::NormTwo);
    u.arrange();
  }

}

#define LOSS_INST_MACRO(SPACE,LOSS)                                     \
  template class Genten::GCPSGD<SptensorT<SPACE>,LOSS>;                 \
  template class Genten::GCPSGD<TensorT<SPACE>,LOSS>;

#define INST_MACRO(SPACE)                                               \
  GENTEN_INST_LOSS(SPACE,LOSS_INST_MACRO)                               \
                                                                        \
  template void gcp_sgd<SptensorT<SPACE> >(                             \
    SptensorT<SPACE>& x,                                                \
    KtensorT<SPACE>& u,                                                 \
    const AlgParams& algParams,                                         \
    ttb_indx& numIters,                                                 \
    ttb_real& resNorm,                                                  \
    PerfHistory& perfInfo,                                              \
    std::ostream& out);                                                 \
                                                                        \
  template void gcp_sgd<TensorT<SPACE> >(                               \
    TensorT<SPACE>& x,                                                  \
    KtensorT<SPACE>& u,                                                 \
    const AlgParams& algParams,                                         \
    ttb_indx& numIters,                                                 \
    ttb_real& resNorm,                                                  \
    PerfHistory& perfInfo,                                              \
    std::ostream& out);

GENTEN_INST(INST_MACRO)
