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

#include "Genten_GCP_SGD.hpp"
#include "Genten_GCP_StratifiedSampler.hpp"
#include "Genten_GCP_SemiStratifiedSampler.hpp"
#include "Genten_GCP_ValueKernels.hpp"
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_KokkosVector.hpp"

#include "Genten_Sptensor.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_RandomMT.hpp"
#include "Genten_MixedFormatOps.hpp"

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

namespace Genten {

  namespace Impl {

    // Version of gcp_sgd that uses a sparse-array approach for computing
    // the gradient without atomics

    template <typename TensorT, typename ExecSpace, typename LossFunction>
    void gcp_sgd_sa_impl(TensorT& X, KtensorT<ExecSpace>& u0,
                         const LossFunction& loss_func,
                         const AlgParams& algParams,
                         ttb_indx& numEpochs,
                         ttb_real& fest,
                         PerfHistory& perfInfo,
                         std::ostream& out)
    {
      typedef KokkosVector<ExecSpace> VectorType;
      typedef typename VectorType::view_type view_type;
      using std::sqrt;
      using std::pow;

      const ttb_indx nd = u0.ndims();
      const ttb_indx nc = u0.ncomponents();
      const ProcessorMap* pmap = u0.getProcessorMap();

      // Constants for the algorithm
      const ttb_real tol = algParams.tol;
      const ttb_real decay = algParams.decay;
      const ttb_real rate = algParams.rate;
      const ttb_indx max_fails = algParams.max_fails;
      const ttb_indx epoch_iters = algParams.epoch_iters;
      const ttb_indx frozen_iters = algParams.frozen_iters;
      const ttb_indx seed = algParams.seed;
      const ttb_indx maxEpochs = algParams.maxiters;
      const ttb_indx printIter = algParams.printitn;
      const bool compute_fit = algParams.compute_fit;

      // ADAM parameters
      const bool use_adam = algParams.step_type == GCP_Step::ADAM;
      const ttb_real beta1 = algParams.adam_beta1;
      const ttb_real beta2 = algParams.adam_beta2;
      const ttb_real eps = algParams.adam_eps;

      // Create sampler
      Genten::SemiStratifiedSampler<ExecSpace,LossFunction> sampler(
        X, algParams, true);
      const ttb_indx tot_num_grad_samples = sampler.totalNumGradSamples();

      // bounds
      constexpr bool has_bounds = (LossFunction::has_lower_bound() ||
                                   LossFunction::has_upper_bound());
      constexpr ttb_real lb = LossFunction::lower_bound();
      constexpr ttb_real ub = LossFunction::upper_bound();

      if (printIter > 0) {
        const ttb_indx nnz = X.global_nnz();
        const ttb_real tsz = X.global_numel_float();
        const ttb_real nz = tsz - nnz;
        out << "\nGCP-SGD (Generalized CP Tensor Decomposition):\n"
            << "  Generalized function type: " << loss_func.name() << std::endl
            << "  Optimization method: " << (use_adam ? "adam\n" : "sgd\n")
            << "  Max iterations (epochs): " << maxEpochs << std::endl
            << "  Iterations per epoch: " << epoch_iters << std::endl
            << "  Learning rate / decay / maxfails: "
            << std::setprecision(1) << std::scientific
            << rate << " " << decay << " " << max_fails << std::endl;
        sampler.print(out);
        out << "  Gradient method: Fused sampling and sparse array MTTKRP\n";
        out << std::endl;
      }

      // Timers -- turn on fences when timing info is requested so we get
      // accurate kernel times
      int num_timers = 0;
      const int timer_sgd = num_timers++;
      const int timer_sort = num_timers++;
      const int timer_sample_f = num_timers++;
      const int timer_sample_g = num_timers++;
      const int timer_fest = num_timers++;
      const int timer_grad = num_timers++;
      const int timer_grad_nzs = num_timers++;
      const int timer_grad_zs = num_timers++;
      const int timer_grad_init = num_timers++;
      const int timer_grad_sort = num_timers++;
      const int timer_grad_scan = num_timers++;
      const int timer_step = num_timers++;
      const int timer_sample_g_z_nz = num_timers++;
      const int timer_sample_g_perm = num_timers++;
      SystemTimer timer(num_timers, algParams.timings, pmap);

      // Start timer for total execution time of the algorithm.
      timer.start(timer_sgd);

      // Distribute the initial guess to have weights of one.
      u0.normalize(Genten::NormTwo);
      u0.distribute();

      // Ktensor-vector for solution
      VectorType u(u0);
      u.copyFromKtensor(u0);
      KtensorT<ExecSpace> ut = u.getKtensor();
      ut.setProcessorMap(pmap);

      // Gradient Ktensor
      IndxArrayT<ExecSpace> gsz(nd, tot_num_grad_samples);
      VectorType g(nc, nd, gsz);
      KtensorT<ExecSpace> gt = g.getKtensor();
      gt.setProcessorMap(pmap);
      Kokkos::View<ttb_indx**,Kokkos::LayoutLeft,ExecSpace> gind("Gradient index", tot_num_grad_samples, nd);
      Kokkos::View<ttb_indx*,ExecSpace> perm("perm", tot_num_grad_samples);

      // Copy Ktensor for restoring previous solution
      VectorType u_prev = u.clone();
      u_prev.set(u);

      // ADAM first (m) and second (v) moment vectors
      VectorType adam_m, adam_v, adam_m_prev, adam_v_prev;
      if (use_adam) {
        adam_m = u.clone();
        adam_v = u.clone();
        adam_m_prev = u.clone();
        adam_v_prev = u.clone();
        adam_m.zero();
        adam_v.zero();
        adam_m_prev.zero();
        adam_v_prev.zero();
      }

      // Initialize sampler (sorting, hashing, ...)
      timer.start(timer_sort);
      RandomMT rng(seed);
      Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(rng.genrnd_int32());
      sampler.initialize(rand_pool, out);
      timer.stop(timer_sort);

      // Sample X for f-estimate
      SptensorT<ExecSpace> X_val, X_grad;
      ArrayT<ExecSpace> w_val, w_grad;
      timer.start(timer_sample_f);
      sampler.sampleTensor(false, ut, loss_func, X_val, w_val);
      timer.stop(timer_sample_f);

      // Objective estimates
      ttb_real fit = 0.0;
      ttb_real x_norm = 0.0;
      timer.start(timer_fest);
      fest = Impl::gcp_value(X_val, ut, w_val, loss_func);
      if (compute_fit) {
        x_norm = X.global_norm();
        ttb_real u_norm = sqrt(u.normFsq());
        ttb_real dot = innerprod(X, ut);
        fit = 1.0 - sqrt(x_norm*x_norm + u_norm*u_norm - 2.0*dot) / x_norm;
      }
      timer.stop(timer_fest);
      ttb_real fest_prev = fest;
      ttb_real fit_prev = fit;

      if (printIter > 0) {
        out << "Begin main loop\n"
            << "Initial f-est: "
            << std::setw(13) << std::setprecision(6) << std::scientific
            << fest;
        if (compute_fit)
          out << ", fit: "
              << std::setw(10) << std::setprecision(3) << std::scientific
              << fit;
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
      ttb_real nuc = 1.0;
      ttb_indx total_iters = 0;
      ttb_indx nfails = 0;
      ttb_real beta1t = 1.0;
      ttb_real beta2t = 1.0;
      for (numEpochs=0; numEpochs<maxEpochs; ++numEpochs) {
        // Gradient step size
        ttb_real step = nuc*rate;

        // Epoch iterations
        for (ttb_indx iter=0; iter<epoch_iters; ++iter) {

          // ADAM step size
          // Note sure if this should be constant for frozen iters?
          ttb_real adam_step = 0.0;
          if (use_adam) {
            beta1t = beta1 * beta1t;
            beta2t = beta2 * beta2t;
            adam_step = step*sqrt(1.0-beta2t) / (1.0-beta1t);
          }

          for (ttb_indx giter=0; giter<frozen_iters; ++giter) {
             ++total_iters;

            // compute gradient
            timer.start(timer_grad);
            timer.start(timer_grad_init);
            g.zero(); // algorithm does not use weights
            timer.stop(timer_grad_init);
            sampler.fusedGradientAndStep(
              u, loss_func, g, gind, perm,
              use_adam, adam_m, adam_v, beta1, beta2, eps,
              use_adam ? adam_step : step,
              has_bounds, lb, ub,
              timer, timer_grad_nzs, timer_grad_zs,
              timer_grad_sort, timer_grad_scan, timer_step);
            timer.stop(timer_grad);
          }
        }

        // compute objective estimate
        timer.start(timer_fest);
        fest = Impl::gcp_value(X_val, ut, w_val, loss_func);
        if (compute_fit) {
          ttb_real u_norm = sqrt(u.normFsq());
          ttb_real dot = innerprod(X, ut);
          fit = 1.0 - sqrt(x_norm*x_norm + u_norm*u_norm - 2.0*dot) / x_norm;
        }
        timer.stop(timer_fest);

        // check convergence
        const bool failed_epoch = fest > fest_prev;

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
              << step;
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
          nuc *= decay;

          // restart from last epoch
          u.set(u_prev);
          fest = fest_prev;
          fit = fit_prev;
          if (use_adam) {
            adam_m.set(adam_m_prev);
            adam_v.set(adam_v_prev);
            beta1t /= pow(beta1,epoch_iters);
            beta2t /= pow(beta2,epoch_iters);
          }
        }
        else {
          // update previous data
          u_prev.set(u);
          fest_prev = fest;
          fit_prev = fit;
          if (use_adam) {
            adam_m_prev.set(adam_m);
            adam_v_prev.set(adam_v);
          }
        }

        if (nfails > max_fails || fest < tol)
          break;
      }
      timer.stop(timer_sgd);

      if (printIter > 0) {
        out << "End main loop\n"
            << "Final f-est: "
            << std::setw(13) << std::setprecision(6) << std::scientific
            << fest;
        if (compute_fit)
          out << ", fit: "
              << std::setw(10) << std::setprecision(3) << std::scientific
              << fit;
        out << std::endl << std::endl
            << "GCP-SGD completed " << total_iters << " iterations in "
            << std::setw(8) << std::setprecision(2) << std::scientific
            << timer.getTotalTime(timer_sgd) << " seconds" << std::endl;
        if (algParams.timings) {
          out << "\tsort/hash: " << timer.getTotalTime(timer_sort)
              << " seconds\n"
              << "\tsample-f:  " << timer.getTotalTime(timer_sample_f)
              << " seconds\n"
              << "\tf-est:     " << timer.getTotalTime(timer_fest)
              << " seconds\n"
              << "\tgradient:  " << timer.getTotalTime(timer_grad)
              << " seconds\n"
              << "\t\tinit:    " << timer.getTotalTime(timer_grad_init)
              << " seconds\n"
              << "\t\tnzs:     " << timer.getTotalTime(timer_grad_nzs)
              << " seconds\n"
              << "\t\tzs:      " << timer.getTotalTime(timer_grad_zs)
              << " seconds\n"
              << "\t\tsort     " << timer.getTotalTime(timer_grad_sort)
              << " seconds\n"
              << "\t\tscan:    " << timer.getTotalTime(timer_grad_scan)
              << " seconds\n"
              << "\tstep/clip: " << timer.getTotalTime(timer_step)
              << " seconds\n";
        }
        out << std::endl;
      }

      u.copyToKtensor(u0);

      // Normalize Ktensor u
      u0.normalize(Genten::NormTwo);
      u0.arrange();
    }

  }


  template<typename TensorT, typename ExecSpace>
  void gcp_sgd_sa(TensorT& x, KtensorT<ExecSpace>& u,
                  const AlgParams& algParams,
                  ttb_indx& numIters,
                  ttb_real& resNorm,
                  PerfHistory& perfInfo,
                  std::ostream& out)
  {
#ifdef HAVE_CALIPER
    cali::Function cali_func("Genten::gcp_sgd");
#endif

    // Check size compatibility of the arguments.
    if (u.isConsistent() == false)
      Genten::error("Genten::gcp_sgd - ktensor u is not consistent");
    if (x.ndims() != u.ndims())
      Genten::error("Genten::gcp_sgd - u and x have different num dims");
    for (ttb_indx  i = 0; i < x.ndims(); i++)
    {
      if (x.size(i) != u[i].nRows())
        Genten::error("Genten::gcp_sgd - u and x have different size");
    }

    // Dispatch implementation based on loss function type
    if (algParams.loss_function_type == GCP_LossFunction::Gaussian)
      Impl::gcp_sgd_sa_impl(x, u, GaussianLossFunction(algParams.loss_eps),
                            algParams, numIters, resNorm, perfInfo, out);
    else if (algParams.loss_function_type == GCP_LossFunction::Rayleigh)
      Impl::gcp_sgd_sa_impl(x, u, RayleighLossFunction(algParams.loss_eps),
                            algParams, numIters, resNorm, perfInfo, out);
    else if (algParams.loss_function_type == GCP_LossFunction::Gamma)
      Impl::gcp_sgd_sa_impl(x, u, GammaLossFunction(algParams.loss_eps),
                            algParams, numIters, resNorm, perfInfo, out);
    else if (algParams.loss_function_type == GCP_LossFunction::Bernoulli)
      Impl::gcp_sgd_sa_impl(x, u, BernoulliLossFunction(algParams.loss_eps),
                            algParams, numIters, resNorm, perfInfo, out);
    else if (algParams.loss_function_type == GCP_LossFunction::Poisson)
      Impl::gcp_sgd_sa_impl(x, u, PoissonLossFunction(algParams.loss_eps),
                           algParams, numIters, resNorm, perfInfo, out);
    else
       Genten::error("Genten::gcp_sgd - unknown loss function");
  }

}

#define INST_MACRO(SPACE)                                               \
  template void gcp_sgd_sa<SptensorT<SPACE>,SPACE>(                     \
    SptensorT<SPACE>& x,                                                \
    KtensorT<SPACE>& u,                                                 \
    const AlgParams& algParams,                                         \
    ttb_indx& numIters,                                                 \
    ttb_real& resNorm,                                                  \
    PerfHistory& perfInfo,                                              \
    std::ostream& out);

GENTEN_INST(INST_MACRO)
