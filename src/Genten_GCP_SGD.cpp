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
#include "Genten_GCP_UniformSampler.hpp"
#include "Genten_GCP_StratifiedSampler.hpp"
#include "Genten_GCP_SemiStratifiedSampler.hpp"
#include "Genten_GCP_ValueKernels.hpp"
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_KokkosVector.hpp"
#include "Genten_GCP_SGD_Step.hpp"
#include "Genten_GCP_SGD_Iter.hpp"
#include "Genten_GCP_SGD_Iter_Async.hpp"

#include "Genten_Sptensor.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_RandomMT.hpp"
#include "Genten_MixedFormatOps.hpp"

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

namespace Genten {

  namespace Impl {

    template <typename TensorT, typename ExecSpace, typename LossFunction>
    void gcp_sgd_impl(TensorT& X, KtensorT<ExecSpace>& u0,
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

      // Check for valid option combinations
      if (algParams.async &&
          algParams.sampling_type != GCP_Sampling::SemiStratified)
        Genten::error("Must use semi-stratified sampling with asynchronous solver!");

      const ttb_indx nd = u0.ndims();
      const ttb_indx nc = u0.ncomponents();
      const ProcessorMap* pmap = u0.getProcessorMap();

      // Constants for the algorithm
      const ttb_real tol = algParams.gcp_tol;
      const ttb_real decay = algParams.decay;
      const ttb_real rate = algParams.rate;
      const ttb_indx max_fails = algParams.max_fails;
      const ttb_indx epoch_iters = algParams.epoch_iters;
      const ttb_indx seed = algParams.seed;
      const ttb_indx maxEpochs = algParams.maxiters;
      const ttb_indx printIter = algParams.printitn;
      const bool compute_fit = algParams.compute_fit;

      // Create sampler
      Sampler<ExecSpace,LossFunction> *sampler = nullptr;
      if (algParams.sampling_type == GCP_Sampling::Uniform)
        sampler = new Genten::UniformSampler<ExecSpace,LossFunction>(
          X, algParams);
      else if (algParams.sampling_type == GCP_Sampling::Stratified)
        sampler = new Genten::StratifiedSampler<ExecSpace,LossFunction>(
          X, algParams);
      else if (algParams.sampling_type == GCP_Sampling::SemiStratified)
        sampler = new Genten::SemiStratifiedSampler<ExecSpace,LossFunction>(
          X, algParams, true);
      else
        Genten::error("Genten::gcp_sgd - unknown sampling type");

      if (printIter > 0) {
        const ttb_indx nnz = X.global_nnz();
        const ttb_real tsz = X.global_numel_float();
        const ttb_real nz = tsz - nnz;
        out << "\nGCP-SGD (Generalized CP Tensor Decomposition)\n\n"
            << "Tensor size: ";
        for (ttb_indx i=0; i<nd; ++i) {
          out << X.size(i) << " ";
          if (i<nd-1)
            out << "x ";
        }
        out << "(" << tsz << " total entries)\n"
            << "Sparse tensor: " << nnz << " ("
            << std::setprecision(1) << std::fixed << 100.0*(nnz/tsz)
            << "%) Nonzeros" << " and ("
            << std::setprecision(1) << std::fixed << 100.0*(nz/tsz)
            << "%) Zeros\n"
            << "Rank: " << nc << std::endl
            << "Generalized function type: " << loss_func.name() << std::endl
            << "Optimization method: " << GCP_Step::names[algParams.step_type]
            << std::endl
            << "Max iterations (epochs): " << maxEpochs << std::endl
            << "Iterations per epoch: " << epoch_iters << std::endl
            << "Learning rate / decay / maxfails: "
            << std::setprecision(1) << std::scientific
            << rate << " " << decay << " " << max_fails << std::endl;
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
      SystemTimer timer(num_timers, algParams.timings, pmap);

      // Start timer for total execution time of the algorithm.
      timer.start(timer_sgd);

      // Distribute the initial guess to have weights of one.
      u0.normalize(Genten::NormTwo);
      u0.distribute();

      // Create iterator
      GCP_SGD_Iter<ExecSpace,LossFunction> *itp = nullptr;
      if (algParams.async)
        itp = new GCP_SGD_Iter_Async<ExecSpace,LossFunction>(u0, algParams);
      else
        itp = new GCP_SGD_Iter<ExecSpace,LossFunction>(u0, algParams);
      GCP_SGD_Iter<ExecSpace,LossFunction>& it = *itp;

      // Get vector/Ktensor for current solution (this is a view of the data)
      VectorType u = it.getSolution();
      KtensorT<ExecSpace> ut = u.getKtensor();
      ut.setProcessorMap(pmap);

      // Copy Ktensor for restoring previous solution
      VectorType u_prev = u.clone();
      u_prev.set(u);

      // Create stepper
      GCP_SGD_Step<ExecSpace,LossFunction> *stepper = nullptr;
      if (algParams.step_type == GCP_Step::ADAM)
        stepper = new AdamStep<ExecSpace,LossFunction>(algParams, u);
      else if (algParams.step_type == GCP_Step::AdaGrad)
        stepper = new AdaGradStep<ExecSpace,LossFunction>(algParams, u);
      else if (algParams.step_type == GCP_Step::AMSGrad)
        stepper = new AMSGradStep<ExecSpace,LossFunction>(algParams, u);
      else if (algParams.step_type == GCP_Step::SGDMomentum)
        stepper = new SGDMomentumStep<ExecSpace,LossFunction>(algParams, u);
      else {
        stepper = new SGDStep<ExecSpace,LossFunction>();
        std::cout << "Using SGD\n";
      }

      // Initialize sampler (sorting, hashing, ...)
      timer.start(timer_sort);
      RandomMT rng(seed);
      Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(rng.genrnd_int32());
      sampler->initialize(rand_pool, out);
      timer.stop(timer_sort);

      // Sample X for f-estimate
      SptensorT<ExecSpace> X_val;
      ArrayT<ExecSpace> w_val;
      timer.start(timer_sample_f);
      sampler->sampleTensor(false, ut, loss_func, X_val, w_val);
      timer.stop(timer_sample_f);

      ttb_real x_norm = X.global_norm();
      auto u_norm = std::sqrt(ut.normFsq());

      // Objective estimates
      ttb_real fit = 0.0;
      timer.start(timer_fest);
      fest = Impl::gcp_value(X_val, ut, w_val, loss_func);
      auto fitter = [&]{
        const auto x_norm2 = x_norm * x_norm;
        const auto u_norm2 = ut.normFsq();
        const auto dot = innerprod(X, ut);
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

      if (printIter > 0) {
        out << "Begin main loop\n"
            << "Initial f-est: "
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
      }

      struct Annealer {
        ttb_real last_returned = 0.0;
        ttb_real last_good = 0.0;
        ttb_real min_lr;
        ttb_real max_lr = 0.0;
        ttb_real warm_up_min;
        ttb_real warm_up_max;
        ttb_real warmup_scale;
        int epoch_internal = 0;
        int cycle_size = 100;
        int warmup_size = 50;
        bool do_warmup = true;

        Annealer(AlgParams const& algParams):
          min_lr(algParams.anneal_min_lr),
          max_lr(algParams.anneal_max_lr),
          warm_up_max(10 * algParams.anneal_max_lr)
        {
            warm_up_min = 0.1 * min_lr;
            const auto term = std::log(warm_up_max/warm_up_min)/warmup_size;
            warmup_scale = std::exp(term);
        }

        ttb_real operator()(int epoch){
           if(do_warmup){
             last_returned = warm_up_min * std::pow(warmup_scale, epoch_internal);
           } else {
              last_returned = min_lr + 0.5 * (max_lr - min_lr) * (1 +
                  std::cos(double(epoch_internal + cycle_size)/cycle_size * M_PI)); 
           }
           ++epoch_internal;
           if(do_warmup && epoch_internal == warmup_size){
             epoch_internal = 0; // Start over
             do_warmup = false;
           }
           return last_returned;
        }

        void failed(){
          if(do_warmup){
            do_warmup = false;
            max_lr = 0.5 * last_good;
          } else {
            min_lr *= 0.1;
            max_lr *= 0.1;
          }
          epoch_internal = 0;
        }

        void success(){
          last_good = last_returned;
        }
      } annealer(algParams);

      // SGD epoch loop
      ttb_real nuc = 1.0;
      ttb_indx nfails = 0;
      ttb_indx total_iters = 0;

      for (numEpochs=0; numEpochs<maxEpochs; ++numEpochs) {
        // Gradient step size
        if(algParams.anneal){
          stepper->setStep(annealer(numEpochs));
        } else {
          stepper->setStep(nuc*rate);
        }

        // Epoch iterations
        it.run(X, loss_func, *sampler, *stepper, total_iters);

        // compute objective estimate
        timer.start(timer_fest);
        fest = Impl::gcp_value(X_val, ut, w_val, loss_func);
        if (compute_fit) {
          fit = fitter();
        }
        timer.stop(timer_fest);

        // check convergence
        bool failed_epoch = fest > fest_prev || std::isnan(fest);

        if (failed_epoch)
          ++nfails;

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
          if (failed_epoch){
            out << ", nfails = " << nfails;
          }
          out << std::endl;
        }

        {
          perfInfo.addEmpty();
          auto& p = perfInfo.lastEntry();
          p.iteration = numEpochs+1;
          p.residual = fest;
          if (compute_fit)
            p.fit = fit;
        }

        if (failed_epoch) {
          nuc *= decay;

          // restart from last epoch
          u.set(u_prev);
          fest = fest_prev;
          fit = fit_prev;
          stepper->setFailed();
          annealer.failed();
        }
        else {
          // update previous data
          u_prev.set(u);
          fest_prev = fest;
          fit_prev = fit;
          stepper->setPassed();
          annealer.success();
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

      // Normalize Ktensor u
      u0.normalize(Genten::NormTwo);
      u0.arrange();

      delete stepper;
      delete sampler;
      delete itp;
    }

  }


  template<typename TensorT, typename ExecSpace>
  void gcp_sgd(TensorT& x, KtensorT<ExecSpace>& u,
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
      Impl::gcp_sgd_impl(x, u, GaussianLossFunction(algParams.loss_eps),
                         algParams, numIters, resNorm, perfInfo, out);
    else if (algParams.loss_function_type == GCP_LossFunction::Rayleigh)
      Impl::gcp_sgd_impl(x, u, RayleighLossFunction(algParams.loss_eps),
                         algParams, numIters, resNorm, perfInfo, out);
    else if (algParams.loss_function_type == GCP_LossFunction::Gamma)
      Impl::gcp_sgd_impl(x, u, GammaLossFunction(algParams.loss_eps),
                         algParams, numIters, resNorm, perfInfo, out);
    else if (algParams.loss_function_type == GCP_LossFunction::Bernoulli)
      Impl::gcp_sgd_impl(x, u, BernoulliLossFunction(algParams.loss_eps),
                         algParams, numIters, resNorm, perfInfo, out);
    else if (algParams.loss_function_type == GCP_LossFunction::Poisson)
      Impl::gcp_sgd_impl(x, u, PoissonLossFunction(algParams.loss_eps),
                         algParams, numIters, resNorm, perfInfo, out);
    else
       Genten::error("Genten::gcp_sgd - unknown loss function");
  }

}

#define INST_MACRO(SPACE)                                               \
  template void gcp_sgd<SptensorT<SPACE>,SPACE>(                        \
    SptensorT<SPACE>& x,                                                \
    KtensorT<SPACE>& u,                                                 \
    const AlgParams& algParams,                                         \
    ttb_indx& numIters,                                                 \
    ttb_real& resNorm,                                                  \
    PerfHistory& perfInfo,                                              \
    std::ostream& out);

GENTEN_INST(INST_MACRO)
