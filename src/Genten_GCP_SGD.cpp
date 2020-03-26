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
#include "Genten_GCP_KokkosVector.hpp"

#include "Genten_Sptensor.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_RandomMT.hpp"
#include "Genten_MixedFormatOps.hpp"

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

namespace Genten {

  namespace Impl {

    template <typename ExecSpace, typename LossFunction>
    class GCP_SGD_Step {
    public:
      typedef GCP::KokkosVector<ExecSpace> VectorType;

      GCP_SGD_Step() = default;

      virtual ~GCP_SGD_Step() {}

      virtual void setStep(const ttb_real step) = 0;

      virtual ttb_real getStep() const = 0;

      virtual void update() = 0;

      virtual void reset() = 0;

      virtual void setPassed() = 0;

      virtual void setFailed() = 0;

      virtual void eval(const VectorType& g, VectorType& u) const = 0;

    };

    template <typename ExecSpace, typename LossFunction>
    class SGDStep : public GCP_SGD_Step<ExecSpace,LossFunction> {
    public:
      typedef GCP_SGD_Step<ExecSpace,LossFunction> BaseType;
      typedef typename BaseType::VectorType VectorType;

      SGDStep() {}

      virtual ~SGDStep() {}

      virtual void setStep(const ttb_real s) { step = s; }

      virtual ttb_real getStep() const { return step; }

      virtual void update() {}

      virtual void reset() {}

      virtual void setPassed() {}

      virtual void setFailed() {}

      virtual void eval(const VectorType& g, VectorType& u) const
      {
        constexpr bool has_bounds = (LossFunction::has_lower_bound() ||
                                     LossFunction::has_upper_bound());
        constexpr ttb_real lb = LossFunction::lower_bound();
        constexpr ttb_real ub = LossFunction::upper_bound();
        const ttb_real sgd_step = step;
        auto uv = u.getView();
        auto gv = g.getView();
        u.apply_func(KOKKOS_LAMBDA(const ttb_indx i)
        {
          ttb_real uu = uv(i);
          uu -= sgd_step*gv(i);
          if (has_bounds)
            uu = uu < lb ? lb : (uu > ub ? ub : uu);
          uv(i) = uu;
        });
      }

    protected:
      ttb_real step;

    };

    template <typename ExecSpace, typename LossFunction>
    class AdamStep : public GCP_SGD_Step<ExecSpace,LossFunction> {
    public:
      typedef GCP_SGD_Step<ExecSpace,LossFunction> BaseType;
      typedef typename BaseType::VectorType VectorType;

      AdamStep(const AlgParams& algParams, const VectorType& u) :
        epoch_iters(algParams.epoch_iters),
        step(0.0),
        beta1(algParams.adam_beta1),
        beta2(algParams.adam_beta2),
        eps(algParams.adam_eps),
        beta1t(1.0),
        beta2t(1.0),
        adam_step(0.0),
        m(u.clone()),
        v(u.clone()),
        m_prev(u.clone()),
        v_prev(u.clone())
      {
        m.zero();
        v.zero();
        m_prev.zero();
        v_prev.zero();
      }

      virtual ~AdamStep() {}

      virtual void setStep(const ttb_real s) { step = s; }

      virtual ttb_real getStep() const { return step; }

      virtual void update()
      {
        beta1t = beta1 * beta1t;
        beta2t = beta2 * beta2t;
        adam_step = step*std::sqrt(1.0-beta2t) / (1.0-beta1t);
      }

      virtual void reset()
      {
        beta1t = 1.0;
        beta2t = 1.0;
        m.zero();
        v.zero();
        m_prev.zero();
        v_prev.zero();
      }

      virtual void setPassed()
      {
        m_prev.set(m);
        v_prev.set(v);
      }

      virtual void setFailed()
      {
        m.set(m_prev);
        v.set(v_prev);
        beta1t /= std::pow(beta1, epoch_iters);
        beta2t /= std::pow(beta2, epoch_iters);
      }

      virtual void eval(const VectorType& g, VectorType& u) const
      {
        using std::sqrt;
        constexpr bool has_bounds = (LossFunction::has_lower_bound() ||
                                     LossFunction::has_upper_bound());
        constexpr ttb_real lb = LossFunction::lower_bound();
        constexpr ttb_real ub = LossFunction::upper_bound();
        const ttb_real adam_step_ = adam_step;
        const ttb_real eps_ = eps;
        const ttb_real beta1_ = beta1;
        const ttb_real beta2_ = beta2;
        auto uv = u.getView();
        auto gv = g.getView();
        auto mv = m.getView();
        auto vv = v.getView();
        u.apply_func(KOKKOS_LAMBDA(const ttb_indx i)
        {
          mv(i) = beta1_*mv(i) + (1.0-beta1_)*gv(i);
          vv(i) = beta2_*vv(i) + (1.0-beta2_)*gv(i)*gv(i);
          ttb_real uu = uv(i);
          uu -= adam_step_*mv(i)/sqrt(vv(i)+eps_);
          if (has_bounds)
            uu = uu < lb ? lb : (uu > ub ? ub : uu);
          uv(i) = uu;
        });
      }

    protected:
      ttb_indx epoch_iters;
      ttb_real step;
      ttb_real beta1;
      ttb_real beta2;
      ttb_real eps;
      ttb_real beta1t;
      ttb_real beta2t;
      ttb_real adam_step;

      VectorType m;
      VectorType v;
      VectorType m_prev;
      VectorType v_prev;
    };

    template <typename ExecSpace>
    struct GCP_SGD_Iter {
      typedef GCP::KokkosVector<ExecSpace> VectorType;
      VectorType u;
      VectorType g;
      KtensorT<ExecSpace> ut;
      KtensorT<ExecSpace> gt;

      SptensorT<ExecSpace> X_grad;
      ArrayT<ExecSpace> w_grad;

      int total_iters;

      int timer_sample_g;
      int timer_grad;
      int timer_grad_nzs;
      int timer_grad_zs;
      int timer_grad_init;
      int timer_step;
      int timer_sample_g_z_nz;
      int timer_sample_g_perm;
      SystemTimer timer;

      template <typename TensorT, typename LossFunction>
      void run(TensorT& X,
               const LossFunction& loss_func,
               const AlgParams& algParams,
               Sampler<ExecSpace,LossFunction>& sampler,
               GCP_SGD_Step<ExecSpace,LossFunction>& stepper)
      {
        for (ttb_indx iter=0; iter<algParams.epoch_iters; ++iter) {

          // Update stepper for next iteration
          stepper.update();

          // sample for gradient
          if (!algParams.fuse) {
            timer.start(timer_sample_g);
            timer.start(timer_sample_g_z_nz);
            sampler.sampleTensor(true, ut, loss_func,  X_grad, w_grad);
            timer.stop(timer_sample_g_z_nz);
            timer.start(timer_sample_g_perm);
            if (algParams.mttkrp_method == MTTKRP_Method::Perm &&
                algParams.mttkrp_all_method == MTTKRP_All_Method::Iterated)
              X_grad.createPermutation();
            timer.stop(timer_sample_g_perm);
            timer.stop(timer_sample_g);
          }

          for (ttb_indx giter=0; giter<algParams.frozen_iters; ++giter) {
            ++total_iters;

            // compute gradient
            timer.start(timer_grad);
            if (algParams.fuse) {
              timer.start(timer_grad_init);
              g.zero(); // algorithm does not use weights
              timer.stop(timer_grad_init);
              sampler.fusedGradient(ut, loss_func, gt, timer, timer_grad_nzs,
                                     timer_grad_zs);
            }
            else {
              gt.weights() = 1.0; // gt is zeroed in mttkrp
              mttkrp_all(X_grad, ut, gt, algParams);
            }
            timer.stop(timer_grad);

            // take step and clip for bounds
            timer.start(timer_step);
            stepper.eval(g, u);
            timer.stop(timer_step);
          }
        }
      }
    };

    template <typename TensorT, typename ExecSpace, typename LossFunction>
    void gcp_sgd_impl(TensorT& X, KtensorT<ExecSpace>& u0,
                      const LossFunction& loss_func,
                      const AlgParams& algParams,
                      ttb_indx& numEpochs,
                      ttb_real& fest,
                      std::ostream& out)
    {
      typedef GCP::KokkosVector<ExecSpace> VectorType;
      typedef typename VectorType::view_type view_type;
      using std::sqrt;
      using std::pow;

      GCP_SGD_Iter<ExecSpace> it;

      const ttb_indx nd = u0.ndims();
      const ttb_indx nc = u0.ncomponents();

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

      // ADAM parameters
      const bool use_adam = algParams.use_adam;

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
          X, algParams);
      else
        Genten::error("Genten::gcp_sgd - unknown sampling type");

      if (printIter > 0) {
        const ttb_indx nnz = X.nnz();
        const ttb_real tsz = X.numel_float();
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
            << "Optimization method: " << (use_adam ? "adam\n" : "sgd\n")
            << "Max iterations (epochs): " << maxEpochs << std::endl
            << "Iterations per epoch: " << epoch_iters << std::endl
            << "Learning rate / decay / maxfails: "
            << std::setprecision(1) << std::scientific
            << rate << " " << decay << " " << max_fails << std::endl;
        sampler->print(out);
        out << "Gradient method: ";
        if (algParams.fuse)
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
      it.timer_sample_g = num_timers++;
      it.timer_grad = num_timers++;
      it.timer_grad_nzs = num_timers++;
      it.timer_grad_zs = num_timers++;
      it.timer_grad_init = num_timers++;
      it.timer_step = num_timers++;
      it.timer_sample_g_z_nz = num_timers++;
      it.timer_sample_g_perm = num_timers++;
      it.timer.init(num_timers, algParams.timings);

      // Start timer for total execution time of the algorithm.
      it.timer.start(timer_sgd);

      // Distribute the initial guess to have weights of one.
      u0.normalize(Genten::NormTwo);
      u0.distribute();

      // Ktensor-vector for solution
      it.u = VectorType(u0);
      it.u.copyFromKtensor(u0);
      it.ut = it.u.getKtensor();

      // Gradient Ktensor
      it.g = it.u.clone();
      it.gt = it.g.getKtensor();

      // Copy Ktensor for restoring previous solution
      VectorType u_prev = it.u.clone();
      u_prev.set(it.u);

      // Create stepper
      GCP_SGD_Step<ExecSpace,LossFunction> *stepper = nullptr;
      if (use_adam)
        stepper = new AdamStep<ExecSpace,LossFunction>(algParams, it.u);
      else
        stepper = new SGDStep<ExecSpace,LossFunction>();

      // Initialize sampler (sorting, hashing, ...)
      it.timer.start(timer_sort);
      RandomMT rng(seed);
      Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(rng.genrnd_int32());
      sampler->initialize(rand_pool, out);
      it.timer.stop(timer_sort);

      // Sample X for f-estimate
      SptensorT<ExecSpace> X_val;
      ArrayT<ExecSpace> w_val;
      it.timer.start(timer_sample_f);
      sampler->sampleTensor(false, it.ut, loss_func, X_val, w_val);
      it.timer.stop(timer_sample_f);

      // Objective estimates
      ttb_real fit = 0.0;
      ttb_real x_norm = 0.0;
      it.timer.start(timer_fest);
      fest = Impl::gcp_value(X_val, it.ut, w_val, loss_func);
      if (compute_fit) {
        x_norm = X.norm();
        ttb_real u_norm = it.u.normFsq();
        ttb_real dot = innerprod(X, it.ut);
        fit = 1.0 - sqrt(x_norm*x_norm + u_norm*u_norm - 2.0*dot) / x_norm;
      }
      it.timer.stop(timer_fest);
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

      // SGD epoch loop
      ttb_real nuc = 1.0;
      ttb_indx nfails = 0;
      it.total_iters = 0;
      for (numEpochs=0; numEpochs<maxEpochs; ++numEpochs) {
        // Gradient step size
        stepper->setStep(nuc*rate);

        // Epoch iterations
        it.run(X, loss_func, algParams, *sampler, *stepper);

        // compute objective estimate
        it.timer.start(timer_fest);
        fest = Impl::gcp_value(X_val, it.ut, w_val, loss_func);
        if (compute_fit) {
          ttb_real u_norm = it.u.normFsq();
          ttb_real dot = innerprod(X, it.ut);
          fit = 1.0 - sqrt(x_norm*x_norm + u_norm*u_norm - 2.0*dot) / x_norm;
        }
        it.timer.stop(timer_fest);

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
              << stepper->getStep();
          out << ", time = "
              << std::setw(8) << std::setprecision(2) << std::scientific
              << it.timer.getTotalTime(timer_sgd) << " sec";
          if (failed_epoch)
            out << ", nfails = " << nfails
                << " (resetting to solution from last epoch)";
          out << std::endl;
        }

        if (failed_epoch) {
          nuc *= decay;

          // restart from last epoch
          it.u.set(u_prev);
          fest = fest_prev;
          fit = fit_prev;
          stepper->setFailed();
        }
        else {
          // update previous data
          u_prev.set(it.u);
          fest_prev = fest;
          fit_prev = fit;
          stepper->setPassed();
        }

        if (nfails > max_fails || fest < tol)
          break;
      }
      it.timer.stop(timer_sgd);

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
            << "GCP-SGD completed " << it.total_iters << " iterations in "
            << std::setw(8) << std::setprecision(2) << std::scientific
            << it.timer.getTotalTime(timer_sgd) << " seconds" << std::endl;
        if (algParams.timings) {
          out << "\tsort/hash: " << it.timer.getTotalTime(timer_sort)
              << " seconds\n"
              << "\tsample-f:  " << it.timer.getTotalTime(timer_sample_f)
              << " seconds\n";
          if (!algParams.fuse) {
            out << "\tsample-g:  "
                << it.timer.getTotalTime(it.timer_sample_g)
                << " seconds\n"
                << "\t\tzs/nzs:   "
                << it.timer.getTotalTime(it.timer_sample_g_z_nz)
                << " seconds\n";
            if (algParams.mttkrp_method == MTTKRP_Method::Perm &&
                algParams.mttkrp_all_method == MTTKRP_All_Method::Iterated) {
              out << "\t\tperm:     "
                  << it.timer.getTotalTime(it.timer_sample_g_perm)
                  << " seconds\n";
            }
          }
          out << "\tf-est:     " << it.timer.getTotalTime(timer_fest)
              << " seconds\n"
              << "\tgradient:  " << it.timer.getTotalTime(it.timer_grad)
              << " seconds\n";
          if (algParams.fuse) {
            out << "\t\tinit:    " << it.timer.getTotalTime(it.timer_grad_init)
                << " seconds\n"
                << "\t\tnzs:     " << it.timer.getTotalTime(it.timer_grad_nzs)
                << " seconds\n"
                << "\t\tzs:      " << it.timer.getTotalTime(it.timer_grad_zs)
                << " seconds\n";
          }
          out << "\tstep/clip: " << it.timer.getTotalTime(it.timer_step)
              << " seconds\n";
        }
      }

      it.u.copyToKtensor(u0);

      // Normalize Ktensor u
      u0.normalize(Genten::NormTwo);
      u0.arrange();

      delete sampler;
    }

  }


  template<typename TensorT, typename ExecSpace>
  void gcp_sgd(TensorT& x, KtensorT<ExecSpace>& u,
               const AlgParams& algParams,
               ttb_indx& numIters,
               ttb_real& resNorm,
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
                         algParams, numIters, resNorm, out);
    else if (algParams.loss_function_type == GCP_LossFunction::Rayleigh)
      Impl::gcp_sgd_impl(x, u, RayleighLossFunction(algParams.loss_eps),
                         algParams, numIters, resNorm, out);
    else if (algParams.loss_function_type == GCP_LossFunction::Gamma)
      Impl::gcp_sgd_impl(x, u, GammaLossFunction(algParams.loss_eps),
                         algParams, numIters, resNorm, out);
    else if (algParams.loss_function_type == GCP_LossFunction::Bernoulli)
      Impl::gcp_sgd_impl(x, u, BernoulliLossFunction(algParams.loss_eps),
                         algParams, numIters, resNorm, out);
    else if (algParams.loss_function_type == GCP_LossFunction::Poisson)
      Impl::gcp_sgd_impl(x, u, PoissonLossFunction(algParams.loss_eps),
                         algParams, numIters, resNorm, out);
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
    std::ostream& out);

GENTEN_INST(INST_MACRO)
