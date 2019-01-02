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


/*!
  @file Genten_GCP.cpp
  @brief GCP algorithm, in template form to allow different data tensor types.
*/

#include <iomanip>
#include <algorithm>
#include <cmath>

#include "Genten_GCP_SGD.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_RandomMT.hpp"
#include "Genten_GCP_ValueKernels.hpp"
#include "Genten_MixedFormatOps.hpp"

#include "Kokkos_Random.hpp"

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

// To do:
//   * add ETI for sampling, value kernels to improve compile times
//   * sort out what is going wrong with non-Gaussian
//      * how do we handle contraints?
//   * investigate HogWild for parallelism over iterations
//   * The step is not an insignificant cost.  Maybe do something like the
//     Kokkos implementation of the ROL vector

namespace Genten {

  namespace Impl {

    template <typename ExecSpace, typename LossFunction>
    void stratified_sample_tensor(const SptensorT<ExecSpace>& X,
                                  const ttb_indx num_samples_nonzeros,
                                  const ttb_indx num_samples_zeros,
                                  const ttb_real weight_nonzeros,
                                  const ttb_real weight_zeros,
                                  const KtensorT<ExecSpace>& u,
                                  const LossFunction& loss_func,
                                  const bool compute_gradient,
                                  SptensorT<ExecSpace>& Y,
                                  ArrayT<ExecSpace>& w,
                                  RandomMT& rng,
                                  const AlgParams& algParams)
    {
      const ttb_indx nnz = X.nnz();
      const ttb_indx nd = X.ndims();
      const ttb_indx tsz = X.numel();

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples_nonzeros + num_samples_zeros;
      if (Y.nnz() < total_samples) {
        Y = SptensorT<ExecSpace>(X.size(), total_samples);
        w = ArrayT<ExecSpace>(total_samples);
      }

#if 1
      // Parallel sampling on the device
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      RandomPool rand_pool(rng.genrnd_int32());
      const ttb_indx nloops = algParams.rng_iters;

      // Generate samples of nonzeros
      const ttb_indx N_nonzeros = (num_samples_nonzeros+nloops-1)/nloops;
      Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,N_nonzeros),
                           KOKKOS_LAMBDA(const ttb_indx k)
      {
        generator_type gen = rand_pool.get_state();
        for (ttb_indx l=0; l<nloops; ++l) {
          const ttb_indx i = k*nloops+l;
          if (i<num_samples_nonzeros) {
            const ttb_indx idx = Rand::draw(gen,0,nnz);
            const auto& ind = X.getSubscripts(idx);
            if (compute_gradient) {
              const ttb_real m_val = compute_Ktensor_value(u, ind);
              Y.value(i) =
                weight_nonzeros * loss_func.deriv(X.value(idx), m_val);
            }
            else {
              Y.value(i) = X.value(idx);
              w[i] = weight_nonzeros;
            }
            for (ttb_indx j=0; j<nd; ++j)
              Y.subscript(i,j) = ind[j];
          }
        }
        rand_pool.free_state(gen);
      }, "Genten::GCP_SGD::Uniform_Sample_Nonzeros");

      // Generate samples of zeros
      typedef Kokkos::View< ttb_indx*, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;
      typedef Kokkos::TeamPolicy<ExecSpace> Policy;
      typedef typename Policy::member_type TeamMember;
      const bool is_cuda = is_cuda_space<ExecSpace>::value;
      const ttb_indx vector_size = 1;
      const ttb_indx team_size = is_cuda ? 128/vector_size : 1;
      const ttb_indx loops_per_team = nloops*team_size;
      const ttb_indx N_zeros =
        (num_samples_zeros+loops_per_team-1)/loops_per_team;
      const size_t bytes = TmpScratchSpace::shmem_size(nd);
      Policy policy(N_zeros,team_size,vector_size);
      Kokkos::parallel_for(policy.set_scratch_size(0,Kokkos::PerThread(bytes)),
                           KOKKOS_LAMBDA(const TeamMember& team)
      {
        const ttb_indx k = team.league_rank()*team_size + team.team_rank();
        TmpScratchSpace ind(team.thread_scratch(0), nd);

        generator_type gen = rand_pool.get_state();
        for (ttb_indx l=0; l<nloops; ++l) {
          const ttb_indx i = k*nloops+l;
          if (i<num_samples_zeros) {

            // Keep generating samples until we get one not in the tensor
            bool found = true;
            while (found) {
              // Generate index
              for (ttb_indx j=0; j<nd; ++j)
                ind[j] = Rand::draw(gen,0,X.size(j));

              // Search for index
              found = (X.index(ind) < nnz);

              // If not found, add it
              if (!found) {
                const ttb_indx idx = num_samples_nonzeros + i;
                for (ttb_indx j=0; j<nd; ++j)
                  Y.subscript(idx,j) = ind[j];
                if (compute_gradient) {
                  const ttb_real m_val = compute_Ktensor_value(u, ind);
                  Y.value(idx) =
                    weight_zeros * loss_func.deriv(ttb_real(0.0), m_val);
                }
                else {
                  Y.value(idx) = 0.0;
                  w[idx] = weight_zeros;
                }
              }
            }

          }
        }
        rand_pool.free_state(gen);
      }, "Genten::GCP_SGD::Uniform_Sample_Zeros");
#else
      // Serial sampling on the host
      typedef typename SptensorT<ExecSpace>::HostMirror Sptensor_host_type;
      typedef typename Sptensor_host_type::exec_space host_exec_space;
      typedef typename ArrayT<ExecSpace>::HostMirror Array_host_type;
      typedef typename KtensorT<ExecSpace>::HostMirror Ktensor_host_type;

      // Copy to host
      Sptensor_host_type X_host = create_mirror_view(host_exec_space(), X);
      Sptensor_host_type Y_host = create_mirror_view(host_exec_space(), Y);
      Array_host_type w_host;
      Ktensor_host_type u_host;
      if (compute_gradient) {
        w_host = create_mirror_view(host_exec_space(), w);
        u_host = create_mirror_view(host_exec_space(), u);
        deep_copy(u_host, u);
      }
      deep_copy(X_host, X);

      // Geneate samples of nonzeros
      for (ttb_indx i=0; i<num_samples_nonzeros; ++i) {
        const ttb_indx idx = ttb_indx(rng.genrnd_double() * nnz);
        const auto& ind = X_host.getSubscripts(idx);
        if (compute_gradient) {
          const ttb_real m_val = compute_Ktensor_value(u_host, ind);
          Y_host.value(i) =
            weight_nonzeros * loss_func.deriv(X.value(idx), m_val);
        }
        else {
          Y_host.value(i) = X_host.value(idx);
          w_host[i] = weight_nonzeros;
        }
        for (ttb_indx j=0; j<nd; ++j)
          Y_host.subscript(i,j) = ind[j];
      }

      // Generate samples of zeros
      IndxArrayT<host_exec_space> ind(nd);
      ttb_indx i=0;
      while (i<num_samples_zeros) {

        // Generate index
        for (ttb_indx j=0; j<nd; ++j)
          ind[j] = ttb_indx(rng.genrnd_double() * X_host.size(j));

        // Search for index
        const bool found = (X_host.index(ind) < nnz);

        // If not found, add it
        if (!found) {
          const ttb_indx idx = num_samples_nonzeros + i;
          for (ttb_indx j=0; j<nd; ++j)
            Y_host.subscript(idx,j) = ind[j];
          if (compute_gradient) {
            const ttb_real m_val = compute_Ktensor_value(u_host, ind);
            Y_host.value(idx) =
              weight_zeros * loss_func.deriv(ttb_real(0.0), m_val);
          }
          else {
            Y_host.value(idx) = 0.0;
            w_host[idx] = weight_zeros;
          }
          ++i;
        }

      }

      // Copy to device
      deep_copy(Y, Y_host);
      if (!compute_gradient)
        deep_copy(w, w_host);
#endif

      if (algParams.mttkrp_method == MTTKRP_Method::Perm) {
        Y.createPermutation();
      }
    }

    template<typename TensorT, typename ExecSpace, typename LossFunction>
    void gcp_sgd_impl(TensorT& X, KtensorT<ExecSpace>& u,
                      const LossFunction& loss_func,
                      const AlgParams& algParams,
                      ttb_indx& numEpochs,
                      ttb_real& fest,
                      std::ostream& out)
    {
      typedef FacMatrixT<ExecSpace> fac_matrix_type;
      typedef typename fac_matrix_type::view_type view_type;
      using std::sqrt;
      using std::pow;

      const ttb_indx nd = u.ndims();
      const ttb_indx nc = u.ncomponents();

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
      ttb_indx num_samples_nonzeros_value =
        algParams.num_samples_nonzeros_value;
      ttb_indx num_samples_zeros_value =
        algParams.num_samples_zeros_value;
      ttb_indx num_samples_nonzeros_grad =
        algParams.num_samples_nonzeros_grad;
      ttb_indx num_samples_zeros_grad =
        algParams.num_samples_zeros_grad;
      ttb_real weight_nonzeros_value = algParams.w_f_nz;
      ttb_real weight_zeros_value = algParams.w_f_z;
      ttb_real weight_nonzeros_grad = algParams.w_g_nz;
      ttb_real weight_zeros_grad = algParams.w_g_z;

      // ADAM parameters
      const bool use_adam = algParams.use_adam;
      const ttb_real beta1 = algParams.adam_beta1;
      const ttb_real beta2 = algParams.adam_beta2;
      const ttb_real eps = algParams.adam_eps;

      // Compute number of samples if necessary
      const ttb_indx nnz = X.nnz();
      const ttb_indx tsz = X.numel();
      const ttb_indx nz = tsz - nnz;
      const ttb_indx ftmp = std::max((nnz+99)/100,ttb_indx(100000));
      const ttb_indx gtmp = std::max((3*nnz+maxEpochs-1)/maxEpochs,
                                     ttb_indx(1000));
      if (num_samples_nonzeros_value == 0)
        num_samples_nonzeros_value = std::min(ftmp, nnz);
      if (num_samples_zeros_value == 0)
        num_samples_zeros_value = std::min(num_samples_nonzeros_value, nz);
      if (num_samples_nonzeros_grad == 0)
        num_samples_nonzeros_grad = std::min(gtmp, nnz);
      if (num_samples_zeros_grad == 0)
        num_samples_zeros_grad = std::min(num_samples_nonzeros_grad, nz);

      // Compute weights if necessary
      if (weight_nonzeros_value < 0.0)
        weight_nonzeros_value =
          ttb_real(nnz) / ttb_real(num_samples_nonzeros_value);
      if (weight_zeros_value < 0.0)
        weight_zeros_value =
          ttb_real(tsz-nnz) / ttb_real(num_samples_zeros_value);
      if (weight_nonzeros_grad < 0.0)
        weight_nonzeros_grad =
          ttb_real(nnz) / ttb_real(num_samples_nonzeros_grad);
      if (weight_zeros_grad < 0.0)
        weight_zeros_grad =
          ttb_real(tsz-nnz) / ttb_real(num_samples_zeros_grad);

      if (printIter > 0) {
        out << "Starting GCP-SGD" << std::endl
            << "\tNum samples f: " << num_samples_nonzeros_value <<" nonzeros, "
            << num_samples_zeros_value << " zeros" << std::endl
            << "\tNum samples g: " << num_samples_nonzeros_grad << " nonzeros, "
            << num_samples_zeros_grad << " zeros" << std::endl
            << "\tWeights f: " << weight_nonzeros_value << " nonzeros, "
            << weight_zeros_value << " zeros" << std::endl
            << "\tWeights g: " << weight_nonzeros_grad << " nonzeros, "
            << weight_zeros_grad << " zeros" << std::endl;
      }

      // Timers
      const int timer_sgd = 0;
      const int timer_sort = 1;
      const int timer_sample_f = 2;
      const int timer_sample_g = 3;
      const int timer_fest = 4;
      const int timer_grad = 5;
      const int timer_step = 6;
      SystemTimer timer(7);

      // Start timer for total execution time of the algorithm.
      timer.start(timer_sgd);

      // Distribute the initial guess to have weights of one.
      u.distribute(0);

      // Gradient Ktensor
      KtensorT<ExecSpace> g(nc, nd);
      for (ttb_indx m=0; m<nd; ++m)
        g.set_factor(m, fac_matrix_type(u[m].nRows(), nc));

      // Copy Ktensor for restoring previous solution
      KtensorT<ExecSpace> u_prev(nc, nd);
      for (ttb_indx m=0; m<nd; ++m)
        u_prev.set_factor(m, fac_matrix_type(u[m].nRows(), nc));
      deep_copy(u_prev, u);

      // ADAM first (m) and second (v) moment vectors
      KtensorT<ExecSpace> adam_m, adam_v, adam_m_prev, adam_v_prev;
      if (use_adam) {
        adam_m = KtensorT<ExecSpace>(nc, nd);
        adam_v = KtensorT<ExecSpace>(nc, nd);
        adam_m_prev = KtensorT<ExecSpace>(nc, nd);
        adam_v_prev = KtensorT<ExecSpace>(nc, nd);
        for (ttb_indx m=0; m<nd; ++m) {
          adam_m.set_factor(m, fac_matrix_type(u[m].nRows(), nc));
          adam_v.set_factor(m, fac_matrix_type(u[m].nRows(), nc));
          adam_m_prev.set_factor(m, fac_matrix_type(u[m].nRows(), nc));
          adam_v_prev.set_factor(m, fac_matrix_type(u[m].nRows(), nc));
        }
      }

      // Sort tensor if necessary
      if (!X.isSorted()) {
        if (printIter > 0)
          out << "Sorting tensor for faster sampling...";
        timer.start(timer_sort);
        X.sort();
        timer.stop(timer_sort);
        if (printIter > 0)
          out << timer.getTotalTime(timer_sort) << " seconds" << std::endl;
      }

      // Sample X for f-estimate
      SptensorT<ExecSpace> X_val, X_grad, Y;
      ArrayT<ExecSpace> w_val, w_grad;
      RandomMT rng(seed);
      timer.start(timer_sample_f);
      Impl::stratified_sample_tensor(
        X, num_samples_nonzeros_value, num_samples_zeros_value,
        weight_nonzeros_value, weight_zeros_value,
        u, loss_func, false,
        X_val, w_val, rng, algParams);
      timer.stop(timer_sample_f);

      // Objective estimates
      ttb_real fit = 0.0;
      ttb_real x_norm2 = 0.0;
      ArrayT<ExecSpace> w_fit;
      timer.start(timer_fest);
      fest = Impl::gcp_value(X_val, u, w_val, loss_func);
      if (compute_fit) {
        w_fit = ArrayT<ExecSpace>(nnz, 1.0);
        fit = Impl::gcp_value(X, u, w_fit, loss_func);
        x_norm2 = X.norm(); x_norm2 *= x_norm2;
        fit = 1.0 - fit / x_norm2;
      }
      timer.stop(timer_fest);
      ttb_real fest_prev = fest;
      ttb_real fit_prev = fit;

      if (printIter > 0) {
        out << "Initial f-est: "
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

          // sample for gradient
          timer.start(timer_sample_g);
          Impl::stratified_sample_tensor(
            X, num_samples_nonzeros_grad, num_samples_zeros_grad,
            weight_nonzeros_grad, weight_zeros_grad,
            u, loss_func, true,
            X_grad, w_grad, rng, algParams);
          timer.stop(timer_sample_g);

          for (ttb_indx giter=0; giter<frozen_iters; ++giter) {
             ++total_iters;

            // compute gradient
            timer.start(timer_grad);
            g.weights() = 1.0;
            for (unsigned m=0; m<nd; ++m)
              mttkrp(X_grad, u, m, g[m], algParams);
            timer.stop(timer_grad);

            // take step
            timer.start(timer_step);
            for (ttb_indx m=0; m<nd; ++m) {
              view_type uv = u[m].view();
              view_type gv = g[m].view();
              if (use_adam) {
                view_type mv = adam_m[m].view();
                view_type vv = adam_v[m].view();
                u[m].apply_func(KOKKOS_LAMBDA(const ttb_indx j,const ttb_indx i)
                {
                  mv(i,j) = beta1*mv(i,j) + (1.0-beta1)*gv(i,j);
                  vv(i,j) = beta2*vv(i,j) + (1.0-beta2)*gv(i,j)*gv(i,j);
                  uv(i,j) -= adam_step*mv(i,j)/sqrt(vv(i,j)+eps);
                });
              }
              else {
                u[m].apply_func(KOKKOS_LAMBDA(const ttb_indx j,const ttb_indx i)
                {
                  uv(i,j) -= step*gv(i,j);
                });
              }
            }
            timer.stop(timer_step);
          }
        }

        // compute objective estimate
        timer.start(timer_fest);
        fest = Impl::gcp_value(X_val, u, w_val, loss_func);
        if (compute_fit) {
          fit = Impl::gcp_value(X, u, w_fit, loss_func);
          fit = 1.0 - fit / x_norm2;
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

        if (failed_epoch) {
          nuc *= decay;

          // restart from last epoch
          deep_copy(u, u_prev);
          fest = fest_prev;
          fit = fit_prev;
          if (use_adam) {
            deep_copy(adam_m, adam_m_prev);
            deep_copy(adam_v, adam_v_prev);
            beta1t /= pow(beta1,epoch_iters);
            beta2t /= pow(beta2,epoch_iters);
          }
        }
        else {
          // update previous data
          deep_copy(u_prev, u);
          fest_prev = fest;
          fit_prev = fit;
          if (use_adam) {
            deep_copy(adam_m_prev, adam_m);
            deep_copy(adam_v_prev, adam_v);
          }
        }

        if (nfails > max_fails || fest < tol)
          break;
      }
      timer.stop(timer_sgd);

      if (printIter > 0) {
         out << "GCP-SGD completed " << total_iters << " iterations in "
             << timer.getTotalTime(timer_sgd) << " seconds" << std::endl
             << "\tsort:     " << timer.getTotalTime(timer_sort) << " seconds\n"
             << "\tsample-f: " << timer.getTotalTime(timer_sample_f) << " seconds\n"
             << "\tsample-g: " << timer.getTotalTime(timer_sample_g) << " seconds\n"
             << "\tf-est:    " << timer.getTotalTime(timer_fest) << " seconds\n"
             << "\tgradient: " << timer.getTotalTime(timer_grad) << " seconds\n"
             << "\tstep:     " << timer.getTotalTime(timer_step) << " seconds\n"
             << "Final f-est: "
             << std::setw(13) << std::setprecision(6) << std::scientific
             << fest;
         if (compute_fit)
           out << ", fit: "
               << std::setw(10) << std::setprecision(3) << std::scientific
               << fit;
         out << std::endl;
      }

      // Normalize Ktensor u
      u.normalize(Genten::NormTwo);
      u.arrange();
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
