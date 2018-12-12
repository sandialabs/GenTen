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

#include "Genten_GCP_SGD.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_RandomMT.hpp"

// Whether to compute gradient tensor "Y" explicitly in GCP kernels
#define COMPUTE_Y 0
#include "Genten_GCP_Kernels.hpp"

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

// To do:
//   * better tensor search in sampling
//   * parallelize kernels/remove host-device transfers
//   * pass in algorithmic parameters
//   * connect to driver
//   * add ADAM

namespace Genten {

  namespace Impl {

    template <typename ExecSpace>
    void uniform_sample_tensor(const SptensorT<ExecSpace>& X,
                               const ttb_indx num_samples_nonzeros,
                               const ttb_indx num_samples_zeros,
                               SptensorT<ExecSpace>& Y,
                               ArrayT<ExecSpace>& w,
                               RandomMT& rng,
                               const AlgParams& algParams)
    {
      typedef typename SptensorT<ExecSpace>::HostMirror Sptensor_host_type;
      typedef typename Sptensor_host_type::exec_space host_exec_space;
      typedef typename ArrayT<ExecSpace>::HostMirror Array_host_type;

      const ttb_indx nnz = X.nnz();
      const ttb_indx nd = X.ndims();
      const ttb_indx tsz = X.numel();

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples_nonzeros + num_samples_zeros;
      if (Y.nnz() < total_samples) {
        Y = SptensorT<ExecSpace>(X.size(), total_samples);
        w = ArrayT<ExecSpace>(total_samples);
      }

      // Copy to host
      // Todo:  make kernels below run on device so don't need to copy
      Sptensor_host_type X_host = create_mirror_view(host_exec_space(), X);
      Sptensor_host_type Y_host = create_mirror_view(host_exec_space(), Y);
      Array_host_type w_host = create_mirror_view(host_exec_space(), w);
      deep_copy(X_host, X);

      // Geneate num_samples_nonzeros samples in the range [0,nnz)
      for (ttb_indx i=0; i<num_samples_nonzeros; ++i) {
        const ttb_indx idx = ttb_indx(rng.genrnd_double() * nnz);
        Y_host.value(i) = X_host.value(idx);
        for (ttb_indx j=0; j<nd; ++j)
          Y_host.subscript(i,j) = X_host.subscript(idx,j);
        w_host[i] = ttb_real(nnz) / ttb_real(num_samples_nonzeros);
      }

      // Generate num_samples_zeros of zeros
      //IndxArrayT<host_exec_space> ind(nd);
      std::vector<ttb_indx> ind(nd);
      ttb_indx i=0;
      while (i<num_samples_zeros) {

        // Generate index
        for (ttb_indx j=0; j<nd; ++j)
          ind[j] = ttb_indx(rng.genrnd_double() * X_host.size(j));

        // Search for index
        bool found = false;
        for (ttb_indx idx=0; idx<nnz; ++idx) {
          bool t = true;
          for (ttb_indx j=0; j<nd; ++j) {
            if (ind[j] != X_host.subscript(idx,j)) {
              t = false;
              break;
            }
          }
          if (t) {
            found = true;
            break;
          }
        }

        // If not found add it
        if (!found) {
          const ttb_indx idx = num_samples_nonzeros + i;
          for (ttb_indx j=0; j<nd; ++j)
            Y_host.subscript(idx,j) = ind[j];
          Y_host.value(idx) = 0.0;
          w_host[idx] = ttb_real(tsz-nnz) / ttb_real(num_samples_zeros);
          ++i;
        }

      }

      // Copy to device
      deep_copy(Y, Y_host);
      deep_copy(w, w_host);
      if (algParams.mttkrp_method == MTTKRP_Method::Perm) {
        Y.setHavePerm(false);
        Y.createPermutation();
      }
    }

    template<typename TensorT, typename ExecSpace, typename LossFunction>
    void gcp_sgd_impl(const TensorT& X, KtensorT<ExecSpace>& u,
                      const LossFunction& loss_func,
                      const ttb_real tol,
                      const ttb_indx maxEpochs,
                      const ttb_indx printIter,
                      ttb_indx& numEpochs,
                      ttb_real& fest,
                      std::ostream& out,
                      const AlgParams& algParams)
    {
      typedef FacMatrixT<ExecSpace> fac_matrix_type;
      typedef typename fac_matrix_type::view_type view_type;

      const ttb_indx nd = u.ndims();
      const ttb_indx nc = u.ncomponents();

      // Constants for the algorithm that should be passed in eventually
      const ttb_real decay = 0.1;
      const ttb_real rate = 1.0e-3;
      const ttb_indx max_fails = 1;
      const ttb_indx epoch_iters = 1000;
      const ttb_indx seed = 12345;

      // Compute number of samples
      const ttb_indx nnz = X.nnz();
      const ttb_indx tsz = X.numel();
      const ttb_indx nz = tsz - nnz;
      const ttb_indx ftmp = std::max((nnz+99)/100,ttb_indx(100000));
      const ttb_indx num_samples_nonzeros_value =
        std::min(ftmp, nnz);
      const ttb_indx num_samples_zeros_value =
        std::min(num_samples_nonzeros_value, nz);
      const ttb_indx gtmp = std::max((3*nnz+maxEpochs-1)/maxEpochs,
                                     ttb_indx(1000));
      const ttb_indx num_samples_nonzeros_grad =
        std::min(gtmp, nnz);
      const ttb_indx num_samples_zeros_grad =
        std::min(num_samples_nonzeros_grad, nz);

      ttb_real nuc = 1.0;
      ttb_indx total_iters = 0;
      ttb_indx nfails = 0;

      // Gradient Ktensor
      KtensorT<ExecSpace> g(nc, nd);
      for (ttb_indx m=0; m<nd; ++m)
        g.set_factor(m, fac_matrix_type(u[m].nRows(), nc));

      // Copy Ktensor for restoring previous solution
      KtensorT<ExecSpace> u_prev(nc, nd);
      for (ttb_indx m=0; m<nd; ++m)
        u_prev.set_factor(m, fac_matrix_type(u[m].nRows(), nc));
      deep_copy(u_prev, u);

      // Sample X for f-estimate
      SptensorT<ExecSpace> X_val, X_grad, Y;
      ArrayT<ExecSpace> w_val, w_grad;
      RandomMT rng(seed);
      Impl::uniform_sample_tensor(
        X, num_samples_nonzeros_value, num_samples_zeros_value,
        X_val, w_val, rng, algParams);

      // Objective estimates
      fest = Impl::gcp_value(X_val, u, w_val, loss_func);
      ttb_real fest_prev = fest;

      // Start timer for total execution time of the algorithm.
      const int timer_sgd = 0;
      SystemTimer timer(1);
      timer.start(timer_sgd);

      if (printIter > 0) {
        out << "Starting GCP-SGD" << std::endl
            << "Initial f-est = "
            << std::setw(13) << std::setprecision(6) << std::scientific
            << fest << std::endl;
      }

      // SGD epoch loop
      for (numEpochs=0; numEpochs<maxEpochs; ++numEpochs) {
        // Gradient step size
        ttb_real step = nuc*rate;

        // Epoch iterations
        for (ttb_indx iter=0; iter<epoch_iters; ++iter) {
          ++total_iters;

          // compute gradient
          Impl::uniform_sample_tensor(
            X, num_samples_nonzeros_grad, num_samples_zeros_grad,
            X_grad, w_grad, rng, algParams);
          Impl::gcp_gradient(X_grad, Y, u, w_grad, loss_func, g, algParams);

          // take step
          for (ttb_indx m=0; m<nd; ++m) {
            view_type uv = u[m].view();
            view_type gv = g[m].view();
            u[m].apply_func(KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i)
            {
              uv(i,j) -= step*gv(i,j);
            });
          }
        }

        // compute objective estimate
        fest = Impl::gcp_value(X_val, u, w_val, loss_func);

        // check convergence
        const bool failed_epoch = fest > fest_prev;

        if (failed_epoch)
          ++nfails;

         // Print progress of the current iteration.
        if ((printIter > 0) && (((numEpochs + 1) % printIter) == 0)) {
          out << "Epoch " << std::setw(3) << numEpochs + 1 << ": f-est = "
              << std::setw(13) << std::setprecision(6) << std::scientific
              << fest
              << ", step = "
              << std::setw(8) << std::setprecision(1) << std::scientific
              << step;
          if (failed_epoch)
            out << ", nfails = " << nfails
                << "( resetting to solution from last epoch)";
          out << std::endl;
        }

        if (failed_epoch) {
          nuc *= decay;

          // restart from last epoch
          deep_copy(u, u_prev);
          fest = fest_prev;
          total_iters -= epoch_iters;  // Why do this?
        }
        else {
          // update previous data
          deep_copy(u_prev, u);
          fest_prev = fest;
        }

        if (nfails > max_fails || fest < tol)
          break;
      }
      timer.stop(timer_sgd);

      if (printIter > 0) {
         out << "GCP-SGD completed " << total_iters << " iterations in "
             << timer.getTotalTime(timer_sgd) << " seconds\n";
         out << "Final f-est = "
             << std::setw(13) << std::setprecision(6) << std::scientific
             << fest << std::endl;
      }

      // Normalize Ktensor u
      u.normalize(Genten::NormTwo);
      u.arrange();
    }

  }


  template<typename TensorT, typename ExecSpace>
  void gcp_sgd(const TensorT& x, KtensorT<ExecSpace>& u,
               const GCP_LossFunction::type loss_function_type,
               const ttb_real loss_eps,
               const ttb_real tol,
               const ttb_indx maxIters,
               const ttb_indx printIter,
               ttb_indx& numIters,
               ttb_real& resNorm,
               std::ostream& out,
               const AlgParams& algParams)
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
    if (loss_function_type == GCP_LossFunction::Gaussian)
      Impl::gcp_sgd_impl(x, u, GaussianLossFunction(loss_eps),
                         tol, maxIters, printIter, numIters, resNorm,
                         out, algParams);
    // else if (loss_function_type == GCP_LossFunction::Rayleigh)
    //   Impl::gcp_sgd_impl(x, u, RayleighLossFunction(loss_eps), params, stream,
    //                      algParams);
    // else if (loss_function_type == GCP_LossFunction::Gamma)
    //   Impl::gcp_sgd_impl(x, u, GammaLossFunction(loss_eps), params, stream,
    //                      algParams);
    // else if (loss_function_type == GCP_LossFunction::Bernoulli)
    //   Impl::gcp_sgd_impl(x, u, BernoulliLossFunction(loss_eps), params, stream,
    //                      algParams);
    // else if (loss_function_type == GCP_LossFunction::Poisson)
    //   Impl::gcp_sgd_impl(x, u, PoissonLossFunction(loss_eps), params, stream,
    //                      algParams);
    else
       Genten::error("Genten::gcp_sgd - unknown loss function");
  }

}

#define INST_MACRO(SPACE)                                               \
  template void gcp_sgd<SptensorT<SPACE>,SPACE>(                        \
    const SptensorT<SPACE>& x,                                          \
    KtensorT<SPACE>& u,                                                 \
    const GCP_LossFunction::type loss_function_type,                    \
    const ttb_real loss_eps,                                            \
    const ttb_real tol,                                                 \
    const ttb_indx maxIters,                                            \
    const ttb_indx printIter,                                           \
    ttb_indx& numIters,                                                 \
    ttb_real& resNorm,                                                  \
    std::ostream& out,                                                  \
    const AlgParams& algParams);

GENTEN_INST(INST_MACRO)
