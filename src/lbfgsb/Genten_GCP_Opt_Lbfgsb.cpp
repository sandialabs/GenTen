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
  @brief CP algorithm, in template form to allow different data tensor types.
*/

#include <assert.h>
#include <cstdio>
#include <iomanip>

#include "Genten_GCP_Opt_Lbfgsb.hpp"
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_KokkosVector.hpp"
#include "Genten_GCP_Model.hpp"
#include "Genten_SystemTimer.hpp"

extern "C" {
#include "lbfgsb.h"
}

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

namespace Genten {

namespace Impl {

extern std::string findTaskString(integer task);

template <typename ExecSpace, typename LossFunction>
void gcp_opt_lbfgsb_impl(const TensorT<ExecSpace>& X,
                         KtensorT<ExecSpace>& u,
                         const LossFunction& loss_func,
                         const AlgParams& algParams,
                         PerfHistory& history)
{
  typedef KokkosVector<ExecSpace> kokkos_vector;

  Genten::SystemTimer timer(1);
  timer.start(0);

  // Distribute the initial guess to have weights of one since the objective
  // does not include gradients w.r.t. weights
  u.distribute(0);

  if (algParams.printitn > 0) {
    const ttb_indx nc = u.ncomponents();
    std::cout << std::endl
              << "GCP-OPT (L-BFGS-B):" << std::endl;
    std::cout << "  CP Rank: " << nc << std::endl
              << "  function type: " << loss_func.name() << std::endl;
    if (loss_func.has_lower_bound())
      std::cout << "  Lower bound: "
                << std::setprecision(2) << std::scientific
                << loss_func.lower_bound() << std::endl;
    if (loss_func.has_upper_bound())
      std::cout << "  Upper bound: "
                << std::setprecision(2) << std::scientific
                << loss_func.upper_bound() << std::endl;
    std::cout  << "  Gradient method: "
               << MTTKRP_All_Method::names[algParams.mttkrp_all_method];
    if (algParams.mttkrp_all_method == MTTKRP_All_Method::Iterated)
      std::cout << " (" << MTTKRP_Method::names[algParams.mttkrp_method] << ")";
    std::cout << " MTTKRP" << std::endl << std::endl;
  }

  // Solution vector
  kokkos_vector z(u); // this is doesn't copy the values
  z.copyFromKtensor(u);
  auto z_host = Kokkos::create_mirror_view(z.getView());
  Kokkos::deep_copy(z_host, z.getView());
  integer n = z.dimension();

  // Bounds
  std::vector<double> lower(n, loss_func.lower_bound());
  std::vector<double> upper(n, loss_func.upper_bound());
  std::vector<integer> nbd(n);
  for (integer i=0; i<n; ++i) {
    if (!loss_func.has_lower_bound() && !loss_func.has_upper_bound())
      nbd[i] = 0;
    else if (!loss_func.has_upper_bound())
      nbd[i] = 1;
    else if (!loss_func.has_lower_bound())
      nbd[i] = 3;
    else
      nbd[i] = 2;
  };

  // function and gradient values
  double f = 0.0;
  kokkos_vector g = z.clone();
  auto g_host = Kokkos::create_mirror_view(g.getView());

  // L-BFGS-B data
  integer m = algParams.memory;
  double factr = algParams.ftol/MACHINE_EPSILON;
  double pgtol = algParams.gtol;
  ttb_indx iterMax = algParams.maxiters;
  ttb_indx total_iterMax = iterMax * algParams.sub_iters;
  std::vector<integer> iwa(3*n);
  std::vector<double> wa(2*m*n + 5*n + 11*m*m + 8*m);
  integer task = START;
  integer iprint = -1; // turn off L-BFGS-B output and use our own
  integer csave = 1; // ??
  const integer LENGTH_LSAVE = 4;
  const integer LENGTH_ISAVE = 44;
  const integer LENGTH_DSAVE = 29;
  logical lsave[LENGTH_LSAVE];
  integer isave[LENGTH_ISAVE];
  double  dsave[LENGTH_DSAVE];

  const bool compute_fit = algParams.compute_fit;

  history.addEmpty();
  GCP_Model<ExecSpace,LossFunction> gcp_model(X, u, loss_func, algParams);

  // Run GCP-OPT
  ttb_indx iters = 0;
  ttb_indx total_iters = 0;
  ttb_indx print_iter = 0;
  while (iters < iterMax && total_iters < total_iterMax) {
    ++total_iters;

    setulb(&n,&m,z_host.data(),lower.data(),upper.data(),nbd.data(),
           &f,g_host.data(),&factr,&pgtol,wa.data(),iwa.data(),&task,&iprint,
           &csave,lsave,isave,dsave);

    if ( IS_FG(task) ) {
      Kokkos::deep_copy(z.getView(), z_host);
      KtensorT<ExecSpace> M = z.getKtensor();
      KtensorT<ExecSpace> G = g.getKtensor();
      gcp_model.update(M);
      f = gcp_model.value(M);
      gcp_model.gradient(G,M);
      deep_copy(g_host, g.getView());

      const ttb_real nrmg = g.normInf();
      const ttb_real time = timer.getTotalTime(0);

      if (history.size() < iters+1)
        history.resize(iters+1);
      history[iters].iteration = iters;
      history[iters].residual = f;
      if (compute_fit)
        history[iters].fit = gcp_model.computeFit(M);
      history[iters].grad_norm = nrmg;
      history[iters].cum_time = time;

      // Suppress printing of inner iterations by only printing when the
      // the outer iterations increments
      if (iters > print_iter) {
        if (algParams.printitn > 0 && (print_iter+1) % algParams.printitn == 0) {
          const auto& h = history[print_iter];
          std::cout << "Iter " << std::setw(5) << print_iter+1
                    << ", f(x) = "
                    << std::setprecision(6) << std::scientific << h.residual;
          if (compute_fit)
            std::cout << ", fit = "
                      << std::setprecision(3) << std::scientific << h.fit;
          std::cout << ", ||grad||_infty = "
                    << std::setprecision(2) << std::scientific << h.grad_norm
                    << ", t = "
                    << std::setprecision(2) << std::scientific << h.cum_time
                    << std::endl;
        }
        print_iter = iters;
      }
      continue;
    }

    if (task==NEW_X) {
      ++iters;
      continue;
    }
    else
      break;
  }

  // Print last iteration
  if (algParams.printitn > 0) {
    const auto& h = history.lastEntry();
    std::cout << "Iter " << std::setw(5) << print_iter+1
              << ", f(x) = "
              << std::setprecision(6) << std::scientific << h.residual;
    if (compute_fit)
      std::cout << ", fit = "
                << std::setprecision(3) << std::scientific << h.fit;
    std::cout << ", ||grad||_infty = "
              << std::setprecision(2) << std::scientific << h.grad_norm
              << ", t = "
              << std::setprecision(2) << std::scientific << h.cum_time
              << std::endl;
  }

  // Normalize Ktensor u
  z.copyToKtensor(u);
  u.normalize(Genten::NormTwo);
  u.arrange();

  timer.stop(0);

  // Compute final fit
  if (algParams.printitn > 0) {
    if (iters >= iterMax)
      std::cout << "Reached maximum number of iterations." << std::endl;
    else if (total_iters >= total_iterMax)
      std::cout << "Reached maximum number of total iterations." << std::endl;
    else
      std::cout << Impl::findTaskString(task) << std::endl;
    if (compute_fit) {
      gcp_model.update(u);
      const ttb_real fit = gcp_model.computeFit(u);
      std::cout << "Final fit = " << std::setprecision(3) << std::scientific
                << fit << std::endl;
    }
    std::cout << "Total time = " << std::setprecision(2) << std::scientific
              << timer.getTotalTime(0) << std::endl
              << std::endl;
  }
}

}

template<typename ExecSpace>
void gcp_opt_lbfgsb(const TensorT<ExecSpace>& x, KtensorT<ExecSpace>& u,
                    const AlgParams& algParams,
                    PerfHistory& history)
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::gcp_opt_lbfgsb");
#endif

  // Check size compatibility of the arguments.
  if (u.isConsistent() == false)
    Genten::error("Genten::gcp_opt - ktensor u is not consistent");
  if (x.ndims() != u.ndims())
    Genten::error("Genten::gcp_opt - u and x have different num dims");
  for (ttb_indx  i = 0; i < x.ndims(); i++)
  {
    if (x.size(i) != u[i].nRows())
      Genten::error("Genten::cp_opt - u and x have different size");
  }

  // Dispatch implementation based on loss function type
  dispatch_loss(algParams, [&](const auto& loss)
  {
    Impl::gcp_opt_lbfgsb_impl(x, u, loss, algParams, history);
  });
}

}

#define INST_MACRO(SPACE)                                               \
  template void gcp_opt_lbfgsb<SPACE>(                                  \
    const TensorT<SPACE>& x,                                            \
    KtensorT<SPACE>& u,                                                 \
    const AlgParams& algParms,                                          \
    PerfHistory& history);

GENTEN_INST(INST_MACRO)
