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
  @file Genten_CP.cpp
  @brief CP algorithm, in template form to allow different data tensor types.
*/

#include <assert.h>
#include <cstdio>
#include <iomanip>

#include "Genten_CP_Opt_Lbfgsb.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_KokkosVector.hpp"
#include "Genten_CP_Model.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_SystemTimer.hpp"

extern "C" {
#include "lbfgsb.h"
}

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

namespace Genten {

  namespace Impl {

    std::string findTaskString(integer task)
    {
      std::string str;
      switch(task) {
      case 209:
        str = "ERROR: N .LE. 0";
        break;
      case 210:
        str = "ERROR: M .LE. 0";
        break;
      case 211:
        str = "ERROR: FACTR .LT. 0";
        break;
      case 3:
        str = "ABNORMAL_TERMINATION_IN_LNSRCH.";
        break;
      case 4:
        str = "RESTART_FROM_LNSRCH.";
        break;
      case 21:
        str = "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL.";
        break;
      case 22:
        str = "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH.";
        break;
      case 31:
        str = "STOP: CPU EXCEEDING THE TIME LIMIT.";
        break;
      case 32:
        str = "STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIM.";
        break;
      case 33:
        str = "STOP: THE PROJECTED GRADIENT IS SUFFICIENTLY SMALL.";
        break;
      case 101:
        str = "WARNING: ROUNDING ERRORS PREVENT PROGRESS";
        break;
      case 102:
        str = "WARNING: XTOL TEST SATISIED";
        break;
      case 103:
        str = "WARNING: STP = STPMAX";
        break;
      case 104:
        str = "WARNING: STP = STPMIN";
        break;
      case 201:
        str = "ERROR: STP .LT. STPMIN";
        break;
      case 202:
        str = "ERROR: STP .GT. STPMAX";
        break;
      case 203:
        str = "ERROR: INITIAL G .GE. ZERO ";
        break;
      case 204:
        str = "ERROR: FTOL .LT. ZERO";
        break;
      case 205:
        str = "ERROR: GTOL .LT. ZERO";
        break;
      case 206:
        str = "ERROR: XTOL .LT. ZERO";
        break;
      case 207:
        str = "ERROR: STPMIN .LT. ZERO";
        break;
      case 208:
        str = "ERROR: STPMAX .LT. STPMIN";
        break;
      case 212:
        str = "ERROR: INVALID NBD";
        break;
      case 213:
        str = "ERROR: NO FEASIBLE SOLUTION";
        break;
      default:
        str = "UNRECOGNIZED EXIT FLAG";
        break;
      }
      return str;
    }
  }

  template<typename TensorT, typename ExecSpace>
  void cp_opt_lbfgsb(const TensorT& X, KtensorT<ExecSpace>& u,
                     const AlgParams& algParams,
                     std::vector<std::vector<ttb_real> >& history)
  {
    typedef KokkosVector<ExecSpace> kokkos_vector;
#ifdef HAVE_CALIPER
    cali::Function cali_func("Genten::cp_opt_lbfgsb");
#endif

    // Check size compatibility of the arguments.
    if (u.isConsistent() == false)
      Genten::error("Genten::cp_opt - ktensor u is not consistent");
    if (X.ndims() != u.ndims())
      Genten::error("Genten::cp_opt - u and x have different num dims");
    for (ttb_indx  i = 0; i < X.ndims(); i++)
    {
      if (X.size(i) != u[i].nRows())
        Genten::error("Genten::cp_opt - u and x have different size");
    }

    Genten::SystemTimer timer(1);
    timer.start(0);

    // Distribute the initial guess to have weights of one since the objective
    // does not include gradients w.r.t. weights
    u.distribute(0);

    // Solution vector
    kokkos_vector z(u); // this is doesn't copy the values
    z.copyFromKtensor(u);
    auto z_host = Kokkos::create_mirror_view(z.getView());
    Kokkos::deep_copy(z_host, z.getView());
    integer n = z.dimension();

    // Bounds
    std::vector<double> lower(n, algParams.lower);
    std::vector<double> upper(n, algParams.upper);
    std::vector<integer> nbd(n);
    for (integer i=0; i<n; ++i) {
      if (algParams.lower == -DBL_MAX && algParams.upper == DBL_MAX)
        nbd[i] = 0;
      else if (algParams.upper == DBL_MAX)
        nbd[i] = 1;
      else if (algParams.lower == -DBL_MAX)
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
    double factr = algParams.factr;
    double pgtol = algParams.pgtol;
    ttb_indx iterMax = algParams.maxiters;
    ttb_indx total_iterMax = algParams.max_total_iters;
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

    const ttb_real nrm_X = X.norm();
    const ttb_real nrm_X_sq = nrm_X*nrm_X;

    history.push_back(std::vector<ttb_real>(3));
    CP_Model<TensorT> cp_model(X, u, algParams);

    // Run CP-OPT
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
        cp_model.update(M);
        f = cp_model.value_and_gradient(G, M);
        deep_copy(g_host, g.getView());

        const ttb_real nrmg = g.normInf();
        const ttb_real time = timer.getTotalTime(0);

        if (history.size() < iters+1)
          history.resize(iters+1);
        history[iters].resize(3);
        history[iters][0] = f;
        history[iters][1] = nrmg;
        history[iters][2] = time;

        // Suppress printing of inner iterations by only printing when the
        // the outer iterations increments
        if (iters > print_iter) {
          if (algParams.printitn > 0 && (print_iter+1) % algParams.printitn == 0) {
            const std::vector<ttb_real>& h = history[print_iter];
            std::cout << "Iter " << std::setw(5) << print_iter+1
                      << ", f(x) = "
                      << std::setprecision(6) << std::scientific << h[0]
                      << ", ||grad||_infty = "
                      << std::setprecision(2) << std::scientific << h[1]
                      << ", t = "
                      << std::setprecision(2) << std::scientific << h[2]
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
      const std::vector<ttb_real>& h = history[print_iter];
      std::cout << "Iter " << std::setw(5) << print_iter+1
                << ", f(x) = "
                << std::setprecision(6) << std::scientific << h[0]
                << ", ||grad||_infty = "
                << std::setprecision(2) << std::scientific << h[1]
                << ", t = "
                << std::setprecision(2) << std::scientific << h[2]
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
      const ttb_real fit = ttb_real(1.0) - f / (ttb_real(0.5)*nrm_X_sq);
      std::cout << "Final fit = " << fit << std::endl;
      std::cout << "Total time = " << timer.getTotalTime(0) << std::endl;
    }
  }

}

#define INST_MACRO(SPACE)                                               \
  template void cp_opt_lbfgsb<SptensorT<SPACE>,SPACE>(                  \
    const SptensorT<SPACE>& x,                                          \
    KtensorT<SPACE>& u,                                                 \
    const AlgParams& algParms,                                          \
    std::vector<std::vector<ttb_real> >& history);                      \
                                                                        \
  template void cp_opt_lbfgsb<TensorT<SPACE>,SPACE>(                    \
    const TensorT<SPACE>& x,                                            \
    KtensorT<SPACE>& u,                                                 \
    const AlgParams& algParms,                                          \
    std::vector<std::vector<ttb_real> >& history);

GENTEN_INST(INST_MACRO)
