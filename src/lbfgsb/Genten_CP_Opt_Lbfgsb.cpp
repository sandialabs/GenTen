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
#include "Genten_GCP_KokkosVector.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_IOtext.hpp"

extern "C" {
#include "lbfgsb.h"
}

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

namespace Genten {

  namespace Impl {
    template<typename TensorT, typename ExecSpace>
    void cp_opt_lbfgsb_fg(const TensorT& X, const KtensorT<ExecSpace>& M,
                          const ttb_real nrm_X_sq, const AlgParams& algParams,
                          ttb_real& f, KtensorT<ExecSpace>& G,
                          FacMatrixT<ExecSpace>& A, FacMatrixT<ExecSpace>& tmp)
    {
      // Compute objective
      const ttb_real nrm_M_sq = M.normFsq();
      const ttb_real ip = innerprod(X,M);
      f = ttb_real(0.5) * (nrm_X_sq + nrm_M_sq) - ip;

      // Compute gradient
      mttkrp_all(X, M, G, algParams);
      A = ttb_real(0.0);
      const ttb_indx nd = M.ndims();
      for (ttb_indx m=0; m<nd; ++m) {
        A.oprod(M.weights());
        for (ttb_indx n=0; n<nd; ++n) {
          if (n != m) {
            tmp = ttb_real(0.0);
            tmp.gramian(M[n], true, Upper);
            A.times(tmp);
          }
        }
        G[m].gemm(false, false, ttb_real(1.0), M[m], A, ttb_real(-1.0));
      }
    }

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
                     const AlgParams& algParams)
  {
    typedef GCP::KokkosVector<ExecSpace> kokkos_vector;
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
    integer m = 5; //fixme
    double factr = 1e7; // fixme
    double pgtol = 1e-5; // fixme
    ttb_indx iterMax = algParams.maxiters;
    ttb_indx total_iterMax = 5000; // fixme
    std::vector<integer> iwa(3*n);
    std::vector<double> wa(2*m*n + 5*n + 11*m*m + 8*m);
    integer task = START;
    integer iprint = -1; // fixme
    integer csave = 1; // ??
    const integer LENGTH_STRING = 60;
    const integer LENGTH_LSAVE = 4;
    const integer LENGTH_ISAVE = 44;
    const integer LENGTH_DSAVE = 29;
    logical lsave[LENGTH_LSAVE];
    integer isave[LENGTH_ISAVE];
    double  dsave[LENGTH_DSAVE];

    const ttb_indx nd = u.ndims();
    const ttb_indx nc = u.ncomponents();
    FacMatrixT<ExecSpace> A(nc,nc);
    FacMatrixT<ExecSpace> tmp(nc,nc);
    const ttb_real nrm_X = X.norm();
    const ttb_real nrm_X_sq = nrm_X*nrm_X;

    // Run CP-OPT
    ttb_indx iters = 1;
    ttb_indx total_iters = 1;
    ttb_indx print_iter = 0;
    while (iters <= iterMax && total_iters <= total_iterMax) {
      ++total_iters;

      setulb(&n,&m,z_host.data(),lower.data(),upper.data(),nbd.data(),
             &f,g_host.data(),&factr,&pgtol,wa.data(),iwa.data(),&task,&iprint,
             &csave,lsave,isave,dsave);

      if ( IS_FG(task) ) {
        Kokkos::deep_copy(z.getView(), z_host);
        KtensorT<ExecSpace> M = z.getKtensor();
        KtensorT<ExecSpace> G = g.getKtensor();
        Impl::cp_opt_lbfgsb_fg(X,M,nrm_X_sq,algParams,f,G,A,tmp);
        deep_copy(g_host, g.getView());

        // Suppress printing of inner iterations by only printing when the
        // the outer iterations increments
        if (algParams.printitn > 0 && iters % algParams.printitn == 0 &&
            iters > print_iter) {
          print_iter = iters;

          // fixme
          ttb_real nrmg = 0.0;
          for (integer i=0; i<n; ++i)
            if (std::abs(g_host(i)) > nrmg)
              nrmg = std::abs(g_host(i));
          std::cout << "Iter " << std::setw(5) << iters
                    << ", f(x) = "
                    << std::setprecision(6) << std::scientific << f
                    << ", ||grad||_infty = "
                    << std::setprecision(2) << std::scientific << nrmg
                    << std::endl;
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

    // Normalize Ktensor u
    u.normalize(Genten::NormTwo);
    u.arrange();

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
    }
  }

}

#define INST_MACRO(SPACE)                                               \
  template void cp_opt_lbfgsb<SptensorT<SPACE>,SPACE>(                  \
    const SptensorT<SPACE>& x,                                          \
    KtensorT<SPACE>& u,                                                 \
    const AlgParams& algParms);                                         \
                                                                        \
  template void cp_opt_lbfgsb<TensorT<SPACE>,SPACE>(                    \
    const TensorT<SPACE>& x,                                            \
    KtensorT<SPACE>& u,                                                 \
    const AlgParams& algParms);

GENTEN_INST(INST_MACRO)
