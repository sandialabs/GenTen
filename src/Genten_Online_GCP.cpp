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

#include <algorithm>
#include <random>

#include "Genten_Online_GCP.hpp"
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_IOtext.hpp"

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

// To do:
// * add something like info struct from matlab code?
// * add normalization options?

namespace Genten {

  template <typename TensorT, typename ExecSpace, typename LossFunction>
  OnlineGCP<TensorT,ExecSpace,LossFunction>::
  OnlineGCP(TensorT& Xinit,
            const KtensorT<ExecSpace>& u,
            const LossFunction& loss_func,
            const AlgParams& algParams_,
            const AlgParams& temporalAlgParams_,
            const AlgParams& spatialAlgParams_,
            std::ostream& out) :
    algParams(algParams_),
    temporalAlgParams(temporalAlgParams_),
    spatialAlgParams(spatialAlgParams_),
    temporalSolver(u,loss_func,u.ndims()-1,u.ndims(),temporalAlgParams_),
    spatialSolver(u,loss_func,0,u.ndims()-1,spatialAlgParams_),
    generator(algParams.seed),
    hist(u,algParams)
  {
    const ttb_indx nc = u.ncomponents();
    const ttb_indx nd = u.ndims();
    if (temporalAlgParams.streaming_solver == GCP_Streaming_Solver::LeastSquares ||
        spatialAlgParams.streaming_solver == GCP_Streaming_Solver::LeastSquares ||
        temporalAlgParams.streaming_solver == GCP_Streaming_Solver::OnlineCP ||
        spatialAlgParams.streaming_solver == GCP_Streaming_Solver::OnlineCP)
    {
      A = FacMatrixT<ExecSpace>(nc,nc);
      tmp = FacMatrixT<ExecSpace>(nc,nc);
    }

    if (spatialAlgParams.streaming_solver == GCP_Streaming_Solver::OnlineCP) {
      P = std::vector< FacMatrixT<ExecSpace> >(nd-1);
      Q = std::vector< FacMatrixT<ExecSpace> >(nd-1);
      for (ttb_indx k=0; k<nd-1; ++k) {
        P[k] = FacMatrixT<ExecSpace>(u[k].nRows(),nc);
        Q[k] = FacMatrixT<ExecSpace>(nc,nc);
      }
      if (algParams.mttkrp_method == MTTKRP_Method::Perm &&
          !Xinit.havePerm())
        Xinit.createPermutation();
      const bool full = algParams.full_gram;
      for (ttb_indx m=0; m<nd-1; ++m) {
        mttkrp(Xinit, u, m, P[m], algParams);
        Q[m].oprod(u.weights());
        for (ttb_indx n=0; n<nd; ++n) {
          if (n != m) {
            tmp = ttb_real(0.0);
            tmp.gramian(u[n], full, Upper);
            Q[m].times(tmp);
          }
        }
      }
    }

    // Print welcome message
    out << "\nOnline-GCP (Online Generalized CP Tensor Decomposition)\n\n"
        << "Rank: " << nc << std::endl
        << "Generalized function type: " << loss_func.name() << std::endl
        << "Streaming window size: " << algParams.window_size << " ("
        << GCP_Streaming_Window_Method::names[algParams.window_method] << "), "
        << "penalty: " << algParams.window_penalty
        << " * ( " << algParams.window_weight << " )^(T-t)" << std::endl;
  }

  template <typename TensorT, typename ExecSpace, typename LossFunction>
  void
  OnlineGCP<TensorT,ExecSpace,LossFunction>::
  init(const TensorT& X, KtensorT<ExecSpace>& u)
  {
    hist.updateHistory(u);

    // Adjust size of temporal mode of u to match X
    const ttb_indx nd = X.ndims();
    const ttb_indx nt = X.size(nd-1);
    if (u.ndims() != nd)
      Genten::error("Genten::online_gcp - u and x have different num dims");
    if (u[nd-1].nRows() > nt) {
      const ttb_indx nc = u.ncomponents();
      const ttb_indx s = u[nd-1].nRows();
      FacMatrixT<ExecSpace> t(nt,nc);
      auto tt = Kokkos::subview(u[nd-1].view(),
                                std::pair<ttb_indx,ttb_indx>(s-nt,s),
                                Kokkos::ALL);
      deep_copy(t.view(),tt);
      u.set_factor(nd-1, t);
    }
  }

  template <typename TensorT, typename ExecSpace, typename LossFunction>
  void
  OnlineGCP<TensorT,ExecSpace,LossFunction>::
  processSlice(TensorT& X,
               KtensorT<ExecSpace>& u,
               ttb_real& fest,
               ttb_real& ften,
               std::ostream& out,
               const bool print)
  {
    ttb_indx num_epoch = 0;

    if (print)
      out << "Solving for temporal mode..." << std::endl;
    if (temporalAlgParams.streaming_solver == GCP_Streaming_Solver::SGD) {
      // No history for temporal solver
      temporalSolver.reset();
      Genten::PerfHistory perf;
      temporalSolver.solve(X, u, algParams.factor_penalty,
                           num_epoch, fest, perf, out, false, false, print);
    }
    else if (temporalAlgParams.streaming_solver ==
             GCP_Streaming_Solver::LeastSquares ||
             temporalAlgParams.streaming_solver ==
             GCP_Streaming_Solver::OnlineCP)
      leastSquaresSolve(true,X,u,fest,ften,out,print);
    else
      Genten::error("Unknown temporal streaming solver method ");

    if (print)
      out << "Updating spatial modes..." << std::endl;
    if (spatialAlgParams.streaming_solver == GCP_Streaming_Solver::SGD) {
      Genten::PerfHistory perf;
      spatialSolver.solve(X, u, hist, algParams.factor_penalty,
                          num_epoch, fest, ften, perf, out, false, false,
                          print);
    }
    else if (spatialAlgParams.streaming_solver ==
             GCP_Streaming_Solver::LeastSquares)
      leastSquaresSolve(false,X,u,fest,ften,out,print);
    else if (spatialAlgParams.streaming_solver ==
             GCP_Streaming_Solver::OnlineCP) {
      if (algParams.mttkrp_method == MTTKRP_Method::Perm &&
          !X.havePerm())
        X.createPermutation();
      const ttb_indx nd = u.ndims();
      const bool full = algParams.full_gram;
      for (ttb_indx m=0; m<nd-1; ++m) {
        mttkrp(X, u, m, P[m], algParams, false);
        A.oprod(u.weights());
        for (ttb_indx n=0; n<nd; ++n) {
          if (n != m) {
            tmp = ttb_real(0.0);
            tmp.gramian(u[n], full, Upper);
            A.times(tmp);
          }
        }
        Q[m].plus(A);
        deep_copy(u[m], P[m]);
        u[m].solveTransposeRHS(Q[m], full, Upper, true, algParams);
        const ttb_real ip = innerprod(X, u);
        const ttb_real nrmx = X.norm();
        const ttb_real nrmusq = u.normFsq();
        fest = nrmx*nrmx + nrmusq - ttb_real(2.0)*ip;
        ften = fest;
        if (print)
          out << "f = " << fest << std::endl;
      }
    }
    else
      Genten::error("Unknown factor matrix solver method ");

    // Update history window
    if (algParams.window_size > 0 &&
        spatialAlgParams.streaming_solver != GCP_Streaming_Solver::OnlineCP) {
      hist.updateHistory(u);
    }
  }

  template <typename TensorT, typename ExecSpace, typename LossFunction>
  void
  OnlineGCP<TensorT,ExecSpace,LossFunction>::
  leastSquaresSolve(const bool temporal,
                    TensorT& X,
                    KtensorT<ExecSpace>& u,
                    ttb_real& fest,
                    ttb_real& ften,
                    std::ostream& out,
                    const bool print)
  {
    // To do:
    //  * innerproduct trick?
    const ttb_indx nd = u.ndims();
    const bool full = algParams.full_gram;

    if (algParams.mttkrp_method == MTTKRP_Method::Perm &&
        !X.havePerm())
      X.createPermutation();

    const ttb_indx mode_beg = temporal ? nd-1 : 0;
    const ttb_indx mode_end = temporal ? nd   : nd-1;
    for (ttb_indx mode=mode_beg; mode<mode_end; ++mode) {

      // compute LHS
      A.oprod(u.weights());
      for (ttb_indx n=0; n<nd; ++n) {
        if (n != mode) {
          tmp = ttb_real(0.0);
          tmp.gramian(u[n], full, Upper);
          A.times(tmp);
        }
      }
      if (algParams.factor_penalty != ttb_real(0.0))
        A.diagonalShift(ttb_real(2.0)*algParams.factor_penalty);

      // Compute RHS
      mttkrp(X, u, mode, u[mode], algParams);

      // Add in history terms.  The Gram matrices need to be recomputed
      // each iteration since u[mode] changes after each solve
      if (!temporal) {
        hist.prepare_least_squares_contributions(u, mode);
        hist.least_squares_contributions(u, mode, A, u[mode]);
      }

      // Solve least-squares system
      u[mode].solveTransposeRHS(A, full, Upper, true, algParams);
    }

    // Compute residuals
    const ttb_real ip = innerprod(X, u);
    const ttb_real nrmx = X.norm();
    const ttb_real nrmusq = u.normFsq();
    ften = nrmx*nrmx + nrmusq - ttb_real(2.0)*ip;
    fest = ften;
    if (!temporal)
      fest += hist.ktensor_fro_objective(u);
    if (algParams.factor_penalty != ttb_real(0.0)) {
      for (ttb_indx i=0; i<nd; ++i)
        fest += algParams.factor_penalty * u[i].normFsq();
    }
    if (print)
      out << "f = " << fest << std::endl;
  }

  template <typename TensorT, typename ExecSpace, typename LossFunction>
  void online_gcp_impl(std::vector<TensorT>& X,
                       TensorT& Xinit,
                       KtensorT<ExecSpace>& u,
                       const LossFunction& loss_func,
                       const AlgParams& algParams,
                       const AlgParams& temporalAlgParams,
                       const AlgParams& spatialAlgParams,
                       std::ostream& out,
                       Array& fest,
                       Array& ften)
  {
    const ttb_indx nd = u.ndims();
    const ttb_indx nc = u.ncomponents();
    const ttb_indx num_slices = X.size();

    if (num_slices == 0)
      return;

    OnlineGCP<TensorT,ExecSpace,LossFunction> ogcp(
      Xinit, u, loss_func, algParams, temporalAlgParams, spatialAlgParams,
      out);

    // Compute total number of time slices
    ttb_indx nt = 0;
    for (ttb_indx i=0; i<num_slices; ++i) {
      if (X[i].ndims() != u.ndims())
        Genten::error("Genten::online_gcp - u and x have different num dims");
      for (ttb_indx j=0; j<nd-1; ++j)
      {
        if (X[i].size(j) != u[j].nRows())
          Genten::error("Genten::online_gcp - u and x have different size");
      }
      nt += X[i].size(nd-1);
    }

    // Allocate factor matrix for time mode
    FacMatrixT<ExecSpace> time_mode(nt, nc);

    // Loop over slices, processing one at a time
    fest = Array(num_slices);
    ften = Array(num_slices);
    ttb_real row = 0;
    for (ttb_indx i=0; i<num_slices; ++i) {
      const bool print =
        (algParams.printitn > 0) &&((i+1) % algParams.printitn == 0);

      if (print)
        out << "\nProcessing slice " << i+1 << " of " << num_slices
            << std::endl;

      if (i == 0) {
        ogcp.init(X[i], u);
      }

      ogcp.processSlice(X[i], u, fest[i], ften[i], out, print);

      // Copy time mode for this slice into time_mode
      const ttb_indx nrow = X[i].size(nd-1);
      auto tm_row = Kokkos::subview(
        time_mode.view(),
        std::pair<ttb_indx,ttb_indx>(row,row+nrow),
        Kokkos::ALL);
      deep_copy(tm_row, u[nd-1].view());
      row += nrow;
    }

    // Set time mode of Ktensor
    u.set_factor(nd-1, time_mode);
  }

  template<typename TensorT, typename ExecSpace>
  void online_gcp(std::vector<TensorT>& X,
                  TensorT& Xinit,
                  KtensorT<ExecSpace>& u,
                  const AlgParams& algParams,
                  const AlgParams& temporalAlgParams,
                  const AlgParams& spatialAlgParams,
                  std::ostream& out,
                  Array& fest,
                  Array& ften)
  {
#ifdef HAVE_CALIPER
    cali::Function cali_func("Genten::online_gcp");
#endif

    // Check size compatibility of the arguments.
    if (u.isConsistent() == false)
      Genten::error("Genten::online_gcp - ktensor u is not consistent");

    // Dispatch implementation based on loss function type
    dispatch_loss(algParams, [&](const auto& loss)
    {
      online_gcp_impl(X, Xinit, u, loss, algParams, temporalAlgParams,
                      spatialAlgParams, out, fest, ften);
    });
  }

}

#define LOSS_INST_MACRO(SPACE,LOSS)                                     \
  template class Genten::OnlineGCP<SptensorT<SPACE>,SPACE,LOSS>;

#define INST_MACRO(SPACE)                                               \
  GENTEN_INST_LOSS(SPACE,LOSS_INST_MACRO)                               \
                                                                        \
  template void online_gcp<SptensorT<SPACE>,SPACE>(                     \
    std::vector<SptensorT<SPACE>>& x,                                   \
    SptensorT<SPACE>& x_init,                                           \
    KtensorT<SPACE>& u,                                                 \
    const AlgParams& algParams,                                         \
    const AlgParams& temporalAlgParams,                                 \
    const AlgParams& spatialAlgParams,                                  \
    std::ostream& out,                                                  \
    Array& fest,                                                        \
    Array& ften);

GENTEN_INST(INST_MACRO)
