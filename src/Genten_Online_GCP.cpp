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
    rand(algParams.seed)
  {
    const ttb_indx nc = u.ncomponents();
    if (temporalAlgParams.streaming_solver == GCP_Streaming_Solver::LeastSquares ||
        temporalAlgParams.streaming_solver == GCP_Streaming_Solver::OnlineCP ||
        spatialAlgParams.streaming_solver == GCP_Streaming_Solver::LeastSquares)
    {
      A = FacMatrixT<ExecSpace>(nc,nc);
      tmp = FacMatrixT<ExecSpace>(nc,nc);
    }

    if (spatialAlgParams.streaming_solver == GCP_Streaming_Solver::OnlineCP) {
      const ttb_indx nd = u.ndims();
      IndxArrayT<ExecSpace> Psz(nd-1), Qsz(nd-1);
      auto Psz_host = create_mirror_view(Psz);
      auto Qsz_host = create_mirror_view(Qsz);
      for (ttb_indx i=0; i<nd-1; ++i) {
        Psz_host[i] = u[i].nRows();
        Qsz_host[i] = nc;
      }
      deep_copy(Psz, Psz_host);
      deep_copy(Qsz, Qsz_host);
      P = FacMatArrayT<ExecSpace>(nd-1, Psz, nc);
      Q = FacMatArrayT<ExecSpace>(nd-1, Qsz, nc);

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
        << "Streaming window size: " << algParams.window_size << "("
        << GCP_Streaming_Window_Method::names[algParams.window_method] << "), "
        << "penalty: " << algParams.window_penalty
        << " * ( " << algParams.window_weight << " )^(T-t)" << std::endl;
  }

  template <typename TensorT, typename ExecSpace, typename LossFunction>
  void
  OnlineGCP<TensorT,ExecSpace,LossFunction>::
  init(const TensorT& X, KtensorT<ExecSpace>& u)
  {
    // to do:
    //   * make sure u is normalized properly with unit weights
    const ttb_indx nc = u.ncomponents();
    const ttb_indx nd = X.ndims();

    // Construct window data -- u contains initial history
    if (algParams.window_size > 0 &&
        spatialAlgParams.streaming_solver != GCP_Streaming_Solver::OnlineCP) {
      up = KtensorT<ExecSpace>(nc, nd);
      for (ttb_indx i=0; i<nd-1; ++i)
        up.set_factor(i, FacMatrixT<ExecSpace>(u[i].nRows(), nc));
      up.set_factor(nd-1, FacMatrixT<ExecSpace>(algParams.window_size, nc));
      deep_copy(up.weights(), u.weights());
      window_idx = IndxArray(algParams.window_size);
      window_val = ArrayT<ExecSpace>(algParams.window_size);
      window_val_host = create_mirror_view(window_val);
      slice_idx = 0;

      // We rely on unused rows of window_val, up[nd-1] being zero,
      // so explicitly initialize them
      window_val = ttb_real(0.0);
      up[nd-1] = ttb_real(0.0);

      updateHistory(u);
    }

    // Adjust size of temporal mode of u to match X
    const ttb_indx nt = X.size(nd-1);
    if (u.ndims() != nd)
      Genten::error("Genten::online_gcp - u and x have different num dims");
    if (u[nd-1].nRows() > nt) {
      const ttb_indx nc = u.ncomponents();
      const ttb_indx s = u[nd-1].nRows();
      FacMatrixT<ExecSpace> ut(nt,nc);
      auto tt = Kokkos::subview(u[nd-1].view(),
                                std::pair<ttb_indx,ttb_indx>(s-nt,s),
                                Kokkos::ALL);
      deep_copy(ut.view(),tt);
      u.set_factor(nd-1, ut);
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
    typedef GCP::KokkosVector<ExecSpace> VectorType;
    typedef typename VectorType::view_type view_type;

    const ttb_indx nd = u.ndims();
    ttb_indx num_epoch = 0;

    if (print)
      out << "Solving for temporal mode..." << std::endl;
    if (temporalAlgParams.streaming_solver == GCP_Streaming_Solver::SGD) {
      // No history for temporal solver
      temporalSolver.reset();
      temporalSolver.solve(X, u, num_epoch, fest, out, false, false, print);
    }
    else if (temporalAlgParams.streaming_solver ==
             GCP_Streaming_Solver::LeastSquares ||
             temporalAlgParams.streaming_solver ==
             GCP_Streaming_Solver::OnlineCP)
      leastSquaresSolve(nd-1,X,u,fest,ften,out,print);
    else
      Genten::error(
        std::string("Unknown temporal streaming solver method ") +
        std::string(GCP_Streaming_Solver::names[temporalAlgParams.streaming_solver]));

    if (print)
      out << "Updating spatial modes..." << std::endl;
    if (spatialAlgParams.streaming_solver == GCP_Streaming_Solver::SGD)
      spatialSolver.solve(X, u, up, window_val, algParams.window_penalty,
                          num_epoch, fest, ften, out, false, false, print);
    else if (spatialAlgParams.streaming_solver ==
             GCP_Streaming_Solver::LeastSquares)
      for (ttb_indx i=0; i<nd-1; ++i)
        leastSquaresSolve(i,X,u,fest,ften,out,print);
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
      Genten::error(
        std::string("Unknown factor matrix solver method ") +
        std::string(GCP_Streaming_Solver::names[spatialAlgParams.streaming_solver]));

    // Update history window
    if (algParams.window_size > 0 &&
        spatialAlgParams.streaming_solver != GCP_Streaming_Solver::OnlineCP) {
      updateHistory(u);
    }
  }

  template <typename TensorT, typename ExecSpace, typename LossFunction>
  void
  OnlineGCP<TensorT,ExecSpace,LossFunction>::
  leastSquaresSolve(const ttb_indx mode,
                    TensorT& X,
                    KtensorT<ExecSpace>& u,
                    ttb_real& fest,
                    ttb_real& ften,
                    std::ostream& out,
                    const bool print)
  {
    // To do:
    //  * penalty, history
    //  * innerproduct trick
    //  * reg terms in fest
    const ttb_indx nd = u.ndims();
    const bool full = algParams.full_gram;
    A.oprod(u.weights());
    for (ttb_indx n=0; n<nd; ++n) {
      if (n != mode) {
        tmp = ttb_real(0.0);
        tmp.gramian(u[n], full, Upper);
        A.times(tmp);
      }
    }
    if (algParams.mttkrp_method == MTTKRP_Method::Perm &&
        !X.havePerm())
      X.createPermutation();
    mttkrp(X, u, mode, u[mode], algParams);
    u[mode].solveTransposeRHS(A, full, Upper, true, algParams);
    const ttb_real ip = innerprod(X, u);
    const ttb_real nrmx = X.norm();
    const ttb_real nrmusq = u.normFsq();
    fest = nrmx*nrmx + nrmusq - ttb_real(2.0)*ip;
    ften = fest;
    if (print)
      out << "f = " << fest << std::endl;
  }

  template <typename TensorT, typename ExecSpace, typename LossFunction>
  void
  OnlineGCP<TensorT,ExecSpace,LossFunction>::
  updateHistory(const KtensorT<ExecSpace>& u)
  {
    const ttb_indx nd = u.ndims();
    ttb_indx num_new = u[nd-1].nRows();

    // Copy spatial modes
    for (ttb_indx i=0; i<nd-1; ++i)
      deep_copy(up[i], u[i]);

    // First copy rows from u[nd-1] into up[nd-1] until history is full
    ttb_indx idx = 0;
    while (idx < num_new && slice_idx+idx < algParams.window_size) {
      auto sub1 = Kokkos::subview(up[nd-1].view(),slice_idx+idx,Kokkos::ALL);
      auto sub2 = Kokkos::subview(u[nd-1].view(),idx,Kokkos::ALL);
      deep_copy(sub1, sub2);
      window_idx[slice_idx+idx] = slice_idx+idx;
      ++idx;
    }

    // Copy in the rest of the rows in u[nd-1] depending on method
    if (idx < num_new) {
      num_new = num_new - idx;
      if (algParams.window_method == GCP_Streaming_Window_Method::Last) {
        // Shift each row of up[nd-1] down by num_new
        for (ttb_indx i=0; i<algParams.window_size-num_new; ++i) {
          auto sub1 = Kokkos::subview(up[nd-1].view(),i,Kokkos::ALL);
          auto sub2 = Kokkos::subview(up[nd-1].view(),i+num_new,Kokkos::ALL);
          deep_copy(sub1, sub2);
          window_idx[i] = window_idx[i+num_new];
        }
        // Copy last num_new rows u[nd-1] into last rows of up[nd-1]
        auto sub1 = Kokkos::subview(
          up[nd-1].view(),
          std::make_pair(algParams.window_size-num_new,algParams.window_size),
          Kokkos::ALL);
        auto sub2 = Kokkos::subview(
          u[nd-1].view(), std::make_pair(idx,idx+num_new), Kokkos::ALL);
        deep_copy(sub1, sub2);
        for (ttb_indx i=0; i<num_new; ++i)
          window_idx[algParams.window_size-num_new+i] = slice_idx+idx+i;
      }
      else if (algParams.window_method ==
               GCP_Streaming_Window_Method::Reservoir) {
        for (ttb_indx i=0; i<num_new; ++i) {
          // Random integer in [0,slice_idx+i)
          const ttb_indx j =
            ttb_indx(rand.genrnd_double()*double(slice_idx+i));

          // Replace entry j in history with slice_idx+i
          if (j < algParams.window_size) {
            auto sub1 = Kokkos::subview(up[nd-1].view(),j,Kokkos::ALL);
            auto sub2 = Kokkos::subview(u[nd-1].view(),idx+i,Kokkos::ALL);
            deep_copy(sub1, sub2);
            window_idx[j] = slice_idx+idx+i;
          }
        }
      }
      else
        Genten::error(
          std::string("Unsupported window method: ") +
          std::string(GCP_Streaming_Window_Method::names[algParams.window_method]));
    }

    // Update slice index counter
    slice_idx += u[nd-1].nRows();

    // Update weighting values
    for (ttb_indx i=0; i<algParams.window_size; ++i)
      window_val_host[i] =
        std::pow(algParams.window_weight, ttb_real(slice_idx-window_idx[i]));
    deep_copy(window_val, window_val_host);
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
      Xinit, u, loss_func, algParams, temporalAlgParams, spatialAlgParams, out);

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
      auto tm_row = Kokkos::subview(time_mode.view(),
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
    if (algParams.loss_function_type == GCP_LossFunction::Gaussian)
      online_gcp_impl(X, Xinit, u, GaussianLossFunction(algParams.loss_eps),
                      algParams, temporalAlgParams, spatialAlgParams,
                      out, fest, ften);
    else if (algParams.loss_function_type == GCP_LossFunction::Rayleigh)
      online_gcp_impl(X, Xinit, u, RayleighLossFunction(algParams.loss_eps),
                      algParams, temporalAlgParams, spatialAlgParams,
                      out, fest, ften);
    else if (algParams.loss_function_type == GCP_LossFunction::Gamma)
      online_gcp_impl(X, Xinit, u, GammaLossFunction(algParams.loss_eps),
                      algParams, temporalAlgParams, spatialAlgParams,
                      out, fest, ften);
    else if (algParams.loss_function_type == GCP_LossFunction::Bernoulli)
      online_gcp_impl(X, Xinit, u, BernoulliLossFunction(algParams.loss_eps),
                      algParams, temporalAlgParams, spatialAlgParams,
                      out, fest, ften);
    else if (algParams.loss_function_type == GCP_LossFunction::Poisson)
      online_gcp_impl(X, Xinit, u, PoissonLossFunction(algParams.loss_eps),
                      algParams, temporalAlgParams, spatialAlgParams,
                      out, fest, ften);
    else
       Genten::error("Genten::gcp_sgd - unknown loss function");
  }

}

#define LOSS_INST_MACRO(SPACE,LOSS)                                     \
  template class Genten::OnlineGCP<SptensorT<SPACE>,SPACE,LOSS>;

#define INST_MACRO(SPACE)                                               \
  LOSS_INST_MACRO(SPACE,GaussianLossFunction)                           \
  LOSS_INST_MACRO(SPACE,RayleighLossFunction)                           \
  LOSS_INST_MACRO(SPACE,GammaLossFunction)                              \
  LOSS_INST_MACRO(SPACE,BernoulliLossFunction)                          \
  LOSS_INST_MACRO(SPACE,PoissonLossFunction)                            \
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
