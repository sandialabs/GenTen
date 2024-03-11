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

#pragma once

#include "Genten_GCP_Sampler.hpp"
#include "Genten_AlgParams.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_GCP_SamplingKernels.hpp"
#include "Genten_GCP_ValueKernels.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_DistKtensorUpdate.hpp"

namespace Genten {

  template <typename TensorType, typename LossFunction>
  class DenseSampler : public Sampler<TensorType,LossFunction> {
  public:

    typedef Sampler<TensorType,LossFunction> base_type;
    typedef typename TensorType::exec_space exec_space;
    typedef typename base_type::pool_type pool_type;

    DenseSampler(const TensorType& X_,
                 const KtensorT<exec_space>& u,
                 const AlgParams& algParams_) :
      X(X_), algParams(algParams_), uh(u.ncomponents(),u.ndims())
    {
      if (!std::is_same<LossFunction, GaussianLossFunction>::value)
        Genten::error("Dense sampler only implemented for Gaussian loss!");

      dku = createKtensorUpdate(X, u, algParams);
      u_overlap = dku->createOverlapKtensor(u);
    }

    virtual ~DenseSampler()
    {
      delete dku;
    }

    virtual void initialize(const pool_type& rand_pool_,
                            const bool printitn,
                            std::ostream& out) override
    {
    }

    virtual ttb_indx getNumGradSamples() const override
    {
      return 0;
    }

    virtual void print(std::ostream& out) override
    {
      out << "Function sampler:  dense\n"
          << "Gradient sampler:  dense"
          << std::endl;
    }

    virtual void sampleTensorF(const KtensorT<exec_space>& u,
                               const LossFunction& loss_func) override
    {
    }

    virtual void sampleTensorG(const KtensorT<exec_space>& u,
                               const StreamingHistory<exec_space>& hist,
                               const LossFunction& loss_func) override
    {
    }

    virtual void prepareGradient(const KtensorT<exec_space>& g) override
    {
      if (g_overlap.isEmpty())
        g_overlap = dku->createOverlapKtensor(g);
    }

    virtual void value(const KtensorT<exec_space>& u,
                       const StreamingHistory<exec_space>& hist,
                       const ttb_real penalty,
                       const LossFunction& loss_func,
                       ttb_real& fest, ttb_real& ften) override
    {
      dku->doImport(u_overlap, u);

      const ttb_indx nd = u.ndims();
      const ttb_real ip = innerprod(X, u_overlap);
      const ttb_real nrmx = X.global_norm();
      const ttb_real nrmusq = u.normFsq();
      ften = nrmx*nrmx + nrmusq - ttb_real(2.0)*ip;
      fest = ften;
      if (hist.do_gcp_loss()) {
        // gcp-loss is the same as ktensor-fro here
        fest += hist.ktensor_fro_objective(u);
      }
      else
        fest += hist.objective(u);
      if (penalty != ttb_real(0.0)) {
        for (ttb_indx i=0; i<nd; ++i)
          fest += penalty * u[i].normFsq();
      }
    }

    virtual void gradient(const KtensorT<exec_space>& ut,
                          const StreamingHistory<exec_space>& hist,
                          const ttb_real penalty,
                          const LossFunction& loss_func,
                          KokkosVector<exec_space>& g,
                          const KtensorT<exec_space>& gt,
                          const ttb_indx mode_beg,
                          const ttb_indx mode_end,
                          SystemTimer& timer,
                          const int timer_init,
                          const int timer_nzs,
                          const int timer_zs,
                          const int timer_grad_mttkrp,
                          const int timer_grad_comm,
                          const int timer_grad_update) override
    {
      timer.start(timer_init);
      dku->initOverlapKtensor(g_overlap);
      timer.stop(timer_init);

      timer.start(timer_grad_comm);
      dku->doImport(u_overlap, ut);
      timer.stop(timer_grad_comm);

      timer.start(timer_grad_mttkrp);
      mttkrp_all(X, u_overlap, g_overlap, mode_beg, mode_end, algParams, false);
      timer.stop(timer_grad_mttkrp);

      timer.start(timer_grad_comm);
      dku->doExport(gt, g_overlap);
      timer.stop(timer_grad_comm);

      const ttb_indx nd = ut.ndims();
      const ttb_indx nc = ut.ncomponents();
      const bool full = true; // Needs to be full for later gemm
      FacMatrixT<exec_space> A(nc,nc);  // To do:  reuse (needs ut in cons.)
      FacMatrixT<exec_space> tmp(nc,nc);
      for (ttb_indx m=mode_beg; m<mode_end; ++m) {
        A.oprod(ut.weights());
        for (ttb_indx n=0; n<nd; ++n) {
          if (n != m) {
            tmp = ttb_real(0.0);
            tmp.gramian(ut[n], full, Upper);
            A.times(tmp);
          }
        }
        if (penalty != 0.0)
          A.diagonalShift(penalty);
        gt[m-mode_beg].gemm(false, false, ttb_real(2.0), ut[m], A,
                            ttb_real(-2.0));
      }
      if (hist.do_gcp_loss()) {
        // gcp-loss is the same as ktensor-fro here
        hist.ktensor_fro_gradient(ut, mode_beg, mode_end, gt);
      }
      else
        hist.gradient(ut, mode_beg, mode_end, gt);
    }

  protected:

    TensorType X;
    AlgParams algParams;
    KtensorT<exec_space> uh;
    std::vector< FacMatrixT<exec_space> > Z1, Z2;
    FacMatrixT<exec_space> tmp, ZZ1, ZZ2;
    KtensorT<exec_space> u_overlap;
    KtensorT<exec_space> g_overlap;
    DistKtensorUpdate<exec_space> *dku;
  };

}
