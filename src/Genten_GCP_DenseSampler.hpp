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

namespace Genten {

  template <typename ExecSpace, typename LossFunction>
  class DenseSampler : public Sampler<ExecSpace,LossFunction> {
  public:

    typedef Sampler<ExecSpace,LossFunction> base_type;
    typedef typename base_type::pool_type pool_type;
    typedef typename base_type::map_type map_type;

    DenseSampler(const SptensorT<ExecSpace>& X_,
                      const AlgParams& algParams_) :
      X(X_), algParams(algParams_), uh(algParams_.rank,X.ndims())
    {
      if (!std::is_same<LossFunction, GaussianLossFunction>::value)
        Genten::error("Dense sampler only implemented for Gaussian loss!");
    }

    virtual ~DenseSampler() {}

    virtual void initialize(const pool_type& rand_pool_,
                            const bool printitn,
                            std::ostream& out) override
    {
    }

    virtual void print(std::ostream& out) override
    {
      out << "Function sampler:  dense\n"
          << "Gradient sampler:  dense"
          << std::endl;
    }

    virtual void sampleTensorF(const KtensorT<ExecSpace>& u,
                               const LossFunction& loss_func) override
    {
    }

    virtual void sampleTensorG(const KtensorT<ExecSpace>& u,
                               const StreamingHistory<ExecSpace>& hist,
                               const LossFunction& loss_func) override
    {
    }

    virtual void prepareGradient() override
    {
    }

    virtual void value(const KtensorT<ExecSpace>& u,
                       const StreamingHistory<ExecSpace>& hist,
                       const ttb_real penalty,
                       const LossFunction& loss_func,
                       ttb_real& fest, ttb_real& ften) override
    {
      const ttb_indx nd = u.ndims();
      const ttb_indx nc = u.ncomponents();
      const ttb_real ip = innerprod(X, u);
      const ttb_real nrmx = X.norm();
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

    virtual void gradient(const KtensorT<ExecSpace>& ut,
                          const StreamingHistory<ExecSpace>& hist,
                          const ttb_real penalty,
                          const LossFunction& loss_func,
                          KokkosVector<ExecSpace>& g,
                          const KtensorT<ExecSpace>& gt,
                          const ttb_indx mode_beg,
                          const ttb_indx mode_end,
                          SystemTimer& timer,
                          const int timer_init,
                          const int timer_nzs,
                          const int timer_zs) override
    {
      timer.start(timer_init);
      gt.weights() = ttb_real(1.0);
      g.zero();
      timer.stop(timer_init);

      mttkrp_all(X, ut, gt, mode_beg, mode_end, algParams, false);
      const ttb_indx nd = ut.ndims();
      const ttb_indx nc = ut.ncomponents();
      const bool full = true; // Needs to be full for later gemm
      FacMatrixT<ExecSpace> A(nc,nc);  // To do:  reuse (needs ut in cons.)
      FacMatrixT<ExecSpace> tmp(nc,nc);
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

    SptensorT<ExecSpace> X;
    AlgParams algParams;
    KtensorT<ExecSpace> uh;
    std::vector< FacMatrixT<ExecSpace> > Z1, Z2;
    FacMatrixT<ExecSpace> tmp, ZZ1, ZZ2;
  };

}
