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

#include <ostream>

#include "Genten_GCP_Sampler.hpp"
#include "Genten_GCP_SGD_Step.hpp"
#include "Genten_KokkosVector.hpp"
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_AlgParams.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_MixedFormatOps.hpp"

namespace Genten {

  namespace Impl {

    template <typename ExecSpace, typename LossFunction>
    class GCP_SGD_Iter {
    public:
      typedef KokkosVector<ExecSpace> VectorType;

      GCP_SGD_Iter(const KtensorT<ExecSpace>& u0,
                   const AlgParams& algParams_) :
        algParams(algParams_)
      {
        // Setup timers
        int num_timers = 0;
        timer_sample_g = num_timers++;
        timer_grad = num_timers++;
        timer_grad_nzs = num_timers++;
        timer_grad_zs = num_timers++;
        timer_grad_init = num_timers++;
        timer_step = num_timers++;
        timer_sample_g_z_nz = num_timers++;
        timer_sample_g_perm = num_timers++;
        timer.init(num_timers, algParams.timings);

        // Ktensor-vector for solution
        u = VectorType(u0);
        u.copyFromKtensor(u0);
        ut = u.getKtensor();

        // Gradient Ktensor
        g = u.clone();
        gt = g.getKtensor();
      }

      virtual ~GCP_SGD_Iter() {}

      virtual VectorType getSolution() const { return u; }

      virtual void run(SptensorT<ExecSpace>& X,
                       const LossFunction& loss_func,
                       Sampler<ExecSpace,LossFunction>& sampler,
                       GCP_SGD_Step<ExecSpace,LossFunction>& stepper,
                       ttb_indx& total_iters)
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

        total_iters += algParams.epoch_iters*algParams.frozen_iters;
      }

      virtual void printTimers(std::ostream& out) const
      {
        if (!algParams.fuse) {
          out << "\tsample-g:  "
              << timer.getTotalTime(timer_sample_g)
              << " seconds\n"
              << "\t\tzs/nzs:   "
              << timer.getTotalTime(timer_sample_g_z_nz)
              << " seconds\n";
          if (algParams.mttkrp_method == MTTKRP_Method::Perm &&
              algParams.mttkrp_all_method == MTTKRP_All_Method::Iterated) {
            out << "\t\tperm:     "
                << timer.getTotalTime(timer_sample_g_perm)
                << " seconds\n";
          }
        }
        out << "\tgradient:  " << timer.getTotalTime(timer_grad)
            << " seconds\n";
        if (algParams.fuse) {
          out << "\t\tinit:    " << timer.getTotalTime(timer_grad_init)
              << " seconds\n"
              << "\t\tnzs:     " << timer.getTotalTime(timer_grad_nzs)
              << " seconds\n"
              << "\t\tzs:      " << timer.getTotalTime(timer_grad_zs)
              << " seconds\n";
        }
        out << "\tstep/clip: " << timer.getTotalTime(timer_step)
            << " seconds\n";
      }

    protected:
      AlgParams algParams;

      int timer_sample_g;
      int timer_grad;
      int timer_grad_nzs;
      int timer_grad_zs;
      int timer_grad_init;
      int timer_step;
      int timer_sample_g_z_nz;
      int timer_sample_g_perm;
      SystemTimer timer;

      VectorType u;
      VectorType g;
      KtensorT<ExecSpace> ut;
      KtensorT<ExecSpace> gt;

      SptensorT<ExecSpace> X_grad;
      ArrayT<ExecSpace> w_grad;
    };

  }

}
