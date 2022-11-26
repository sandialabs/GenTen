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
#include "Genten_GCP_SS_Grad.hpp"
#include "Genten_GCP_SS_Grad_SA.hpp"
#include "Genten_KokkosVector.hpp"

// to do:
//   * replace modes array with mode_beg, mode_end

namespace Genten {

  template <typename ExecSpace, typename LossFunction>
  class SemiStratifiedSampler : public Sampler<ExecSpace,LossFunction> {
  public:

    typedef Sampler<ExecSpace,LossFunction> base_type;
    typedef typename base_type::pool_type pool_type;
    typedef typename base_type::map_type map_type;

    SemiStratifiedSampler(const SptensorT<ExecSpace>& X_,
                          const AlgParams& algParams_,
                          bool distribute_num_samples) :
      X(X_), algParams(algParams_), uh(algParams_.rank,X.ndims())
    {
      global_num_samples_nonzeros_value = algParams.num_samples_nonzeros_value;
      global_num_samples_zeros_value = algParams.num_samples_zeros_value;
      global_num_samples_nonzeros_grad = algParams.num_samples_nonzeros_grad;
      global_num_samples_zeros_grad = algParams.num_samples_zeros_grad;
      weight_nonzeros_value = algParams.w_f_nz;
      weight_zeros_value = algParams.w_f_z;
      weight_nonzeros_grad = algParams.w_g_nz;
      weight_zeros_grad = algParams.w_g_z;

      // Compute number of samples if necessary
      const ttb_indx nnz = X.global_nnz();
      const ttb_real tsz = X.global_numel_float();
      const ttb_real nz = tsz - nnz;
      const ttb_indx maxEpochs = algParams.maxiters;
      const ttb_indx ftmp = std::max((nnz+99)/100,ttb_indx(100000));
      const ttb_indx gtmp = std::max((3*nnz+maxEpochs-1)/maxEpochs,
                                     ttb_indx(1000));
      if (global_num_samples_nonzeros_value == 0)
        global_num_samples_nonzeros_value = std::min(ftmp, nnz);
      if (global_num_samples_zeros_value == 0)
        global_num_samples_zeros_value =
          ttb_indx(std::min(ttb_real(global_num_samples_nonzeros_value), nz));
      if (global_num_samples_nonzeros_grad == 0)
        global_num_samples_nonzeros_grad = std::min(gtmp, nnz);
      if (global_num_samples_zeros_grad == 0)
        global_num_samples_zeros_grad =
          ttb_indx(std::min(ttb_real(global_num_samples_nonzeros_grad), nz));

      // Compute local number of samples by distributing them evenly across
      // processors (might be better to weight according to number of nonzeros)
      const ProcessorMap* pmap = X.getProcessorMap();
      const ttb_indx lnnz = X.nnz();
      const ttb_real lsz = X.numel_float();
      const ttb_real lnz = lsz - lnnz;
      const ttb_indx np = pmap != nullptr ? pmap->gridSize() : 1;
      num_samples_nonzeros_value = global_num_samples_nonzeros_value / np;
      num_samples_zeros_value = global_num_samples_zeros_value / np;
      num_samples_nonzeros_grad = global_num_samples_nonzeros_grad / np;
      num_samples_zeros_grad = global_num_samples_zeros_grad / np;

      // Don't sample more zeros/nonzeros than we actually have locally
      num_samples_nonzeros_value = std::min(num_samples_nonzeros_value, lnnz);
      num_samples_zeros_value = std::min(ttb_real(num_samples_zeros_value),
                                         lnz);
      num_samples_nonzeros_grad = std::min(num_samples_nonzeros_grad, lnnz);
      num_samples_zeros_grad = std::min(ttb_real(num_samples_zeros_grad),
                                        lnz);

      // Compute global number of samples actually used
      if (pmap != nullptr) {
        global_num_samples_nonzeros_value =
          pmap->gridAllReduce(num_samples_nonzeros_value);
        global_num_samples_zeros_value =
          pmap->gridAllReduce(num_samples_zeros_value);
        global_num_samples_nonzeros_grad =
          pmap->gridAllReduce(num_samples_nonzeros_grad);
        global_num_samples_zeros_grad =
          pmap->gridAllReduce(num_samples_zeros_grad);
      }
      else {
        global_num_samples_nonzeros_value = num_samples_nonzeros_value;
        global_num_samples_zeros_value = num_samples_zeros_value;
        global_num_samples_nonzeros_grad = num_samples_nonzeros_grad;
        global_num_samples_zeros_grad = num_samples_zeros_grad;
      }

      // Compute weights if necessary
      if (weight_nonzeros_value < 0.0)
        weight_nonzeros_value = global_num_samples_nonzeros_value == 0 ? 0.0 :
          ttb_real(nnz) / ttb_real(global_num_samples_nonzeros_value);
      if (weight_zeros_value < 0.0)
        weight_zeros_value = global_num_samples_zeros_value == 0 ? 0.0 :
          ttb_real(tsz-nnz) / ttb_real(global_num_samples_zeros_value);
      if (weight_nonzeros_grad < 0.0)
        weight_nonzeros_grad = global_num_samples_nonzeros_grad == 0 ? 0.0 :
          ttb_real(nnz) / ttb_real(global_num_samples_nonzeros_grad);
      if (weight_zeros_grad < 0.0)
        weight_zeros_grad = global_num_samples_zeros_grad == 0 ? 0.0 :
          ttb_real(tsz) / ttb_real(global_num_samples_zeros_grad);

      nz_percent =
        ttb_real(global_num_samples_nonzeros_grad * algParams.epoch_iters) /
        ttb_real(nnz) * ttb_real(100.0);
    }

    virtual ~SemiStratifiedSampler() {}

    virtual void initialize(const pool_type& rand_pool_,
                            const bool printitn,
                            std::ostream& out) override
    {
      rand_pool = rand_pool_;

      // Sort/hash tensor if necessary for faster sampling
      if (printitn > 0) {
        if (algParams.hash)
          out << "Hashing tensor for faster sampling...";
        else
          out << "Sorting tensor for faster sampling...";
      }
      SystemTimer timer(1, algParams.timings);
      timer.start(0);
      if (algParams.hash)
        hash_map = this->buildHashMap(X,out);
      else if (!X.isSorted())
        X.sort();
      timer.stop(0);
      if (printitn > 0)
        out << timer.getTotalTime(0) << " seconds" << std::endl;
    }

    virtual void print(std::ostream& out) override
    {
      out << "  Function sampler:  stratified with "
          << global_num_samples_nonzeros_value
          << " nonzero and " << global_num_samples_zeros_value
          << " zero samples\n"
          << "  Gradient sampler:  semi-stratified with "
          << global_num_samples_nonzeros_grad
          << " nonzero and " << global_num_samples_zeros_grad
          << " zero samples\n"
          << "  Gradient nonzero samples per epoch: "
          << global_num_samples_nonzeros_grad*algParams.epoch_iters
          << " (" << std::setprecision(1) << std::fixed << nz_percent << "%)"
          << std::endl;
    }

    virtual void sampleTensorF(const KtensorT<ExecSpace>& u,
                               const LossFunction& loss_func) override
    {
      if (algParams.hash)
        Impl::stratified_sample_tensor_hash(
          X, hash_map,
          num_samples_nonzeros_value, num_samples_zeros_value,
          weight_nonzeros_value, weight_zeros_value,
          u, loss_func, false,
          Yf, wf, rand_pool, algParams);
      else
        Impl::stratified_sample_tensor(
          X, num_samples_nonzeros_value, num_samples_zeros_value,
          weight_nonzeros_value, weight_zeros_value,
          u, loss_func, false,
          Yf, wf, rand_pool, algParams);
    }

    virtual void sampleTensorG(const KtensorT<ExecSpace>& u,
                               const StreamingHistory<ExecSpace>& hist,
                               const LossFunction& loss_func) override
    {
      if (!algParams.fuse) {
        Impl::semi_stratified_sample_tensor(
          X, num_samples_nonzeros_grad, num_samples_zeros_grad,
          weight_nonzeros_grad, weight_zeros_grad,
          u, loss_func, true,
          Yg, wg, rand_pool, algParams);

        if (hist.do_gcp_loss()) {
          // Create uh, u with time mode replaced by time mode of up
          // This should all just be view assignments, so should be fast
          KtensorT<ExecSpace> uh;
          uh.weights() = u.weights();
          const ttb_indx nd = u.ndims();
          for (ttb_indx i=0; i<nd-1; ++i)
            uh.set_factor(i, u[i]);
          uh.set_factor(nd-1, hist.up[nd-1]);

          Impl::stratified_ktensor_grad(
            Yg, num_samples_nonzeros_grad, num_samples_zeros_grad,
            weight_nonzeros_grad, weight_zeros_grad,
            uh, hist.up, hist.window_val, hist.window_penalty, loss_func,
            Yh, algParams);
        }
      }
    }

    virtual void prepareGradient() override
    {
      if (!algParams.fuse &&
          algParams.mttkrp_method == MTTKRP_Method::Perm &&
          algParams.mttkrp_all_method == MTTKRP_All_Method::Iterated) {
        Yg.createPermutation();
        if (Yh.nnz() > 0)
          Yh.createPermutation();
      }
    }

    virtual void value(const KtensorT<ExecSpace>& u,
                       const StreamingHistory<ExecSpace>& hist,
                       const ttb_real penalty,
                       const LossFunction& loss_func,
                       ttb_real& fest, ttb_real& ften) override
    {
      if (!hist.do_gcp_loss()) {
        ften = Impl::gcp_value(Yf, u, wf, loss_func);
        fest = ften + hist.objective(u);
      }
      else {
        ttb_real fhis = 0.0;
        Impl::gcp_value(Yf, u, hist.up, hist.window_val, hist.window_penalty,
                        wf, loss_func, ften, fhis);
        fest = ften + fhis;
      }
      if (penalty != ttb_real(0.0)) {
        const ttb_indx nd = u.ndims();
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

      if (algParams.fuse) {
        if (!hist.do_gcp_loss()) {
          Impl::gcp_sgd_ss_grad(
            X, ut, loss_func,
            num_samples_nonzeros_grad, num_samples_zeros_grad,
            weight_nonzeros_grad, weight_zeros_grad,
            gt, rand_pool, algParams,
            timer, timer_nzs, timer_zs);
          hist.gradient(ut, mode_beg, mode_end, gt);
        }
        else {
          // Create modes array
          IndxArrayT<ExecSpace> modes(mode_end-mode_beg);
          auto modes_host = create_mirror_view(modes);
          for (ttb_indx i=mode_beg; i<mode_end; ++i)
            modes_host[i-mode_beg] = i;
          deep_copy(modes, modes_host);

          // Create uh, u with time mode replaced by time mode of up
          // This should all just be view assignments, so should be fast
          uh.weights() = ut.weights();
          const ttb_indx nd = ut.ndims();
          for (ttb_indx i=0; i<nd-1; ++i)
            uh.set_factor(i, ut[i]);
          uh.set_factor(nd-1, hist.up[nd-1]);

          Impl::gcp_sgd_ss_grad(
            X, ut, uh, hist.up, loss_func,
            num_samples_nonzeros_grad, num_samples_zeros_grad,
            weight_nonzeros_grad, weight_zeros_grad,
            hist.window_val, hist.window_penalty, modes,
            gt, rand_pool, algParams,
            timer, timer_nzs, timer_zs);
        }
      }
      else {
        mttkrp_all(Yg, ut, gt, mode_beg, mode_end, algParams, false);
        if (Yh.nnz() > 0) {
          // Create uh, u with time mode replaced by time mode of up
          // This should all just be view assignments, so should be fast
          uh.weights() = ut.weights();
          const ttb_indx nd = ut.ndims();
          for (ttb_indx i=0; i<nd-1; ++i)
            uh.set_factor(i, ut[i]);
          uh.set_factor(nd-1, hist.up[nd-1]);

          mttkrp_all(Yh, uh, gt, mode_beg, mode_end, algParams, false);
        }
        else
          hist.gradient(ut, mode_beg, mode_end, gt);
      }

      if (penalty != 0.0)
        for (ttb_indx i=mode_beg; i<mode_end; ++i)
          gt[i-mode_beg].plus(ut[i], ttb_real(2.0)*penalty);
    }

    ttb_indx totalNumGradSamples() const {
      return num_samples_nonzeros_grad + num_samples_zeros_grad;
    }

    void fusedGradientAndStep(const KokkosVector<ExecSpace>& u,
                              const LossFunction& loss_func,
                              KokkosVector<ExecSpace>& g,
                              const KtensorT<ExecSpace>& gt,
                              const Kokkos::View<ttb_indx**,Kokkos::LayoutLeft,ExecSpace>& gind,
                              const Kokkos::View<ttb_indx*,ExecSpace>& perm,
                              const bool use_adam,
                              const KokkosVector<ExecSpace>& adam_m,
                              const KokkosVector<ExecSpace>& adam_v,
                              const ttb_real beta1,
                              const ttb_real beta2,
                              const ttb_real eps,
                              const ttb_real step,
                              const bool has_bounds,
                              const ttb_real lb,
                              const ttb_real ub,
                              SystemTimer& timer,
                              const int timer_init,
                              const int timer_nzs,
                              const int timer_zs,
                              const int timer_sort,
                              const int timer_scan,
                              const int timer_step)
    {
      timer.start(timer_init);
      gt.weights() = 1.0;
      g.zero();
      timer.stop(timer_init);

      Impl::gcp_sgd_ss_grad_sa(
        X, u, loss_func,
        num_samples_nonzeros_grad, num_samples_zeros_grad,
        weight_nonzeros_grad, weight_zeros_grad,
        g, gind, perm, use_adam, adam_m, adam_v, beta1, beta2, eps, step,
        has_bounds, lb, ub,
        rand_pool, algParams,
        timer, timer_nzs, timer_zs, timer_sort, timer_scan, timer_step);
    }

    pool_type& getRandPool() { return rand_pool; }
    ttb_indx getNumSamplesZerosGrad() const { return num_samples_zeros_grad; }
    ttb_indx getNumSamplesNonzerosGrad() const { return num_samples_nonzeros_grad; }
    ttb_real getWeightZerosGrad() const { return weight_zeros_grad; }
    ttb_real getWeightNonzerosGrad() const { return weight_nonzeros_grad; }

  protected:

    SptensorT<ExecSpace> X;
    SptensorT<ExecSpace> Yf;
    SptensorT<ExecSpace> Yg;
    SptensorT<ExecSpace> Yh;
    ArrayT<ExecSpace> wf;
    ArrayT<ExecSpace> wg;
    pool_type rand_pool;
    AlgParams algParams;
    ttb_indx num_samples_nonzeros_value;
    ttb_indx num_samples_zeros_value;
    ttb_indx num_samples_nonzeros_grad;
    ttb_indx num_samples_zeros_grad;
    ttb_indx global_num_samples_nonzeros_value;
    ttb_indx global_num_samples_zeros_value;
    ttb_indx global_num_samples_nonzeros_grad;
    ttb_indx global_num_samples_zeros_grad;
    ttb_real weight_nonzeros_value;
    ttb_real weight_zeros_value;
    ttb_real weight_nonzeros_grad;
    ttb_real weight_zeros_grad;
    ttb_real nz_percent;
    map_type hash_map;
    KtensorT<ExecSpace> uh;
  };

}
