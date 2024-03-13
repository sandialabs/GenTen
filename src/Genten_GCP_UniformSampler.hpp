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
  class UniformSampler {};

  template <typename ExecSpace, typename LossFunction>
  class UniformSampler<SptensorT<ExecSpace>,LossFunction> :
    public Sampler<SptensorT<ExecSpace>,LossFunction> {
  public:

    typedef Sampler<SptensorT<ExecSpace>,LossFunction> base_type;
    typedef typename base_type::pool_type pool_type;
    typedef typename base_type::map_type map_type;

    UniformSampler(const SptensorT<ExecSpace>& X_,
                   const KtensorT<ExecSpace>& u,
                   const AlgParams& algParams_) :
      X(X_), algParams(algParams_), uh(u.ncomponents(),u.ndims())
    {
      global_num_samples_nonzeros_value = algParams.num_samples_nonzeros_value;
      global_num_samples_zeros_value = algParams.num_samples_zeros_value;
      global_num_samples_grad = algParams.num_samples_nonzeros_grad;
      weight_nonzeros_value = algParams.w_f_nz;
      weight_zeros_value = algParams.w_f_z;
      weight_grad = algParams.w_g_nz;

      // Compute number of samples if necessary
      const ttb_indx nnz = X.global_nnz();
      const ttb_real tsz = X.global_numel_float();
      const ttb_real nz = tsz - nnz;
      const ttb_indx maxEpochs = algParams.maxiters;
      const ttb_indx ftmp = std::max((nnz+99)/100,ttb_indx(100000));
      if (global_num_samples_nonzeros_value == 0)
        global_num_samples_nonzeros_value = std::min(ftmp, nnz);
      else if (global_num_samples_nonzeros_value == INT_MAX)
        global_num_samples_nonzeros_value = nnz;

      if (global_num_samples_zeros_value == 0)
        global_num_samples_zeros_value =
          ttb_indx(std::min(ttb_real(global_num_samples_nonzeros_value), nz));
      else if (global_num_samples_zeros_value == INT_MAX)
        global_num_samples_zeros_value = nz;

      if (global_num_samples_grad == 0)
        global_num_samples_grad =
          ttb_indx(std::min(std::max(ttb_real(10.0)*tsz/maxEpochs,
                                     ttb_real(1e3)), tsz));
      else if (global_num_samples_grad == INT_MAX)
        global_num_samples_grad = ttb_indx(tsz);

      // Compute local number of samples by distributing them evenly across
      // processors (might be better to weight according to number of nonzeros)
      const ProcessorMap* pmap = X.getProcessorMap();
      const ttb_indx lnnz = X.nnz();
      const ttb_real lsz = X.numel_float();
      const ttb_real lnz = lsz - lnnz;
      const ttb_indx np = pmap != nullptr ? pmap->gridSize() : 1;
      num_samples_nonzeros_value = static_cast<ttb_indx>(global_num_samples_nonzeros_value * (static_cast<ttb_real>(lnnz)/static_cast<ttb_real>(nnz)));
      num_samples_zeros_value = static_cast<ttb_indx>(global_num_samples_zeros_value * (lnz/nz));
      num_samples_grad = static_cast<ttb_indx>(global_num_samples_grad * (lsz/tsz));

      // Adjust number of samples uniformly across procs to handle remainder.
      // This is not perfect as the number of samples is locally limited by the
      // the number of local nonzeros/zeros.
      ttb_indx rank = pmap != nullptr ? pmap->gridRank() : 0;
      if (rank < global_num_samples_nonzeros_value-num_samples_nonzeros_value*np)
        ++num_samples_nonzeros_value;
      if (rank < global_num_samples_zeros_value-num_samples_zeros_value*np)
        ++num_samples_zeros_value;
      if (rank < global_num_samples_grad-num_samples_grad*np)
        ++num_samples_grad;

      // Don't sample more zeros/nonzeros than we actually have locally
      num_samples_nonzeros_value = std::min(num_samples_nonzeros_value, lnnz);
      num_samples_zeros_value = std::min(num_samples_zeros_value,
                                         ttb_indx(lnz));
      num_samples_grad = std::min(num_samples_grad, ttb_indx(lsz));

      // Compute global number of samples actually used
      if (pmap != nullptr) {
        global_num_samples_nonzeros_value =
          pmap->gridAllReduce(num_samples_nonzeros_value);
        global_num_samples_zeros_value =
          pmap->gridAllReduce(num_samples_zeros_value);
        global_num_samples_grad =
          pmap->gridAllReduce(num_samples_grad);
      }
      else {
        global_num_samples_nonzeros_value = num_samples_nonzeros_value;
        global_num_samples_zeros_value = num_samples_zeros_value;
        global_num_samples_grad = num_samples_grad;
      }

      // Compute weights if necessary
      if (weight_nonzeros_value < 0.0)
        weight_nonzeros_value = global_num_samples_nonzeros_value == 0 ? 0.0 :
          ttb_real(nnz) / ttb_real(global_num_samples_nonzeros_value);
      if (weight_zeros_value < 0.0)
        weight_zeros_value = global_num_samples_zeros_value == 0 ? 0.0 :
          ttb_real(tsz-nnz) / ttb_real(global_num_samples_zeros_value);
      if (weight_grad < 0.0)
        weight_grad = global_num_samples_grad == 0 ? 0.0 :
          tsz / ttb_real(global_num_samples_grad);

      grad_percent = ttb_real(global_num_samples_grad * algParams.epoch_iters) /
        ttb_real(tsz) * ttb_real(100.0);

      dku_F = createKtensorUpdate(Yf, u, algParams);
      dku_G = createKtensorUpdate(Yg, u, algParams);

      if (algParams.dist_update_method != Dist_Update_Method::Tpetra) {
        u_overlap_F = dku_F->createOverlapKtensor(u);
        u_overlap_G = dku_G->createOverlapKtensor(u);
      }
    }

    virtual ~UniformSampler()
    {
      delete dku_F;
      delete dku_G;
    }

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

    virtual ttb_indx getNumGradSamples() const override
    {
      return global_num_samples_grad;
    }

    virtual void print(std::ostream& out) override
    {
      out << "  Function sampler:  stratified with "
          << global_num_samples_nonzeros_value
          << " nonzero and " << global_num_samples_zeros_value
          << " zero samples\n"
          << "  Gradient sampler:  uniform with " << global_num_samples_grad
          << " samples\n"
          << "  Gradient samples per epoch: "
          << global_num_samples_grad*algParams.epoch_iters
          << " (" << std::setprecision(1) << std::fixed << grad_percent << "%)"
          << std::endl;
    }

    virtual void sampleTensorF(const KtensorT<ExecSpace>& u,
                               const LossFunction& loss_func) override
    {
      if (algParams.dist_update_method == Dist_Update_Method::Tpetra) {
        if (algParams.hash)
          Impl::stratified_sample_tensor_tpetra(
            X, Impl::HashSearcher<ExecSpace>(this->X.impl(), hash_map),
            num_samples_nonzeros_value, num_samples_zeros_value,
            weight_nonzeros_value, weight_zeros_value,
            u, Impl::StratifiedGradient<LossFunction>(loss_func), false,
            Yf, wf, u_overlap_F, rand_pool, algParams);
        else
          Impl::stratified_sample_tensor_tpetra(
            X, Impl::SortSearcher<ExecSpace>(this->X.impl()),
            num_samples_nonzeros_value, num_samples_zeros_value,
            weight_nonzeros_value, weight_zeros_value,
            u, Impl::StratifiedGradient<LossFunction>(loss_func), false,
            Yf, wf, u_overlap_F, rand_pool, algParams);
      }
      else if (algParams.dist_update_method == Dist_Update_Method::OneSided ||
               algParams.dist_update_method == Dist_Update_Method::TwoSided) {
        if (algParams.hash)
          Impl::stratified_sample_tensor_onesided(
            X, Impl::HashSearcher<ExecSpace>(this->X.impl(), hash_map),
            num_samples_nonzeros_value, num_samples_zeros_value,
            weight_nonzeros_value, weight_zeros_value,
            u, Impl::StratifiedGradient<LossFunction>(loss_func), false,
            Yf, wf, *dku_F, u_overlap_F, rand_pool, algParams);
        else
          Impl::stratified_sample_tensor_onesided(
            X, Impl::SortSearcher<ExecSpace>(this->X.impl()),
            num_samples_nonzeros_value, num_samples_zeros_value,
            weight_nonzeros_value, weight_zeros_value,
            u, Impl::StratifiedGradient<LossFunction>(loss_func), false,
            Yf, wf, *dku_F, u_overlap_F, rand_pool, algParams);
      }
      else {
        dku_F->doImport(u_overlap_F, u);
        if (algParams.hash)
          Impl::stratified_sample_tensor(
            X, Impl::HashSearcher<ExecSpace>(this->X.impl(), hash_map),
            num_samples_nonzeros_value, num_samples_zeros_value,
            weight_nonzeros_value, weight_zeros_value,
            u_overlap_F, Impl::StratifiedGradient<LossFunction>(loss_func),
            false, Yf, wf, rand_pool, algParams);
        else
          Impl::stratified_sample_tensor(
            X, Impl::SortSearcher<ExecSpace>(this->X.impl()),
            num_samples_nonzeros_value, num_samples_zeros_value,
            weight_nonzeros_value, weight_zeros_value,
            u_overlap_F, Impl::StratifiedGradient<LossFunction>(loss_func),
            false, Yf, wf, rand_pool, algParams);
      }

      if (algParams.dist_update_method != Dist_Update_Method::OneSided &&
          algParams.dist_update_method != Dist_Update_Method::TwoSided)
        dku_F->updateTensor(Yf);
    }

    virtual void sampleTensorG(const KtensorT<ExecSpace>& u,
                               const StreamingHistory<ExecSpace>& hist,
                               const LossFunction& loss_func) override
    {
      if (algParams.dist_update_method == Dist_Update_Method::Tpetra) {
        if (algParams.hash)
          Impl::uniform_sample_tensor_tpetra(
            X, Impl::HashSearcher<ExecSpace>(this->X.impl(), hash_map),
            num_samples_grad, weight_grad, u, loss_func, true,
            Yg, wg, u_overlap_G, rand_pool, algParams);
        else
          Impl::uniform_sample_tensor_tpetra(
            X, Impl::SortSearcher<ExecSpace>(this->X.impl()),
            num_samples_grad, weight_grad, u, loss_func, true,
            Yg, wg, u_overlap_G, rand_pool, algParams);
      }
      else if (algParams.dist_update_method == Dist_Update_Method::OneSided ||
               algParams.dist_update_method == Dist_Update_Method::TwoSided) {
        if (algParams.hash)
          Impl::uniform_sample_tensor_onesided(
            X, Impl::HashSearcher<ExecSpace>(this->X.impl(), hash_map),
            num_samples_grad, weight_grad, u, loss_func, true,
            Yg, wg, *dku_G, u_overlap_G, rand_pool, algParams);
        else
          Impl::uniform_sample_tensor_onesided(
            X, Impl::SortSearcher<ExecSpace>(this->X.impl()),
            num_samples_grad, weight_grad, u, loss_func, true,
            Yg, wg, *dku_G, u_overlap_G, rand_pool, algParams);
      }
      else {
        dku_G->doImport(u_overlap_G, u);
        if (algParams.hash)
          Impl::uniform_sample_tensor(
            X, Impl::HashSearcher<ExecSpace>(this->X.impl(), hash_map),
            num_samples_grad, weight_grad, u_overlap_G, loss_func, true,
            Yg, wg, rand_pool, algParams);
        else
          Impl::uniform_sample_tensor(
            X, Impl::SortSearcher<ExecSpace>(this->X.impl()),
            num_samples_grad, weight_grad, u_overlap_G, loss_func, true,
            Yg, wg, rand_pool, algParams);
      }

      if (hist.do_gcp_loss()) {
        // Create uh, u with time mode replaced by time mode of up
        // This should all just be view assignments, so should be fast
        uh.weights() = u_overlap_G.weights();
        const ttb_indx nd = u.ndims();
        for (ttb_indx i=0; i<nd-1; ++i)
          uh.set_factor(i, u_overlap_G[i]);
        uh.set_factor(nd-1, hist.up[nd-1]);

        Impl::stratified_ktensor_grad(
          Yg, num_samples_grad, ttb_indx(0),
          weight_grad, ttb_real(0.0),
          uh, hist.up, hist.window_val, hist.window_penalty, loss_func,
          Yh, algParams);
      }

      if (algParams.dist_update_method != Dist_Update_Method::OneSided &&
          algParams.dist_update_method != Dist_Update_Method::TwoSided)
        dku_G->updateTensor(Yg);
    }

    virtual void prepareGradient(const KtensorT<ExecSpace>& gt) override
    {
      if (algParams.mttkrp_method == MTTKRP_Method::Perm &&
          algParams.mttkrp_all_method == MTTKRP_All_Method::Iterated) {
        Yg.createPermutation();
        if (Yh.nnz() > 0)
          Yh.createPermutation();
      }
      if (gt_overlap.isEmpty() || dku_G->overlapDependsOnTensor())
        gt_overlap = dku_G->createOverlapKtensor(gt);
    }

    virtual void value(const KtensorT<ExecSpace>& u,
                       const StreamingHistory<ExecSpace>& hist,
                       const ttb_real penalty,
                       const LossFunction& loss_func,
                       ttb_real& fest, ttb_real& ften) override
    {
      dku_F->doImport(u_overlap_F, u);

      if (!hist.do_gcp_loss()) {
        ften = Impl::gcp_value(Yf, u_overlap_F, wf, loss_func);
        fest = ften + hist.objective(u);
      }
      else {
        ttb_real fhis = 0.0;
        Impl::gcp_value(Yf, u_overlap_F, hist.up, hist.window_val,
                        hist.window_penalty, wf, loss_func, ften, fhis);
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
                          const int timer_zs,
                          const int timer_grad_mttkrp,
                          const int timer_grad_comm,
                          const int timer_grad_update) override
    {
      timer.start(timer_init);
      dku_G->initOverlapKtensor(gt_overlap);
      timer.stop(timer_init);

      // We are cheating here by not importing ut, since we know it happened
      // when sampling G if it was necessary

      timer.start(timer_grad_mttkrp);
      mttkrp_all(Yg, u_overlap_G, gt_overlap, mode_beg, mode_end, algParams,
                 false);
      timer.stop(timer_grad_mttkrp);

      if (Yh.nnz() > 0) {
        // Create uh, u with time mode replaced by time mode of up
        // This should all just be view assignments, so should be fast
        uh.weights() = u_overlap_G.weights();
        const ttb_indx nd = ut.ndims();
        for (ttb_indx i=0; i<nd-1; ++i)
          uh.set_factor(i, u_overlap_G[i]);
        uh.set_factor(nd-1, hist.up[nd-1]);

        mttkrp_all(Yh, uh, gt_overlap, mode_beg, mode_end, algParams, false);
      }

      timer.start(timer_grad_comm);
      dku_G->doExport(gt, gt_overlap);
      timer.stop(timer_grad_comm);

      if (Yh.nnz() == 0)
        hist.gradient(ut, mode_beg, mode_end, gt);

      if (penalty != 0.0)
        for (ttb_indx i=mode_beg; i<mode_end; ++i)
          gt[i-mode_beg].plus(ut[i], ttb_real(2.0)*penalty);
    }

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
    ttb_indx num_samples_grad;
    ttb_indx global_num_samples_nonzeros_value;
    ttb_indx global_num_samples_zeros_value;
    ttb_indx global_num_samples_grad;
    ttb_real weight_nonzeros_value;
    ttb_real weight_zeros_value;
    ttb_real weight_grad;
    ttb_real grad_percent;
    map_type hash_map;
    KtensorT<ExecSpace> uh;
    KtensorT<ExecSpace> u_overlap_F;
    KtensorT<ExecSpace> u_overlap_G;
    KtensorT<ExecSpace> gt_overlap;
    DistKtensorUpdate<ExecSpace> *dku_F;
    DistKtensorUpdate<ExecSpace> *dku_G;
  };

  template <typename ExecSpace, typename LossFunction>
  class UniformSampler<TensorT<ExecSpace>,LossFunction> :
    public Sampler<TensorT<ExecSpace>,LossFunction> {
  public:

    typedef Sampler<TensorT<ExecSpace>,LossFunction> base_type;
    typedef typename base_type::pool_type pool_type;

    UniformSampler(const TensorT<ExecSpace>& X_,
                   const KtensorT<ExecSpace>& u,
                   const AlgParams& algParams_) :
      X(X_), algParams(algParams_), uh(u.ncomponents(),u.ndims())
    {
      global_num_samples_value = algParams.num_samples_nonzeros_value;
      global_num_samples_grad = algParams.num_samples_nonzeros_grad;
      weight_value = algParams.w_f_nz;
      weight_grad = algParams.w_g_nz;

      // Compute number of samples if necessary
      const ttb_real tsz = X.global_numel_float();
      const ttb_indx maxEpochs = algParams.maxiters;
      if (global_num_samples_value == 0)
        global_num_samples_value =
          ttb_indx(std::min(std::max(tsz/ttb_real(10.0),
                                     ttb_real(1e6)), tsz));
      else if (global_num_samples_value == INT_MAX)
        global_num_samples_value = ttb_indx(tsz);

      if (global_num_samples_grad == 0)
        global_num_samples_grad =
          ttb_indx(std::min(std::max(ttb_real(10.0)*tsz/maxEpochs,
                                     ttb_real(1e3)), tsz));
      else if (global_num_samples_grad == INT_MAX)
        global_num_samples_grad = ttb_indx(tsz);

      // Compute local number of samples by distributing them evenly across
      // processors (might be better to weight according to number of nonzeros)
      const ProcessorMap* pmap = /*X.getProcessorMap()*/ nullptr;
      const ttb_real lsz = X.numel_float();
      const ttb_indx np = pmap != nullptr ? pmap->gridSize() : 1;
      num_samples_value = global_num_samples_value / np;
      num_samples_grad = global_num_samples_grad / np;

      // Don't sample more nonzeros than we actually have locally
      num_samples_value = std::min(num_samples_value, ttb_indx(lsz));
      num_samples_grad = std::min(num_samples_grad, ttb_indx(lsz));

      // Compute global number of samples actually used
      if (pmap != nullptr) {
        global_num_samples_value = pmap->gridAllReduce(num_samples_value);
        global_num_samples_grad = pmap->gridAllReduce(num_samples_grad);
      }
      else {
        global_num_samples_value = num_samples_value;
        global_num_samples_grad = num_samples_grad;
      }

      // Compute weights if necessary
      if (weight_value < 0.0)
        weight_value = global_num_samples_value == 0 ? 0.0 :
          tsz / ttb_real(global_num_samples_value);
      if (weight_grad < 0.0)
        weight_grad = global_num_samples_grad == 0 ? 0.0 :
          tsz / ttb_real(global_num_samples_grad);

      grad_percent = ttb_real(global_num_samples_grad * algParams.epoch_iters) /
        ttb_real(tsz) * ttb_real(100.0);

      dku_F = createKtensorUpdate(Yf, u, algParams);
      dku_G = createKtensorUpdate(Yg, u, algParams);

      if (algParams.dist_update_method != Dist_Update_Method::Tpetra) {
        u_overlap_F = dku_F->createOverlapKtensor(u);
        u_overlap_G = dku_G->createOverlapKtensor(u);
      }
    }

    virtual ~UniformSampler()
    {
      delete dku_F;
      delete dku_G;
    }

    virtual void initialize(const pool_type& rand_pool_,
                            const bool printitn,
                            std::ostream& out) override
    {
      rand_pool = rand_pool_;
    }

    virtual ttb_indx getNumGradSamples() const override
    {
      return global_num_samples_grad;
    }

    virtual void print(std::ostream& out) override
    {
      out << "  Function sampler:  uniform with "
          << global_num_samples_value
          << " samples\n"
          << "  Gradient sampler:  uniform with " << global_num_samples_grad
          << " samples\n"
          << "  Gradient samples per epoch: "
          << global_num_samples_grad*algParams.epoch_iters
          << " (" << std::setprecision(1) << std::fixed << grad_percent << "%)"
          << std::endl;
    }

    virtual void sampleTensorF(const KtensorT<ExecSpace>& u,
                               const LossFunction& loss_func) override
    {
      if (X.has_left_impl()) {
        if (algParams.dist_update_method == Dist_Update_Method::Tpetra) {
          Impl::uniform_sample_tensor_tpetra(
            X, Impl::denseSearcher(X.left_impl()),
            num_samples_value, weight_value,
            u, loss_func, false,
            Yf, wf, u_overlap_F, rand_pool, algParams);
        }
        else if (algParams.dist_update_method == Dist_Update_Method::OneSided ||
                 algParams.dist_update_method == Dist_Update_Method::TwoSided) {
          Impl::uniform_sample_tensor_onesided(
            X, Impl::denseSearcher(X.left_impl()),
            num_samples_value, weight_value,
            u, loss_func, false,
            Yf, wf, *dku_F, u_overlap_F, rand_pool, algParams);
        }
        else {
          Impl::uniform_sample_tensor(
            X, Impl::denseSearcher(X.left_impl()),
            num_samples_value, weight_value,
            u, loss_func, false,
            Yf, wf, rand_pool, algParams);
          u_overlap_F = u;
        }
      }
      else {
        if (algParams.dist_update_method == Dist_Update_Method::Tpetra) {
          Impl::uniform_sample_tensor_tpetra(
            X, Impl::denseSearcher(X.right_impl()),
            num_samples_value, weight_value,
            u, loss_func, false,
            Yf, wf, u_overlap_F, rand_pool, algParams);
        }
        else if (algParams.dist_update_method == Dist_Update_Method::OneSided ||
                 algParams.dist_update_method == Dist_Update_Method::TwoSided) {
          Impl::uniform_sample_tensor_onesided(
            X, Impl::denseSearcher(X.right_impl()),
            num_samples_value, weight_value,
            u, loss_func, false,
            Yf, wf, *dku_F, u_overlap_F, rand_pool, algParams);
        }
        else {
          Impl::uniform_sample_tensor(
            X, Impl::denseSearcher(X.right_impl()),
            num_samples_value, weight_value,
            u, loss_func, false,
            Yf, wf, rand_pool, algParams);
          u_overlap_F = u;
        }
      }

      if (algParams.dist_update_method != Dist_Update_Method::OneSided &&
          algParams.dist_update_method != Dist_Update_Method::TwoSided)
        dku_F->updateTensor(Yf);
    }

    virtual void sampleTensorG(const KtensorT<ExecSpace>& u,
                               const StreamingHistory<ExecSpace>& hist,
                               const LossFunction& loss_func) override
    {
      if (X.has_left_impl()) {
        if (algParams.dist_update_method == Dist_Update_Method::Tpetra) {
          Impl::uniform_sample_tensor_tpetra(
            X, Impl::denseSearcher(X.left_impl()),
            num_samples_grad, weight_grad, u, loss_func, true,
            Yg, wg, u_overlap_G, rand_pool, algParams);
        }
        else if (algParams.dist_update_method == Dist_Update_Method::OneSided ||
                 algParams.dist_update_method == Dist_Update_Method::TwoSided) {
          Impl::uniform_sample_tensor_onesided(
            X, Impl::denseSearcher(X.left_impl()),
            num_samples_grad, weight_grad, u, loss_func, true,
            Yg, wg, *dku_G, u_overlap_G, rand_pool, algParams);
        }
        else {
          Impl::uniform_sample_tensor(
            X, Impl::denseSearcher(X.left_impl()),
            num_samples_grad, weight_grad, u, loss_func, true,
            Yg, wg, rand_pool, algParams);
          u_overlap_G = u;
        }
      }
      else {
        if (algParams.dist_update_method == Dist_Update_Method::Tpetra) {
          Impl::uniform_sample_tensor_tpetra(
            X, Impl::denseSearcher(X.right_impl()),
            num_samples_grad, weight_grad, u, loss_func, true,
            Yg, wg, u_overlap_G, rand_pool, algParams);
        }
        else if (algParams.dist_update_method == Dist_Update_Method::OneSided ||
                 algParams.dist_update_method == Dist_Update_Method::TwoSided) {
          Impl::uniform_sample_tensor_onesided(
            X, Impl::denseSearcher(X.right_impl()),
            num_samples_grad, weight_grad, u, loss_func, true,
            Yg, wg, *dku_G, u_overlap_G, rand_pool, algParams);
        }
        else {
          Impl::uniform_sample_tensor(
            X, Impl::denseSearcher(X.right_impl()),
            num_samples_grad, weight_grad, u, loss_func, true,
            Yg, wg, rand_pool, algParams);
          u_overlap_G = u;
        }
      }

      if (hist.do_gcp_loss()) {
        // Create uh, u with time mode replaced by time mode of up
        // This should all just be view assignments, so should be fast
        uh.weights() = u_overlap_G.weights();
        const ttb_indx nd = u.ndims();
        for (ttb_indx i=0; i<nd-1; ++i)
          uh.set_factor(i, u_overlap_G[i]);
        uh.set_factor(nd-1, hist.up[nd-1]);

        Impl::stratified_ktensor_grad(
          Yg, num_samples_grad, ttb_indx(0),
          weight_grad, ttb_real(0.0),
          uh, hist.up, hist.window_val, hist.window_penalty, loss_func,
          Yh, algParams);
      }

      if (algParams.dist_update_method != Dist_Update_Method::OneSided &&
          algParams.dist_update_method != Dist_Update_Method::TwoSided)
        dku_G->updateTensor(Yg);
    }

    virtual void prepareGradient(const KtensorT<ExecSpace>& gt) override
    {
      if (algParams.mttkrp_method == MTTKRP_Method::Perm &&
          algParams.mttkrp_all_method == MTTKRP_All_Method::Iterated) {
        Yg.createPermutation();
        if (Yh.nnz() > 0)
          Yh.createPermutation();
      }
      gt_overlap = dku_G->createOverlapKtensor(gt);
    }

    virtual void value(const KtensorT<ExecSpace>& u,
                       const StreamingHistory<ExecSpace>& hist,
                       const ttb_real penalty,
                       const LossFunction& loss_func,
                       ttb_real& fest, ttb_real& ften) override
    {
      dku_F->doImport(u_overlap_F, u);

      if (!hist.do_gcp_loss()) {
        ften = Impl::gcp_value(Yf, u_overlap_F, wf, loss_func);
        fest = ften + hist.objective(u);
      }
      else {
        ttb_real fhis = 0.0;
        Impl::gcp_value(Yf, u_overlap_F, hist.up, hist.window_val,
                        hist.window_penalty, wf, loss_func, ften, fhis);
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
                          const int timer_zs,
                          const int timer_grad_mttkrp,
                          const int timer_grad_comm,
                          const int timer_grad_update) override
    {
      timer.start(timer_init);
      gt_overlap.weights() = ttb_real(1.0);
      gt_overlap.setMatrices(0.0);
      timer.stop(timer_init);

      // We are cheating here by not importing ut, since we know it happened
      // when sampling G if it was necessary

      timer.start(timer_grad_mttkrp);
      mttkrp_all(Yg, u_overlap_G, gt_overlap, mode_beg, mode_end, algParams,
                 false);
      timer.stop(timer_grad_mttkrp);

      if (Yh.nnz() > 0) {
        // Create uh, u with time mode replaced by time mode of up
        // This should all just be view assignments, so should be fast
        uh.weights() = u_overlap_G.weights();
        const ttb_indx nd = ut.ndims();
        for (ttb_indx i=0; i<nd-1; ++i)
          uh.set_factor(i, u_overlap_G[i]);
        uh.set_factor(nd-1, hist.up[nd-1]);

        mttkrp_all(Yh, uh, gt_overlap, mode_beg, mode_end, algParams, false);
      }

      timer.start(timer_grad_comm);
      dku_G->doExport(gt, gt_overlap);
      timer.stop(timer_grad_comm);

      if (Yh.nnz() == 0)
        hist.gradient(ut, mode_beg, mode_end, gt);

      if (penalty != 0.0)
        for (ttb_indx i=mode_beg; i<mode_end; ++i)
          gt[i-mode_beg].plus(ut[i], ttb_real(2.0)*penalty);
    }

  protected:

    TensorT<ExecSpace> X;
    SptensorT<ExecSpace> Yf;
    SptensorT<ExecSpace> Yg;
    SptensorT<ExecSpace> Yh;
    ArrayT<ExecSpace> wf;
    ArrayT<ExecSpace> wg;
    pool_type rand_pool;
    AlgParams algParams;
    ttb_indx num_samples_value;
    ttb_indx num_samples_grad;
    ttb_indx global_num_samples_value;
    ttb_indx global_num_samples_grad;
    ttb_real weight_value;
    ttb_real weight_grad;
    ttb_real grad_percent;
    KtensorT<ExecSpace> uh;
    KtensorT<ExecSpace> u_overlap_F;
    KtensorT<ExecSpace> u_overlap_G;
    KtensorT<ExecSpace> gt_overlap;
    DistKtensorUpdate<ExecSpace> *dku_F;
    DistKtensorUpdate<ExecSpace> *dku_G;
  };

}
