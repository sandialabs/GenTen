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

#include <cmath>

#include "Genten_AlgParams.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_KokkosVector.hpp"
#include "Genten_GCP_LossFunctions.hpp"

#include "impl/Kokkos_Atomic_Generic.hpp"

namespace Genten {

  namespace Impl {

    template <typename LossFunction>
    struct BoundUpdate {
      KOKKOS_INLINE_FUNCTION
      ttb_real apply(const ttb_real& u, const ttb_real& delta) const {
        ttb_real un = u + delta;
        if (LossFunction::has_lower_bound() && un < LossFunction::lower_bound())
          un = LossFunction::lower_bound();
        if (LossFunction::has_upper_bound() && un > LossFunction::upper_bound())
          un = LossFunction::upper_bound();
        return un;
      }
    };

    template <typename ExecSpace, typename LossFunction>
    class GCP_SGD_Step {
    public:
      typedef KokkosVector<ExecSpace> VectorType;

      GCP_SGD_Step() = default;

      virtual ~GCP_SGD_Step() {}

      virtual void setStep(const ttb_real step) = 0;

      virtual ttb_real getStep() const = 0;

      virtual void update() = 0;

      virtual void reset() = 0;

      virtual void setPassed() = 0;

      virtual void setFailed() = 0;

      virtual void setNumSamples(const ttb_indx num_samples) = 0;

      virtual void eval(const VectorType& g, VectorType& u) const = 0;

    protected:

      KOKKOS_INLINE_FUNCTION
      void update_u_async(const unsigned dim, const ttb_indx row,
                          const unsigned col, const ttb_real delta,
                          const KtensorT<ExecSpace>& u) const
      {
        // Without bounds, do simple atomic update
        if (!LossFunction::has_lower_bound() &&
            !LossFunction::has_upper_bound())
          Kokkos::atomic_add(&u[dim].entry(row,col), delta);

        else {
#if 1
          // The fastest but least safe version.  Only requires a single atomic,
          // but bounds could be violated.  Compute a step truncated to bounds.
          const ttb_real uold = u[dim].entry(row,col);
          ttb_real unew = uold + delta;
          if (LossFunction::has_lower_bound() &&
              unew < LossFunction::lower_bound())
            unew = LossFunction::lower_bound();
          if (LossFunction::has_upper_bound() &&
              unew > LossFunction::upper_bound())
            unew = LossFunction::upper_bound();
          const ttb_real delta2 = unew - uold;
          Kokkos::atomic_add(&u[dim].entry(row,col), delta2);
#elif 1
          // Safer but slower.  Only do atomic max/min if necessary.
          const ttb_real uold =
            Kokkos::atomic_fetch_add(&u[dim].entry(row,col), delta);
          const ttb_real unew = uold + delta;
          if (LossFunction::has_lower_bound() &&
              unew < LossFunction::lower_bound())
            Kokkos::atomic_fetch_max(&u[dim].entry(row,col),
                                     LossFunction::lower_bound());
          if (LossFunction::has_upper_bound() &&
              unew > LossFunction::upper_bound())
            Kokkos::atomic_fetch_min(&u[dim].entry(row,col),
                                     LossFunction::upper_bound());
#elif 1
          // Slowest version.  Always do atomic max/min.
          Kokkos::atomic_add(&u[dim].entry(row,col), delta);
          if (LossFunction::has_lower_bound())
            Kokkos::atomic_fetch_max(&u[dim].entry(row,col),
                                     LossFunction::lower_bound());
          if (LossFunction::has_upper_bound())
            Kokkos::atomic_fetch_min(&u[dim].entry(row,col),
                                     LossFunction::upper_bound());
#else
          // Probably even slower.  Apply bounds while updating
          if (LossFunction::has_lower_bound() ||
              LossFunction::has_upper_bound())
            Kokkos::Impl::atomic_oper_fetch(BoundUpdate<LossFunction>(),
                                            &u[dim].entry(row,col),
                                            delta);
          else
            Kokkos::atomic_add(&u[dim].entry(row,col), delta);
#endif
        }
      }

    };

    template <typename ExecSpace, typename LossFunction>
    class SGDStep : public GCP_SGD_Step<ExecSpace,LossFunction> {
    public:
      typedef GCP_SGD_Step<ExecSpace,LossFunction> BaseType;
      typedef typename BaseType::VectorType VectorType;

      SGDStep() {}

      virtual ~SGDStep() {}

      virtual void setStep(const ttb_real s) { step = s; }

      virtual ttb_real getStep() const { return step; }

      virtual void update() {}

      virtual void reset() {}

      virtual void setPassed() {}

      virtual void setFailed() {}

      virtual void setNumSamples(const ttb_indx num_samples) {}

      virtual void eval(const VectorType& g, VectorType& u) const
      {
        constexpr bool has_bounds = (LossFunction::has_lower_bound() ||
                                     LossFunction::has_upper_bound());
        constexpr ttb_real lb = LossFunction::lower_bound();
        constexpr ttb_real ub = LossFunction::upper_bound();
        const ttb_real sgd_step = step;
        auto uv = u.getView();
        auto gv = g.getView();
        u.apply_func(KOKKOS_LAMBDA(const ttb_indx i)
        {
          ttb_real uu = uv(i);
          uu -= sgd_step*gv(i);
          if (has_bounds)
            uu = uu < lb ? lb : (uu > ub ? ub : uu);
          uv(i) = uu;
        });
      }

      template <typename TeamMember>
      KOKKOS_INLINE_FUNCTION
      void update_async(const ttb_indx num_iters, const TeamMember& team) const {}

      KOKKOS_INLINE_FUNCTION
      void eval_async(const unsigned dim, const ttb_indx row,
                      const unsigned col, const ttb_real g,
                      const KtensorT<ExecSpace>& u) const
      {
        // Update u incorporating bounds
        GCP_SGD_Step<ExecSpace,LossFunction>::update_u_async(
          dim, row, col, -step*g, u);
      }

    protected:
      ttb_real step;

    };

    struct AdamOp {
      ttb_real beta;

      KOKKOS_INLINE_FUNCTION
      AdamOp(const ttb_real& beta_) : beta(beta_) {}

      KOKKOS_INLINE_FUNCTION
      ttb_real apply(const ttb_real& m, const ttb_real& g) const {
        return beta*m + (1.0-beta)*g;
      }
    };

#if 0

    // Version of ADAM that keeps an iteration counter for threads to compute
    // the bias correction for asynchronous.  This appears to not work
    // very well, probably due to the sparse updates not being equivalent
    // to the dense updates applied in synchronous ADAM

    template <typename ExecSpace, typename LossFunction>
    class AdamStep : public GCP_SGD_Step<ExecSpace,LossFunction> {
    public:
      typedef GCP_SGD_Step<ExecSpace,LossFunction> BaseType;
      typedef typename BaseType::VectorType VectorType;

      AdamStep(const AlgParams& algParams, const VectorType& u) :
        epoch_iters(algParams.epoch_iters),
        num_samples_per_it(0),
        step(0.0),
        beta1(algParams.adam_beta1),
        beta2(algParams.adam_beta2),
        eps(algParams.adam_eps),
        beta1t(1.0),
        beta2t(1.0),
        adam_step(0.0),
        m(u.clone()),
        v(u.clone()),
        m_prev(u.clone()),
        v_prev(u.clone()),
        mt(m.getKtensor()),
        vt(v.getKtensor()),
        total_samples("total_samples")
      {
        m.zero();
        v.zero();
        m_prev.zero();
        v_prev.zero();
        Kokkos::deep_copy(total_samples, 0);
      }

      virtual ~AdamStep() {}

      virtual void setStep(const ttb_real s) { step = s; }

      virtual ttb_real getStep() const { return step; }

      virtual void update()
      {
        beta1t = beta1 * beta1t;
        beta2t = beta2 * beta2t;
        adam_step = step*std::sqrt(1.0-beta2t) / (1.0-beta1t);
      }

      virtual void reset()
      {
        beta1t = 1.0;
        beta2t = 1.0;
        m.zero();
        v.zero();
        m_prev.zero();
        v_prev.zero();
        Kokkos::deep_copy(total_samples, 0);
      }

      virtual void setPassed()
      {
        m_prev.set(m);
        v_prev.set(v);
      }

      virtual void setFailed()
      {
        m.set(m_prev);
        v.set(v_prev);
        beta1t /= std::pow(beta1, epoch_iters);
        beta2t /= std::pow(beta2, epoch_iters);

        auto total_samples_host = Kokkos::create_mirror_view(total_samples);
        Kokkos::deep_copy(total_samples_host, total_samples);
        total_samples_host() -= epoch_iters*num_samples_per_it;
        if (total_samples_host() < 0)
          total_samples_host() = 0;
        Kokkos::deep_copy(total_samples, total_samples_host);
      }

      virtual void setNumSamples(const ttb_indx num_samples) {
        num_samples_per_it = num_samples;
      }

      virtual void eval(const VectorType& g, VectorType& u) const
      {
        using std::sqrt;
        constexpr bool has_bounds = (LossFunction::has_lower_bound() ||
                                     LossFunction::has_upper_bound());
        constexpr ttb_real lb = LossFunction::lower_bound();
        constexpr ttb_real ub = LossFunction::upper_bound();
        const ttb_real adam_step_ = adam_step;
        const ttb_real eps_ = eps;
        const ttb_real beta1_ = beta1;
        const ttb_real beta2_ = beta2;
        auto uv = u.getView();
        auto gv = g.getView();
        auto mv = m.getView();
        auto vv = v.getView();
        u.apply_func(KOKKOS_LAMBDA(const ttb_indx i)
        {
          mv(i) = beta1_*mv(i) + (1.0-beta1_)*gv(i);
          vv(i) = beta2_*vv(i) + (1.0-beta2_)*gv(i)*gv(i);
          ttb_real uu = uv(i);
          uu -= adam_step_*mv(i)/sqrt(vv(i)+eps_);
          if (has_bounds)
            uu = uu < lb ? lb : (uu > ub ? ub : uu);
          uv(i) = uu;
        });
      }

      template <typename TeamMember>
      KOKKOS_INLINE_FUNCTION
      void update_async(const ttb_indx num_iters, const TeamMember& team) const
      {
        Kokkos::single(Kokkos::PerThread( team ), [&]()
        {
          Kokkos::atomic_add(&total_samples(), ptrdiff_t(num_iters));
        });
      }

      KOKKOS_INLINE_FUNCTION
      void eval_async(const unsigned dim, const ttb_indx row,
                      const unsigned col, const ttb_real g,
                      const KtensorT<ExecSpace>& u) const
      {
        using std::sqrt;
        using std::pow;
        using std::abs;

        // Compute our iteration index
        const ttb_indx ts = total_samples();
        const ttb_indx it = (ts+num_samples_per_it-1) / num_samples_per_it;

        // Compute exponential weighting
        const ttb_real beta1t_ = pow(beta1, ttb_real(it+1));
        const ttb_real beta2t_ = pow(beta2, ttb_real(it+1));
        const ttb_real adam_step_ = step*sqrt(1.0-beta2t_) / (1.0-beta1t_);

        // This seems to generally work ok, but doesn't converge as well as
        // synchronous

        // Update moments
        const ttb_real mn =
          Kokkos::Impl::atomic_oper_fetch(AdamOp(beta1),
                                          &mt[dim].entry(row,col),
                                          g);
        const ttb_real vn =
          Kokkos::Impl::atomic_oper_fetch(AdamOp(beta2),
                                          &vt[dim].entry(row,col),
                                          g*g);

        // Update to u
        const ttb_real delta = -adam_step_*mn/(sqrt(abs(vn))+eps);

        // Update u incorporating bounds
        if (LossFunction::has_lower_bound() ||
            LossFunction::has_upper_bound())
          Kokkos::Impl::atomic_oper_fetch(BoundUpdate<LossFunction>(),
                                          &u[dim].entry(row,col),
                                          delta);
        else
          Kokkos::atomic_add(&u[dim].entry(row,col), delta);
      }

    protected:
      ttb_indx epoch_iters;
      ttb_indx num_samples_per_it;
      ttb_real step;
      ttb_real beta1;
      ttb_real beta2;
      ttb_real eps;
      ttb_real beta1t;
      ttb_real beta2t;
      ttb_real adam_step;

      VectorType m;
      VectorType v;
      VectorType m_prev;
      VectorType v_prev;
      KtensorT<ExecSpace> mt;
      KtensorT<ExecSpace> vt;

      // Specifically using signed integer here to allow for negative
      Kokkos::View<ptrdiff_t,ExecSpace> total_samples;
    };

#else

    // Version of ADAM that keeps an iteration counter for each individual DOF
    // to compute the bias correction for asynchronous.  This appears to work
    // somewhat better than the above, but still doesn't solve the sparse update
    // problem

    template <typename ExecSpace, typename LossFunction>
    class AdamStep : public GCP_SGD_Step<ExecSpace,LossFunction> {
    public:
      typedef GCP_SGD_Step<ExecSpace,LossFunction> BaseType;
      typedef typename BaseType::VectorType VectorType;

      AdamStep(const AlgParams& algParams, const VectorType& u) :
        epoch_iters(algParams.epoch_iters),
        num_samples_per_it(0),
        step(0.0),
        beta1(algParams.adam_beta1),
        beta2(algParams.adam_beta2),
        eps(algParams.adam_eps),
        beta1t(1.0),
        beta2t(1.0),
        adam_step(0.0),
        m(u.clone()),
        v(u.clone()),
        t(u.clone()),
        m_prev(u.clone()),
        v_prev(u.clone()),
        t_prev(u.clone()),
        mt(m.getKtensor()),
        vt(v.getKtensor()),
        tt(t.getKtensor())
      {
        m.zero();
        v.zero();
        t.zero();
        m_prev.zero();
        v_prev.zero();
        t_prev.zero();
      }

      virtual ~AdamStep() {}

      virtual void setStep(const ttb_real s) { step = s; }

      virtual ttb_real getStep() const { return step; }

      virtual void update()
      {
        beta1t = beta1 * beta1t;
        beta2t = beta2 * beta2t;
        adam_step = step*std::sqrt(1.0-beta2t) / (1.0-beta1t);
      }

      virtual void reset()
      {
        beta1t = 1.0;
        beta2t = 1.0;
        m.zero();
        v.zero();
        t.zero();
        m_prev.zero();
        v_prev.zero();
        t_prev.zero();
      }

      virtual void setPassed()
      {
        m_prev.set(m);
        v_prev.set(v);
        t_prev.set(t);
      }

      virtual void setFailed()
      {
        m.set(m_prev);
        v.set(v_prev);
        t.set(t_prev);
        beta1t /= std::pow(beta1, epoch_iters);
        beta2t /= std::pow(beta2, epoch_iters);
      }

      virtual void setNumSamples(const ttb_indx num_samples) {
        num_samples_per_it = num_samples;
      }

      virtual void eval(const VectorType& g, VectorType& u) const
      {
        using std::sqrt;
        constexpr bool has_bounds = (LossFunction::has_lower_bound() ||
                                     LossFunction::has_upper_bound());
        constexpr ttb_real lb = LossFunction::lower_bound();
        constexpr ttb_real ub = LossFunction::upper_bound();
        const ttb_real adam_step_ = adam_step;
        const ttb_real eps_ = eps;
        const ttb_real beta1_ = beta1;
        const ttb_real beta2_ = beta2;
        auto uv = u.getView();
        auto gv = g.getView();
        auto mv = m.getView();
        auto vv = v.getView();
        u.apply_func(KOKKOS_LAMBDA(const ttb_indx i)
        {
          mv(i) = beta1_*mv(i) + (1.0-beta1_)*gv(i);
          vv(i) = beta2_*vv(i) + (1.0-beta2_)*gv(i)*gv(i);
          ttb_real uu = uv(i);
          uu -= adam_step_*mv(i)/sqrt(vv(i)+eps_);
          if (has_bounds)
            uu = uu < lb ? lb : (uu > ub ? ub : uu);
          uv(i) = uu;
        });
      }

      template <typename TeamMember>
      KOKKOS_INLINE_FUNCTION
      void update_async(const ttb_indx num_iters, const TeamMember& team) const
      {
      }

      KOKKOS_INLINE_FUNCTION
      void eval_async(const unsigned dim, const ttb_indx row,
                      const unsigned col, const ttb_real g,
                      const KtensorT<ExecSpace>& u) const
      {
#if defined(__SYCL_DEVICE_ONLY__)
        using sycl::pow;
#else
        using std::pow;
#endif

        using std::sqrt;
        using std::abs;

        // Compute exponential weighting
        const ttb_real t_ =
          Kokkos::atomic_fetch_add(&tt[dim].entry(row,col), ttb_real(1.0))+1.0;
        const ttb_real beta1t_ = pow(beta1, t_);
        const ttb_real beta2t_ = pow(beta2, t_);
        const ttb_real adam_step_ = step*sqrt(1.0-beta2t_) / (1.0-beta1t_);

        // Update moments
        const ttb_real mn =
          Kokkos::Impl::atomic_oper_fetch(AdamOp(beta1),
                                          &mt[dim].entry(row,col),
                                          g);
        const ttb_real vn =
          Kokkos::Impl::atomic_oper_fetch(AdamOp(beta2),
                                          &vt[dim].entry(row,col),
                                          g*g);

        // Update to u
        const ttb_real delta = -adam_step_*mn/(sqrt(abs(vn))+eps);

        // Update u incorporating bounds
        if (LossFunction::has_lower_bound() ||
            LossFunction::has_upper_bound())
          Kokkos::Impl::atomic_oper_fetch(BoundUpdate<LossFunction>(),
                                          &u[dim].entry(row,col),
                                          delta);
        else
          Kokkos::atomic_add(&u[dim].entry(row,col), delta);
      }

    protected:
      ttb_indx epoch_iters;
      ttb_indx num_samples_per_it;
      ttb_real step;
      ttb_real beta1;
      ttb_real beta2;
      ttb_real eps;
      ttb_real beta1t;
      ttb_real beta2t;
      ttb_real adam_step;

      VectorType m;
      VectorType v;
      VectorType t;
      VectorType m_prev;
      VectorType v_prev;
      VectorType t_prev;
      KtensorT<ExecSpace> mt;
      KtensorT<ExecSpace> vt;
      KtensorT<ExecSpace> tt;
    };

#endif

    template <typename ExecSpace, typename LossFunction>
    class AMSGradStep : public GCP_SGD_Step<ExecSpace,LossFunction> {
    public:
      typedef GCP_SGD_Step<ExecSpace,LossFunction> BaseType;
      typedef typename BaseType::VectorType VectorType;

      AMSGradStep(const AlgParams& algParams, const VectorType& u) :
        epoch_iters(algParams.epoch_iters),
        num_samples_per_it(0),
        step(0.0),
        beta1(algParams.adam_beta1),
        beta2(algParams.adam_beta2),
        eps(algParams.adam_eps),
        beta1t(1.0),
        beta2t(1.0),
        adam_step(0.0),
        m(u.clone()),
        v(u.clone()),
        w(u.clone()),
        m_prev(u.clone()),
        v_prev(u.clone()),
        w_prev(u.clone()),
        mt(m.getKtensor()),
        vt(v.getKtensor()),
        wt(w.getKtensor()),
        total_samples("total_samples")
      {
        m.zero();
        v.zero();
        w.zero();
        m_prev.zero();
        v_prev.zero();
        w_prev.zero();
        Kokkos::deep_copy(total_samples, 0);
      }

      virtual ~AMSGradStep() {}

      virtual void setStep(const ttb_real s) { step = s; }

      virtual ttb_real getStep() const { return step; }

      virtual void update()
      {
        beta1t = beta1 * beta1t;
        beta2t = beta2 * beta2t;
        adam_step = step*std::sqrt(1.0-beta2t) / (1.0-beta1t);
      }

      virtual void reset()
      {
        beta1t = 1.0;
        beta2t = 1.0;
        m.zero();
        v.zero();
        w.zero();
        m_prev.zero();
        v_prev.zero();
        w_prev.zero();
        Kokkos::deep_copy(total_samples, 0);
      }

      virtual void setPassed()
      {
        m_prev.set(m);
        v_prev.set(v);
        w_prev.set(w);
      }

      virtual void setFailed()
      {
        m.set(m_prev);
        v.set(v_prev);
        w.set(w_prev);
        beta1t /= std::pow(beta1, epoch_iters);
        beta2t /= std::pow(beta2, epoch_iters);

        auto total_samples_host = Kokkos::create_mirror_view(total_samples);
        Kokkos::deep_copy(total_samples_host, total_samples);
        total_samples_host() -= epoch_iters*num_samples_per_it;
        if (total_samples_host() < 0)
          total_samples_host() = 0;
        Kokkos::deep_copy(total_samples, total_samples_host);
      }

      virtual void setNumSamples(const ttb_indx num_samples) {
        num_samples_per_it = num_samples;
      }

      virtual void eval(const VectorType& g, VectorType& u) const
      {
        using std::sqrt;
        constexpr bool has_bounds = (LossFunction::has_lower_bound() ||
                                     LossFunction::has_upper_bound());
        constexpr ttb_real lb = LossFunction::lower_bound();
        constexpr ttb_real ub = LossFunction::upper_bound();
        const ttb_real adam_step_ = adam_step;
        const ttb_real eps_ = eps;
        const ttb_real beta1_ = beta1;
        const ttb_real beta2_ = beta2;
        auto uv = u.getView();
        auto gv = g.getView();
        auto mv = m.getView();
        auto vv = v.getView();
        auto wv = w.getView();
        u.apply_func(KOKKOS_LAMBDA(const ttb_indx i)
        {
          mv(i) = beta1_*mv(i) + (1.0-beta1_)*gv(i);
          vv(i) = beta2_*vv(i) + (1.0-beta2_)*gv(i)*gv(i);
          wv(i) = vv(i) > wv(i) ? vv(i) : wv(i);
          ttb_real uu = uv(i);
          uu -= adam_step_*mv(i)/sqrt(wv(i)+eps_);
          if (has_bounds)
            uu = uu < lb ? lb : (uu > ub ? ub : uu);
          uv(i) = uu;
        });
      }

      template <typename TeamMember>
      KOKKOS_INLINE_FUNCTION
      void update_async(const ttb_indx num_iters, const TeamMember& team) const
      {
        Kokkos::single(Kokkos::PerThread( team ), [&]()
        {
          Kokkos::atomic_add(&total_samples(), ptrdiff_t(num_iters));
        });
      }

      KOKKOS_INLINE_FUNCTION
      void eval_async(const unsigned dim, const ttb_indx row,
                      const unsigned col, const ttb_real g,
                      const KtensorT<ExecSpace>& u) const
      {
#if defined(__SYCL_DEVICE_ONLY__)
        using cl::sycl::pow;
#else
        using std::pow;
#endif

        using std::sqrt;
        using std::abs;

        // Compute our iteration index
        const ttb_indx ts = total_samples();
        const ttb_indx it = (ts+num_samples_per_it-1) / num_samples_per_it;

        // Compute exponential weighting
        const ttb_real beta1t_ = pow(beta1, ttb_real(it+1));
        const ttb_real beta2t_ = pow(beta2, ttb_real(it+1));
        const ttb_real adam_step_ = step*sqrt(1.0-beta2t_) / (1.0-beta1t_);

        // Update moments
        const ttb_real mn =
          Kokkos::Impl::atomic_oper_fetch(AdamOp(beta1),
                                          &mt[dim].entry(row,col),
                                          g);
        const ttb_real vn =
          Kokkos::Impl::atomic_oper_fetch(AdamOp(beta2),
                                          &vt[dim].entry(row,col),
                                          g*g);
        const ttb_real wn =
          Kokkos::atomic_max_fetch(&wt[dim].entry(row,col), vn);

        // Update to u
        const ttb_real delta = -adam_step_*mn/(sqrt(abs(wn))+eps);

        // Update u incorporating bounds
        if (LossFunction::has_lower_bound() ||
            LossFunction::has_upper_bound())
          Kokkos::Impl::atomic_oper_fetch(BoundUpdate<LossFunction>(),
                                          &u[dim].entry(row,col),
                                          delta);
        else
          Kokkos::atomic_add(&u[dim].entry(row,col), delta);
      }

    protected:
      ttb_indx epoch_iters;
      ttb_indx num_samples_per_it;
      ttb_real step;
      ttb_real beta1;
      ttb_real beta2;
      ttb_real eps;
      ttb_real beta1t;
      ttb_real beta2t;
      ttb_real adam_step;

      VectorType m;
      VectorType v;
      VectorType w;
      VectorType m_prev;
      VectorType v_prev;
      VectorType w_prev;
      KtensorT<ExecSpace> mt;
      KtensorT<ExecSpace> vt;
      KtensorT<ExecSpace> wt;

      // Specifically using signed integer here to allow for negative
      Kokkos::View<ptrdiff_t,ExecSpace> total_samples;
    };

    template <typename ExecSpace, typename LossFunction>
    class AdaGradStep : public GCP_SGD_Step<ExecSpace,LossFunction> {
    public:
      typedef GCP_SGD_Step<ExecSpace,LossFunction> BaseType;
      typedef typename BaseType::VectorType VectorType;

      AdaGradStep(const AlgParams& algParams, const VectorType& u) :
        step(0.0),
        eps(algParams.adam_eps),
        s(u.clone()),
        s_prev(u.clone()),
        st(s.getKtensor())
      {
        s.zero();
        s_prev.zero();
      }

      virtual ~AdaGradStep() {}

      virtual void setStep(const ttb_real s) { step = s; }

      virtual ttb_real getStep() const { return step; }

      virtual void update() {}

      virtual void reset()
      {
        s.zero();
        s_prev.zero();
      }

      virtual void setPassed()
      {
        s_prev.set(s);
      }

      virtual void setFailed()
      {
        s.set(s_prev);
      }

      virtual void setNumSamples(const ttb_indx num_samples) {}

      virtual void eval(const VectorType& g, VectorType& u) const
      {
        using std::sqrt;
        constexpr bool has_bounds = (LossFunction::has_lower_bound() ||
                                     LossFunction::has_upper_bound());
        constexpr ttb_real lb = LossFunction::lower_bound();
        constexpr ttb_real ub = LossFunction::upper_bound();
        const ttb_real step_ = step;
        const ttb_real eps_ = eps;
        auto uv = u.getView();
        auto gv = g.getView();
        auto sv = s.getView();
        u.apply_func(KOKKOS_LAMBDA(const ttb_indx i)
        {
          const ttb_real gg = gv(i);
          ttb_real ss = sv(i);
          ttb_real uu = uv(i);
          ss += gg*gg;
          uu -= step_*gg/sqrt(ss+eps_);
          if (has_bounds)
            uu = uu < lb ? lb : (uu > ub ? ub : uu);
          sv(i) = ss;
          uv(i) = uu;
        });
      }

      template <typename TeamMember>
      KOKKOS_INLINE_FUNCTION
      void update_async(const ttb_indx num_iters, const TeamMember& team) const {}

      KOKKOS_INLINE_FUNCTION
      void eval_async(const unsigned dim, const ttb_indx row,
                      const unsigned col, const ttb_real g,
                      const KtensorT<ExecSpace>& u) const
      {
        using std::sqrt;

        // Update sum-of-squares of gradient
        ttb_real ss = Kokkos::atomic_fetch_add(&st[dim].entry(row,col), g*g);

        // Our update to u
        ttb_real delta = -step*g/sqrt(ss+g*g+eps);

        // Update u incorporating bounds
        GCP_SGD_Step<ExecSpace,LossFunction>::update_u_async(
          dim, row, col, delta, u);
      }

    protected:
      ttb_real step;
      ttb_real eps;

      VectorType s;
      VectorType s_prev;
      KtensorT<ExecSpace> st;
    };

  }

}
