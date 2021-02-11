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

#include "Genten_GCP_ValueKernels.hpp"
#include "Genten_SimdKernel.hpp"

namespace Genten {

  namespace Impl {

    template <typename ExecSpace, typename loss_type>
    struct GCP_Value {
      typedef SptensorT<ExecSpace> tensor_type;
      typedef KtensorT<ExecSpace> Ktensor_type;
      typedef ArrayT<ExecSpace> weights_type;

      const tensor_type XX;
      const Ktensor_type MM;
      const weights_type ww;
      const loss_type ff;

      ttb_real value;

      GCP_Value(const tensor_type& X_, const Ktensor_type& M_,
                const weights_type& w_, const loss_type& f_) :
        XX(X_), MM(M_), ww(w_), ff(f_) {}

      template <unsigned FBS, unsigned VS>
      void run()
      {
        typedef typename tensor_type::exec_space exec_space;
        typedef Kokkos::TeamPolicy<exec_space> Policy;
        typedef typename Policy::member_type TeamMember;

        const tensor_type X = XX;
        const Ktensor_type M = MM;
        const weights_type w = ww;
        const loss_type f = ff;

        static const bool is_cuda = Genten::is_cuda_space<exec_space>::value;
        static const unsigned RowBlockSize = 128;
        static const unsigned FacBlockSize = FBS;
        static const unsigned VectorSize = is_cuda ? VS : 1;
        static const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;

        const ttb_indx nnz = X.nnz();
        const ttb_indx N = (nnz+RowBlockSize-1)/RowBlockSize;

        Policy policy(N, TeamSize, VectorSize);
        ttb_real v = 0.0;
        Kokkos::parallel_reduce("GCP_RolObjective::value",
                                policy, KOKKOS_LAMBDA(const TeamMember& team,
                                                      ttb_real& d)
        {
          for (ttb_indx ii=team.team_rank(); ii<RowBlockSize; ii+=TeamSize) {
            const ttb_indx i = team.league_rank()*RowBlockSize + ii;
            if (i >= nnz)
              continue;

            // Compute Ktensor value
            ttb_real m_val =
              compute_Ktensor_value<exec_space, FacBlockSize, VectorSize>(
                M, X, i);

            // Evaluate link function
            d += w[i] * f.value(X.value(i), m_val);
          }
        }, v);
        Kokkos::fence();  // ensure v is updated before using it
        value = v;
      }
    };

    template <typename ExecSpace, typename loss_type,
              unsigned TeamSize, unsigned VectorSize, unsigned FacBlockSize,
              unsigned RowBlockSize>
    struct GCP_ValueHistoryFunctor {
      typedef SptensorT<ExecSpace> tensor_type;
      typedef KtensorT<ExecSpace> Ktensor_type;
      typedef ArrayT<ExecSpace> array_type;

      typedef Kokkos::TeamPolicy<ExecSpace> Policy;
      typedef typename Policy::member_type TeamMember;
      typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

      const tensor_type X;
      const Ktensor_type M;
      const Ktensor_type Mt;
      const Ktensor_type Mprev;
      const array_type window;
      const ttb_real window_penalty;
      const array_type w;
      const loss_type f;

      typedef ttb_real value_type[];
      static constexpr int value_count = 2;

      GCP_ValueHistoryFunctor(const tensor_type& X_,
                              const Ktensor_type& M_,
                              const Ktensor_type& Mt_,
                              const Ktensor_type& Mprev_,
                              const array_type& window_,
                              const ttb_real window_penalty_,
                              const array_type& w_,
                              const loss_type& f_) :
        X(X_), M(M_), Mt(Mt_), Mprev(Mprev_), window(window_),
        window_penalty(window_penalty_), w(w_), f(f_)
      {}

      KOKKOS_INLINE_FUNCTION
      void operator() (const TeamMember& team, value_type d) const
      {
        /*const*/ unsigned nd = M.ndims();
        /*const*/ ttb_indx nnz = X.nnz();
        /*const*/ ttb_indx nh = window.size();
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        for (ttb_indx ii=team.team_rank(); ii<RowBlockSize; ii+=TeamSize) {
          /*const*/ ttb_indx i = team.league_rank()*RowBlockSize + ii;
          if (i >= nnz)
            continue;

          // Compute Ktensor value
          ttb_real m_val =
            compute_Ktensor_value<ExecSpace, FacBlockSize, VectorSize>(
              M, X, i);

          // Evaluate link function
          d[0] += w[i] * f.value(X.value(i), m_val);

          // Add in history term
          for (ttb_indx h=0; h<nh; ++h) {
            // Modify index for history -- use broadcast form to force
            // warp sync so that ind is updated before used by other threads
            int sync = 0;
            Kokkos::single( Kokkos::PerThread( team ), [&] (int& s)
            {
              for (ttb_indx m=0; m<nd-1; ++m)
                ind[m] = X.subscript(i,m);
              ind[nd-1] = h;
              s = 1;
            }, sync);

            // Compute Yt value
            const ttb_real mt_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(
                Mt, ind);
            const ttb_real mp_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(
                Mprev, ind);
            d[1] +=
              window_penalty * window[h] * w[i] * f.value(mp_val, mt_val);
          }
        }
      }

      KOKKOS_INLINE_FUNCTION
      void join(volatile value_type dst, const volatile value_type src) const
      {
        dst[0] += src[0];
        dst[1] += src[1];
      }

      KOKKOS_INLINE_FUNCTION
      void init(value_type val) const
      {
        val[0] = ttb_real(0.0);
        val[1] = ttb_real(0.0);
      }
    };

    template <typename ExecSpace, typename loss_type>
    struct GCP_ValueHistory {
      typedef SptensorT<ExecSpace> tensor_type;
      typedef KtensorT<ExecSpace> Ktensor_type;
      typedef ArrayT<ExecSpace> array_type;

      const tensor_type X;
      const Ktensor_type M;
      Ktensor_type Mt;
      const Ktensor_type Mprev;
      const array_type window;
      const ttb_real window_penalty;
      const array_type w;
      const loss_type f;

      ttb_real val_ten, val_his;

      GCP_ValueHistory(const tensor_type& X_,
                       const Ktensor_type& M_,
                       const Ktensor_type& Mprev_,
                       const array_type& window_,
                       const ttb_real window_penalty_,
                       const array_type& w_,
                       const loss_type& f_) :
        X(X_), M(M_), Mprev(Mprev_), window(window_),
        window_penalty(window_penalty_), w(w_), f(f_),
        val_ten(ttb_real(0.0)), val_his(ttb_real(0.0))
      {
        const ttb_indx nd = M.ndims();
        const ttb_indx nc = M.ncomponents();
        Mt = Ktensor_type(nc, nd);
        for (ttb_indx i=0; i<nd-1; ++i) {
          Genten::FacMatrixT<ExecSpace>v(M[i].nRows(), nc);
          deep_copy(v, M[i]);
          Mt.set_factor(i, v);
        }
        Genten::FacMatrixT<ExecSpace>v(Mprev[nd-1].nRows(), nc);
        deep_copy(v, Mprev[nd-1]);
        Mt.set_factor(nd-1, v);
        Mt.setWeights(1.0);
      }

      template <unsigned FBS, unsigned VS>
      void run()
      {
        typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

        static const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;
        static const unsigned RowBlockSize = 128;
        static const unsigned FacBlockSize = FBS;
        //static const unsigned VectorSize = is_cuda ? VS : 1;
        // Use VectorSize = 1 even for Cuda to workaround not-implemented issue
        // in Kokkos
        static const unsigned VectorSize = 1;
        static const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;

        /*const*/ unsigned nd = M.ndims();
        const ttb_indx nnz = X.nnz();
        const ttb_indx N = (nnz+RowBlockSize-1)/RowBlockSize;
        const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

        /*const*/ ttb_indx nh = window.size();
        if (Mprev.ndims() > 0 && Mprev.ncomponents() > 0) {
          if (Mt[nd-1].nRows() != nh)
            Genten::error("GCP_ValueHistory::run():  temporal mode size of ktensor M (" + std::to_string(Mt[nd-1].nRows()) + ") does not match given history window (" + std::to_string(nh) + ")!");
          if (Mprev[nd-1].nRows() != nh)
            Genten::error("GCP_ValueHistory::run():  temporal mode size of ktensor Mprev (" + std::to_string(Mt[nd-1].nRows()) + ") does not match given history window (" + std::to_string(nh) + ")!");
        }

        GCP_ValueHistoryFunctor<ExecSpace,loss_type,TeamSize,VectorSize,FacBlockSize,RowBlockSize> func(X, M, Mt, Mprev, window, window_penalty, w, f);
        Kokkos::TeamPolicy<ExecSpace> policy(N, TeamSize, VectorSize);
        ttb_real value[2];
        Kokkos::parallel_reduce(
          "GCP_ValueHistory",
          policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
          func, value);
        Kokkos::fence();  // ensure value is updated before using it
        val_ten = value[0];
        val_his = value[1];
      }
    };

    template <typename ExecSpace, typename loss_type>
    ttb_real gcp_value(const SptensorT<ExecSpace>& X,
                       const KtensorT<ExecSpace>& M,
                       const ArrayT<ExecSpace>& w,
                       const loss_type& f)
    {
#if 1
      GCP_Value<ExecSpace,loss_type> kernel(X,M,w,f);
      run_row_simd_kernel(kernel, M.ncomponents());
      return kernel.value;
#else
      const ttb_indx nnz = X.nnz();
      const unsigned nd = M.ndims();
      const unsigned nc = M.ncomponents();
      Kokkos::RangePolicy<ExecSpace> policy(0, nnz);
      ttb_real v = 0.0;
      Kokkos::parallel_reduce("GCP_RolObjective::value", policy,
                              KOKKOS_LAMBDA(const ttb_indx i, ttb_real& d)
      {
        // Compute Ktensor value
        ttb_real m_val = 0.0;
        for (unsigned j=0; j<nc; ++j) {
          ttb_real tmp = M.weights(j);
          for (unsigned m=0; m<nd; ++m)
            tmp *= M[m].entry(X.subscript(i,m),j);
          m_val += tmp;
        }

        // Evaluate link function
        d += w[i] * f.value(X.value(i), m_val);
      }, v);
      Kokkos::fence();  // ensure v is updated before using it
      return v;
#endif
    }

    template <typename ExecSpace, typename loss_type>
    void gcp_value(const SptensorT<ExecSpace>& X,
                   const KtensorT<ExecSpace>& M,
                   const KtensorT<ExecSpace>& Mprev,
                   const ArrayT<ExecSpace>& window,
                   const ttb_real window_penalty,
                   const ArrayT<ExecSpace>& w,
                   const loss_type& f,
                   ttb_real& val_ten,
                   ttb_real& val_his)
    {
      if (Mprev.ndims() > 0 && Mprev.ncomponents() > 0) {
        GCP_ValueHistory<ExecSpace,loss_type> kernel(
          X,M,Mprev,window,window_penalty,w,f);
        run_row_simd_kernel(kernel, M.ncomponents());
        val_ten = kernel.val_ten;
        val_his = kernel.val_his;
      }
      else {
        GCP_Value<ExecSpace,loss_type> kernel(X,M,w,f);
        run_row_simd_kernel(kernel, M.ncomponents());
        val_ten = kernel.value;
        val_his = 0.0;
      }
    }

  }

}

#include "Genten_GCP_LossFunctions.hpp"

#define LOSS_INST_MACRO(SPACE,LOSS)                                     \
  template ttb_real                                                     \
  Impl::gcp_value<SPACE,LOSS>(const SptensorT<SPACE>& X,                \
                              const KtensorT<SPACE>& M,                 \
                              const ArrayT<SPACE>& w,                   \
                              const LOSS& f);                           \
                                                                        \
  template void                                                         \
  Impl::gcp_value<SPACE,LOSS>(const SptensorT<SPACE>& X,                \
                              const KtensorT<SPACE>& M,                 \
                              const KtensorT<SPACE>& Mprev,             \
                              const ArrayT<SPACE>& window,              \
                              const ttb_real window_penalty,            \
                              const ArrayT<SPACE>& w,                   \
                              const LOSS& f,                            \
                              ttb_real& val_ten,                        \
                              ttb_real& val_his);

#define INST_MACRO(SPACE)                                               \
  LOSS_INST_MACRO(SPACE,GaussianLossFunction)                           \
  LOSS_INST_MACRO(SPACE,RayleighLossFunction)                           \
  LOSS_INST_MACRO(SPACE,GammaLossFunction)                              \
  LOSS_INST_MACRO(SPACE,BernoulliLossFunction)                          \
  LOSS_INST_MACRO(SPACE,PoissonLossFunction)

GENTEN_INST(INST_MACRO)
