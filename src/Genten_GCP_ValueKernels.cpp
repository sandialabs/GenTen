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

        static const bool is_gpu = Genten::is_gpu_space<exec_space>::value;
        static const unsigned RowBlockSize = 128;
        static const unsigned FacBlockSize = FBS;
        static const unsigned VectorSize = is_gpu ? VS : 1;
        static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;

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
                team, M, X, i);

            // Evaluate link function
            d += w[i] * f.value(X.value(i), m_val);
          }
        }, v);
        Kokkos::fence();  // ensure v is updated before using it
        value = v;
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

  }

}

#include "Genten_GCP_LossFunctions.hpp"

#define LOSS_INST_MACRO(SPACE,LOSS)                                     \
  template ttb_real                                                     \
  Impl::gcp_value<SPACE,LOSS>(const SptensorT<SPACE>& X,                \
                              const KtensorT<SPACE>& M,                 \
                              const ArrayT<SPACE>& w,                   \
                              const LOSS& f);

#define INST_MACRO(SPACE)                                               \
  LOSS_INST_MACRO(SPACE,GaussianLossFunction)                           \
  LOSS_INST_MACRO(SPACE,RayleighLossFunction)                           \
  LOSS_INST_MACRO(SPACE,GammaLossFunction)                              \
  LOSS_INST_MACRO(SPACE,BernoulliLossFunction)                          \
  LOSS_INST_MACRO(SPACE,PoissonLossFunction)

GENTEN_INST(INST_MACRO)
