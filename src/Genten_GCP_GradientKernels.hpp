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

#include "Genten_Tensor.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_SimdKernel.hpp"
#include "Genten_AlgParams.hpp"

namespace Genten {

  namespace Impl {

    template <typename ExecSpace, typename Layout, typename loss_type>
    struct GCP_Grad_Tensor {
      typedef TensorImpl<ExecSpace,Layout> tensor_type;
      typedef KtensorImpl<ExecSpace> Ktensor_type;

      const tensor_type XX;
      const Ktensor_type MM;
      const ttb_real ww;
      const loss_type ff;
      const tensor_type YY;

      GCP_Grad_Tensor(const tensor_type& X_, const Ktensor_type& M_,
                      const ttb_real w_, const loss_type& f_,
                      const tensor_type& Y_) :
        XX(X_), MM(M_), ww(w_), ff(f_), YY(Y_) {}

      template <unsigned FBS, unsigned VS>
      void run() const
      {
        typedef typename tensor_type::exec_space exec_space;
        typedef Kokkos::TeamPolicy<exec_space> Policy;
        typedef typename Policy::member_type TeamMember;
        typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

        const auto X = XX;
        const Ktensor_type M = MM;
        const ttb_real w = ww;
        const loss_type f = ff;
        const auto Y = YY;

        static const bool is_gpu = Genten::is_gpu_space<exec_space>::value;
        static const unsigned RowBlockSize = 128;
        static const unsigned FacBlockSize = FBS;
        static const unsigned VectorSize = is_gpu ? VS : 1;
        static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;

        /*const*/ ttb_indx nnz = X.numel();
        /*const*/ unsigned nd = M.ndims();
        const ttb_indx N = (nnz+RowBlockSize-1)/RowBlockSize;
        const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

        Policy policy(N, TeamSize, VectorSize);
        Kokkos::parallel_for("GCP_Gradient: Y eval",
                             policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
                             KOKKOS_LAMBDA(const TeamMember& team)
        {
          TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
          ttb_indx *ind = &(team_ind(team.team_rank(),0));

          for (ttb_indx ii=team.team_rank(); ii<RowBlockSize; ii+=TeamSize) {
            const ttb_indx i = team.league_rank()*RowBlockSize + ii;
            if (i >= nnz)
              continue;

            // Compute subscripts for this entry
            Kokkos::single( Kokkos::PerThread( team ), [&] ()
            {
              X.ind2sub(ind,i);
            });

            // Compute Ktensor value
            ttb_real m_val =
              compute_Ktensor_value<exec_space, FacBlockSize, VectorSize>(
                team, M, ind);

            // Evaluate link function derivative
            Kokkos::single( Kokkos::PerThread( team ), [&] ()
            {
              Y[i] = w * f.deriv(X[i], m_val);
            });
          }
        });
      }
    };

    template <typename ExecSpace, typename loss_type>
    void gcp_gradient(const TensorT<ExecSpace>& X,
                      TensorT<ExecSpace>& Y,
                      const KtensorT<ExecSpace>& M,
                      const ttb_real w,
                      const loss_type& f,
                      const KtensorT<ExecSpace>& G,
                      const AlgParams& algParams)
    {
      // Compute Y tensor
      {
        GENTEN_TIME_MONITOR("GCP_Gradient: Y eval");

        if (X.has_left_impl()) {
          // Resize Y if necessary
          if (Y.numel() != X.numel())
            Y = TensorT<ExecSpace>(X.size(),0.0,TensorLayout::Left);

          GCP_Grad_Tensor<ExecSpace,Impl::TensorLayoutLeft,loss_type> kernel(
            X.left_impl(),M.impl(),w,f,Y.left_impl());
          run_row_simd_kernel(kernel, M.ncomponents());
        }
        else {
          // Resize Y if necessary
          if (Y.numel() != X.numel())
            Y = TensorT<ExecSpace>(X.size(),0.0,TensorLayout::Right);

          GCP_Grad_Tensor<ExecSpace,Impl::TensorLayoutRight,loss_type> kernel(
            X.right_impl(),M.impl(),w,f,Y.right_impl());
          run_row_simd_kernel(kernel, M.ncomponents());
        }
      }

      // Compute gradient
      {
        GENTEN_TIME_MONITOR("GCP_Gradient: mttkrp");
        G.weights() = 1.0;
        mttkrp_all(Y, M, G, algParams);
      }
    }

  }

}
