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

#include <algorithm>

#include "Genten_GCP_SS_Grad_SA.hpp"
#include "Genten_SimdKernel.hpp"
#include "Genten_KokkosAlgs.hpp"

namespace Genten {

  namespace Impl {

    // Gradient kernel for gcp_sgd that combines sampling and mttkrp for
    // computing the gradient.  It also does each mode in the MTTKRP for each
    // nonzero, rather than doing a full MTTKRP for each mode.  This speeds it
    // up significantly.  This is based on a sparse array approach combining
    // sampling, MTTKRP, step, and project in one kernel that doesn't rely on
    // atomics.
    template <unsigned FBS, unsigned VS,
              typename ExecSpace, typename loss_type>
    void gcp_sgd_ss_grad_sa_kernel(
      const SptensorImpl<ExecSpace>& X,
      const KtensorImpl<ExecSpace>& M,
      const loss_type& f,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorImpl<ExecSpace>& G,
      const Kokkos::View<ttb_indx**,Kokkos::LayoutLeft,ExecSpace>& Gind,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams,
      SystemTimer& timer,
      const int timer_nzs,
      const int timer_zs)
    {
      typedef Kokkos::TeamPolicy<ExecSpace> Policy;
      typedef typename Policy::member_type TeamMember;
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

      static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
      static const unsigned RowBlockSize = 1;
      static const unsigned FacBlockSize = FBS;
      static const unsigned VectorSize = is_gpu ? VS : 1;
      static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;
      static const unsigned RowsPerTeam = TeamSize * RowBlockSize;

      /*const*/ unsigned nd = M.ndims();
      /*const*/ unsigned nc = M.ncomponents();
      /*const*/ ttb_indx ns_nz = num_samples_nonzeros;
      /*const*/ ttb_indx ns_z = num_samples_zeros;
      /*const*/ ttb_indx nnz = X.nnz();
      const ttb_indx N_nz = (ns_nz+RowsPerTeam-1)/RowsPerTeam;
      const ttb_indx N_z = (ns_z+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

      timer.start(timer_nzs);
      Policy policy_nz(N_nz, TeamSize, VectorSize);
      Kokkos::parallel_for(
        "gcp_sgd_ss_grad_sa_nonzero_kernel",
        policy_nz.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          /*const*/ ttb_indx idx = offset + ii;
          if (idx >= ns_nz)
            continue;

          // Generate random tensor index
          ttb_real x_val = 0.0;
          Kokkos::single( Kokkos::PerThread( team ), [&] (ttb_real& xv)
          {
            const ttb_indx i = Rand::draw(gen,0,nnz);
            for (ttb_indx m=0; m<nd; ++m)
              ind[m] = X.subscript(i,m);
            xv = X.value(i);
          }, x_val);

          // Compute Ktensor value
          const ttb_real m_val =
            compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, M, ind);

          // Compute Y value
          const ttb_real y_val =
            weight_nonzeros * ( f.deriv(x_val, m_val) -
                                f.deriv(ttb_real(0.0), m_val) );

          auto row_func = [&](auto j, auto nj, auto Nj, auto n) {
            typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;

            auto tmp = TVM::make(team, nj, y_val);
            for (unsigned m=0; m<nd; ++m) {
              if (m != n)
                tmp *= &(M[m].entry(ind[m],j));
            }
            tmp.store(&(G[n].entry(idx,j)));
          };

          for (unsigned n=0; n<nd; ++n) {
            Kokkos::single( Kokkos::PerThread( team ), [&] ()
            {
              Gind(idx,n) = ind[n];
            });
            for (unsigned j=0; j<nc; j+=FacBlockSize) {
              if (j+FacBlockSize <= nc) {
                const unsigned nj = FacBlockSize;
                row_func(j, nj, std::integral_constant<unsigned,nj>(), n);
              }
              else {
                const unsigned nj = nc-j;
                row_func(j, nj, std::integral_constant<unsigned,0>(), n);
              }
            } // j
          } // n
        } // i
        rand_pool.free_state(gen);
      });
      timer.stop(timer_nzs);

      timer.start(timer_zs);
      Policy policy_z(N_z, TeamSize, VectorSize);
      Kokkos::parallel_for(
        "gcp_sgd_ss_grad_sa_zero_kernel",
        policy_z.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          /*const*/ ttb_indx idx = offset + ii;
          if (idx >= ns_z)
            continue;

          // Generate index -- use broadcast form to force warp sync
          // so that ind is updated before used by other threads
          int sync = 0;
          Kokkos::single( Kokkos::PerThread( team ), [&] (int& s)
          {
            for (ttb_indx m=0; m<nd; ++m)
              ind[m] = Rand::draw(gen,0,X.size(m));
            s = 1;
          }, sync);

          // Compute Ktensor value
          const ttb_real m_val =
            compute_Ktensor_value<ExecSpace, FacBlockSize, VectorSize>(team, M, ind);

          // Compute Y value
          const ttb_real y_val = weight_zeros * f.deriv(ttb_real(0.0), m_val);

          auto row_func = [&](auto j, auto nj, auto Nj, auto n) {
            typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;

            auto tmp = TVM::make(team, nj, y_val);
            for (unsigned m=0; m<nd; ++m) {
              if (m != n)
                tmp *= &(M[m].entry(ind[m],j));
            }
            tmp.store(&(G[n].entry(idx+ns_nz,j)));
          };

          for (unsigned n=0; n<nd; ++n) {
            Kokkos::single( Kokkos::PerThread( team ), [&] ()
            {
              Gind(idx+ns_nz,n) = ind[n];
            });
            for (unsigned j=0; j<nc; j+=FacBlockSize) {
              if (j+FacBlockSize <= nc) {
                const unsigned nj = FacBlockSize;
                row_func(j, nj, std::integral_constant<unsigned,nj>(), n);
              }
              else {
                const unsigned nj = nc-j;
                row_func(j, nj, std::integral_constant<unsigned,0>(), n);
              }
            } // j
          } // n
        } // i
        rand_pool.free_state(gen);
      });
      timer.stop(timer_zs);
    }

    template <typename ExecSpace, typename loss_type>
    struct GCP_SS_Grad_SA {
      typedef ExecSpace exec_space;
      typedef SptensorImpl<exec_space> tensor_type;
      typedef KtensorImpl<exec_space> Ktensor_type;
      typedef Kokkos::View<ttb_indx**,Kokkos::LayoutLeft,ExecSpace> grad_index_type;

      const tensor_type X;
      const Ktensor_type M;
      const loss_type f;
      const ttb_indx num_samples_nonzeros;
      const ttb_indx num_samples_zeros;
      const ttb_real weight_nonzeros;
      const ttb_real weight_zeros;
      const Ktensor_type G;
      const grad_index_type Gind;
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool;
      const AlgParams algParams;
      SystemTimer& timer;
      const int timer_nzs;
      const int timer_zs;

      GCP_SS_Grad_SA(const tensor_type& X_, const Ktensor_type& M_,
                     const loss_type& f_,
                     const ttb_indx num_samples_nonzeros_,
                     const ttb_indx num_samples_zeros_,
                     const ttb_real weight_nonzeros_,
                     const ttb_real weight_zeros_,
                     const Ktensor_type& G_,
                     const grad_index_type& Gind_,
                     Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool_,
                     const AlgParams& algParams_,
                     SystemTimer& timer_,
                     const int timer_nzs_,
                     const int timer_zs_) :
        X(X_), M(M_), f(f_),
        num_samples_nonzeros(num_samples_nonzeros_),
        num_samples_zeros(num_samples_zeros_),
        weight_nonzeros(weight_nonzeros_),
        weight_zeros(weight_zeros_),
        G(G_), Gind(Gind_), rand_pool(rand_pool_), algParams(algParams_),
        timer(timer_), timer_nzs(timer_nzs_), timer_zs(timer_zs_) {}

      template <unsigned FBS, unsigned VS>
      void run() const {
        gcp_sgd_ss_grad_sa_kernel<FBS,VS>(
          X,M,f,num_samples_nonzeros,num_samples_zeros,
          weight_nonzeros,weight_zeros,G,Gind,rand_pool,algParams,
          timer,timer_nzs,timer_zs);
      }
    };

    template <typename ExecSpace, typename loss_type>
    void gcp_sgd_ss_grad_sa(
      const SptensorT<ExecSpace>& X,
      const KokkosVector<ExecSpace>& M,
      const loss_type& f,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KokkosVector<ExecSpace>& G,
      const Kokkos::View<ttb_indx**,Kokkos::LayoutLeft,ExecSpace>& Gind,
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
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams,
      SystemTimer& timer,
      const int timer_nzs,
      const int timer_zs,
      const int timer_sort,
      const int timer_scan,
      const int timer_step)
    {
      // Compute sparse gradient
      KtensorImpl<ExecSpace> Mt = M.getKtensor().impl();
      KtensorImpl<ExecSpace> Gt = G.getKtensor().impl();
      GCP_SS_Grad_SA<ExecSpace,loss_type> kernel(
        X.impl(),Mt,f,num_samples_nonzeros,num_samples_zeros,
        weight_nonzeros,weight_zeros,Gt,Gind,rand_pool,algParams,
        timer,timer_nzs,timer_zs);
      run_row_simd_kernel(kernel, Mt.ncomponents());

      if (X.getProcessorMap() != nullptr) {
        Kokkos::fence();
        for (ttb_indx n=0; n<X.ndims(); ++n)
          X.getProcessorMap()->subGridAllReduce(n, Gt[n].view().data(),
                                                Gt[n].view().span());
      }

      const ttb_indx ns = Gind.extent(0);
      const ttb_indx nd = Gind.extent(1);
      KtensorImpl<ExecSpace> mt = adam_m.getKtensor().impl();
      KtensorImpl<ExecSpace> vt = adam_v.getKtensor().impl();
      for (ttb_indx n=0; n<nd; ++n) {
        // Keys for dimension n
        Kokkos::View<ttb_indx*,Kokkos::LayoutLeft,ExecSpace> Gind_n =
          Kokkos::subview(Gind, Kokkos::ALL, n);

        // Sort keys
        timer.start(timer_sort);
        Genten::perm_sort(perm, Gind_n);
        timer.stop(timer_sort);

        // Segmented reduction based on sorted Gind key
        timer.start(timer_scan);
        Genten::key_scan(Gt[n].view(), Gind_n, perm);
        timer.stop(timer_scan);

        // Step
        timer.start(timer_step);
        typedef Genten::SpaceProperties<ExecSpace> Prop;
        const unsigned R = Mt.ncomponents();
        const unsigned vector_size = Prop::is_gpu ? R : 1;
        const unsigned team_size = Prop::is_gpu ? 256/vector_size : 1;
        const ttb_indx league_size = (ns+team_size-1)/team_size;
        typedef Kokkos::TeamPolicy<ExecSpace> Policy;
        typedef typename Policy::member_type TeamMember;
        Policy policy(league_size, team_size, vector_size);
        Kokkos::parallel_for("Genten::Impl::gcp_sgd_ss_grad_sa::step_clip",
                             policy, KOKKOS_LAMBDA(const TeamMember& team)
        {
          using std::sqrt;
          const ttb_indx i =
            team.league_rank()*team.team_size() + team.team_rank();
          if (i >= ns) return;
          const ttb_indx p = perm(i);
          const ttb_indx row = Gind_n(p);
          if (i == ns-1 || row != Gind_n(perm(i+1))) {
            if (use_adam) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, R),
                                   [&] (const unsigned& j)
              {
                ttb_real g = Gt[n].entry(p,j);
                ttb_real& u = Mt[n].entry(row,j);
                ttb_real& m = mt[n].entry(row,j);
                ttb_real& v = vt[n].entry(row,j);
                m = beta1*m + (1.0-beta1)*g;
                v = beta2*v + (1.0-beta2)*g*g;
                u -= step*m/(sqrt(v+eps));
                if (has_bounds)
                  u = u < lb ? lb : (u > ub ? ub : u);
              });
            }
            else {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, R),
                                   [&] (const unsigned& j)
              {
                ttb_real g = Gt[n].entry(p,j);
                ttb_real& u = Mt[n].entry(row,j);
                u -= step*g;
                if (has_bounds)
                  u = u < lb ? lb : (u > ub ? ub : u);
              });
            }
          }
        });
        timer.stop(timer_step);
      }
    }

  }

}

#define LOSS_INST_MACRO(SPACE,LOSS)                                     \
  template void Impl::gcp_sgd_ss_grad_sa(                               \
    const SptensorT<SPACE>& X,                                          \
    const KokkosVector<SPACE>& M,                                       \
    const LOSS& f,                                                      \
    const ttb_indx num_samples_nonzeros,                                \
    const ttb_indx num_samples_zeros,                                   \
    const ttb_real weight_nonzeros,                                     \
    const ttb_real weight_zeros,                                        \
    const KokkosVector<SPACE>& G,                                       \
    const Kokkos::View<ttb_indx**,Kokkos::LayoutLeft,SPACE>& Gind,      \
    const Kokkos::View<ttb_indx*,SPACE>& perm,                          \
    const bool use_adam,                                                \
    const KokkosVector<SPACE>& adam_m,                                  \
    const KokkosVector<SPACE>& adam_v,                                  \
    const ttb_real beta1,                                               \
    const ttb_real beta2,                                               \
    const ttb_real eps,                                                 \
    const ttb_real step,                                                \
    const bool has_bounds,                                              \
    const ttb_real lb,                                                  \
    const ttb_real ub,                                                  \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams,                                         \
    SystemTimer& timer,                                                 \
    const int timer_nzs,                                                \
    const int timer_zs,                                                 \
    const int timer_sort,                                               \
    const int timer_scan,                                               \
    const int timer_step);
