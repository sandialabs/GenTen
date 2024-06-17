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

#include "Genten_GCP_SS_Grad.hpp"
#include "Genten_SimdKernel.hpp"
#include "Genten_GCP_SamplingKernels.hpp"
#include "Genten_Array.hpp"
#include "Genten_MixedFormatOps.hpp"

// This is a locally-modified version of Kokkos_ScatterView.hpp which we
// need until the changes are moved into Kokkos
#include "Genten_Kokkos_ScatterView.hpp"

namespace Genten {

  namespace Impl {

    // Gradient kernel for gcp_sgd that combines sampling and mttkrp for
    // computing the gradient.  It also does each mode in the MTTKRP for each
    // nonzero, rather than doing a full MTTKRP for each mode.  This speeds it
    // up significantly.  Uses scatter view.  Because of problems with scatter
    // view, doesn't work on GPU.
    template <int Dupl, int Cont, unsigned FBS, unsigned VS,
              typename ExecSpace, typename loss_type>
    void gcp_sgd_ss_grad_sv_kernel(
      const SptensorImpl<ExecSpace>& X,
      const KtensorImpl<ExecSpace>& M,
      const loss_type& f,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorImpl<ExecSpace>& G,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams,
      SystemTimer& timer,
      const int timer_nzs,
      const int timer_zs)
    {
      using Kokkos::Experimental::create_scatter_view;
      using Kokkos::Experimental::ScatterView;
      using Kokkos::Experimental::ScatterSum;

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

      static_assert(!is_gpu,
                    "Cannot call gcp_sgd_ss_grad_sv_kernel for Cuda, HIP or SYCL space!");

      /*const*/ unsigned nd = M.ndims();
      /*const*/ unsigned nc = M.ncomponents();
      /*const*/ ttb_indx ns_nz = num_samples_nonzeros;
      /*const*/ ttb_indx ns_z = num_samples_zeros;
      /*const*/ ttb_indx nnz = X.nnz();
      const ttb_indx N_nz = (ns_nz+RowsPerTeam-1)/RowsPerTeam;
      const ttb_indx N_z = (ns_z+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

      typedef ScatterView<ttb_real**,Kokkos::LayoutRight,ExecSpace,ScatterSum,Dupl,Cont> ScatterViewType;
      ScatterViewType *sa = new ScatterViewType[nd];
      for (unsigned n=0; n<nd; ++n)
        sa[n] = ScatterViewType(G[n].view());

      timer.start(timer_nzs);
      Policy policy_nz(N_nz, TeamSize, VectorSize);
      Kokkos::parallel_for(
        "gcp_sgd_ss_grad_sv_nonzero_kernel",
        policy_nz.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx idx = offset + ii;
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

          auto row_func = [&](auto j, auto nj, auto Nj, auto k, auto n, auto& ga)
          {
            typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;

            auto tmp = TVM::make(team, nj, y_val);
            for (unsigned m=0; m<nd; ++m) {
              if (m != n)
                tmp *= &(M[m].entry(ind[m],j));
            }
            //Kokkos::atomic_add(&(G[n].entry(k,j)), tmp);
            ga(k,j) += tmp;
          };

          for (unsigned n=0; n<nd; ++n) {
            auto ga = sa[n].access();
            const ttb_indx k = ind[n];
            for (unsigned j=0; j<nc; j+=FacBlockSize) {
              if (j+FacBlockSize <= nc) {
                const unsigned nj = FacBlockSize;
                row_func(j, nj, std::integral_constant<unsigned,nj>(), k, n, ga);
              }
              else {
                const unsigned nj = nc-j;
                row_func(j, nj, std::integral_constant<unsigned,0>(), k, n, ga);
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
        "gcp_sgd_ss_grad_sv_zero_kernel",
        policy_z.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx idx = offset + ii;
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

          auto row_func = [&](auto j, auto nj, auto Nj, auto k, auto n, auto& ga)
          {
            typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;

            auto tmp = TVM::make(team, nj, y_val);
            for (unsigned m=0; m<nd; ++m) {
              if (m != n)
                tmp *= &(M[m].entry(ind[m],j));
            }
            //Kokkos::atomic_add(&(G[n].entry(k,j)), tmp);
            ga(k,j) += tmp;
          };

          for (unsigned n=0; n<nd; ++n) {
            auto ga = sa[n].access();
            const ttb_indx k = ind[n];
            for (unsigned j=0; j<nc; j+=FacBlockSize) {
              if (j+FacBlockSize <= nc) {
                const unsigned nj = FacBlockSize;
                row_func(j, nj, std::integral_constant<unsigned,nj>(), k, n, ga);
              }
              else {
                const unsigned nj = nc-j;
                row_func(j, nj, std::integral_constant<unsigned,0>(), k, n, ga);
              }
            } // j
          } // n
        } // i
        rand_pool.free_state(gen);
      });
      timer.stop(timer_zs);

      for (unsigned n=0; n<nd; ++n)
        sa[n].contribute_into(G[n].view());
      delete [] sa;
    }

    // Gradient kernel for gcp_sgd that combines sampling and mttkrp for
    // computing the gradient.  It also does each mode in the MTTKRP for each
    // nonzero, rather than doing a full MTTKRP for each mode.  This speeds it
    // up significantly.  Obviously it only works with atomic writes.
    template <unsigned FBS, unsigned VS,
              typename ExecSpace, typename loss_type>
    void gcp_sgd_ss_grad_atomic_kernel(
      const SptensorImpl<ExecSpace>& X,
      const KtensorImpl<ExecSpace>& M,
      const loss_type& f,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorImpl<ExecSpace>& G,
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
        "gcp_sgd_ss_grad_atomic_nonzero_kernel",
        policy_nz.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx idx = offset + ii;
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

          auto row_func = [&](auto j, auto nj, auto Nj, auto k, auto n) {
            typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;

            auto tmp = TVM::make(team, nj, y_val);
            for (unsigned m=0; m<nd; ++m) {
              if (m != n)
                tmp *= &(M[m].entry(ind[m],j));
            }
            Kokkos::atomic_add(&(G[n].entry(k,j)), tmp);
          };

          for (unsigned n=0; n<nd; ++n) {
            const ttb_indx k = ind[n];
            for (unsigned j=0; j<nc; j+=FacBlockSize) {
              if (j+FacBlockSize <= nc) {
                const unsigned nj = FacBlockSize;
                row_func(j, nj, std::integral_constant<unsigned,nj>(), k, n);
              }
              else {
                const unsigned nj = nc-j;
                row_func(j, nj, std::integral_constant<unsigned,0>(), k, n);
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
        "gcp_sgd_ss_grad_atomic_zero_kernel",
        policy_z.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx idx = offset + ii;
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

          auto row_func = [&](auto j, auto nj, auto Nj, auto k, auto n) {
            typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;

            auto tmp = TVM::make(team, nj, y_val);

            for (unsigned m=0; m<nd; ++m) {
              if (m != n)
                tmp *= &(M[m].entry(ind[m],j));
            }
            Kokkos::atomic_add(&(G[n].entry(k,j)), tmp);
          };

          for (unsigned n=0; n<nd; ++n) {
            const ttb_indx k = ind[n];
            for (unsigned j=0; j<nc; j+=FacBlockSize) {
              if (j+FacBlockSize <= nc) {
                const unsigned nj = FacBlockSize;
                row_func(j, nj, std::integral_constant<unsigned,nj>(), k, n);
              }
              else {
                const unsigned nj = nc-j;
                row_func(j, nj, std::integral_constant<unsigned,0>(), k, n);
              }
            } // j
          } // n
        } // i
        rand_pool.free_state(gen);
      });
      timer.stop(timer_zs);
    }

    template <typename ExecSpace, typename loss_type>
    struct GCP_SS_Grad {
      typedef ExecSpace exec_space;
      typedef SptensorImpl<exec_space> tensor_type;
      typedef KtensorImpl<exec_space> Ktensor_type;

      const tensor_type X;
      const Ktensor_type M;
      const loss_type f;
      const ttb_indx num_samples_nonzeros;
      const ttb_indx num_samples_zeros;
      const ttb_real weight_nonzeros;
      const ttb_real weight_zeros;
      const Ktensor_type G;
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool;
      const AlgParams algParams;
      SystemTimer& timer;
      const int timer_nzs;
      const int timer_zs;

      GCP_SS_Grad(const tensor_type& X_, const Ktensor_type& M_,
                  const loss_type& f_,
                  const ttb_indx num_samples_nonzeros_,
                  const ttb_indx num_samples_zeros_,
                  const ttb_real weight_nonzeros_,
                  const ttb_real weight_zeros_,
                  const Ktensor_type& G_,
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
        G(G_), rand_pool(rand_pool_), algParams(algParams_),
        timer(timer_), timer_nzs(timer_nzs_), timer_zs(timer_zs_) {}

      template <unsigned FBS, unsigned VS>
      void run() const {
        using Kokkos::Experimental::ScatterDuplicated;
        using Kokkos::Experimental::ScatterNonDuplicated;
        using Kokkos::Experimental::ScatterAtomic;
        using Kokkos::Experimental::ScatterNonAtomic;

        MTTKRP_All_Method::type method = algParams.mttkrp_all_method;

        if (method == MTTKRP_All_Method::Single)
          gcp_sgd_ss_grad_sv_kernel<ScatterNonDuplicated,ScatterNonAtomic,FBS,VS>(
            X,M,f,num_samples_nonzeros,num_samples_zeros,
            weight_nonzeros,weight_zeros,G,rand_pool,algParams,
            timer,timer_nzs,timer_zs);
        else if (method == MTTKRP_All_Method::Atomic)
          gcp_sgd_ss_grad_sv_kernel<ScatterNonDuplicated,ScatterAtomic,FBS,VS>(
            X,M,f,num_samples_nonzeros,num_samples_zeros,
            weight_nonzeros,weight_zeros,G,rand_pool,algParams,
            timer,timer_nzs,timer_zs);
        else if (method == MTTKRP_All_Method::Duplicated)
          gcp_sgd_ss_grad_sv_kernel<ScatterDuplicated,ScatterNonAtomic,FBS,VS>(
            X,M,f,num_samples_nonzeros,num_samples_zeros,
            weight_nonzeros,weight_zeros,G,rand_pool,algParams,
            timer,timer_nzs,timer_zs);
        else if (method == MTTKRP_All_Method::Iterated)
          Genten::error("Cannot use iterated MTTKRP method in fused stratified-sampling/MTTKRP kernel!");
      }
    };

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    // Specialization for Cuda, HIP and SYCL that always uses atomics and doesn't call
    // gcp_sgd_ss_grad_sv_kernel, which won't run on the GPU
    template <typename loss_type>
    struct GCP_SS_Grad<Kokkos_GPU_Space,loss_type> {
      typedef Kokkos_GPU_Space exec_space;
      typedef SptensorImpl<exec_space> tensor_type;
      typedef KtensorImpl<exec_space> Ktensor_type;

      const tensor_type X;
      const Ktensor_type M;
      const loss_type f;
      const ttb_indx num_samples_nonzeros;
      const ttb_indx num_samples_zeros;
      const ttb_real weight_nonzeros;
      const ttb_real weight_zeros;
      const Ktensor_type G;
      Kokkos::Random_XorShift64_Pool<exec_space>& rand_pool;
      const AlgParams algParams;
      SystemTimer& timer;
      const int timer_nzs;
      const int timer_zs;

      GCP_SS_Grad(const tensor_type& X_, const Ktensor_type& M_,
                  const loss_type& f_,
                  const ttb_indx num_samples_nonzeros_,
                  const ttb_indx num_samples_zeros_,
                  const ttb_real weight_nonzeros_,
                  const ttb_real weight_zeros_,
                  const Ktensor_type& G_,
                  Kokkos::Random_XorShift64_Pool<exec_space>& rand_pool_,
                  const AlgParams& algParams_,
                  SystemTimer& timer_,
                  const int timer_nzs_,
                  const int timer_zs_) :
        X(X_), M(M_), f(f_),
        num_samples_nonzeros(num_samples_nonzeros_),
        num_samples_zeros(num_samples_zeros_),
        weight_nonzeros(weight_nonzeros_),
        weight_zeros(weight_zeros_),
        G(G_), rand_pool(rand_pool_), algParams(algParams_),
        timer(timer_), timer_nzs(timer_nzs_), timer_zs(timer_zs_) {}

      template <unsigned FBS, unsigned VS>
      void run() const {
        if (algParams.mttkrp_all_method != MTTKRP_All_Method::Atomic)
          Genten::error("MTTKRP-All method must be atomic on Cuda, HIP or SYCL!");

        gcp_sgd_ss_grad_atomic_kernel<FBS,VS>(
          X,M,f,num_samples_nonzeros,num_samples_zeros,
          weight_nonzeros,weight_zeros,G,rand_pool,algParams,
          timer,timer_nzs,timer_zs);
      }
    };
#endif

    template <typename ExecSpace, typename loss_type>
    void gcp_sgd_ss_grad(
      const SptensorT<ExecSpace>& X,
      const KtensorT<ExecSpace>& M,
      const loss_type& f,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& G,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams,
      SystemTimer& timer,
      const int timer_nzs,
      const int timer_zs)
    {
      GCP_SS_Grad<ExecSpace,loss_type> kernel(
        X.impl(),M.impl(),f,num_samples_nonzeros,num_samples_zeros,
        weight_nonzeros,weight_zeros,G.impl(),rand_pool,algParams,
        timer,timer_nzs,timer_zs);
      run_row_simd_kernel(kernel, M.ncomponents());
    }

    template <typename ExecSpace, typename Searcher, typename Gradient>
    void gcp_sgd_ss_grad_onesided(
      const SptensorT<ExecSpace>& X,
      const KtensorT<ExecSpace>& u,
      const Searcher& searcher,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const Gradient& gradient,
      const AlgParams& algParams,
      DistKtensorUpdate<ExecSpace> *dku_ptr,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      KtensorT<ExecSpace>& u_overlapped,
      const KtensorT<ExecSpace>& G,
      const KtensorT<ExecSpace>& G_overlap,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool)
    {
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;

      // Cast to one-sided k-tensor update
      KtensorOneSidedUpdate<ExecSpace> *dku =
        dynamic_cast<KtensorOneSidedUpdate<ExecSpace>*>(dku_ptr);
      if (dku == nullptr)
        Genten::error("Cast to KtensorOneSidedUpdate failed!");

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples_nonzeros + num_samples_zeros;
      if (Y.ndims() == 0 || Y.nnz() < total_samples) {
        Y = SptensorT<ExecSpace>(X.size(), total_samples);
      }

      // auto X = create_mirror_view(Xd);
      // auto u = create_mirror_view(ud);
      // auto Y = create_mirror_view(Yd);
      // auto u_overlapped = create_mirror_view(u_overlappedd);
      // deep_copy(X, Xd);
      // deep_copy(u, ud);

      const unsigned nd = X.ndims();
      const unsigned nc = u.ncomponents();

      GENTEN_START_TIMER("sample tensor");
      dku->copyToWindows(u);
      dku->lockWindows();

      // Generate samples of nonzeros
      for (ttb_indx idx=0; idx<num_samples_nonzeros; ++idx) {
        generator_type gen = rand_pool.get_state();
        const ttb_indx i = Rand::draw(gen,0,X.nnz());
        for (ttb_indx m=0; m<nd; ++m) {
          const ttb_indx row = X.subscript(i,m);
          Y.subscript(idx,m) = row;
          dku->importRow(m, row, u, u_overlapped);
        }
        Y.value(idx) = X.value(i); // We need x_val later in gradient
        rand_pool.free_state(gen);
      }

      // Generate samples of zeros
      IndxArray ind(nd);
      for (ttb_indx idx=0; idx<num_samples_zeros; ++idx) {
        generator_type gen = rand_pool.get_state();
        // Keep generating samples until we get one not in the tensor
        bool found = true;
        while (found) {
          // Generate index
          for (ttb_indx m=0; m<nd; ++m)
            ind[m] = Rand::draw(gen,0,X.size(m));

          // Search for index
          found = searcher.search(ind);
        }

        // Add new nonzero
        for (ttb_indx m=0; m<nd; ++m) {
          Y.subscript(num_samples_nonzeros+idx,m) = ind[m];
          dku->importRow(m, ind[m], u, u_overlapped);
        }
        rand_pool.free_state(gen);
      }

      dku->unlockWindows();
      GENTEN_STOP_TIMER("sample tensor");

      GENTEN_START_TIMER("MTTKRP-all");
      dku->zeroOutWindows();
      dku->lockWindows();

      Genten::ArrayT<ExecSpace> grad_tmp(nc, 0.0);
      const ttb_indx nnz = Y.nnz();
      for (ttb_indx idx=0; idx<nnz; ++idx) {

        auto ind = Y.getSubscripts(idx);

        // Compute Ktensor value
        const ttb_real m_val = compute_Ktensor_value(u_overlapped, ind);

        // Compute gradient tensor value
        const ttb_real y_val = idx < num_samples_nonzeros ?
          gradient.evalNonZero(Y.value(idx), m_val, weight_nonzeros) :
          gradient.evalZero(m_val, weight_zeros);

        // Do MTTKRP
        for (unsigned n=0; n<nd; ++n) {
          for (unsigned j=0; j<nc; ++j) {
            ttb_real tmp = y_val;
            for (unsigned m=0; m<nd; ++m) {
              if (m != n)
                tmp *= u_overlapped[m].entry(ind[m],j);
            }
            grad_tmp[j] = tmp;
          }
          dku->exportRow(n, ind[n], grad_tmp, G);
        }
      }

      dku->unlockWindows();
      dku->copyFromWindows(G);
      GENTEN_STOP_TIMER("MTTKRP-all");
    }

  }

    // template <typename ExecSpace, typename Searcher, typename Gradient>
  //   void gcp_sgd_ss_grad_onesided(
  //     const SptensorT<ExecSpace>& X,
  //     const KtensorT<ExecSpace>& u,
  //     const Searcher& searcher,
  //     const ttb_indx num_samples_nonzeros,
  //     const ttb_indx num_samples_zeros,
  //     const ttb_real weight_nonzeros,
  //     const ttb_real weight_zeros,
  //     const Gradient& gradient,
  //     const AlgParams& algParams,
  //     DistKtensorUpdate<ExecSpace> *dku_ptr,
  //     SptensorT<ExecSpace>& Y,
  //     ArrayT<ExecSpace>& w,
  //     KtensorT<ExecSpace>& u_overlap,
  //     const KtensorT<ExecSpace>& G,
  //     const KtensorT<ExecSpace>& G_overlap,
  //     Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool)
  //   {
  //     typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
  //     typedef typename RandomPool::generator_type generator_type;
  //     typedef Kokkos::rand<generator_type, ttb_indx> Rand;

  //     // Cast to one-sided k-tensor update
  //     typedef KtensorOneSidedUpdate<ExecSpace> dku_type;
  //     dku_type *dku = dynamic_cast<dku_type*>(dku_ptr);
  //     if (dku == nullptr)
  //       Genten::error("Cast to KtensorOneSidedUpdate failed!");

  //     // Resize Y if necessary
  //     const ttb_indx total_samples = num_samples_nonzeros + num_samples_zeros;
  //     if (Y.ndims() == 0 || Y.nnz() < total_samples) {
  //       Y = SptensorT<ExecSpace>(X.size(), total_samples);
  //       Y.allocGlobalSubscripts();
  //       deep_copy(Y.getLowerBounds(), X.getLowerBounds());
  //       deep_copy(Y.getUpperBounds(), X.getUpperBounds());
  //       Y.setProcessorMap(X.getProcessorMap());
  //     }

  //     // auto X = create_mirror_view(Xd);
  //     // auto u = create_mirror_view(ud);
  //     // auto Y = create_mirror_view(Yd);
  //     // auto u_overlapped = create_mirror_view(u_overlappedd);
  //     // deep_copy(X, Xd);
  //     // deep_copy(u, ud);

  //     const unsigned nd = X.ndims();

  //     using unordered_map_type = typename dku_type::unordered_map_type;
  //     dku->maps.resize(nd);
  //     for (unsigned n=0; n<nd; ++n)
  //       dku->maps[n] = unordered_map_type(u_overlap[n].nRows());

  //     GENTEN_START_TIMER("sample tensor");
  //     dku->copyToWindows(u);
  //     dku->lockWindows();

  //     // Generate samples of nonzeros
  //     for (ttb_indx idx=0; idx<num_samples_nonzeros; ++idx) {
  //       generator_type gen = rand_pool.get_state();
  //       const ttb_indx i = Rand::draw(gen,0,X.nnz());
  //       for (ttb_indx m=0; m<nd; ++m) {
  //         const ttb_indx row = X.globalSubscript(i,m);
  //         Y.globalSubscript(idx,m) = row;
  //         if (!dku->maps[m].exists(row)) {
  //           dku->importRow(m, row, u, u_overlap);
  //           unsigned p = dku->find_proc_for_row(m, row);
  //           gt_assert(!dku->maps[m].insert(row,p).failed());
  //         }
  //       }
  //       Y.value(idx) = X.value(i); // We need x_val later in gradient
  //       rand_pool.free_state(gen);
  //     }

  //     // Generate samples of zeros
  //     IndxArray ind(nd);
  //     for (ttb_indx idx=0; idx<num_samples_zeros; ++idx) {
  //       generator_type gen = rand_pool.get_state();
  //       // Keep generating samples until we get one not in the tensor
  //       bool found = true;
  //       while (found) {
  //         // Generate index
  //         for (ttb_indx m=0; m<nd; ++m)
  //           ind[m] = Rand::draw(gen,X.lowerBound(m),X.upperBound(m));

  //         // Search for index
  //         found = searcher.search(ind);
  //       }

  //       // Add new nonzero
  //       for (ttb_indx m=0; m<nd; ++m) {
  //         const ttb_indx row = ind[m];
  //         Y.globalSubscript(num_samples_nonzeros+idx,m) = row;
  //         if (!dku->maps[m].exists(row)) {
  //           dku->importRow(m, row, u, u_overlap);
  //           unsigned p = dku->find_proc_for_row(m, row);
  //           gt_assert(!dku->maps[m].insert(row,p).failed());
  //         }
  //       }
  //       rand_pool.free_state(gen);
  //     }

  //     dku->unlockWindows();
  //     dku->fenceWindows();
  //     GENTEN_STOP_TIMER("sample tensor");

  //     // Update tensor in DKU
  //     //dku->updateTensor(Y);

  //     // Import u to overlapped tensor map
  //     //dku->doImport(u_overlap, u);

  //     const ttb_indx nnz = Y.nnz();
  //     for (ttb_indx idx=0; idx<nnz; ++idx) {

  //       auto ind = Y.getSubscripts(idx);

  //       // Compute Ktensor value
  //       const ttb_real m_val = compute_Ktensor_value(u_overlap, ind);

  //       // Compute gradient tensor value
  //       const ttb_real y_val = idx < num_samples_nonzeros ?
  //         gradient.evalNonZero(Y.value(idx), m_val, weight_nonzeros) :
  //         gradient.evalZero(m_val, weight_zeros);

  //       Y.value(idx) = y_val;
  //     }

  //     mttkrp_all(Y, u_overlap, G_overlap, algParams);
  //     dku_ptr->doExport(G, G_overlap);
  //   }

  // }

}

#include "Genten_GCP_SS_Grad_Streaming_Def.hpp"

#define LOSS_INST_MACRO(SPACE,LOSS)                                     \
  template void Impl::gcp_sgd_ss_grad(                                  \
    const SptensorT<SPACE>& X,                                          \
    const KtensorT<SPACE>& M,                                           \
    const LOSS& f,                                                      \
    const ttb_indx num_samples_nonzeros,                                \
    const ttb_indx num_samples_zeros,                                   \
    const ttb_real weight_nonzeros,                                     \
    const ttb_real weight_zeros,                                        \
    const KtensorT<SPACE>& G,                                           \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams,                                         \
    SystemTimer& timer,                                                 \
    const int timer_nzs,                                                \
    const int timer_zs);                                                \
                                                                        \
  template void Impl::gcp_sgd_ss_grad(                                  \
    const SptensorT<SPACE>& X,                                          \
    const KtensorT<SPACE>& M,                                           \
    const KtensorT<SPACE>& Mt,                                          \
    const KtensorT<SPACE>& Mprev,                                       \
    const LOSS& f,                                                      \
    const ttb_indx num_samples_nonzeros,                                \
    const ttb_indx num_samples_zeros,                                   \
    const ttb_real weight_nonzeros,                                     \
    const ttb_real weight_zeros,                                        \
    const ArrayT<SPACE>& window,                                        \
    const ttb_real window_penalty,                                      \
    const IndxArrayT<SPACE>& modes,                                     \
    const KtensorT<SPACE>& G,                                           \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams,                                         \
    SystemTimer& timer,                                                 \
    const int timer_nzs,                                                \
    const int timer_zs);                                                \
  template void Impl::gcp_sgd_ss_grad_onesided(                         \
    const SptensorT<SPACE>& Xd,                                         \
    const KtensorT<SPACE>& ud,                                          \
    const Impl::SemiStratifiedSearcher<SPACE>& searcher,                \
    const ttb_indx num_samples_nonzeros,                                \
    const ttb_indx num_samples_zeros,                                   \
    const ttb_real weight_nonzeros,                                     \
    const ttb_real weight_zeros,                                        \
    const Impl::SemiStratifiedGradient<LOSS>& gradient,                 \
    const AlgParams& algParams,                                         \
    DistKtensorUpdate<SPACE> *dku_ptr,                                  \
    SptensorT<SPACE>& Yd,                                               \
    ArrayT<SPACE>& w,                                                   \
    KtensorT<SPACE>& ud_overlap,                                        \
    const KtensorT<SPACE>& Gd,                                          \
    const KtensorT<SPACE>& Gd_overlap,                                  \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool);
