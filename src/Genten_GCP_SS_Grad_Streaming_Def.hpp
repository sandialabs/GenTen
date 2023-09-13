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
      const SptensorT<ExecSpace>& Xd,
      const KtensorT<ExecSpace>& Md,
      const KtensorT<ExecSpace>& Mtd,
      const KtensorT<ExecSpace>& Mprevd,
      const loss_type& f,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const ArrayT<ExecSpace>& window,
      const ttb_real window_penalty,
      const IndxArrayT<ExecSpace>& modes,
      const KtensorT<ExecSpace>& Gd,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams,
      SystemTimer& timer,
      const int timer_nzs,
      const int timer_zs)
    {
      const auto X = Xd.impl();
      const auto M = Md.impl();
      const auto Mt = Mtd.impl();
      const auto Mprev = Mprevd.impl();
      const auto G = Gd.impl();

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

      // Note, G is length d

      /*const*/ unsigned d = modes.size();
      /*const*/ unsigned nd = M.ndims();
      /*const*/ unsigned nc = M.ncomponents();
      /*const*/ ttb_indx ns_nz = num_samples_nonzeros;
      /*const*/ ttb_indx ns_z = num_samples_zeros;
      /*const*/ ttb_indx nnz = X.nnz();
      const ttb_indx N_nz = (ns_nz+RowsPerTeam-1)/RowsPerTeam;
      const ttb_indx N_z = (ns_z+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);
      /*const*/ ttb_indx nh = window.size();

      if (Mprev.ndims() > 0 && Mprev.ncomponents() > 0) {
        if (Mt[nd-1].nRows() != nh)
          Genten::error("gcp_sgd_ss_grad_sv_kernel():  temporal mode size of ktensor M (" + std::to_string(Mt[nd-1].nRows()) + ") does not match given history window (" + std::to_string(nh) + ")!");
        if (Mprev[nd-1].nRows() != nh)
          Genten::error("gcp_sgd_ss_grad_sv_kernel():  temporal mode size of ktensor Mprev (" + std::to_string(Mt[nd-1].nRows()) + ") does not match given history window (" + std::to_string(nh) + ")!");
      }

      typedef ScatterView<ttb_real**,Kokkos::LayoutRight,ExecSpace,ScatterSum,Dupl,Cont> ScatterViewType;
      ScatterViewType *sa = new ScatterViewType[d];
      for (unsigned n=0; n<d; ++n)
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
            ga(k,j) += tmp;
          };

          for (unsigned n=0; n<d; ++n) {
            auto ga = sa[n].access();
            const unsigned nn = modes[n];
            const ttb_indx k = ind[nn];
            for (unsigned j=0; j<nc; j+=FacBlockSize) {
              if (j+FacBlockSize <= nc) {
                const unsigned nj = FacBlockSize;
                row_func(j, nj, std::integral_constant<unsigned,nj>(), k, nn, ga);
              }
              else {
                const unsigned nj = nc-j;
                row_func(j, nj, std::integral_constant<unsigned,0>(), k, nn, ga);
              }
            } // j
          } // n

          // Add in history term
          for (ttb_indx h=0; h<nh; ++h) {
            // Modify index for history -- use broadcast form to force warp sync
            // so that ind is updated before used by other threads
            int sync = 0;
            Kokkos::single( Kokkos::PerThread( team ), [&] (int& s)
            {
              ind[nd-1] = h;
              s = 1;
            }, sync);

            // Compute Yt value
            const ttb_real mt_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, Mt, ind);
            const ttb_real mp_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, Mprev, ind);
            // Here we are using the corresponding formula for stratified
            // sampling, since we don't know if this corresponds to a true
            // zero or nonzero in the history slice
            ttb_real yt_val = window_penalty * window[h] * weight_nonzeros * (
              f.deriv(mp_val, mt_val) );

            auto row_func = [&](auto j, auto nj, auto Nj, auto k, auto n, auto& ga)
            {
              typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;

              auto tmp = TVM::make(team, nj, yt_val);
              for (unsigned m=0; m<nd; ++m) {
                if (m != n)
                  tmp *= &(Mt[m].entry(ind[m],j));
              }
              ga(k,j) += tmp;
            };

            for (unsigned n=0; n<d; ++n) {
              auto ga = sa[n].access();
              const unsigned nn = modes[n];
              const ttb_indx k = ind[nn];
              for (unsigned j=0; j<nc; j+=FacBlockSize) {
                if (j+FacBlockSize <= nc) {
                  const unsigned nj = FacBlockSize;
                  row_func(j, nj, std::integral_constant<unsigned,nj>(), k, nn, ga);
                }
                else {
                  const unsigned nj = nc-j;
                  row_func(j, nj, std::integral_constant<unsigned,0>(), k, nn, ga);
                }
              } // j
            } // n
          } // h

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
            ga(k,j) += tmp;
          };

          for (unsigned n=0; n<d; ++n) {
            auto ga = sa[n].access();
            const unsigned nn = modes[n];
            const ttb_indx k = ind[nn];
            for (unsigned j=0; j<nc; j+=FacBlockSize) {
              if (j+FacBlockSize <= nc) {
                const unsigned nj = FacBlockSize;
                row_func(j, nj, std::integral_constant<unsigned,nj>(), k, nn, ga);
              }
              else {
                const unsigned nj = nc-j;
                row_func(j, nj, std::integral_constant<unsigned,0>(), k, nn, ga);
              }
            } // j
          } // n

          // Add in history term
          for (ttb_indx h=0; h<nh; ++h) {
            // Modify index for history -- use broadcast form to force warp sync
            // so that ind is updated before used by other threads
            int sync = 0;
            Kokkos::single( Kokkos::PerThread( team ), [&] (int& s)
            {
              ind[nd-1] = h;
              s = 1;
            }, sync);

            // Compute Yt value
            const ttb_real mt_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, Mt, ind);
            const ttb_real mp_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, Mprev, ind);
            // Here we are using the corresponding formula for stratified
            // sampling, since we don't know if this corresponds to a true
            // zero or nonzero in the history slice
            ttb_real yt_val = window_penalty * window[h] * weight_zeros * (
              f.deriv(mp_val, mt_val) );

            auto row_func = [&](auto j, auto nj, auto Nj, auto k, auto n, auto& ga)
            {
              typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;

              auto tmp = TVM::make(team, nj, yt_val);
              for (unsigned m=0; m<nd; ++m) {
                if (m != n)
                  tmp *= &(Mt[m].entry(ind[m],j));
              }
              ga(k,j) += tmp;
            };

            for (unsigned n=0; n<d; ++n) {
              auto ga = sa[n].access();
              const unsigned nn = modes[n];
              const ttb_indx k = ind[nn];
              for (unsigned j=0; j<nc; j+=FacBlockSize) {
                if (j+FacBlockSize <= nc) {
                  const unsigned nj = FacBlockSize;
                  row_func(j, nj, std::integral_constant<unsigned,nj>(), k, nn, ga);
                }
                else {
                  const unsigned nj = nc-j;
                  row_func(j, nj, std::integral_constant<unsigned,0>(), k, nn, ga);
                }
              } // j
            } // n
          } // h

        } // i
        rand_pool.free_state(gen);
      });
      timer.stop(timer_zs);

      for (unsigned n=0; n<d; ++n)
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
      const SptensorT<ExecSpace>& Xd,
      const KtensorT<ExecSpace>& Md,
      const KtensorT<ExecSpace>& Mtd,
      const KtensorT<ExecSpace>& Mprevd,
      const loss_type& f,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const ArrayT<ExecSpace>& window,
      const ttb_real window_penalty,
      const IndxArrayT<ExecSpace>& modes,
      const KtensorT<ExecSpace>& Gd,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams,
      SystemTimer& timer,
      const int timer_nzs,
      const int timer_zs)
    {
      const auto X = Xd.impl();
      const auto M = Md.impl();
      const auto Mt = Mtd.impl();
      const auto Mprev = Mprevd.impl();
      const auto G = Gd.impl();

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

      // Note, G is length d

      /*const*/ unsigned d = modes.size();
      /*const*/ unsigned nd = M.ndims();
      /*const*/ unsigned nc = M.ncomponents();
      /*const*/ ttb_indx ns_nz = num_samples_nonzeros;
      /*const*/ ttb_indx ns_z = num_samples_zeros;
      /*const*/ ttb_indx nnz = X.nnz();
      const ttb_indx N_nz = (ns_nz+RowsPerTeam-1)/RowsPerTeam;
      const ttb_indx N_z = (ns_z+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);
      /*const*/ ttb_indx nh = window.size();

      if (Mprev.ndims() > 0 && Mprev.ncomponents() > 0) {
        if (Mt[nd-1].nRows() != nh)
          Genten::error("gcp_sgd_ss_grad_atomic_kernel():  temporal mode size of ktensor M does not match given history window!");
        if (Mprev[nd-1].nRows() != nh)
          Genten::error("gcp_sgd_ss_grad_atomic_kernel():  temporal mode size of ktensor Mprev does not match given history window!");
      }

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

          auto row_func = [&](auto j, auto nj, auto Nj, auto k, auto n, auto nn) {
            typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;

            auto tmp = TVM::make(team, nj, y_val);
            for (unsigned m=0; m<nd; ++m) {
              if (m != nn)
                tmp *= &(M[m].entry(ind[m],j));
            }
            Kokkos::atomic_add(&(G[n].entry(k,j)), tmp);
          };

          for (unsigned n=0; n<d; ++n) {
            const unsigned nn = modes[n];
            const ttb_indx k = ind[nn];
            for (unsigned j=0; j<nc; j+=FacBlockSize) {
              if (j+FacBlockSize <= nc) {
                const unsigned nj = FacBlockSize;
                row_func(j, nj, std::integral_constant<unsigned,nj>(), k, n, nn);
              }
              else {
                const unsigned nj = nc-j;
                row_func(j, nj, std::integral_constant<unsigned,0>(), k, n, nn);
              }
            } // j
          } // n

          // Add in history term
          for (ttb_indx h=0; h<nh; ++h) {
            // Modify index for history -- use broadcast form to force warp sync
            // so that ind is updated before used by other threads
            int sync = 0;
            Kokkos::single( Kokkos::PerThread( team ), [&] (int& s)
            {
              ind[nd-1] = h;
              s = 1;
            }, sync);

            // Compute Yt value
            const ttb_real mt_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, Mt, ind);
            const ttb_real mp_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, Mprev, ind);
            ttb_real yt_val = window_penalty * window[h] * weight_nonzeros * (
                f.deriv(mp_val, mt_val) );

            auto row_func = [&](auto j, auto nj, auto Nj, auto k, auto n, auto nn) {
              typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;

              auto tmp = TVM::make(team, nj, yt_val);
              for (unsigned m=0; m<nd; ++m) {
                if (m != nn)
                  tmp *= &(Mt[m].entry(ind[m],j));
              }
              Kokkos::atomic_add(&(G[n].entry(k,j)), tmp);
            };

            for (unsigned n=0; n<d; ++n) {
              const unsigned nn = modes[n];
              const ttb_indx k = ind[nn];
              for (unsigned j=0; j<nc; j+=FacBlockSize) {
                if (j+FacBlockSize <= nc) {
                  const unsigned nj = FacBlockSize;
                  row_func(j, nj, std::integral_constant<unsigned,nj>(), k, n, nn);
                }
                else {
                  const unsigned nj = nc-j;
                  row_func(j, nj, std::integral_constant<unsigned,0>(), k, n, nn);
                }
              } // j
            } // n
          } // h

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

          auto row_func = [&](auto j, auto nj, auto Nj, auto k, auto n, auto nn) {
            typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;

            auto tmp = TVM::make(team, nj, y_val);
            for (unsigned m=0; m<nd; ++m) {
              if (m != nn)
                tmp *= &(M[m].entry(ind[m],j));
            }
            Kokkos::atomic_add(&(G[n].entry(k,j)), tmp);
          };

          for (unsigned n=0; n<d; ++n) {
            const unsigned nn = modes[n];
            const ttb_indx k = ind[nn];
            for (unsigned j=0; j<nc; j+=FacBlockSize) {
              if (j+FacBlockSize <= nc) {
                const unsigned nj = FacBlockSize;
                row_func(j, nj, std::integral_constant<unsigned,nj>(), k, n, nn);
              }
              else {
                const unsigned nj = nc-j;
                row_func(j, nj, std::integral_constant<unsigned,0>(), k, n, nn);
              }
            } // j
          } // n

          // Add in history term
          for (ttb_indx h=0; h<nh; ++h) {
            // Modify index for history -- use broadcast form to force warp sync
            // so that ind is updated before used by other threads
            int sync = 0;
            Kokkos::single( Kokkos::PerThread( team ), [&] (int& s)
            {
              ind[nd-1] = h;
              s = 1;
            }, sync);

            // Compute Yt value
            const ttb_real mt_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, Mt, ind);
            const ttb_real mp_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, Mprev, ind);
            ttb_real yt_val = window_penalty * window[h] * weight_zeros * (
                f.deriv(mp_val, mt_val) );

            auto row_func = [&](auto j, auto nj, auto Nj, auto k, auto n, auto nn) {
              typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;

              auto tmp = TVM::make(team, nj, yt_val);
              for (unsigned m=0; m<nd; ++m) {
                if (m != nn)
                  tmp *= &(Mt[m].entry(ind[m],j));
              }
              Kokkos::atomic_add(&(G[n].entry(k,j)), tmp);
            };

            for (unsigned n=0; n<d; ++n) {
              const unsigned nn = modes[n];
              const ttb_indx k = ind[nn];
              for (unsigned j=0; j<nc; j+=FacBlockSize) {
                if (j+FacBlockSize <= nc) {
                  const unsigned nj = FacBlockSize;
                  row_func(j, nj, std::integral_constant<unsigned,nj>(), k, n, nn);
                }
                else {
                  const unsigned nj = nc-j;
                  row_func(j, nj, std::integral_constant<unsigned,0>(), k, n, nn);
                }
              } // j
            } // n
          } // h
        } // i
        rand_pool.free_state(gen);
      });
      timer.stop(timer_zs);
    }

    template <typename ExecSpace, typename loss_type>
    struct GCP_SS_Grad_Str {
      typedef ExecSpace exec_space;
      typedef SptensorT<exec_space> tensor_type;
      typedef KtensorT<exec_space> Ktensor_type;
      typedef ArrayT<exec_space> Array_type;
      typedef IndxArrayT<exec_space> IndxArray_type;

      const tensor_type X;
      const Ktensor_type M;
      const Ktensor_type Mt;
      const Ktensor_type Mprev;
      const loss_type f;
      const ttb_indx num_samples_nonzeros;
      const ttb_indx num_samples_zeros;
      const ttb_real weight_nonzeros;
      const ttb_real weight_zeros;
      const Array_type window;
      const ttb_real window_penalty;
      const IndxArray_type modes;
      const Ktensor_type G;
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool;
      const AlgParams algParams;
      SystemTimer& timer;
      const int timer_nzs;
      const int timer_zs;

      GCP_SS_Grad_Str(const tensor_type& X_,
                      const Ktensor_type& M_,
                      const Ktensor_type& Mt_,
                      const Ktensor_type& Mprev_,
                      const loss_type& f_,
                      const ttb_indx num_samples_nonzeros_,
                      const ttb_indx num_samples_zeros_,
                      const ttb_real weight_nonzeros_,
                      const ttb_real weight_zeros_,
                      const Array_type window_,
                      const ttb_real window_penalty_,
                      const IndxArray_type& modes_,
                      const Ktensor_type& G_,
                      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool_,
                      const AlgParams& algParams_,
                      SystemTimer& timer_,
                      const int timer_nzs_,
                      const int timer_zs_) :
        X(X_), M(M_), Mt(Mt_), Mprev(Mprev_), f(f_),
        num_samples_nonzeros(num_samples_nonzeros_),
        num_samples_zeros(num_samples_zeros_),
        weight_nonzeros(weight_nonzeros_),
        weight_zeros(weight_zeros_),
        window(window_), window_penalty(window_penalty_),
        modes(modes_),
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
            X,M,Mt,Mprev,f,num_samples_nonzeros,num_samples_zeros,
            weight_nonzeros,weight_zeros,window,window_penalty,
            modes,G,rand_pool,algParams,
            timer,timer_nzs,timer_zs);
        else if (method == MTTKRP_All_Method::Atomic)
          gcp_sgd_ss_grad_sv_kernel<ScatterNonDuplicated,ScatterAtomic,FBS,VS>(
            X,M,Mt,Mprev,f,num_samples_nonzeros,num_samples_zeros,
            weight_nonzeros,weight_zeros,window,window_penalty,
            modes,G,rand_pool,algParams,
            timer,timer_nzs,timer_zs);
        else if (method == MTTKRP_All_Method::Duplicated)
          gcp_sgd_ss_grad_sv_kernel<ScatterDuplicated,ScatterNonAtomic,FBS,VS>(
            X,M,Mt,Mprev,f,num_samples_nonzeros,num_samples_zeros,
            weight_nonzeros,weight_zeros,window,window_penalty,
            modes,G,rand_pool,algParams,
            timer,timer_nzs,timer_zs);
        else if (method == MTTKRP_All_Method::Iterated)
          Genten::error("Cannot use iterated MTTKRP method in fused stratified-sampling/MTTKRP kernel!");
      }
    };

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    // Specialization for Cuda, HIP and SYCL that always uses atomics and doesn't call
    // gcp_sgd_ss_grad_sv_kernel, which won't run on the GPU
    template <typename loss_type>
    struct GCP_SS_Grad_Str<Kokkos_GPU_Space,loss_type> {
      typedef Kokkos_GPU_Space exec_space;
      typedef SptensorT<exec_space> tensor_type;
      typedef KtensorT<exec_space> Ktensor_type;
      typedef ArrayT<exec_space> Array_type;
      typedef IndxArrayT<exec_space> IndxArray_type;

      const tensor_type X;
      const Ktensor_type M;
      const Ktensor_type Mt;
      const Ktensor_type Mprev;
      const loss_type f;
      const ttb_indx num_samples_nonzeros;
      const ttb_indx num_samples_zeros;
      const ttb_real weight_nonzeros;
      const ttb_real weight_zeros;
      const Array_type window;
      const ttb_real window_penalty;
      const IndxArray_type modes;
      const Ktensor_type G;
      Kokkos::Random_XorShift64_Pool<exec_space>& rand_pool;
      const AlgParams algParams;
      SystemTimer& timer;
      const int timer_nzs;
      const int timer_zs;

      GCP_SS_Grad_Str(const tensor_type& X_,
                      const Ktensor_type& M_,
                      const Ktensor_type& Mt_,
                      const Ktensor_type& Mprev_,
                      const loss_type& f_,
                      const ttb_indx num_samples_nonzeros_,
                      const ttb_indx num_samples_zeros_,
                      const ttb_real weight_nonzeros_,
                      const ttb_real weight_zeros_,
                      const Array_type window_,
                      const ttb_real window_penalty_,
                      const IndxArray_type& modes_,
                      const Ktensor_type& G_,
                      Kokkos::Random_XorShift64_Pool<exec_space>& rand_pool_,
                      const AlgParams& algParams_,
                      SystemTimer& timer_,
                      const int timer_nzs_,
                      const int timer_zs_) :
        X(X_), M(M_), Mt(Mt_), Mprev(Mprev_), f(f_),
        num_samples_nonzeros(num_samples_nonzeros_),
        num_samples_zeros(num_samples_zeros_),
        weight_nonzeros(weight_nonzeros_),
        weight_zeros(weight_zeros_),
        window(window_), window_penalty(window_penalty_),
        modes(modes_),
        G(G_), rand_pool(rand_pool_), algParams(algParams_),
        timer(timer_), timer_nzs(timer_nzs_), timer_zs(timer_zs_) {}

      template <unsigned FBS, unsigned VS>
      void run() const {
        if (algParams.mttkrp_all_method != MTTKRP_All_Method::Atomic)
          Genten::error("MTTKRP-All method must be atomic on Cuda, HIP or SYCL!");

        gcp_sgd_ss_grad_atomic_kernel<FBS,VS>(
          X,M,Mt,Mprev,f,num_samples_nonzeros,num_samples_zeros,
          weight_nonzeros,weight_zeros,window,window_penalty,
          modes,G,rand_pool,algParams,
          timer,timer_nzs,timer_zs);
      }
    };
#endif

    template <typename ExecSpace, typename loss_type>
    void gcp_sgd_ss_grad(
      const SptensorT<ExecSpace>& X,
      const KtensorT<ExecSpace>& M,
      const KtensorT<ExecSpace>& Mt,
      const KtensorT<ExecSpace>& Mprev,
      const loss_type& f,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const ArrayT<ExecSpace>& window,
      const ttb_real window_penalty,
      const IndxArrayT<ExecSpace>& modes,
      const KtensorT<ExecSpace>& G,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams,
      SystemTimer& timer,
      const int timer_nzs,
      const int timer_zs)
    {
      GCP_SS_Grad_Str<ExecSpace,loss_type> kernel(
        X,M,Mt,Mprev,f,num_samples_nonzeros,num_samples_zeros,
        weight_nonzeros,weight_zeros,window,window_penalty,
        modes,G,rand_pool,algParams,
        timer,timer_nzs,timer_zs);
      run_row_simd_kernel(kernel, M.ncomponents());
    }

  }

}
