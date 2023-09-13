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

#include <type_traits>

#include "Genten_Util.hpp"
#include "Genten_FacMatrix.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_TinyVec.hpp"
#include "Genten_AlgParams.hpp"
#include "Genten_SimdKernel.hpp"

// This is a locally-modified version of Kokkos_ScatterView.hpp which we
// need until the changes are moved into Kokkos
#include "Genten_Kokkos_ScatterView.hpp"

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

//-----------------------------------------------------------------------------
//  Method:  mttkrp, Sptensor X, output to FacMatrix
//-----------------------------------------------------------------------------

namespace Genten {
namespace Impl {

// MTTKRP kernel for Sptensor
template <int Dupl, int Cont, unsigned FBS, unsigned VS, typename ExecSpace>
void
mttkrp_kernel(const SptensorImpl<ExecSpace>& X,
              const KtensorImpl<ExecSpace>& u,
              const unsigned n,
              const FacMatrixT<ExecSpace>& v,
              const AlgParams& algParams,
              const bool zero_v)
{
  if (zero_v)
    v = ttb_real(0.0);

  using Kokkos::Experimental::create_scatter_view;
  using Kokkos::Experimental::ScatterView;
  using Kokkos::Experimental::ScatterSum;

  static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
  static const unsigned FacBlockSize = FBS;
  static const unsigned VectorSize = is_gpu ? VS : 1;
  static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;
  /*const*/ unsigned RowBlockSize = algParams.mttkrp_nnz_tile_size;
  const unsigned RowsPerTeam = TeamSize * RowBlockSize;

  /*const*/ unsigned nd = u.ndims();
  /*const*/ unsigned nc_total = u.ncomponents();
  /*const*/ ttb_indx nnz = X.nnz();
  const ttb_indx N = (nnz+RowsPerTeam-1)/RowsPerTeam;

  typedef Kokkos::TeamPolicy<ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  Policy policy(N, TeamSize, VectorSize);

  // Use factor matrix tile size as requested by the user, or all columns if
  // unspecified
  const unsigned FacTileSize =
    algParams.mttkrp_duplicated_factor_matrix_tile_size > 0 ? algParams.mttkrp_duplicated_factor_matrix_tile_size : nc_total;
  for (unsigned nc_beg=0; nc_beg<nc_total; nc_beg += FacTileSize) {
    const unsigned nc =
      nc_beg+FacTileSize <= nc_total ? FacTileSize : nc_total-nc_beg;
    const unsigned nc_end = nc_beg+nc;
    auto vv = Kokkos::subview(v.view(),Kokkos::ALL,
                              std::make_pair(nc_beg,nc_end));
    auto sv = create_scatter_view<ScatterSum,Dupl,Cont>(vv);
    Kokkos::parallel_for("mttkrp_kernel",
                         policy, KOKKOS_LAMBDA(const TeamMember& team)
    {
      auto va = sv.access();

      // Loop over tensor non-zeros with a large stride on the GPU to
      // reduce atomic contention when the non-zeros are in a nearly sorted
      // order (often the first dimension of the tensor).  This is similar
      // to an approach used in ParTi (https://github.com/hpcgarage/ParTI)
      // by Jaijai Li.
      ttb_indx offset;
      ttb_indx stride;
      if (is_gpu) {
        offset = team.league_rank()*TeamSize+team.team_rank();
        stride = team.league_size()*TeamSize;
      }
      else {
        offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        stride = 1;
      }

      auto row_func = [&](auto j, auto nj, auto Nj) {
        typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx i = offset + ii*stride;
          if (i >= nnz)
            continue;

          const ttb_indx k = X.subscript(i,n);
          const ttb_real x_val = X.value(i);

          // MTTKRP for dimension n
          auto tmp = TVM::make(team, nj, x_val);
          tmp *= &(u.weights(nc_beg+j));
          for (unsigned m=0; m<nd; ++m) {
            if (m != n)
              tmp *= &(u[m].entry(X.subscript(i,m),nc_beg+j));
          }
          va(k,j) += tmp;
        }
      };

      for (unsigned j=0; j<nc; j+=FacBlockSize) {
        if (j+FacBlockSize <= nc) {
          const unsigned nj = FacBlockSize;
          row_func(j, nj, std::integral_constant<unsigned,nj>());
        }
        else {
          const unsigned nj = nc-j;
          row_func(j, nj, std::integral_constant<unsigned,0>());
        }
      }
    });

    sv.contribute_into(vv);
  }
}

// MTTKRP kernel for Sptensor_perm
template <unsigned FBS, unsigned VS, typename ExecSpace>
void
mttkrp_kernel_perm(const SptensorImpl<ExecSpace>& X,
                   const KtensorImpl<ExecSpace>& u,
                   const unsigned n,
                   const FacMatrixT<ExecSpace>& v,
                   const AlgParams& algParams,
                   const bool zero_v)
{
  if (zero_v)
    v = ttb_real(0.0);

  static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
  static const unsigned FacBlockSize = FBS;
  static const unsigned VectorSize = is_gpu ? VS : 1;
  static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;
  /*const*/ unsigned RowBlockSize = algParams.mttkrp_nnz_tile_size;
  const unsigned RowsPerTeam = TeamSize * RowBlockSize;

  /*const*/ unsigned nd = u.ndims();
  /*const*/ unsigned nc = u.ncomponents();
  /*const*/ ttb_indx nnz = X.nnz();
  const ttb_indx N = (nnz+RowsPerTeam-1)/RowsPerTeam;

  typedef Kokkos::TeamPolicy<ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  Policy policy(N, TeamSize, VectorSize);
  Kokkos::parallel_for("mttkrp_kernel",
                       policy, KOKKOS_LAMBDA(const TeamMember& team)
  {
    /*const*/ ttb_indx invalid_row = ttb_indx(-1);
    /*const*/ ttb_indx i_block =
      (team.league_rank()*TeamSize + team.team_rank())*RowBlockSize;

    auto row_func = [&](auto j, auto nj, auto Nj) {
      typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;
      auto val = TVM::make(team, nj, 0.0);
      auto tmp = TVM::make(team, nj, 0.0);

      ttb_indx row_prev = invalid_row;
      ttb_indx row = invalid_row;
      ttb_indx first_row = invalid_row;
      ttb_indx p = invalid_row;
      ttb_real x_val = 0.0;

      for (unsigned ii=0; ii<RowBlockSize; ++ii) {
        /*const*/ ttb_indx i = i_block+ii;

        if (i >= nnz)
          row = invalid_row;
        else {
          p = X.getPerm(i,n);
          x_val = X.value(p);
          row = X.subscript(p,n);
        }

        if (ii == 0)
          first_row = row;

        // If we got a different row index, add in result
        if (row != row_prev) {
          if (row_prev != invalid_row) {
            if (row_prev == first_row) // Only need atomics for first/last row
              Kokkos::atomic_add(&v.entry(row_prev,j), val);
            else
              val.store_plus(&v.entry(row_prev,j));
            val.broadcast(0.0);
          }
          row_prev = row;
        }

        if (row != invalid_row) {
          // Start tmp equal to the weights.
          tmp.load(&(u.weights(j)));
          tmp *= x_val;

          for (unsigned m=0; m<nd; ++m) {
            if (m != n)
              tmp *= &(u[m].entry(X.subscript(p,m),j));
          }
          val += tmp;
        }
      }

      // Sum in last row
      if (row != invalid_row) {
        Kokkos::atomic_add(&v.entry(row,j), val);
      }
    };

    for (unsigned j=0; j<nc; j+=FacBlockSize) {
      if (j+FacBlockSize <= nc) {
        const unsigned nj = FacBlockSize;
        row_func(j, nj, std::integral_constant<unsigned,nj>());
      }
      else {
        const unsigned nj = nc-j;
        row_func(j, nj, std::integral_constant<unsigned,0>());
      }
    }
  });
}

template <typename ExecSpace>
struct MTTKRP_Kernel {
  const SptensorImpl<ExecSpace> X;
  const KtensorImpl<ExecSpace> u;
  const ttb_indx n;
  const FacMatrixT<ExecSpace> v;
  const AlgParams algParams;
  const bool zero_v;

  MTTKRP_Kernel(const SptensorImpl<ExecSpace>& X_,
                const KtensorImpl<ExecSpace>& u_,
                const ttb_indx n_,
                const FacMatrixT<ExecSpace>& v_,
                const AlgParams& algParams_,
                const bool zero_v_) :
    X(X_), u(u_), n(n_), v(v_), algParams(algParams_), zero_v(zero_v_) {}

  template <unsigned FBS, unsigned VS>
  void run() const {
    using Kokkos::Experimental::ScatterDuplicated;
    using Kokkos::Experimental::ScatterNonDuplicated;
    using Kokkos::Experimental::ScatterAtomic;
    using Kokkos::Experimental::ScatterNonAtomic;
    typedef SpaceProperties<ExecSpace> space_prop;

    MTTKRP_Method::type method = algParams.mttkrp_method;

    if (space_prop::is_gpu &&
        (method == MTTKRP_Method::Single ||
         method == MTTKRP_Method::Duplicated))
      Genten::error("Single and duplicated MTTKRP methods are invalid on Cuda, HIP or SYCL!");

    // Check if Perm is selected, that perm is computed
    if (method == MTTKRP_Method::Perm && !X.havePerm())
      Genten::error("Perm MTTKRP method selected, but permutation array not computed!");

    if (method == MTTKRP_Method::Single)
      mttkrp_kernel<ScatterNonDuplicated,ScatterNonAtomic,FBS,VS>(
        X,u,n,v,algParams,zero_v);
    else if (method == MTTKRP_Method::Atomic)
      mttkrp_kernel<ScatterNonDuplicated,ScatterAtomic,FBS,VS>(
        X,u,n,v,algParams,zero_v);
    else if (method == MTTKRP_Method::Duplicated) {
      // Only use "Duplicated" if the mode length * concurrency is sufficiently
      // small.  Taken from "Sparse Tensor Factorization on Many-Core Processors
      // with High-Bandwidth Memory" by Smith, Park and Karypis.  It's not
      // clear this is really the right choice.  Seems like it should also take
      // into account R = u.ncomponents() and last-level cache size.
      const ttb_indx P = SpaceProperties<ExecSpace>::concurrency();
      const ttb_indx nnz = X.nnz();
      const ttb_indx N = X.size(n);
      const ttb_real gamma = algParams.mttkrp_duplicated_threshold;
      if (gamma < 0.0 || (static_cast<ttb_real>(N*P) <= gamma*nnz))
        mttkrp_kernel<ScatterDuplicated,ScatterNonAtomic,FBS,VS>(
          X,u,n,v,algParams,zero_v);
      else
        mttkrp_kernel<ScatterNonDuplicated,ScatterAtomic,FBS,VS>(
          X,u,n,v,algParams,zero_v);
    }
    else if (method == MTTKRP_Method::Perm)
      mttkrp_kernel_perm<FBS,VS>(X,u,n,v,algParams,zero_v);
    else
      Genten::error(std::string("Unknown MTTKRP method:  ") +
                    std::string(MTTKRP_Method::names[method]));
  }
};

// MTTKRP kernel for Sptensor for all modes simultaneously
// Because of problems with ScatterView, doesn't work on the GPU
template <int Dupl, int Cont, typename ExecSpace>
struct MTTKRP_All_Kernel {
  const SptensorImpl<ExecSpace> XX;
  const KtensorImpl<ExecSpace> uu;
  const KtensorImpl<ExecSpace> vv;
  const ttb_indx mode_beg;
  const ttb_indx mode_end;
  const AlgParams algParams;
  const bool zero_v;

  MTTKRP_All_Kernel(const SptensorImpl<ExecSpace>& X_,
                    const KtensorImpl<ExecSpace>& u_,
                    const KtensorImpl<ExecSpace>& v_,
                    const ttb_indx mode_beg_,
                    const ttb_indx mode_end_,
                    const AlgParams& algParams_,
                    const bool zero_v_) :
    XX(X_), uu(u_), vv(v_), mode_beg(mode_beg_), mode_end(mode_end_),
    algParams(algParams_), zero_v(zero_v_) {}

  template <unsigned FBS, unsigned VS>
  void run() const {
    const SptensorImpl<ExecSpace> X = XX;
    const KtensorImpl<ExecSpace> u = uu;
    const KtensorImpl<ExecSpace> v = vv;
    /*const*/ unsigned mb = mode_beg;
    /*const*/ unsigned me = mode_end;
    /*const*/ unsigned nm = me-mb;

    if (zero_v)
      v.setMatrices(0.0);

    using Kokkos::Experimental::create_scatter_view;
    using Kokkos::Experimental::ScatterView;
    using Kokkos::Experimental::ScatterSum;

    static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
    static const unsigned FacBlockSize = FBS;
    static const unsigned VectorSize = is_gpu ? VS : 1;
    static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;
    /*const*/ unsigned RowBlockSize = algParams.mttkrp_nnz_tile_size;
    const unsigned RowsPerTeam = TeamSize * RowBlockSize;

    static_assert(!is_gpu, "Cannot call mttkrp_all_kernel for Cuda, HIP or SYCL space!");

    /*const*/ unsigned nd = u.ndims();
    /*const*/ unsigned nc_total = u.ncomponents();
    /*const*/ ttb_indx nnz = X.nnz();
    const ttb_indx N = (nnz+RowsPerTeam-1)/RowsPerTeam;

    typedef Kokkos::TeamPolicy<ExecSpace> Policy;
    typedef typename Policy::member_type TeamMember;
    Policy policy(N, TeamSize, VectorSize);

    typedef ScatterView<ttb_real**,Kokkos::LayoutRight,ExecSpace,ScatterSum,Dupl,Cont> ScatterViewType;

    // Use factor matrix tile size as requested by the user, or all columns if
    // unspecified
    const unsigned FacTileSize =
      algParams.mttkrp_duplicated_factor_matrix_tile_size > 0 ? algParams.mttkrp_duplicated_factor_matrix_tile_size : nc_total;
    for (unsigned nc_beg=0; nc_beg<nc_total; nc_beg += FacTileSize) {
      const unsigned nc =
        nc_beg+FacTileSize <= nc_total ? FacTileSize : nc_total-nc_beg;
      const unsigned nc_end = nc_beg+nc;
      ScatterViewType *sa = new ScatterViewType[nm];
      for (unsigned n=0; n<nm; ++n) {
        auto vv = Kokkos::subview(v[n].view(),Kokkos::ALL,
                                  std::make_pair(nc_beg,nc_end));
        sa[n] = ScatterViewType(vv);
      }
      Kokkos::parallel_for("mttkrp_all_kernel",
                           policy, KOKKOS_LAMBDA(const TeamMember& team)
      {
        // Loop over tensor non-zeros with a large stride on the GPU to
        // reduce atomic contention when the non-zeros are in a nearly sorted
        // order (often the first dimension of the tensor).  This is similar
        // to an approach used in ParTi (https://github.com/hpcgarage/ParTI)
        // by Jaijai Li.
        ttb_indx offset;
        ttb_indx stride;
        if (is_gpu) {
          offset = team.league_rank()*TeamSize+team.team_rank();
          stride = team.league_size()*TeamSize;
        }
        else {
          offset =
            (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
          stride = 1;
        }

        auto row_func = [&](auto j, auto nj, auto Nj) {
          typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;
          for (unsigned ii=0; ii<RowBlockSize; ++ii) {
            const ttb_indx i = offset + ii*stride;
            if (i >= nnz)
              continue;

            const ttb_real x_val = X.value(i);

            // MTTKRP for dimension n
            for (unsigned n=0; n<nm; ++n) {
              const ttb_indx k = X.subscript(i,n+mb);
              auto va = sa[n].access();
              auto tmp = TVM::make(team, nj, x_val);
              tmp *= &(u.weights(nc_beg+j));
              for (unsigned m=0; m<nd; ++m) {
                if (m != n+mb)
                  tmp *= &(u[m].entry(X.subscript(i,m),nc_beg+j));
              }
              va(k,j) += tmp;
            }
          }
        };

        for (unsigned j=0; j<nc; j+=FacBlockSize) {
          if (j+FacBlockSize <= nc) {
            const unsigned nj = FacBlockSize;
            row_func(j, nj, std::integral_constant<unsigned,nj>());
          }
        else {
          const unsigned nj = nc-j;
          row_func(j, nj, std::integral_constant<unsigned,0>());
        }
        }
      });

      for (unsigned n=0; n<nm; ++n) {
        auto vv = Kokkos::subview(v[n].view(),Kokkos::ALL,
                                  std::make_pair(nc_beg,nc_end));
        sa[n].contribute_into(vv);
      }
      delete [] sa;
    }
  }
};

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
// Specialization for Cuda, HIP or SYCL that always uses atomics and doesn't call
// mttkrp_all_kernel, which won't run on the GPU
template <int Dupl, int Cont>
struct MTTKRP_All_Kernel<Dupl, Cont, Kokkos_GPU_Space> {
  typedef Kokkos_GPU_Space ExecSpace;

  const SptensorImpl<ExecSpace> XX;
  const KtensorImpl<ExecSpace> uu;
  const KtensorImpl<ExecSpace> vv;
  const ttb_indx mode_beg;
  const ttb_indx mode_end;
  const AlgParams algParams;
  const bool zero_v;

  MTTKRP_All_Kernel(const SptensorImpl<ExecSpace>& X_,
                    const KtensorImpl<ExecSpace>& u_,
                    const KtensorImpl<ExecSpace>& v_,
                    const ttb_indx mode_beg_,
                    const ttb_indx mode_end_,
                    const AlgParams& algParams_,
                    const bool zero_v_) :
    XX(X_), uu(u_), vv(v_), mode_beg(mode_beg_), mode_end(mode_end_),
    algParams(algParams_), zero_v(zero_v_) {}

  template <unsigned FBS, unsigned VS>
  void run() const {
    const SptensorImpl<ExecSpace> X = XX;
    const KtensorImpl<ExecSpace> u = uu;
    const KtensorImpl<ExecSpace> v = vv;
    /*const*/ unsigned mb = mode_beg;
    /*const*/ unsigned me = mode_end;
    /*const*/ unsigned nm = me-mb;

    if (algParams.mttkrp_all_method != MTTKRP_All_Method::Atomic)
      Genten::error("MTTKRP-All method must be atomic on Cuda, HIP and SYCL!");

    if (zero_v)
      v.setMatrices(0.0);

    static const unsigned FacBlockSize = FBS;
    static const unsigned VectorSize = VS;
    static const unsigned TeamSize = 128/VectorSize;
    /*const*/ unsigned RowBlockSize = algParams.mttkrp_nnz_tile_size;
    const unsigned RowsPerTeam = TeamSize * RowBlockSize;

    /*const*/ unsigned nd = u.ndims();
    /*const*/ unsigned nc = u.ncomponents();
    /*const*/ ttb_indx nnz = X.nnz();
    const ttb_indx N = (nnz+RowsPerTeam-1)/RowsPerTeam;

    typedef Kokkos::TeamPolicy<ExecSpace> Policy;
    typedef typename Policy::member_type TeamMember;
    Policy policy(N, TeamSize, VectorSize);

    Kokkos::parallel_for("mttkrp_all_kernel",
                         policy, KOKKOS_LAMBDA(const TeamMember& team)
    {
      // Loop over tensor non-zeros with a large stride on the GPU to
      // reduce atomic contention when the non-zeros are in a nearly sorted
      // order (often the first dimension of the tensor).  This is similar
      // to an approach used in ParTi (https://github.com/hpcgarage/ParTI)
      // by Jaijai Li.
      ttb_indx offset = team.league_rank()*TeamSize+team.team_rank();
      ttb_indx stride = team.league_size()*TeamSize;

      auto row_func = [&](auto j, auto nj, auto Nj) {
        typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx i = offset + ii*stride;
          if (i >= nnz)
            continue;

          const ttb_real x_val = X.value(i);

          // MTTKRP for dimension n
          for (unsigned n=0; n<nm; ++n) {
            const ttb_indx k = X.subscript(i,n+mb);
            auto tmp = TVM::make(team, nj, x_val);
            tmp *= &(u.weights(j));
            for (unsigned m=0; m<nd; ++m) {
              if (m != n+mb)
                tmp *= &(u[m].entry(X.subscript(i,m),j));
            }
            Kokkos::atomic_add(&v[n].entry(k,j), tmp);
          }
        }
      };

      for (unsigned j=0; j<nc; j+=FacBlockSize) {
        if (j+FacBlockSize <= nc) {
          const unsigned nj = FacBlockSize;
          row_func(j, nj, std::integral_constant<unsigned,nj>());
        }
        else {
          const unsigned nj = nc-j;
          row_func(j, nj, std::integral_constant<unsigned,0>());
        }
      }
    });
  }
};
#endif

template <typename ExecSpace, unsigned FacBlockSize,
          unsigned TeamSize, unsigned VectorSize>
struct MTTKRP_OrigKokkosKernelBlock {
  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  typedef Kokkos::View< ttb_real**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

  const SptensorImpl<ExecSpace>& X;
  const KtensorImpl<ExecSpace>& u;
  const unsigned n;
  const unsigned nd;
  const FacMatrixT<ExecSpace>& v;
  const ttb_indx i;

  const TeamMember& team;
  const unsigned team_index;
  TmpScratchSpace tmp;

  const ttb_indx k;
  const ttb_real x_val;
  const ttb_real* lambda;

  static inline Policy policy(const ttb_indx nnz_) {
    const ttb_indx N = (nnz_+TeamSize-1)/TeamSize;
    Policy policy(N,TeamSize,VectorSize);
    size_t bytes = TmpScratchSpace::shmem_size(TeamSize,FacBlockSize);
    return policy.set_scratch_size(0,Kokkos::PerTeam(bytes));
  }

  KOKKOS_INLINE_FUNCTION
  MTTKRP_OrigKokkosKernelBlock(const SptensorImpl<ExecSpace>& X_,
                               const KtensorImpl<ExecSpace>& u_,
                               const unsigned n_,
                               const FacMatrixT<ExecSpace>& v_,
                               const ttb_indx i_,
                               const TeamMember& team_) :
    X(X_), u(u_), n(n_), nd(u.ndims()), v(v_), i(i_),
    team(team_), team_index(team.team_rank()),
    tmp(team.team_scratch(0), TeamSize, FacBlockSize),
    k(X.subscript(i,n)), x_val(X.value(i)),
    lambda(&u.weights(0))
    {}

  template <unsigned Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const unsigned j, const unsigned nj_)
  {
     // nj.value == Nj_ if Nj_ > 0 and nj_ otherwise
    Kokkos::Impl::integral_nonzero_constant<unsigned, Nj> nj(nj_);

    const ttb_real *l = &lambda[j];
    ttb_real *v_kj = &v.entry(k,j);

    // Start tmp equal to the weights.
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,unsigned(nj.value)),
                         [&] (const unsigned& jj)
    {
      tmp(team_index,jj) = x_val * l[jj];
    });

    for (unsigned m=0; m<nd; ++m) {
      if (m != n) {
        // Update tmp array with elementwise product of row i
        // from the m-th factor matrix.  Length of the row is nc.
        const ttb_real *row = &(u[m].entry(X.subscript(i,m),j));
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,unsigned(nj.value)),
                             [&] (const unsigned& jj)
        {
          tmp(team_index,jj) *= row[jj];
        });
      }
    }

    // Update output by adding tmp array.
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,unsigned(nj.value)),
                         [&] (const unsigned& jj)
    {
      Kokkos::atomic_add(v_kj+jj, tmp(team_index,jj));
    });
  }
};

template <typename ExecSpace, unsigned FacBlockSize>
void orig_kokkos_mttkrp_kernel(const SptensorImpl<ExecSpace>& X,
                               const KtensorImpl<ExecSpace>& u,
                               const ttb_indx n,
                               const FacMatrixT<ExecSpace>& v)
{
  // Compute team and vector sizes, depending on the architecture
  const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
  const unsigned VectorSize = is_gpu ? (FacBlockSize <= 16 ? FacBlockSize : 16) : 1;
  const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;

  const unsigned nc = u.ncomponents();
  const ttb_indx nnz = X.nnz();

  typedef MTTKRP_OrigKokkosKernelBlock<ExecSpace, FacBlockSize, TeamSize, VectorSize> Kernel;
  typedef typename Kernel::TeamMember TeamMember;

  Kokkos::parallel_for("Genten::mttkrp_kernel",
                       Kernel::policy(nnz),
                       KOKKOS_LAMBDA(TeamMember team)
  {
    const ttb_indx i = team.league_rank()*team.team_size()+team.team_rank();
    if (i >= nnz)
      return;

    MTTKRP_OrigKokkosKernelBlock<ExecSpace, FacBlockSize, TeamSize, VectorSize> kernel(X, u, n, v, i, team);

    for (unsigned j=0; j<nc; j+=FacBlockSize) {
      if (j+FacBlockSize <= nc)
        kernel.template run<FacBlockSize>(j, FacBlockSize);
      else
        kernel.template run<0>(j, nc-j);
    }

  });

  return;
}

template <typename ExecSpace>
void orig_kokkos_mttkrp(const SptensorImpl<ExecSpace>& X,
                        const KtensorImpl<ExecSpace>& u,
                        const ttb_indx n,
                        const FacMatrixT<ExecSpace>& v,
                        const bool zero_v)
{
  const ttb_indx nc = u.ncomponents();     // Number of components
  const ttb_indx nd = u.ndims();           // Number of dimensions

  gt_assert(X.ndims() == nd);
  gt_assert(u.isConsistent());
  for (ttb_indx i = 0; i < nd; i++)
  {
    if (i != n)
      gt_assert(u[i].nRows() == X.size(i));
  }

  // Resize and initialize the output factor matrix to zero.
  gt_assert( v.nRows() == X.size(n) );
  gt_assert( v.nCols() == nc );
  if (zero_v)
    v = ttb_real(0.0);

  // Call kernel with factor block size determined from nc
  if (nc == 1)
    orig_kokkos_mttkrp_kernel<ExecSpace,1>(X,u,n,v);
  else if (nc == 2)
    orig_kokkos_mttkrp_kernel<ExecSpace,2>(X,u,n,v);
  else if (nc <= 4)
    orig_kokkos_mttkrp_kernel<ExecSpace,4>(X,u,n,v);
  else if (nc <= 8)
    orig_kokkos_mttkrp_kernel<ExecSpace,8>(X,u,n,v);
  else if (nc <= 16)
    orig_kokkos_mttkrp_kernel<ExecSpace,16>(X,u,n,v);
  else
    orig_kokkos_mttkrp_kernel<ExecSpace,32>(X,u,n,v);

  if (u.getProcessorMap() != nullptr) {
    Kokkos::fence();
    u.getProcessorMap()->subGridAllReduce(n,v.view().data(), v.view().span());
  }

  return;
}

}

template <typename ExecSpace>
void mttkrp(const SptensorT<ExecSpace>& X,
            const KtensorT<ExecSpace>& u,
            const ttb_indx n,
            const FacMatrixT<ExecSpace>& v,
            const AlgParams& algParams,
            const bool zero_v)
{
  GENTEN_TIME_MONITOR("MTTKRP");
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::mttkrp");
#endif

  const ttb_indx nc = u.ncomponents();     // Number of components
  const ttb_indx nd = u.ndims();           // Number of dimensions

  gt_assert(X.ndims() == nd);
  gt_assert(u.isConsistent());
  for (ttb_indx i = 0; i < nd; i++)
  {
    if (i != n)
      gt_assert(u[i].nRows() == X.size(i));
  }
  gt_assert( v.nRows() == X.size(n) );
  gt_assert( v.nCols() == nc );

  if (algParams.mttkrp_method == MTTKRP_Method::OrigKokkos) {
    Impl::orig_kokkos_mttkrp(X.impl(),u.impl(),n,v,zero_v);
  }
  else {
    Impl::MTTKRP_Kernel<ExecSpace> kernel(X.impl(),u.impl(),n,v,algParams,zero_v);
    Impl::run_row_simd_kernel(kernel, nc);
  }
}

template <typename ExecSpace>
void mttkrp_all(const SptensorT<ExecSpace>& X,
                const KtensorT<ExecSpace>& u,
                const KtensorT<ExecSpace>& v,
                const ttb_indx mode_beg,
                const ttb_indx mode_end,
                const AlgParams& algParams,
                const bool zero_v)
{
  GENTEN_TIME_MONITOR("MTTKRP-all");
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::mttkrp_all");
#endif

  const ttb_indx nc = u.ncomponents();     // Number of components
  const ttb_indx nd = u.ndims();           // Number of dimensions

  gt_assert(X.ndims() == nd);
  gt_assert(v.ncomponents() == nc);
  gt_assert(u.isConsistent());
  for (ttb_indx i=0; i<nd; ++i) {
    gt_assert(u[i].nRows() == X.size(i));
  }
  gt_assert(mode_beg <= mode_end);
  gt_assert(mode_end <= nd);
  gt_assert(v.ndims() == (mode_end - mode_beg));
  for (ttb_indx i=mode_beg; i<mode_end; ++i) {
    gt_assert(v[i-mode_beg].nRows() == X.size(i));
  }

  using Kokkos::Experimental::ScatterDuplicated;
  using Kokkos::Experimental::ScatterNonDuplicated;
  using Kokkos::Experimental::ScatterAtomic;
  using Kokkos::Experimental::ScatterNonAtomic;
  typedef SpaceProperties<ExecSpace> space_prop;

  MTTKRP_All_Method::type method = algParams.mttkrp_all_method;

  if (space_prop::is_gpu &&
      (method == MTTKRP_All_Method::Single ||
       method == MTTKRP_All_Method::Duplicated))
    Genten::error("Single and duplicated MTTKRP-All methods are invalid on Cuda, HIP and SYCL!");

  if (algParams.mttkrp_all_method == MTTKRP_All_Method::Iterated) {
    for (ttb_indx n=mode_beg; n<mode_end; ++n)
      mttkrp(X, u, n, v[n-mode_beg], algParams, zero_v);
  }
  else if (method == MTTKRP_All_Method::Single) {
    Impl::MTTKRP_All_Kernel<ScatterNonDuplicated,ScatterNonAtomic,ExecSpace> kernel(X.impl(),u.impl(),v.impl(),mode_beg,mode_end,algParams, zero_v);
    Impl::run_row_simd_kernel(kernel, nc);
  }
  else if (method == MTTKRP_All_Method::Atomic) {
    Impl::MTTKRP_All_Kernel<ScatterNonDuplicated,ScatterAtomic,ExecSpace> kernel(X.impl(),u.impl(),v.impl(),mode_beg,mode_end,algParams, zero_v);
    Impl::run_row_simd_kernel(kernel, nc);
  }
  else if (method == MTTKRP_All_Method::Duplicated) {
    Impl::MTTKRP_All_Kernel<ScatterDuplicated,ScatterNonAtomic,ExecSpace> kernel(X.impl(),u.impl(),v.impl(),mode_beg,mode_end,algParams, zero_v);
    Impl::run_row_simd_kernel(kernel, nc);
  }
  else
    Genten::error(std::string("Unknown MTTKRP-all method:  ") +
                  std::string(MTTKRP_All_Method::names[method]));
}

}
