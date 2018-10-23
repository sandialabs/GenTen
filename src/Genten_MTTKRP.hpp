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

#include <assert.h>

#include <type_traits>

#include "Genten_Util.hpp"
#include "Genten_FacMatrix.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_TinyVec.hpp"

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

#define USE_SCATTER_VIEW 0

namespace Genten {
namespace Impl {

template <unsigned VS, typename Func>
void run_row_simd_kernel_impl(Func& f, const unsigned nc)
{
  static const unsigned VS4 = 4*VS;
  static const unsigned VS3 = 3*VS;
  static const unsigned VS2 = 2*VS;
  static const unsigned VS1 = 1*VS;

  if (nc > VS3)
    f.template run<VS4,VS>();
  else if (nc > VS2)
    f.template run<VS3,VS>();
  else if (nc > VS1)
    f.template run<VS2,VS>();
  else
    f.template run<VS1,VS>();
}

template <typename Func>
void run_row_simd_kernel(Func& f, const unsigned nc)
{
  if (nc >= 96)
    run_row_simd_kernel_impl<32>(f, nc);
  else if (nc >= 48)
    run_row_simd_kernel_impl<16>(f, nc);
  else if (nc >= 8)
    run_row_simd_kernel_impl<8>(f, nc);
  else if (nc >= 4)
    run_row_simd_kernel_impl<4>(f, nc);
  else if (nc >= 2)
    run_row_simd_kernel_impl<2>(f, nc);
  else
    run_row_simd_kernel_impl<1>(f, nc);
}

}
}

#if USE_SCATTER_VIEW
// This is a locally-modified version of Kokkos_ScatterView.hpp which we
// need until the changes are moved into Kokkos
#include "Genten_Kokkos_ScatterView.hpp"

namespace Kokkos {
namespace Impl {
namespace Experimental {
// Specialization of ReduceDuplicates for rank-2, LayoutRight dst views:
//   * allows vectorization over 2nd dimension
//   * works for padded dst views
// Requires the locally modified ScatterView header included above.
template <typename SrcViewType, typename DstViewType>
struct ReduceDuplicates<
  SrcViewType,
  DstViewType,
  Kokkos::Experimental::ScatterSum,
  typename std::enable_if<
    unsigned(SrcViewType::rank) == 3 &&
    unsigned(DstViewType::rank) == 2 &&
    (std::is_same< typename SrcViewType::array_layout, LayoutRight >::value ||
     std::is_same< typename SrcViewType::array_layout, LayoutStride >::value) &&
    (std::is_same< typename DstViewType::array_layout, LayoutRight >::value ||
     std::is_same< typename DstViewType::array_layout, LayoutStride >::value)
  >::type >
{
  ReduceDuplicates(const SrcViewType& src,
                   const DstViewType& dst,
                   const size_t stride,
                   const size_t start,
                   const size_t n,
                   const std::string& name)
  {
    run(src,dst,stride,start,n,name);
  }

  void run(const SrcViewType& src,
           const DstViewType& dst,
           const size_t stride_in,
           const size_t start,
           const size_t n_in,
           const std::string& name)
  {
    typedef typename DstViewType::value_type ValueType;
    typedef typename DstViewType::execution_space ExecSpace;
    typedef TeamPolicy<ExecSpace, size_t> policy_type;
    typedef typename policy_type::member_type member_type;

    const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;

    const size_t n0 = src.extent(0);
    const size_t n1 = src.extent(1);
    const size_t n2 = src.extent(2);

    const size_t vector_size = is_cuda ? 16 : 1;
    const size_t team_size = is_cuda ? 256/vector_size : 1;
    const size_t row_block_size = 128;
    const size_t N1 = (n1+row_block_size-1) / row_block_size;
    policy_type policy(N1,team_size,vector_size);
    Kokkos::parallel_for( policy, KOKKOS_LAMBDA(const member_type& team)
    {
      for (size_t ii=team.team_rank(); ii<row_block_size; ii+=team_size) {
        const size_t i = team.league_rank()*row_block_size + ii;
        if (i < n1) {
          ValueType* dst_i = &dst(i,0);
          for (size_t k=start; k<n0; ++k) {
            const ValueType* src_ki = &src(k,i,0);
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, n2),
                                 [&] (const size_t j)
            {
              dst_i[j] += src_ki[j];
            });
          }
        }
      }
    }, "reduce_"+name );
  }
};
} } }
#endif

//-----------------------------------------------------------------------------
//  Method:  mttkrp, Sptensor X, output to FacMatrix
//-----------------------------------------------------------------------------

namespace Genten {
namespace Impl {

#if USE_SCATTER_VIEW

// MTTKRP kernel for Sptensor
template <unsigned FBS, unsigned VS, typename SparseTensor, typename ExecSpace>
typename std::enable_if<
    std::is_base_of< SptensorT     <ExecSpace>, SparseTensor >::value &&
  ! std::is_base_of< SptensorT_perm<ExecSpace>, SparseTensor >::value
  >::type
mttkrp_kernel(const SparseTensor& X,
              const Genten::KtensorT<ExecSpace>& u,
              const unsigned n,
              const Genten::FacMatrixT<ExecSpace>& v,
              const AlgParams& algParams)
{
  v = ttb_real(0.0);

  using Kokkos::Experimental::create_scatter_view;
  using Kokkos::Experimental::ScatterView;
  using Kokkos::Experimental::ScatterSum;
  // using Kokkos::Experimental::ScatterDuplicated;
  // using Kokkos::Experimental::ScatterNonDuplicated;
  // using Kokkos::Experimental::ScatterAtomic;
  // using Kokkos::Experimental::ScatterNonAtomic;

  static const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;
  static const unsigned RowBlockSize = 128;
  static const unsigned FacBlockSize = FBS;
  static const unsigned VectorSize = is_cuda ? VS : 1;
  static const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;

  /*const*/ unsigned nd = u.ndims();
  /*const*/ unsigned nc_total = u.ncomponents();
  /*const*/ ttb_indx nnz = X.nnz();
  const unsigned RowsPerTeam = TeamSize * RowBlockSize;
  const ttb_indx N = (nnz+RowsPerTeam-1)/RowsPerTeam;

  typedef Kokkos::TeamPolicy<ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  Policy policy(N, TeamSize, VectorSize);

  // Use factor matrix tile size as requested by the user, or all columns if
  // unspecified
  const unsigned FacTileSize =
    algParams.MTTKRPFactorMatrixTileSize > 0 ? algParams.MTTKRPFactorMatrixTileSize : nc_total;
  for (unsigned nc_beg=0; nc_beg<nc_total; nc_beg += FacTileSize) {
    /*
      const int dupl =
      is_serial ? ScatterNonDuplicated : ScatterDuplicated;
      const int cont = ScatterNonAtomic ;
      auto sv = create_scatter_view<ScatterSum,dupl,cont>(v.view());
    */
    const unsigned nc =
      nc_beg+FacTileSize <= nc_total ? FacTileSize : nc_total-nc_beg;
    const unsigned nc_end = nc_beg+nc;
    auto vv = Kokkos::subview(v.view(),Kokkos::ALL,
                              std::make_pair(nc_beg,nc_end));
    auto sv = create_scatter_view(vv);
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team)
    {
      auto va = sv.access();

      // Loop over tensor non-zeros with a large stride on the GPU to
      // reduce atomic contention when the non-zeros are in a nearly sorted
      // order (often the first dimension of the tensor).  This is similar
      // to an approach used in ParTi (https://github.com/hpcgarage/ParTI)
      // by Jaijai Li.
      ttb_indx offset;
      ttb_indx stride;
      if (is_cuda) {
        offset = team.league_rank()*TeamSize+team.team_rank();
        stride = team.league_size()*TeamSize;
      }
      else {
        offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        stride = 1;
      }

      auto row_func = [&](auto j, auto nj, auto Nj) {
        typedef TinyVec<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TV;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx i = offset + ii*stride;
          if (i >= nnz)
            continue;

          const ttb_indx k = X.subscript(i,n);
          const ttb_real x_val = X.value(i);

          // MTTKRP for dimension n
          TV tmp(nj, x_val);
          tmp *= &(u.weights(nc_beg+j));
          for (unsigned m=0; m<nd; ++m) {
            if (m != n)
              tmp *= &(u[m].entry(X.subscript(i,m),nc_beg+j));
          }
          va(k,j) += tmp;
        }
      };

      for (unsigned j=0; j<nc; j+=FacBlockSize) {
        if (j+FacBlockSize < nc) {
          const unsigned nj = FacBlockSize;
          row_func(j, nj, std::integral_constant<unsigned,nj>());
        }
        else {
          const unsigned nj = nc-j;
          row_func(j, nj, std::integral_constant<unsigned,0>());
        }
      }
    }, "mttkrp_kernel");

    sv.contribute_into(vv);
  }
}

#else

// MTTKRP kernel for Sptensor
template <unsigned FBS, unsigned VS, typename SparseTensor, typename ExecSpace>
typename std::enable_if<
    std::is_base_of< SptensorT     <ExecSpace>, SparseTensor >::value &&
  ! std::is_base_of< SptensorT_perm<ExecSpace>, SparseTensor >::value
  >::type
mttkrp_kernel(const SparseTensor& X,
              const Genten::KtensorT<ExecSpace>& u,
              const unsigned n,
              const Genten::FacMatrixT<ExecSpace>& v,
              const AlgParams& algParams)
{
  v = ttb_real(0.0);

  static const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;
  static const unsigned RowBlockSize = 128;
  static const unsigned FacBlockSize = FBS;
  static const unsigned VectorSize = is_cuda ? VS : 1;
  static const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;

  /*const*/ unsigned nd = u.ndims();
  /*const*/ unsigned nc = u.ncomponents();
  /*const*/ ttb_indx nnz = X.nnz();
  const unsigned RowsPerTeam = TeamSize * RowBlockSize;
  const ttb_indx N = (nnz+RowsPerTeam-1)/RowsPerTeam;

  typedef Kokkos::TeamPolicy<ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  Policy policy(N, TeamSize, VectorSize);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team)
  {
    // Loop over tensor non-zeros with a large stride on the GPU to
    // reduce atomic contention when the non-zeros are in a nearly sorted
    // order (often the first dimension of the tensor).  This is similar to
    // an approach used in ParTi (https://github.com/hpcgarage/ParTI)
    // by Jaijai Li.
    ttb_indx offset;
    ttb_indx stride;
    if (Genten::is_cuda_space<ExecSpace>::value) {
      offset = team.league_rank()*TeamSize+team.team_rank();
      stride = team.league_size()*TeamSize;
    }
    else {
      offset = (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
      stride = 1;
    }

    auto row_func = [&](auto j, auto nj, auto Nj) {
      typedef TinyVec<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TV;

      for (unsigned ii=0; ii<RowBlockSize; ++ii) {
        const ttb_indx i = offset + ii*stride;
        if (i >= nnz)
          continue;

        const ttb_indx k = X.subscript(i,n);
        const ttb_real x_val = X.value(i);

        // MTTKRP for dimension n
        TV tmp(nj, x_val);
        tmp *= &(u.weights(j));
        for (unsigned m=0; m<nd; ++m) {
          if (m != n)
            tmp *= &(u[m].entry(X.subscript(i,m),j));
        }
        Kokkos::atomic_add(&v.entry(k,j), tmp);
      }
    };

    for (unsigned j=0; j<nc; j+=FacBlockSize) {
      if (j+FacBlockSize < nc) {
        const unsigned nj = FacBlockSize;
        row_func(j, nj, std::integral_constant<unsigned,nj>());
      }
      else {
        const unsigned nj = nc-j;
        row_func(j, nj, std::integral_constant<unsigned,0>());
      }
    }
  }, "mttkrp_kernel");
}

#endif

// MTTKRP kernel for Sptensor_perm
template <unsigned FBS, unsigned VS, typename SparseTensor, typename ExecSpace>
typename std::enable_if<
  std::is_base_of< SptensorT_perm<ExecSpace>, SparseTensor >::value
  >::type
mttkrp_kernel(const SparseTensor& X,
              const Genten::KtensorT<ExecSpace>& u,
              const unsigned n,
              const Genten::FacMatrixT<ExecSpace>& v,
              const AlgParams& algParams)
{
  v = ttb_real(0.0);

  static const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;
  static const unsigned RowBlockSize = 128;
  static const unsigned FacBlockSize = FBS;
  static const unsigned VectorSize = is_cuda ? VS : 1;
  static const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;

  /*const*/ unsigned nd = u.ndims();
  /*const*/ unsigned nc = u.ncomponents();
  /*const*/ ttb_indx nnz = X.nnz();
  static const unsigned RowsPerTeam = TeamSize * RowBlockSize;
  const ttb_indx N = (nnz+RowsPerTeam-1)/RowsPerTeam;

  typedef Kokkos::TeamPolicy<ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  Policy policy(N, TeamSize, VectorSize);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team)
  {
    /*const*/ ttb_indx invalid_row = ttb_indx(-1);
    /*const*/ ttb_indx i_block =
      (team.league_rank()*TeamSize + team.team_rank())*RowBlockSize;

    auto row_func = [&](auto j, auto nj, auto Nj) {
      typedef TinyVec<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TV;
      TV val(nj, 0.0), tmp(nj, 0.0);

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
      if (j+FacBlockSize < nc) {
        const unsigned nj = FacBlockSize;
        row_func(j, nj, std::integral_constant<unsigned,nj>());
      }
      else {
        const unsigned nj = nc-j;
        row_func(j, nj, std::integral_constant<unsigned,0>());
      }
    }
  }, "mttkrp_kernel");
}

template <typename SparseTensor>
struct MTTKRP_Kernel {
  typedef typename SparseTensor::exec_space ExecSpace;

  const SparseTensor X;
  const Genten::KtensorT<ExecSpace> u;
  const ttb_indx n;
  const Genten::FacMatrixT<ExecSpace> v;
  const AlgParams algParams;

  MTTKRP_Kernel(const SparseTensor& X_,
                const Genten::KtensorT<ExecSpace>& u_,
                const ttb_indx n_,
                const Genten::FacMatrixT<ExecSpace>& v_,
                const AlgParams& algParams_) :
    X(X_), u(u_), n(n_), v(v_), algParams(algParams_) {}

  template <unsigned FBS, unsigned VS>
  void run() const {
    mttkrp_kernel<FBS,VS>(X,u,n,v,algParams);
  }
};

}
}

template <typename SparseTensor, typename ExecSpace>
void Genten::mttkrp(
  const SparseTensor& X,
  const Genten::KtensorT<ExecSpace>& u,
  const ttb_indx n,
  const Genten::FacMatrixT<ExecSpace>& v,
  const AlgParams& algParams)
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::mttkrp");
#endif

  const ttb_indx nc = u.ncomponents();     // Number of components
  const ttb_indx nd = u.ndims();           // Number of dimensions

  assert(X.ndims() == nd);
  assert(u.isConsistent());
  for (ttb_indx i = 0; i < nd; i++)
  {
    if (i != n)
      assert(u[i].nRows() == X.size(i));
  }
  assert( v.nRows() == X.size(n) );
  assert( v.nCols() == nc );

  Genten::Impl::MTTKRP_Kernel<SparseTensor> kernel(X,u,n,v,algParams);
  Genten::Impl::run_row_simd_kernel(kernel, nc);
}

// Version of mttkrp using a permutation array to improve locality of writes,
// and reduce atomic throughput needs.  This version is uses a rowptr array
// and a parallel_for over rows.
template <typename ExecSpace>
void Genten::mttkrp(const Genten::SptensorT_row<ExecSpace>& X,
                    const Genten::KtensorT<ExecSpace>& u,
                    const ttb_indx n,
                    const Genten::FacMatrixT<ExecSpace>& v,
                    const AlgParams& algParams)
{
#ifdef HAVE_CALIPER
    cali::Function cali_func("Genten::mttkrp_row");
#endif

  typedef typename ExecSpace::size_type size_type;

  const ttb_indx nc = u.ncomponents();      // Number of components
  const size_type nd = u.ndims();           // Number of dimensions

  assert(X.ndims() == nd);
  assert(u.isConsistent());
  for (size_type i = 0; i < nd; i++)
  {
    if (i != n)
      assert(u[i].nRows() == X.size(i));
  }

  // Resize and initialize the output factor matrix to zero.
  //v = FacMatrixT<ExecSpace>(X.size(n), nc);
  assert( v.nRows() == X.size(n) );
  assert( v.nCols() == nc );
  v = ttb_real(0.0);

  // Loop thru the nonzeros of the sparse tensor.  The inner loop updates
  // an entire row at a time, and is run only for nonzero elements.
  // Use team-based parallel-for.  Team is required for scratch memory and
  // will be useful for GPU.
  typedef Kokkos::TeamPolicy <ExecSpace> Policy;

  // Compute team and vector sizes, depending on the architecture
  const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;

  // Use the largest power of 2 <= nc, with a maximum of 64 for the vector size.
  const size_type VectorSize =
    nc == 1 ? 1 : std::min(64,2 << (int(std::log2(nc))-1));
  const size_type TeamSize =
    is_cuda ? 128/VectorSize : 1;
  const ttb_indx Nrow = X.size(n);
  const ttb_indx LeagueSize = (Nrow+TeamSize-1)/TeamSize;
  Policy policy(LeagueSize,TeamSize,VectorSize);

  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(typename Policy::member_type team)
  {
    const size_type league_index = team.league_rank();
    const size_type team_size = team.team_size();
    const size_type team_index = team.team_rank();
    const ttb_indx row = league_index*team_size+team_index;
    if (row >= Nrow)
      return;

    /*const*/ ttb_indx i_begin = X.getPermRowBegin(row,n);
    /*const*/ ttb_indx i_end = X.getPermRowBegin(row+1,n);
    if (i_end == i_begin)
      return;

    const size_type k = X.subscript(X.getPerm(i_begin,n),n);

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,nc),
                         [&] (const size_type& j)
    {
      const ttb_real w = u.weights(j);

      ttb_real val = 0.0;
      for (ttb_indx i=i_begin; i<i_end; ++i) {
        const ttb_indx p = X.getPerm(i,n);

        // Start val equal to the weights.
        ttb_real tmp = X.value(p) * w;

        for (size_type m=0; m<nd; ++m) {
          if (m != n) {
            // Update tmp with elementwise product of row i
            // from the m-th factor matrix.
            tmp *= u[m].entry(X.subscript(p,m),j);
          }
        }

        val += tmp;
      }

      // Add in result for this row
      v.entry(k,j) += val;

    });

  }, "Genten::mttkrp_row_kernel");

  return;
}
