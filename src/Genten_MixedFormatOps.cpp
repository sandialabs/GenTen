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

/*!
 * Methods in this file perform operations between objects of mixed formats.
 * In most cases the method could be moved into a particular class, but
 * then similar methods become disconnected, and it could force a
 * fundamental class like Tensor to include knowledge of a derived class
 * like Ktensor.
 */

#include <assert.h>

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
    std::is_same< typename SrcViewType::array_layout, LayoutRight >::value &&
    std::is_same< typename DstViewType::array_layout, LayoutRight >::value
  >::type >
{
  ReduceDuplicates(const SrcViewType& src,
                   const DstViewType& dst,
                   const size_t stride_in,
                   const size_t start,
                   const size_t n_in,
                   const std::string& name)
  {
    run(src,dst,stride_in,start,n_in,name);
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
//  Method:  innerprod, Sptensor and Ktensor with alternate weights
//-----------------------------------------------------------------------------

namespace Genten {
namespace Impl {

template <typename ExecSpace,
          unsigned RowBlockSize, unsigned FacBlockSize,
          unsigned TeamSize, unsigned VectorSize>
struct InnerProductKernel {

  typedef Kokkos::TeamPolicy<ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  typedef Kokkos::View< ttb_real**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

  const Genten::SptensorT<ExecSpace>& s;
  const Genten::KtensorT<ExecSpace>& u;
  const Genten::ArrayT<ExecSpace>& lambda;
  const ttb_indx nnz;
  const unsigned nd;

  const TeamMember& team;
  const unsigned team_index;
  const unsigned team_size;
  TmpScratchSpace tmp;
  const ttb_indx i_block;

  static inline Policy policy(const ttb_indx nnz_) {
    const ttb_indx N = (nnz_+RowBlockSize-1)/RowBlockSize;
    Policy policy(N,TeamSize,VectorSize);
    size_t bytes = TmpScratchSpace::shmem_size(RowBlockSize,FacBlockSize);
    return policy.set_scratch_size(0,Kokkos::PerTeam(bytes));
  }

  KOKKOS_INLINE_FUNCTION
  InnerProductKernel(const Genten::SptensorT<ExecSpace>& s_,
                     const Genten::KtensorT<ExecSpace>& u_,
                     const Genten::ArrayT<ExecSpace>& lambda_,
                     const TeamMember& team_) :
    s(s_), u(u_), lambda(lambda_),
    nnz(s.nnz()), nd(u.ndims()),
    team(team_), team_index(team.team_rank()), team_size(team.team_size()),
    tmp(team.team_scratch(0), RowBlockSize, FacBlockSize),
    i_block(team.league_rank()*RowBlockSize)
    {}

  template <unsigned Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const unsigned j, const unsigned nj_, ttb_real& d)
  {
    // nj.value == Nj_ if Nj_ > 0 and nj_ otherwise
    Kokkos::Impl::integral_nonzero_constant<unsigned, Nj> nj(nj_);

    const ttb_real *l = &lambda[j];

    for (unsigned ii=team_index; ii<RowBlockSize; ii+=team_size) {
      const ttb_indx i = i_block + ii;
      const ttb_real s_val = i < nnz ? s.value(i) : 0.0;

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,unsigned(nj.value)),
                           [&] (const unsigned& jj)
      {
        tmp(ii,jj) = s_val * l[jj];
      });

      if (i < nnz) {
        for (unsigned m=0; m<nd; ++m) {
          const ttb_real *row = &(u[m].entry(s.subscript(i,m),j));
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,unsigned(nj.value)),
                               [&] (const unsigned& jj)
          {
            tmp(ii,jj) *= row[jj];
          });
        }
      }

    }

    // Do the inner product with 3 levels of parallelism
    ttb_real t = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange( team, RowBlockSize ),
                            [&] ( const unsigned& k, ttb_real& t_outer )
    {
      ttb_real update_outer = 0.0;

      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,unsigned(nj.value)),
                              [&] (const unsigned& jj, ttb_real& t_inner)
      {
        t_inner += tmp(k,jj);
      }, update_outer);

      Kokkos::single( Kokkos::PerThread( team ), [&] ()
      {
        t_outer += update_outer;
      });

    }, t);

    Kokkos::single( Kokkos::PerTeam( team ), [&] ()
    {
      d += t;
    });
  }
};

// Specialization of InnerProductKernel to TeamSize == VectorSize == 1
// (for, e.g., KNL).  Overall this is about 10% faster on KNL.  We could use a
// RangePolicy here, but the TeamPolicy seems to be about 25% faster on KNL.
template <typename ExecSpace,
          unsigned RowBlockSize, unsigned FacBlockSize>
struct InnerProductKernel<ExecSpace,RowBlockSize,FacBlockSize,1,1> {

  typedef Kokkos::TeamPolicy<ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;

  const Genten::SptensorT<ExecSpace>& s;
  const Genten::KtensorT<ExecSpace>& u;
  const Genten::ArrayT<ExecSpace>& lambda;
  const ttb_indx nnz;
  const unsigned nd;
  const ttb_indx i_block;

  alignas(64) ttb_real val[FacBlockSize];
  alignas(64) ttb_real tmp[FacBlockSize];

  static inline Policy policy(const ttb_indx nnz_) {
    const ttb_indx N = (nnz_+RowBlockSize-1)/RowBlockSize;
    Policy policy(N,1,1);
    return policy;
  }

  KOKKOS_INLINE_FUNCTION
  InnerProductKernel(const Genten::SptensorT<ExecSpace>& s_,
                     const Genten::KtensorT<ExecSpace>& u_,
                     const Genten::ArrayT<ExecSpace>& lambda_,
                     const TeamMember& team_) :
    s(s_), u(u_), lambda(lambda_), nnz(s.nnz()), nd(u.ndims()),
    i_block(team_.league_rank()*RowBlockSize)
    {}

  template <unsigned Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const unsigned j, const unsigned nj_, ttb_real& d)
  {
    // nj.value == Nj if Nj > 0 and nj_ otherwise
    Kokkos::Impl::integral_nonzero_constant<unsigned, Nj> nj(nj_);

    const ttb_real *l = &lambda[j];

    for (ttb_indx jj=0; jj<nj.value; ++jj)
      val[jj] = 0.0;

    for (unsigned ii=0; ii<RowBlockSize; ++ii) {
      const ttb_indx i = i_block + ii;

      if (i < nnz) {
        const ttb_real s_val = s.value(i);
        for (ttb_indx jj=0; jj<nj.value; ++jj)
          tmp[jj] = s_val * l[jj];

        for (unsigned m=0; m<nd; ++m) {
          const ttb_real *row = &(u[m].entry(s.subscript(i,m),j));
          for (ttb_indx jj=0; jj<nj.value; ++jj)
            tmp[jj] *= row[jj];
        }

        for (ttb_indx jj=0; jj<nj.value; ++jj)
          val[jj] += tmp[jj];
      }
    }

    for (ttb_indx jj=0; jj<nj.value; ++jj)
      d += val[jj];
  }
};

template <typename ExecSpace, unsigned FacBlockSize>
ttb_real innerprod_kernel(const Genten::SptensorT<ExecSpace>& s,
                          const Genten::KtensorT<ExecSpace>& u,
                          const Genten::ArrayT<ExecSpace>& lambda)
{
  // Compute team and vector sizes, depending on the architecture
  const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;

  const unsigned VectorSize =
    is_cuda ? (FacBlockSize <= 16 ? FacBlockSize : 16) : 1;
  const unsigned TeamSize =
    is_cuda ? 128/VectorSize : 1;
  const unsigned RowBlockSize =
    is_cuda ? TeamSize : 32;

  typedef InnerProductKernel<ExecSpace,RowBlockSize,FacBlockSize,TeamSize,VectorSize> Kernel;
  typedef typename Kernel::TeamMember TeamMember;

  // Do the inner product
  ttb_real dTotal = 0.0;
  Kokkos::parallel_reduce("Genten::innerprod_kernel",
                          Kernel::policy(s.nnz()),
                          KOKKOS_LAMBDA(TeamMember team, ttb_real& d)
  {
    // For some reason using the above typedef causes a really strange
    // compiler error with NVCC 8.0 + GCC 4.9.2
    InnerProductKernel<ExecSpace,RowBlockSize,FacBlockSize,TeamSize,VectorSize> kernel(s,u,lambda,team);

    const unsigned nc = u.ncomponents();
    for (unsigned j=0; j<nc; j+=FacBlockSize) {
      if (j+FacBlockSize <= nc)
        kernel.template run<FacBlockSize>(j, FacBlockSize, d);
      else
        kernel.template run<0>(j, nc-j, d);
    }

  }, dTotal);
  Kokkos::fence();

  return dTotal;
}

}
}

template <typename ExecSpace>
ttb_real Genten::innerprod(const Genten::SptensorT<ExecSpace>& s,
                           const Genten::KtensorT<ExecSpace>& u,
                           const Genten::ArrayT<ExecSpace>& lambda)
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::innerprod");
#endif

  const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;

  const ttb_indx nc = u.ncomponents();               // Number of components
  const ttb_indx nd = u.ndims();                     // Number of dimensions

  // Check on sizes
  assert(nd == s.ndims());
  assert(u.isConsistent(s.size()));
  assert(nc == lambda.size());

  // Call kernel with factor block size determined from nc
  ttb_real d = 0.0;
  if (nc == 1)
    d = Impl::innerprod_kernel<ExecSpace,1>(s,u,lambda);
  else if (nc == 2)
    d = Impl::innerprod_kernel<ExecSpace,2>(s,u,lambda);
  else if (nc <= 4)
    d = Impl::innerprod_kernel<ExecSpace,4>(s,u,lambda);
  else if (nc <= 8)
    d = Impl::innerprod_kernel<ExecSpace,8>(s,u,lambda);
  else if (nc <= 16)
    d = Impl::innerprod_kernel<ExecSpace,16>(s,u,lambda);
  else if (nc < 64 || !is_cuda)
    d = Impl::innerprod_kernel<ExecSpace,32>(s,u,lambda);
  else
    d = Impl::innerprod_kernel<ExecSpace,64>(s,u,lambda);

  return d;
}

//-----------------------------------------------------------------------------
//  Method:  mttkrp, Sptensor X, output to FacMatrix
//-----------------------------------------------------------------------------

namespace Genten {
namespace Impl {

template <typename SparseTensor, typename Result,
          unsigned RowBlockSize, unsigned FacBlockSize,
          unsigned TeamSize, unsigned VectorSize>
struct MTTKRP_KernelBlock;

#if USE_SCATTER_VIEW

// Specialization of MTTKRP_KernelBlock for Sptensor
template <typename ExecSpace, typename Result, unsigned RowBlockSize,
          unsigned FacBlockSize, unsigned TeamSize, unsigned VectorSize>
struct MTTKRP_KernelBlock<
  Genten::SptensorT<ExecSpace>, Result,
  RowBlockSize, FacBlockSize, TeamSize, VectorSize>
{
  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;

  const Genten::SptensorT<ExecSpace> X;
  const Genten::KtensorT<ExecSpace> u;
  const unsigned n;
  const unsigned nd;
  const ttb_indx nnz;
  const Result v;
  const TeamMember team;

  static inline Policy policy(const ttb_indx nnz_) {
    const unsigned RowsPerTeam = TeamSize * RowBlockSize;
    const ttb_indx N = (nnz_+RowsPerTeam-1)/RowsPerTeam;
    Policy policy(N,TeamSize,VectorSize);
    return policy;
  }

  KOKKOS_INLINE_FUNCTION
  MTTKRP_KernelBlock(const Genten::SptensorT<ExecSpace>& X_,
                     const Genten::KtensorT<ExecSpace>& u_,
                     const unsigned n_,
                     const Result& v_,
                     const TeamMember& team_) :
    X(X_), u(u_), n(n_), nd(u.ndims()), nnz(X.nnz()), v(v_), team(team_)
    {}

  template <unsigned Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const unsigned j, const unsigned nj)
  {
    typedef Genten::TinyVec<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj, VectorSize> TV;

    const ttb_real* lambda = &u.weights(0);
    auto sv = v.access();

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
    for (unsigned ii=0; ii<RowBlockSize; ++ii) {
      const ttb_indx i = offset + ii*stride;
      if (i >= nnz)
        return;

      const ttb_indx k = X.subscript(i,n);
      const ttb_real x_val = X.value(i);

      // Start tmp equal to the weights.
      TV tmp(nj, x_val);
      tmp *= lambda+j;

      for (unsigned m=0; m<nd; ++m) {
        if (m != n) {
          // Update tmp array with elementwise product of row i
          // from the m-th factor matrix.  Length of the row is nc.
          const ttb_real *row = &(u[m].entry(X.subscript(i,m),j));
          tmp *= row;
        }
      }

      // Update output by adding tmp array.
      sv(k,j) += tmp;
    }
  }
};

#else

// Specialization of MTTKRP_KernelBlock for Sptensor
template <typename ExecSpace, unsigned RowBlockSize, unsigned FacBlockSize,
          unsigned TeamSize, unsigned VectorSize>
struct MTTKRP_KernelBlock<
  Genten::SptensorT<ExecSpace>, Genten::FacMatrixT<ExecSpace>,
  RowBlockSize, FacBlockSize, TeamSize, VectorSize>
{
  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;

  const Genten::SptensorT<ExecSpace> X;
  const Genten::KtensorT<ExecSpace> u;
  const unsigned n;
  const unsigned nd;
  const ttb_indx nnz;
  const Genten::FacMatrixT<ExecSpace> v;
  const TeamMember team;

  static inline Policy policy(const ttb_indx nnz_) {
    const unsigned RowsPerTeam = TeamSize * RowBlockSize;
    const ttb_indx N = (nnz_+RowsPerTeam-1)/RowsPerTeam;
    Policy policy(N,TeamSize,VectorSize);
    return policy;
  }

  KOKKOS_INLINE_FUNCTION
  MTTKRP_KernelBlock(const Genten::SptensorT<ExecSpace>& X_,
                     const Genten::KtensorT<ExecSpace>& u_,
                     const unsigned n_,
                     const Genten::FacMatrixT<ExecSpace>& v_,
                     const TeamMember& team_) :
    X(X_), u(u_), n(n_), nd(u.ndims()), nnz(X.nnz()), v(v_), team(team_)
    {}

  template <unsigned Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const unsigned j, const unsigned nj)
  {
    typedef Genten::TinyVec<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj, VectorSize> TV;

    const ttb_real* lambda = &u.weights(0);

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
    for (unsigned ii=0; ii<RowBlockSize; ++ii) {
      const ttb_indx i = offset + ii*stride;
      if (i >= nnz)
        return;

      const ttb_indx k = X.subscript(i,n);
      const ttb_real x_val = X.value(i);

      // Start tmp equal to the weights.
      TV tmp(nj, x_val);
      tmp *= lambda+j;

      for (unsigned m=0; m<nd; ++m) {
        if (m != n) {
          // Update tmp array with elementwise product of row i
          // from the m-th factor matrix.  Length of the row is nc.
          const ttb_real *row = &(u[m].entry(X.subscript(i,m),j));
          tmp *= row;
        }
      }

      // Update output by adding tmp array.
      Kokkos::atomic_add(&v.entry(k,j), tmp);
    }
  }
};

#endif

// Specialization of MTTKRP_KernelBlock for Sptensor_perm
template <typename ExecSpace, unsigned RowBlockSize, unsigned FacBlockSize,
          unsigned TeamSize, unsigned VectorSize>
struct MTTKRP_KernelBlock<
  Genten::SptensorT_perm<ExecSpace>, Genten::FacMatrixT<ExecSpace>,
  RowBlockSize, FacBlockSize, TeamSize, VectorSize>
{
  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;

  const Genten::SptensorT_perm<ExecSpace> X;
  const Genten::KtensorT<ExecSpace> u;
  const unsigned n;
  const unsigned nd;
  const ttb_indx nnz;
  const Genten::FacMatrixT<ExecSpace> v;
  const ttb_indx i_block;

  static inline Policy policy(const ttb_indx nnz_) {
    const unsigned RowsPerTeam = TeamSize * RowBlockSize;
    const ttb_indx N = (nnz_+RowsPerTeam-1)/RowsPerTeam;
    Policy policy(N,TeamSize,VectorSize);
    return policy;
  }

  KOKKOS_INLINE_FUNCTION
  MTTKRP_KernelBlock(const Genten::SptensorT_perm<ExecSpace>& X_,
                     const Genten::KtensorT<ExecSpace>& u_,
                     const ttb_indx n_,
                     const Genten::FacMatrixT<ExecSpace>& v_,
                     const TeamMember& team) :
    X(X_), u(u_), n(n_), nd(u.ndims()), nnz(X.nnz()), v(v_),
    i_block(team.league_rank()*RowBlockSize*TeamSize + RowBlockSize*team.team_rank())
    {}

  template <ttb_indx Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const unsigned j, const unsigned nj) const {
    const ttb_indx invalid_row = ttb_indx(-1);

    ttb_indx row_prev = invalid_row;
    ttb_indx row = invalid_row;
    ttb_indx first_row = invalid_row;
    ttb_indx p = invalid_row;
    ttb_real x_val = 0.0;

    typedef Genten::TinyVec<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj, VectorSize> TV;
    TV val(nj, 0.0), tmp(nj, 0.0);

    const ttb_real* lambda = &u.weights(0);

    for (unsigned ii=0; ii<RowBlockSize; ++ii) {
      const ttb_indx i = i_block+ii;

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
        tmp.load(lambda+j);
        tmp *= x_val;

        for (unsigned m=0; m<nd; ++m) {
          if (m != n) {
            // Update tmp array with elementwise product of row i
            // from the m-th factor matrix.
            // Note:  Doing this load through texture cache on the GPU
            // could help performance.  However we would have to get rid
            // of pointer to use the MemoryRandomAccess memory trait.
            const ttb_real *rowptr = &(u[m].entry(X.subscript(p,m),j));
            tmp *= rowptr;
          }
        }

        val += tmp;
      }
    }

    // Sum in last row
    if (row != invalid_row) {
      Kokkos::atomic_add(&v.entry(row,j), val);
    }
  }
};

template <unsigned VS, typename SparseTensor>
void mttkrp_kernel(
  const SparseTensor& X,
  const Genten::KtensorT<typename SparseTensor::exec_space>& u,
  const ttb_indx n,
  const Genten::FacMatrixT<typename SparseTensor::exec_space>& v)
{
  typedef typename SparseTensor::exec_space ExecSpace;

  // Compute team and vector sizes, depending on the architecture
  static const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;
  static const unsigned FacBlockSize = 128;
  static const unsigned VectorSize = is_cuda ? VS : 1;
  static const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;
  static const unsigned RowBlockSize = 128;

  static const unsigned VS4 = 4*VS;
  static const unsigned VS3 = 3*VS;
  static const unsigned VS2 = 2*VS;
  static const unsigned VS1 = 1*VS;

  typedef MTTKRP_KernelBlock<SparseTensor, Genten::FacMatrixT<ExecSpace>, RowBlockSize, FacBlockSize, TeamSize, VectorSize> Kernel;
  typedef typename Kernel::TeamMember TeamMember;

  const unsigned nc = u.ncomponents();
  if (nc > VS3) {
    Kokkos::parallel_for(Kernel::policy(X.nnz()),
                         KOKKOS_LAMBDA(TeamMember team)
    {
      MTTKRP_KernelBlock<SparseTensor, Genten::FacMatrixT<ExecSpace>, RowBlockSize, VS4, TeamSize, VectorSize> kernel(X, u, n, v, team);
      for (unsigned j=0; j<nc; j+=VS4) {
        if (j+VS4 <= nc)
          kernel.template run<VS4>(j, VS4);
        else
          kernel.template run<0>(j, nc-j);
      }

    }, "Genten::mttkrp_kernel");
  }
  else if (nc > VS2) {
    Kokkos::parallel_for(Kernel::policy(X.nnz()),
                         KOKKOS_LAMBDA(TeamMember team)
    {
      MTTKRP_KernelBlock<SparseTensor, Genten::FacMatrixT<ExecSpace>, RowBlockSize, VS3, TeamSize, VectorSize> kernel(X, u, n, v, team);
      for (unsigned j=0; j<nc; j+=VS3) {
        if (j+VS3 <= nc)
          kernel.template run<VS3>(j, VS3);
        else
          kernel.template run<0>(j, nc-j);
      }

    }, "Genten::mttkrp_kernel");
  }
  else if (nc > VS1) {
    Kokkos::parallel_for(Kernel::policy(X.nnz()),
                         KOKKOS_LAMBDA(TeamMember team)
    {
      MTTKRP_KernelBlock<SparseTensor, Genten::FacMatrixT<ExecSpace>, RowBlockSize, VS2, TeamSize, VectorSize> kernel(X, u, n, v, team);
      for (unsigned j=0; j<nc; j+=VS2) {
        if (j+VS2 <= nc)
          kernel.template run<VS2>(j, VS2);
        else
          kernel.template run<0>(j, nc-j);
      }

    }, "Genten::mttkrp_kernel");
  }
  else {
    Kokkos::parallel_for(Kernel::policy(X.nnz()),
                         KOKKOS_LAMBDA(TeamMember team)
    {
      MTTKRP_KernelBlock<SparseTensor, Genten::FacMatrixT<ExecSpace>, RowBlockSize, VS1, TeamSize, VectorSize> kernel(X, u, n, v, team);
      for (unsigned j=0; j<nc; j+=VS1) {
        if (j+VS1 <= nc)
          kernel.template run<VS1>(j, VS1);
        else
          kernel.template run<0>(j, nc-j);
      }

    }, "Genten::mttkrp_kernel");
  }
}

#if USE_SCATTER_VIEW

template <unsigned VS, typename ExecSpace>
void mttkrp_kernel(
  const Genten::SptensorT<ExecSpace>& X,
  const Genten::KtensorT<ExecSpace>& u,
  const ttb_indx n,
  const Genten::FacMatrixT<ExecSpace>& v)
{
  // Compute team and vector sizes, depending on the architecture
  static const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;
// #if defined(KOKKOS_HAVE_SERIAL)
//   const bool is_serial = std::is_same<ExecSpace,Kokkos::Serial>::value;
// #else
//   const bool is_serial = false;
// #endif
  static const unsigned FacBlockSize = 128;
  static const unsigned VectorSize = is_cuda ? VS : 1;
  static const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;
  static const unsigned RowBlockSize = 128;

  static const unsigned VS4 = 4*VS;
  static const unsigned VS3 = 3*VS;
  static const unsigned VS2 = 2*VS;
  static const unsigned VS1 = 1*VS;

  using Kokkos::Experimental::create_scatter_view;
  using Kokkos::Experimental::ScatterView;
  using Kokkos::Experimental::ScatterSum;
  // using Kokkos::Experimental::ScatterDuplicated;
  // using Kokkos::Experimental::ScatterNonDuplicated;
  // using Kokkos::Experimental::ScatterAtomic;
  // using Kokkos::Experimental::ScatterNonAtomic;
  using Kokkos::LayoutRight;

  auto vv = v.view();

  /*
  const int dupl =
    is_serial ? ScatterNonDuplicated : ScatterDuplicated;
  const int cont = ScatterNonAtomic ;
  auto sv = create_scatter_view<ScatterSum,dupl,cont>(vv);
  */

  // For some reason this doesn't work when matching the partial specialization
  // of MTTKRP_KernelBlock, so we have to spell out the type of ScatterView
  // explicitly:
  // auto sv = create_scatter_view(vv);
  // typedef decltype(sv) ScatterViewType;
  typedef ScatterView<ttb_real**,LayoutRight,ExecSpace,ScatterSum> ScatterViewType;
  ScatterViewType sv = create_scatter_view(vv);

  typedef Genten::SptensorT<ExecSpace> SparseTensor;
  typedef MTTKRP_KernelBlock<SparseTensor, ScatterViewType, RowBlockSize, FacBlockSize, TeamSize, VectorSize> Kernel;
  typedef typename Kernel::TeamMember TeamMember;

  const unsigned nc = u.ncomponents();
  if (nc > VS3) {
    Kokkos::parallel_for(Kernel::policy(X.nnz()),
                         KOKKOS_LAMBDA(TeamMember team)
    {
      MTTKRP_KernelBlock<SparseTensor, ScatterViewType, RowBlockSize, VS4, TeamSize, VectorSize> kernel(X, u, n, sv, team);
      for (unsigned j=0; j<nc; j+=VS4) {
        if (j+VS4 <= nc)
          kernel.template run<VS4>(j, VS4);
        else
          kernel.template run<0>(j, nc-j);
      }

    }, "Genten::mttkrp_kernel");
  }
  else if (nc > VS2) {
    Kokkos::parallel_for(Kernel::policy(X.nnz()),
                         KOKKOS_LAMBDA(TeamMember team)
    {
      MTTKRP_KernelBlock<SparseTensor, ScatterViewType, RowBlockSize, VS3, TeamSize, VectorSize> kernel(X, u, n, sv, team);
      for (unsigned j=0; j<nc; j+=VS3) {
        if (j+VS3 <= nc)
          kernel.template run<VS3>(j, VS3);
        else
          kernel.template run<0>(j, nc-j);
      }

    }, "Genten::mttkrp_kernel");
  }
  else if (nc > VS1) {
    Kokkos::parallel_for(Kernel::policy(X.nnz()),
                         KOKKOS_LAMBDA(TeamMember team)
    {
      MTTKRP_KernelBlock<SparseTensor, ScatterViewType, RowBlockSize, VS2, TeamSize, VectorSize> kernel(X, u, n, sv, team);
      for (unsigned j=0; j<nc; j+=VS2) {
        if (j+VS2 <= nc)
          kernel.template run<VS2>(j, VS2);
        else
          kernel.template run<0>(j, nc-j);
      }

    }, "Genten::mttkrp_kernel");
  }
  else {
    Kokkos::parallel_for(Kernel::policy(X.nnz()),
                         KOKKOS_LAMBDA(TeamMember team)
    {
      MTTKRP_KernelBlock<SparseTensor, ScatterViewType, RowBlockSize, VS1, TeamSize, VectorSize> kernel(X, u, n, sv, team);
      for (unsigned j=0; j<nc; j+=VS1) {
        if (j+VS1 <= nc)
          kernel.template run<VS1>(j, VS1);
        else
          kernel.template run<0>(j, nc-j);
      }

    }, "Genten::mttkrp_kernel");
  }

  sv.contribute_into(vv);
}

#endif

}
}

template <typename SparseTensor, typename ExecSpace>
void Genten::mttkrp(
  const SparseTensor& X,
  const Genten::KtensorT<ExecSpace>& u,
  const ttb_indx n,
  const Genten::FacMatrixT<ExecSpace>& v)
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

  // Resize and initialize the output factor matrix to zero.
  assert( v.nRows() == X.size(n) );
  assert( v.nCols() == nc );
  v = ttb_real(0.0);

  if (nc >= 96)
    Impl::mttkrp_kernel<32>(X, u, n, v);
  else if (nc >= 48)
    Impl::mttkrp_kernel<16>(X, u, n, v);
  else if (nc >= 8)
    Impl::mttkrp_kernel<8>(X, u, n, v);
  else if (nc >= 4)
    Impl::mttkrp_kernel<4>(X, u, n, v);
  else if (nc >= 2)
    Impl::mttkrp_kernel<2>(X, u, n, v);
  else
    Impl::mttkrp_kernel<1>(X, u, n, v);

  return;
}

// Version of mttkrp using a permutation array to improve locality of writes,
// and reduce atomic throughput needs.  This version is uses a rowptr array
// and a parallel_for over rows.
template <typename ExecSpace>
void Genten::mttkrp(const Genten::SptensorT_row<ExecSpace>& X,
                    const Genten::KtensorT<ExecSpace>& u,
                    const ttb_indx n,
                    const Genten::FacMatrixT<ExecSpace>& v)
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

#define INST_MACRO(SPACE)                                               \
  template                                                              \
  ttb_real innerprod<>(const Genten::SptensorT<SPACE>& s,               \
                       const Genten::KtensorT<SPACE>& u,                \
                       const Genten::ArrayT<SPACE>& lambda);            \
  template                                                              \
  void mttkrp<>(const Genten::SptensorT<SPACE>& X,                      \
                const Genten::KtensorT<SPACE>& u,                       \
                const ttb_indx n,                                       \
                const Genten::FacMatrixT<SPACE>& v);                    \
  template                                                              \
  void mttkrp<>(const Genten::SptensorT_perm<SPACE>& X,                 \
                const Genten::KtensorT<SPACE>& u,                       \
                const ttb_indx n,                                       \
                const Genten::FacMatrixT<SPACE>& v);
GENTEN_INST(INST_MACRO)
