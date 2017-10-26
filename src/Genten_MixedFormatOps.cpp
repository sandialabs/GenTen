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

#define USE_NEW_MTTKRP 1
#define USE_NEW_MTTKRP_PERM 1
#define USE_NEW_IP 1


//-----------------------------------------------------------------------------
//  Method:  innerprod, Sptensor and Ktensor with alternate weights
//-----------------------------------------------------------------------------

#if USE_NEW_IP == 1

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
#if defined(KOKKOS_HAVE_CUDA)
  const bool is_cuda = std::is_same<ExecSpace,Kokkos::Cuda>::value;
#else
  const bool is_cuda = false;
#endif

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
  Kokkos::parallel_reduce(Kernel::policy(s.nnz()),
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

  return dTotal;
}

}
}

template <typename ExecSpace>
ttb_real Genten::innerprod(const Genten::SptensorT<ExecSpace>& s,
                           const Genten::KtensorT<ExecSpace>& u,
                           const Genten::ArrayT<ExecSpace>& lambda)
{
#if defined(KOKKOS_HAVE_CUDA)
  const bool is_cuda = std::is_same<ExecSpace,Kokkos::Cuda>::value;
#else
  const bool is_cuda = false;
#endif

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

#else

template <typename ExecSpace>
ttb_real Genten::innerprod(const Genten::SptensorT<ExecSpace>& s,
                           const Genten::KtensorT<ExecSpace>& u,
                           const Genten::ArrayT<ExecSpace>& lambda)
{
  typedef typename ExecSpace::size_type size_type;

  const ttb_indx nc = u.ncomponents();               // Number of components
  const size_type nd = u.ndims();                     // Number of dimensions

  // Check on sizes
  assert(nd == s.ndims());
  assert(u.isConsistent(s.size()));
  assert(nc == lambda.size());

  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  typedef Kokkos::View< ttb_real**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

  // Compute team and vector sizes, depending on the architecture
#if defined(KOKKOS_HAVE_CUDA)
  const bool is_cuda = std::is_same<ExecSpace,Kokkos::Cuda>::value;
#else
  const bool is_cuda = false;
#endif
#if defined(KOKKOS_HAVE_OPENMP)
  const bool is_openmp = std::is_same<ExecSpace,Kokkos::OpenMP>::value;
  const size_type thread_pool_size =
    is_openmp ? Kokkos::OpenMP::thread_pool_size(2) : 1;
#else
  const size_type thread_pool_size = 1;
#endif
  // Use the largest power of 2 <= nc, with a maximum of 16 for the vector size.
  const size_type VectorSize =
    nc == 1 ? 1 : std::min(16,2 << int(std::log2(nc))-1);
  const size_type TeamSize =
    is_cuda ? 128/VectorSize : thread_pool_size;

  // For a maximum BlockSize=64, VectorSize=16, and TeamSize=8, the kernel
  // needs 8*64*8 = 4K shared memory per CUDA block.  Most modern GPUs have
  // between 48K and 64K shared memory per SM, allowing between 12 and 16
  // blocks per SM, which is typically enough for 75% occupancy.
  const size_type BlockSize = std::min(size_type(64),size_type(nc));

  // compute how much scratch memory (in bytes) is needed
  const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,BlockSize);

  ttb_indx nnz = s.nnz();
  ttb_indx N = (nnz+TeamSize-1)/TeamSize;
  Policy policy(N,TeamSize,VectorSize);

  // Do the inner product
  ttb_real dTotal = 0.0;

  Kokkos::parallel_reduce(policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
                          KOKKOS_LAMBDA(typename Policy::member_type team,
                                        ttb_real& d)
  {
    const size_type team_size = team.team_size();
    const size_type team_index = team.team_rank();
    const ttb_indx i = team.league_rank()*team_size+team_index;

    // Store local product in scratch array of length nc
    TmpScratchSpace tmp(team.team_scratch(0), team_size, BlockSize);

    const ttb_real s_val = i < nnz ? s.value(i) : 0.0;

    for (ttb_indx j_block=0; j_block<nc; j_block+=BlockSize) {

      // Start tmp equal to the weights.
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,BlockSize),
                           [&] (const size_type& jj)
      {
        const ttb_indx j = j_block+jj;
        tmp(team_index,jj) = j < nc ? s_val * lambda[j] : 0.0;
      });

      if (i < nnz) {
        for (size_type m=0; m<nd; ++m) {
          const ttb_real *row = u[m].rowptr(s.subscript(i,m));
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,BlockSize),
                               [&] (const size_type& jj)
          {
            const ttb_indx j = j_block+jj;
            if (j<nc)
              tmp(team_index,jj) *= row[j];
          });
        }
      }

      // Do the inner product with 3 levels of parallelism
      ttb_real t = 0.0;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange( team, team_size ),
                              [&] ( const size_type& k, ttb_real& t_outer )
      {
        ttb_real update_outer = 0.0;

        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,BlockSize),
                                [&] (const size_type& jj, ttb_real& t_inner)
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

  }, dTotal);

  return dTotal;
}

#endif


//-----------------------------------------------------------------------------
//  Method:  mttkrp, Sptensor X, output to FacMatrix
//-----------------------------------------------------------------------------

#if USE_NEW_MTTKRP

namespace Genten {
namespace Impl {

template <typename ExecSpace, unsigned FacBlockSize,
          unsigned TeamSize, unsigned VectorSize>
struct MTTKRP_KernelBlock {

  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  typedef Kokkos::View< ttb_real**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

  const Genten::SptensorT<ExecSpace>& X;
  const Genten::KtensorT<ExecSpace>& u;
  const unsigned n;
  const unsigned nd;
  const Genten::FacMatrixT<ExecSpace>& v;
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
  MTTKRP_KernelBlock(const Genten::SptensorT<ExecSpace>& X_,
                     const Genten::KtensorT<ExecSpace>& u_,
                     const unsigned n_,
                     const Genten::FacMatrixT<ExecSpace>& v_,
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
void mttkrp_kernel(const Genten::SptensorT<ExecSpace>& X,
                   const Genten::KtensorT<ExecSpace>& u,
                   const ttb_indx n,
                   const Genten::FacMatrixT<ExecSpace>& v)
{
  // Compute team and vector sizes, depending on the architecture
#if defined(KOKKOS_HAVE_CUDA)
  const bool is_cuda = std::is_same<ExecSpace,Kokkos::Cuda>::value;
#else
  const bool is_cuda = false;
#endif
  const unsigned VectorSize = is_cuda ? (FacBlockSize <= 16 ? FacBlockSize : 16) : 1;
  const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;

  const unsigned nc = u.ncomponents();
  const ttb_indx nnz = X.nnz();

  typedef MTTKRP_KernelBlock<ExecSpace, FacBlockSize, TeamSize, VectorSize> Kernel;
  typedef typename Kernel::TeamMember TeamMember;

  Kokkos::parallel_for(Kernel::policy(nnz),
                       KOKKOS_LAMBDA(TeamMember team)
  {
    const ttb_indx i = team.league_rank()*team.team_size()+team.team_rank();
    if (i >= nnz)
      return;

    MTTKRP_KernelBlock<ExecSpace, FacBlockSize, TeamSize, VectorSize> kernel(X, u, n, v, i, team);

    for (unsigned j=0; j<nc; j+=FacBlockSize) {
      if (j+FacBlockSize <= nc)
        kernel.template run<FacBlockSize>(j, FacBlockSize);
      else
        kernel.template run<0>(j, nc-j);
    }

  });

  return;
}

}
}

template <typename ExecSpace>
void Genten::mttkrp(const Genten::SptensorT<ExecSpace>& X,
                    const Genten::KtensorT<ExecSpace>& u,
                    const ttb_indx n,
                    const Genten::FacMatrixT<ExecSpace>& v)
{
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
  //v = FacMatrixT<ExecSpace>(X.size(n), nc);
  assert( v.nRows() == X.size(n) );
  assert( v.nCols() == nc );
  v = ttb_real(0.0);

  // Call kernel with factor block size determined from nc
  if (nc == 1)
    Impl::mttkrp_kernel<ExecSpace,1>(X,u,n,v);
  else if (nc == 2)
    Impl::mttkrp_kernel<ExecSpace,2>(X,u,n,v);
  else if (nc <= 4)
    Impl::mttkrp_kernel<ExecSpace,4>(X,u,n,v);
  else if (nc <= 8)
    Impl::mttkrp_kernel<ExecSpace,8>(X,u,n,v);
  else if (nc <= 16)
    Impl::mttkrp_kernel<ExecSpace,16>(X,u,n,v);
  else
    Impl::mttkrp_kernel<ExecSpace,32>(X,u,n,v);

  return;
}

#else

template <typename ExecSpace>
void Genten::mttkrp(const Genten::SptensorT<ExecSpace>& X,
                    const Genten::KtensorT<ExecSpace>& u,
                    const  ttb_indx n,
                    const Genten::FacMatrixT<ExecSpace>& v)
{
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
  typedef Kokkos::View< ttb_real**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

  // Compute team and vector sizes, depending on the architecture
#if defined(KOKKOS_HAVE_CUDA)
  const bool is_cuda = std::is_same<ExecSpace,Kokkos::Cuda>::value;
#else
  const bool is_cuda = false;
#endif
#if defined(KOKKOS_HAVE_OPENMP)
  const bool is_openmp = std::is_same<ExecSpace,Kokkos::OpenMP>::value;
  const size_type thread_pool_size =
    is_openmp ? Kokkos::OpenMP::thread_pool_size(2) : 1;
#else
  const size_type thread_pool_size = 1;
#endif
  // Use the largest power of 2 <= nc, with a maximum of 8 for the vector size.
  const size_type VectorSize =
    nc == 1 ? 1 : std::min(8,2 << int(std::log2(nc))-1);
  const size_type TeamSize =
    is_cuda ? 128/VectorSize : thread_pool_size;

  // For a maximum BlockSize=16 and TeamSize=16, the kernel
  // needs 8*16*16 = 2K shared memory per CUDA block.  Most modern GPUs have
  // between 48K and 64K shared memory per SM, allowing between 24 and 32
  // blocks per SM, which is typically enough for 100% occupancy.
  const size_type BlockSize = std::min(size_type(16),size_type(nc));

  // compute how much scratch memory (in bytes) is needed
  const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,BlockSize);

  ttb_indx nnz = X.nnz();
  ttb_indx N = (nnz+TeamSize-1)/TeamSize;
  Policy policy(N,TeamSize,VectorSize);

  Kokkos::parallel_for(policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
                       KOKKOS_LAMBDA(typename Policy::member_type team)
  {
    const size_type team_size = team.team_size();
    const size_type team_index = team.team_rank();
    const ttb_indx i = team.league_rank()*team_size+team_index;
    if (i >= nnz)
      return;

    // Store local product in scratch array of length nc
    TmpScratchSpace tmp(team.team_scratch(0), team_size, BlockSize);

    ttb_real x_val = X.value(i);
    ttb_indx k = X.subscript(i,n);

    for (ttb_indx j_block=0; j_block<nc; j_block+=BlockSize) {

      // Start tmp equal to the weights.
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,BlockSize),
                           [&] (const size_type& jj)
      {
        const ttb_indx j = j_block+jj;
        if (j<nc)
          tmp(team_index,jj) = x_val * u.weights(j);
      });

      for (size_type m=0; m<nd; ++m) {
        if (m != n) {
          // Update tmp array with elementwise product of row i
          // from the m-th factor matrix.  Length of the row is nc.
          const ttb_real *row = u[m].rowptr(X.subscript(i,m));
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,BlockSize),
                               [&] (const size_type& jj)
          {
            const ttb_indx j = j_block+jj;
            if (j<nc)
              tmp(team_index,jj) *= row[j];
          });
        }
      }

      // Update output by adding tmp array.
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,BlockSize),
                           [&] (const size_type& jj)
      {
        const ttb_indx j = j_block+jj;
        if (j<nc)
          Kokkos::atomic_add(&v.entry(k,j), tmp(team_index,jj));
      });
    }
  });

  return;
}

#endif

namespace Genten {
namespace Impl {
template <typename ExecSpace>
void mttkrp_perm(const Genten::SptensorT_perm<ExecSpace>& X,
                 const Genten::KtensorT<ExecSpace>& u,
                 ttb_indx n,
                 const Genten::FacMatrixT<ExecSpace>& v);

#if !USE_NEW_MTTKRP_PERM && defined(KOKKOS_HAVE_CUDA)
void mttkrp_perm(const Genten::SptensorT_perm<Kokkos::Cuda>& X,
                 const Genten::KtensorT<Kokkos::Cuda>& u,
                 ttb_indx n,
                 const Genten::FacMatrixT<Kokkos::Cuda>& v);
#endif
}
}

// Version of mttkrp using a permutation array to improve locality of writes,
// and reduce atomic throughput needs.  This version is geared towards CPUs
// and processes a large block of nonzeros before performing a segmented
// reduction across the corresponding rows.
template <typename ExecSpace>
void Genten::mttkrp(const Genten::SptensorT_perm<ExecSpace>& X,
                    const Genten::KtensorT<ExecSpace>& u,
                    const ttb_indx n,
                    const Genten::FacMatrixT<ExecSpace>& v)
{
  Impl::mttkrp_perm(X,u,n,v);
}

namespace Genten {
namespace Impl {

#if USE_NEW_MTTKRP_PERM

template <typename ExecSpace, unsigned RowBlockSize, unsigned FacBlockSize,
          unsigned TeamSize, unsigned VectorSize>
struct MTTKRP_PermKernelBlock {

  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;

  const Genten::SptensorT_perm<ExecSpace>& X;
  const Genten::KtensorT<ExecSpace>& u;
  const unsigned n;
  const unsigned nd;
  const ttb_indx nnz;
  const Genten::FacMatrixT<ExecSpace>& v;
  const ttb_indx i_block;

  static inline Policy policy(const ttb_indx nnz_) {
    const unsigned RowsPerTeam = TeamSize * RowBlockSize;
    const ttb_indx N = (nnz_+RowsPerTeam-1)/RowsPerTeam;
    Policy policy(N,TeamSize,VectorSize);
    return policy;
  }

  KOKKOS_INLINE_FUNCTION
  MTTKRP_PermKernelBlock(const Genten::SptensorT_perm<ExecSpace>& X_,
                         const Genten::KtensorT<ExecSpace>& u_,
                         const ttb_indx n_,
                         const Genten::FacMatrixT<ExecSpace>& v_,
                         const TeamMember& team) :
    X(X_), u(u_), n(n_), nd(u.ndims()), nnz(X.nnz()), v(v_),
    i_block(team.league_rank()*RowBlockSize*TeamSize + RowBlockSize*team.team_rank()) {}

  template <ttb_indx Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const unsigned j, const unsigned nj) {
    const ttb_indx invalid_row = ttb_indx(-1);

    ttb_indx row_prev = invalid_row;
    ttb_indx row = invalid_row;
    ttb_indx first_row = invalid_row;
    ttb_indx p = invalid_row;
    ttb_real x_val = 0.0;

    typedef Genten::TinyVec<ttb_real, unsigned, FacBlockSize, Nj, VectorSize> TV;
    TV val(nj, 0.0), tmp(nj, 0.0), row_vec(nj);

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
            const ttb_real *rowptr = u[m].rowptr(X.subscript(p,m));
            row_vec.load(rowptr+j);
            tmp *= row_vec;
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

#else

template <typename ExecSpace, unsigned RowBlockSize, unsigned FacBlockSize,
          unsigned TeamSize, unsigned VectorSize>
struct MTTKRP_PermKernelBlock {

  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;

  const Genten::SptensorT_perm<ExecSpace>& X;
  const Genten::KtensorT<ExecSpace>& u;
  const ttb_indx n;
  const ttb_indx nd;
  const ttb_indx nnz;
  const Genten::FacMatrixT<ExecSpace>& v;
  const ttb_indx i_block;

  // Align arrays to 64 byte boundary, using new C++11 syntax
  alignas(64) ttb_real val[FacBlockSize];
  alignas(64) ttb_real tmp[FacBlockSize];

  static inline Policy policy(const ttb_indx nnz_) {
    const unsigned RowsPerTeam = TeamSize * RowBlockSize;
    const ttb_indx N = (nnz_+RowsPerTeam-1)/RowsPerTeam;
    Policy policy(N,TeamSize,VectorSize);
    return policy;
  }

  KOKKOS_INLINE_FUNCTION
  MTTKRP_PermKernelBlock(const Genten::SptensorT_perm<ExecSpace>& X_,
                         const Genten::KtensorT<ExecSpace>& u_,
                         const ttb_indx n_,
                         const Genten::FacMatrixT<ExecSpace>& v_,
                         const TeamMember& team) :
    X(X_), u(u_), n(n_), nd(u.ndims()), nnz(X.nnz()), v(v_),
    i_block(team.league_rank()*RowBlockSize*TeamSize + RowBlockSize*team.team_rank()) {}

  template <ttb_indx Nj_>
  KOKKOS_INLINE_FUNCTION
  void run(const ttb_indx j_block, const ttb_indx nj_) {
    const ttb_indx invalid_row = ttb_indx(-1);

    ttb_indx row_prev = invalid_row;
    ttb_indx row = invalid_row;
    ttb_indx first_row = invalid_row;
    ttb_indx p = invalid_row;
    ttb_real x_val = 0.0;

    // nj.value == Nj_ if Nj_ > 0 and nj_ otherwise
    Kokkos::Impl::integral_nonzero_constant<ttb_indx, Nj_> nj(nj_);

#pragma ivdep
    for (ttb_indx jj=0; jj<nj.value; ++jj) {
      val[jj] = 0.0;
      tmp[jj] = 0.0;
    }

    for (ttb_indx ii=0; ii<RowBlockSize; ++ii) {
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
          if (row_prev == first_row) { // Only need atomics for first and last row in block
#pragma ivdep
            for (ttb_indx jj=0; jj<nj.value; ++jj)
            {
              const ttb_indx j = j_block+jj;
              Kokkos::atomic_add(&v.entry(row_prev,j), val[jj]);
              val[jj] = 0.0;
            }
          }
          else {
#pragma ivdep
            for (ttb_indx jj=0; jj<nj.value; ++jj)
            {
              const ttb_indx j = j_block+jj;
              v.entry(row_prev,j) += val[jj];
              val[jj] = 0.0;
            }
          }
        }
        row_prev = row;
      }

      if (row != invalid_row) {
        // Start tmp equal to the weights.
#pragma ivdep
        for (ttb_indx jj=0; jj<nj.value; ++jj)
        {
          const ttb_indx j = j_block+jj;
          tmp[jj] = x_val * u.weights(j);
        }

        for (ttb_indx m=0; m<nd; ++m) {
          if (m != n) {
            // Update tmp array with elementwise product of row i
            // from the m-th factor matrix.
            const ttb_real *rowptr = u[m].rowptr(X.subscript(p,m));
#pragma ivdep
            for (ttb_indx jj=0; jj<nj.value; ++jj)
            {
              const ttb_indx j = j_block+jj;
              tmp[jj] *= rowptr[j];
            }
          }
        }

#pragma ivdep
        for (ttb_indx jj=0; jj<nj.value; ++jj)
        {
          val[jj] += tmp[jj];
        }
      }
    }

    // Sum in last row
    if (row != invalid_row) {
#pragma ivdep
      for (ttb_indx jj=0; jj<nj.value; ++jj)
      {
        const ttb_indx j = j_block+jj;
        Kokkos::atomic_add(&v.entry(row,j), val[jj]);
      }
    }
  }
};

#endif

template <typename ExecSpace, ttb_indx FacBlockSize>
void mttkrp_perm_kernel(const Genten::SptensorT_perm<ExecSpace>& X,
                        const Genten::KtensorT<ExecSpace>& u,
                        const ttb_indx n,
                        const Genten::FacMatrixT<ExecSpace>& v)
{
  // Compute team and vector sizes, depending on the architecture
#if defined(KOKKOS_HAVE_CUDA)
  const bool is_cuda = std::is_same<ExecSpace,Kokkos::Cuda>::value;
#else
  const bool is_cuda = false;
#endif
  const unsigned VectorSize = is_cuda ? (FacBlockSize <= 32 ? FacBlockSize : 32) : 1;
  const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;
  const unsigned RowBlockSize = 128;

  typedef MTTKRP_PermKernelBlock<ExecSpace, RowBlockSize, FacBlockSize, TeamSize, VectorSize> Kernel;
  typedef typename Kernel::TeamMember TeamMember;

  Kokkos::parallel_for(Kernel::policy(X.nnz()),
                       KOKKOS_LAMBDA(TeamMember team)
  {
    MTTKRP_PermKernelBlock<ExecSpace, RowBlockSize, FacBlockSize, TeamSize, VectorSize> kernel(X, u, n, v, team);
    const unsigned nc = u.ncomponents();
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
void mttkrp_perm(const Genten::SptensorT_perm<ExecSpace>& X,
                 const Genten::KtensorT<ExecSpace>& u,
                 const ttb_indx n,
                 const Genten::FacMatrixT<ExecSpace>& v)
{
#if defined(KOKKOS_HAVE_CUDA)
  const bool is_cuda = std::is_same<ExecSpace,Kokkos::Cuda>::value;
#else
  const bool is_cuda = false;
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
  //v = FacMatrixT<ExecSpace>(X.size(n), nc);
  assert( v.nRows() == X.size(n) );
  assert( v.nCols() == nc );
  v = ttb_real(0.0);

  if (nc == 1)
    Impl::mttkrp_perm_kernel<ExecSpace,1>(X,u,n,v);
  else if (nc == 2)
    Impl::mttkrp_perm_kernel<ExecSpace,2>(X,u,n,v);
  else if (nc <= 4)
    Impl::mttkrp_perm_kernel<ExecSpace,4>(X,u,n,v);
  else if (nc <= 8)
    Impl::mttkrp_perm_kernel<ExecSpace,8>(X,u,n,v);
  else if (nc <= 16)
    Impl::mttkrp_perm_kernel<ExecSpace,16>(X,u,n,v);
  else if (nc < 64 || !is_cuda)
    Impl::mttkrp_perm_kernel<ExecSpace,32>(X,u,n,v);
  else if (nc < 128)
    Impl::mttkrp_perm_kernel<ExecSpace,64>(X,u,n,v);
  else
    Impl::mttkrp_perm_kernel<ExecSpace,128>(X,u,n,v);

  return;
}

#if !USE_NEW_MTTKRP_PERM && defined(KOKKOS_HAVE_CUDA)
// Version of mttkrp using a permutation array to improve locality of writes,
// and reduce atomic throughput needs.  This version is geared towards GPUs
// and performs a team-size segmented reduction while processing a large block
// of nonzeros.  This is a pure-Cuda implementation of the same kernel above,
// and it appears to therefore be somewhat faster.
void mttkrp_perm(const Genten::SptensorT_perm<Kokkos::Cuda>& X,
                 const Genten::KtensorT<Kokkos::Cuda>& u,
                 const ttb_indx n,
                 const Genten::FacMatrixT<Kokkos::Cuda>& v)
{
  typedef Kokkos::Cuda ExecSpace;

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
  //v = FacMatrixT<ExecSpace>(X.size(n), nc);
  assert( v.nRows() == X.size(n) );
  assert( v.nCols() == nc );
  v = ttb_real(0.0);

  const ttb_indx nnz = X.nnz();
  const ttb_indx RowBlockSize = 128;
  const int FacBlockSize = std::min(128, 2 << int(std::log2(nc)));
  const ttb_indx TeamSize = 128 / FacBlockSize;
  const ttb_indx RowsPerTeam = TeamSize * RowBlockSize;
  const ttb_indx N = (nnz+RowsPerTeam-1)/RowsPerTeam;

  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  Policy policy(N,TeamSize,FacBlockSize);

  Kokkos::parallel_for(policy,
                       [=]__device__(typename Policy::member_type team)
  {
    const ttb_indx i_block = blockIdx.x*RowsPerTeam + RowBlockSize*threadIdx.y;
    const ttb_indx invalid_row = ttb_indx(-1);

    for (ttb_indx j=threadIdx.x; j<nc; j+=FacBlockSize) {
      ttb_indx row_prev = invalid_row;
      ttb_indx row = invalid_row;
      ttb_indx first_row = invalid_row;
      ttb_indx p = invalid_row;
      ttb_real x_val = 0.0;

      ttb_real val = 0.0;
      ttb_real tmp = 0.0;

      for (int ii=0; ii<RowBlockSize; ++ii) {
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
            if (row_prev == first_row) { // Only need atomics for first and last row in block
              Kokkos::atomic_add(&v.entry(row_prev,j), val);
              val = 0.0;
            }
            else {
              v.entry(row_prev,j) += val;
              val = 0.0;
            }
          }
          row_prev = row;
        }

        if (row != invalid_row) {
          // Start tmp equal to the weights.
          tmp = x_val * u.weights(j);

          for (int m=0; m<nd; ++m) {
            if (m != n) {
              // Update tmp array with elementwise product of row i
              // from the m-th factor matrix.
              const ttb_real *rowptr = u[m].rowptr(X.subscript(p,m));
              tmp *= rowptr[j];
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

  });

  return;
}
#endif

}
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
#if defined(KOKKOS_HAVE_CUDA)
  const bool is_cuda = std::is_same<ExecSpace,Kokkos::Cuda>::value;
#else
  const bool is_cuda = false;
#endif
  // Use the largest power of 2 <= nc, with a maximum of 64 for the vector size.
  const size_type VectorSize =
    nc == 1 ? 1 : std::min(64,2 << int(std::log2(nc))-1);
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

    const ttb_indx i_begin = X.getPermRowBegin(row,n);
    const ttb_indx i_end = X.getPermRowBegin(row+1,n);
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

  });

  return;
}

#define INST_MACRO(SPACE)                                               \
  template                                                              \
  ttb_real innerprod<SPACE>(const Genten::SptensorT<SPACE>& s,          \
                            const Genten::KtensorT<SPACE>& u,           \
                            const Genten::ArrayT<SPACE>& lambda);       \
  template                                                              \
  void mttkrp<SPACE>(const Genten::SptensorT<SPACE>& X,                 \
                     const Genten::KtensorT<SPACE>& u,                  \
                     const ttb_indx n,                                  \
                     const Genten::FacMatrixT<SPACE>& v);               \
  template                                                              \
  void mttkrp<SPACE>(const Genten::SptensorT_perm<SPACE>& X,            \
                     const Genten::KtensorT<SPACE>& u,                  \
                     const ttb_indx n,                                  \
                     const Genten::FacMatrixT<SPACE>& v);               \
  template                                                              \
  void mttkrp<SPACE>(const Genten::SptensorT_row<SPACE>& X,             \
                     const Genten::KtensorT<SPACE>& u,                  \
                     const ttb_indx n,                                  \
                     const Genten::FacMatrixT<SPACE>& v);
GENTEN_INST(INST_MACRO)
