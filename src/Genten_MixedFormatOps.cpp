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

#include "Genten_Util.h"
#include "Genten_FacMatrix.h"
#include "Genten_Ktensor.h"
#include "Genten_MixedFormatOps.h"
#include "Genten_Sptensor.h"
#include "Genten_TinyVec.h"

#define USE_NEW_MTTKRP 1
#define USE_NEW_MTTKRP_PERM 1

ttb_real Genten::innerprod(const Genten::Sptensor & s,
                           const Genten::Ktensor  & u)
{
  ttb_indx nc = u.ncomponents();              // Number of components
  ttb_indx nd = u.ndims();                    // Number of dimensions

  // Check on sizes
  assert(nd == s.ndims());
  assert(u.isConsistent(s.size()));

  // Do the inner product
  ttb_real dTotal = 0.0;
  IndxArray sub(nd);
  for (ttb_indx i = 0; i < s.nnz(); i ++)
  {
    s.getSubscripts (i,sub);
    dTotal += s.value(i) * u.entry (sub);
  }

  return( dTotal );
}


//-----------------------------------------------------------------------------
//  Method:  innerprod, Sptensor and Ktensor with alternate weights
//-----------------------------------------------------------------------------
ttb_real Genten::innerprod(const Genten::Sptensor & s,
                                 Genten::Ktensor  & u,
                           const Genten::Array    & lambda)
{
  typedef Kokkos::DefaultExecutionSpace ExecSpace;
  typedef typename ExecSpace::size_type size_type;

  const ttb_indx nc = u.ncomponents();               // Number of components
  const size_type nd = u.ndims();                     // Number of dimensions

  // Check on sizes
  assert(nd == s.ndims());
  assert(u.isConsistent(s.size()));
  assert(nc == lambda.size());

  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  typedef Kokkos::View< ttb_real**, Kokkos::LayoutRight, ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

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
                          KOKKOS_LAMBDA(Policy::member_type team, ttb_real& d)
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

//-----------------------------------------------------------------------------
//  Method:  mttkrp, Sptensor X, output to Ktensor mode n
//-----------------------------------------------------------------------------
void Genten::mttkrp(const Genten::Sptensor& X,
                    Genten::Ktensor& u,
                    ttb_indx n)
{
  mttkrp (X, u, n, u[n]);
  return;
}
void Genten::mttkrp(const Genten::Sptensor_perm& X,
                    Genten::Ktensor& u,
                    ttb_indx n)
{
  mttkrp (X, u, n, u[n]);
  return;
}
void Genten::mttkrp(const Genten::Sptensor_row& X,
                    Genten::Ktensor& u,
                    ttb_indx n)
{
  mttkrp (X, u, n, u[n]);
  return;
}


//-----------------------------------------------------------------------------
//  Method:  mttkrp, Sptensor X, output to FacMatrix
//-----------------------------------------------------------------------------

#if USE_NEW_MTTKRP

namespace Genten {

template <unsigned FacBlockSize, unsigned VectorSize>
struct MTTKRP_KernelBlock {

  const Genten::Sptensor& X;
  const Genten::Ktensor& u;
  const unsigned n;
  const unsigned nd;
  const ttb_indx i;
  const Genten::FacMatrix& v;

  const ttb_indx k;
  const ttb_real x_val;
  const ttb_real* lambda;

  KOKKOS_INLINE_FUNCTION
  MTTKRP_KernelBlock(const Genten::Sptensor& X_,
                     const Genten::Ktensor& u_,
                     const unsigned n_,
                     const unsigned nd_,
                     const ttb_indx i_,
                     const Genten::FacMatrix& v_) :
    X(X_), u(u_), n(n_), nd(nd_), i(i_), v(v_),
    k(X.subscript(i,n)), x_val(X.value(i)),
    lambda(&u.weights(0))
    {}

  template <unsigned Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const unsigned j, const unsigned nj)
  {
    typedef Genten::TinyVec<ttb_real, unsigned, FacBlockSize, Nj, VectorSize> TV;

    TV tmp(nj), row_vec(nj);

    // Start tmp equal to the weights.
    tmp.load(lambda+j);
    tmp *= x_val;

    for (unsigned m=0; m<nd; ++m) {
      if (m != n) {
        // Update tmp array with elementwise product of row i
        // from the m-th factor matrix.  Length of the row is nc.
        const ttb_real *row = u[m].rowptr(X.subscript(i,m));
        row_vec.load(row+j);
        tmp *= row_vec;
      }
    }

    // Update output by adding tmp array.
    Kokkos::atomic_add(&v.entry(k,j), tmp);
  }
};

template <unsigned FacBlockSize>
void mttkrp_kernel(const Genten::Sptensor& X,
                   const Genten::Ktensor& u,
                   const ttb_indx n,
                   Genten::FacMatrix& v)
{
  typedef Kokkos::DefaultExecutionSpace ExecSpace;
  typedef Kokkos::TeamPolicy <ExecSpace> Policy;

  // Compute team and vector sizes, depending on the architecture
#if defined(KOKKOS_HAVE_CUDA)
  const bool is_cuda = std::is_same<ExecSpace,Kokkos::Cuda>::value;
#else
  const bool is_cuda = false;
#endif
#if defined(KOKKOS_HAVE_OPENMP)
  const bool is_openmp = std::is_same<ExecSpace,Kokkos::OpenMP>::value;
  const unsigned thread_pool_size =
    is_openmp ? Kokkos::OpenMP::thread_pool_size(2) : 1;
#else
  const unsigned thread_pool_size = 1;
#endif

  const unsigned VectorSize = is_cuda ? (FacBlockSize <= 8 ? FacBlockSize : 8) : 1;
  const unsigned TeamSize = is_cuda ? 256/VectorSize : thread_pool_size;

  const unsigned nc = u.ncomponents();
  const unsigned nd = u.ndims();
  const ttb_indx nnz = X.nnz();
  const ttb_indx N = (nnz+TeamSize-1)/TeamSize;

  Kokkos::parallel_for(Policy(N,TeamSize,VectorSize),
                       KOKKOS_LAMBDA(typename Policy::member_type team)
  {
    const ttb_indx i = team.league_rank()*team.team_size()+team.team_rank();
    if (i >= nnz)
      return;

    MTTKRP_KernelBlock<FacBlockSize, VectorSize> kernel(X, u, n, nd, i, v);

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

void Genten::mttkrp(const Genten::Sptensor& X,
                    const Genten::Ktensor& u,
                    ttb_indx n,
                    Genten::FacMatrix& v)
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
  v = FacMatrix(X.size(n), nc);

  // Call kernel with factor block size determined from nc
  if (nc == 1)
    mttkrp_kernel<1>(X,u,n,v);
  else if (nc <= 2)
    mttkrp_kernel<2>(X,u,n,v);
  else if (nc <= 4)
    mttkrp_kernel<4>(X,u,n,v);
  else if (nc <= 8)
    mttkrp_kernel<8>(X,u,n,v);
  else if (nc <= 16)
    mttkrp_kernel<16>(X,u,n,v);
  else
    mttkrp_kernel<32>(X,u,n,v);

  return;
}

#else

void Genten::mttkrp(const Genten::Sptensor  & X,
                 const Genten::Ktensor   & u,
                       ttb_indx         n,
                       Genten::FacMatrix & v)
{
  std::cout << "calling old mttkrp..." << std::endl;

  typedef Kokkos::DefaultExecutionSpace ExecSpace;
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
  v = FacMatrix(X.size(n), nc);

  // Loop thru the nonzeros of the sparse tensor.  The inner loop updates
  // an entire row at a time, and is run only for nonzero elements.
  // Use team-based parallel-for.  Team is required for scratch memory and
  // will be useful for GPU.
  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  typedef Kokkos::View< ttb_real**, Kokkos::LayoutRight, ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

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
                       KOKKOS_LAMBDA(Policy::member_type team)
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

#if !USE_NEW_MTTKRP_PERM && defined(KOKKOS_HAVE_CUDA)
namespace Genten {
void mttkrp_perm_cuda(const Genten::Sptensor_perm& X,
                      const Genten::Ktensor& u,
                      ttb_indx n,
                      Genten::FacMatrix& v);
}
#endif
namespace Genten {
void mttkrp_perm_general(const Genten::Sptensor_perm& X,
                         const Genten::Ktensor& u,
                         ttb_indx n,
                         Genten::FacMatrix& v);
}

// Version of mttkrp using a permutation array to improve locality of writes,
// and reduce atomic throughput needs.  This version is geared towards CPUs
// and processes a large block of nonzeros before performing a segmented
// reduction across the corresponding rows.
void Genten::mttkrp(const Genten::Sptensor_perm& X,
                    const Genten::Ktensor& u,
                    ttb_indx n,
                    Genten::FacMatrix& v)
{
#if !USE_NEW_MTTKRP_PERM && defined(KOKKOS_HAVE_CUDA)
  typedef Kokkos::DefaultExecutionSpace ExecSpace;
  if (std::is_same<ExecSpace,Kokkos::Cuda>::value)
    mttkrp_perm_cuda(X,u,n,v);
  else
#endif
    mttkrp_perm_general(X,u,n,v);
}

namespace Genten {


#if USE_NEW_MTTKRP_PERM

template <ttb_indx RowBlockSize, ttb_indx FacBlockSize, ttb_indx VectorSize>
struct MTTKRP_PermKernelBlock {

  const Genten::Sptensor_perm& X;
  const Genten::Ktensor& u;
  const ttb_indx n;
  const ttb_indx nd;
  const ttb_indx nnz;
  const Genten::FacMatrix& v;

  KOKKOS_INLINE_FUNCTION
  MTTKRP_PermKernelBlock(const Genten::Sptensor_perm& X_,
                         const Genten::Ktensor& u_,
                         const ttb_indx n_,
                         const ttb_indx nd_,
                         const ttb_indx nnz_,
                         const Genten::FacMatrix& v_) :
    X(X_), u(u_), n(n_), nd(nd_), nnz(nnz_), v(v_) {}

  template <ttb_indx Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const ttb_indx i_block, const ttb_indx j, const ttb_indx nj) {
    const ttb_indx invalid_row = ttb_indx(-1);

    ttb_indx row_prev = invalid_row;
    ttb_indx row = invalid_row;
    ttb_indx first_row = invalid_row;
    ttb_indx p = invalid_row;
    ttb_real x_val = 0.0;

    typedef Genten::TinyVec<ttb_real, unsigned, FacBlockSize, Nj, VectorSize> TV;
    TV val(nj, 0.0), tmp(nj, 0.0), row_vec(nj);

    const ttb_real* lambda = &u.weights(0);

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

        for (ttb_indx m=0; m<nd; ++m) {
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

template <ttb_indx RowBlockSize, ttb_indx FacBlockSize, ttb_indx VectorSize>
struct MTTKRP_PermKernelBlock {

  const Genten::Sptensor_perm& X;
  const Genten::Ktensor& u;
  const ttb_indx n;
  const ttb_indx nd;
  const ttb_indx nnz;
  const Genten::FacMatrix& v;

  // ttb_real val[FacBlockSize]  __attribute__((aligned(64)));
  // ttb_real tmp[FacBlockSize]  __attribute__((aligned(64)));

  // Align arrays to 64 byte boundary, using new C++11 syntax
  alignas(64) ttb_real val[FacBlockSize];
  alignas(64) ttb_real tmp[FacBlockSize];

  KOKKOS_INLINE_FUNCTION
  MTTKRP_PermKernelBlock(const Genten::Sptensor_perm& X_,
                         const Genten::Ktensor& u_,
                         const ttb_indx n_,
                         const ttb_indx nd_,
                         const ttb_indx nnz_,
                         const Genten::FacMatrix& v_) :
    X(X_), u(u_), n(n_), nd(nd_), nnz(nnz_), v(v_) {}

  template <ttb_indx Nj_>
  KOKKOS_INLINE_FUNCTION
  void run(const ttb_indx i_block, const ttb_indx j_block, const ttb_indx nj_) {
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

template <ttb_indx FacBlockSize>
void mttkrp_perm_general_kernel(const Genten::Sptensor_perm& X,
                                const Genten::Ktensor& u,
                                const ttb_indx n,
                                Genten::FacMatrix& v)
{
  typedef Kokkos::DefaultExecutionSpace ExecSpace;
  typedef Kokkos::TeamPolicy <ExecSpace> Policy;

  // Compute team and vector sizes, depending on the architecture
#if defined(KOKKOS_HAVE_CUDA)
  const bool is_cuda = std::is_same<ExecSpace,Kokkos::Cuda>::value;
#else
  const bool is_cuda = false;
#endif
#if defined(KOKKOS_HAVE_OPENMP)
  const bool is_openmp = std::is_same<ExecSpace,Kokkos::OpenMP>::value;
  const unsigned thread_pool_size =
    is_openmp ? Kokkos::OpenMP::thread_pool_size(2) : 1;
#else
  const unsigned thread_pool_size = 1;
#endif

  const unsigned VectorSize = is_cuda ? (FacBlockSize <= 8 ? FacBlockSize : 8) : 1;
  //const unsigned VectorSize = is_cuda ? FacBlockSize : 1;
  const unsigned TeamSize = is_cuda ? 256/VectorSize : thread_pool_size;

  const ttb_indx nc = u.ncomponents();
  const ttb_indx nd = u.ndims();
  const ttb_indx nnz = X.nnz();
  const ttb_indx RowBlockSize = 128;
  const ttb_indx RowsPerTeam = TeamSize * RowBlockSize;
  const ttb_indx N = (nnz+RowsPerTeam-1)/RowsPerTeam;

  Kokkos::parallel_for(Policy(N,TeamSize,VectorSize),
                       KOKKOS_LAMBDA(typename Policy::member_type team)
  {
    const ttb_indx i =
      team.league_rank()*RowsPerTeam + RowBlockSize*team.team_rank();

    MTTKRP_PermKernelBlock<RowBlockSize, FacBlockSize, VectorSize> kernel(
      X, u, n, nd, nnz, v);

    for (ttb_indx j=0; j<nc; j+=FacBlockSize) {
      if (j+FacBlockSize <= nc)
        kernel.template run<FacBlockSize>(i, j, FacBlockSize);
      else
        kernel.template run<0>(i, j, nc-j);
    }

  });

  return;
}

}

void Genten::mttkrp_perm_general(const Genten::Sptensor_perm& X,
                                 const Genten::Ktensor& u,
                                 ttb_indx n,
                                 Genten::FacMatrix& v)
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
  v = FacMatrix(X.size(n), nc);


  if (sizeof(ttb_real) == 4) {
    // For float
    if (nc <= 4)
      mttkrp_perm_general_kernel<4>(X,u,n,v);
    else if (nc <= 8)
      mttkrp_perm_general_kernel<8>(X,u,n,v);
    else if (nc <= 16)
      mttkrp_perm_general_kernel<16>(X,u,n,v);
    else
      mttkrp_perm_general_kernel<32>(X,u,n,v);
  }
  else {
    // For double or anything else, using 16 for 16 <= nc <= 64 seems to be
    // signficantly faster than 32
    if (nc <= 4)
      mttkrp_perm_general_kernel<4>(X,u,n,v);
    else if (nc <= 8)
      mttkrp_perm_general_kernel<8>(X,u,n,v);
    else if (nc <= 64)
      mttkrp_perm_general_kernel<16>(X,u,n,v);
    else
      mttkrp_perm_general_kernel<32>(X,u,n,v);
  }

  // mttkrp_perm_general_kernel<16>(X,u,n,v);

  return;
}

#if !USE_NEW_MTTKRP_PERM && defined(KOKKOS_HAVE_CUDA)
// Version of mttkrp using a permutation array to improve locality of writes,
// and reduce atomic throughput needs.  This version is geared towards GPUs
// and performs a team-size segmented reduction while processing a large block
// of nonzeros.  This is a pure-Cuda implementation of the same kernel above,
// and it appears to therefore be somewhat faster.

void Genten::mttkrp_perm_cuda(const Genten::Sptensor_perm& X,
                              const Genten::Ktensor& u,
                              const ttb_indx n,
                              Genten::FacMatrix& v)
{
  typedef Kokkos::DefaultExecutionSpace ExecSpace;

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
  v = FacMatrix(X.size(n), nc);

  const ttb_indx nnz = X.nnz();
  const ttb_indx RowBlockSize = 128;
  const int FacBlockSize = std::min(128, 2 << int(std::log2(nc)));
  const ttb_indx TeamSize = 128 / FacBlockSize;
  const ttb_indx RowsPerTeam = TeamSize * RowBlockSize;
  const ttb_indx N = (nnz+RowsPerTeam-1)/RowsPerTeam;

  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  Policy policy(N,TeamSize,FacBlockSize);

  Kokkos::parallel_for(policy,
                       [=]__device__(Policy::member_type team)
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

// Version of mttkrp using a permutation array to improve locality of writes,
// and reduce atomic throughput needs.  This version is uses a rowptr array
// and a parallel_for over rows.
void Genten::mttkrp(const Genten::Sptensor_row   & X,
                 const Genten::Ktensor & u,
                 ttb_indx                    n,
                 Genten::FacMatrix &     v)
{
  typedef Kokkos::DefaultExecutionSpace ExecSpace;
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
  v = FacMatrix(X.size(n), nc);

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

  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(Policy::member_type team)
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
