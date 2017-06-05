//@HEADER
// ************************************************************************
//     Genten Tensor Toolbox
//     Software package for tensor math by Sandia National Laboratories
//
// Sandia National Laboratories is a multimission laboratory managed
// and operated by National Technology and Engineering Solutions of Sandia,
// LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
// U.S. Department of Energy’s National Nuclear Security Administration under
// contract DE-NA0003525.
//
// Copyright 20XX, Sandia Corporation.
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

#include "Genten_FacMatrix.h"
#include "Genten_Ktensor.h"
#include "Genten_MixedFormatOps.h"
#include "Genten_Sptensor.h"
#include "Genten_Util.h"

#include "Kokkos_Core.hpp"

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
void Genten::mttkrp(const Genten::Sptensor & X,
                       Genten::Ktensor  & u,
                       ttb_indx               n)
{
  mttkrp (X, u, n, u[n]);
  return;
}
void Genten::mttkrp(const Genten::Sptensor_perm  & X,
                       Genten::Ktensor & u,
                       ttb_indx              n)
{
  mttkrp (X, u, n, u[n]);
  return;
}
void Genten::mttkrp(const Genten::Sptensor_row   & X,
                       Genten::Ktensor & u,
                       ttb_indx              n)
{
  mttkrp (X, u, n, u[n]);
  return;
}


//-----------------------------------------------------------------------------
//  Method:  mttkrp, Sptensor X, output to FacMatrix
//-----------------------------------------------------------------------------
void Genten::mttkrp(const Genten::Sptensor  & X,
                 const Genten::Ktensor   & u,
                       ttb_indx         n,
                       Genten::FacMatrix & v)
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

#if defined(KOKKOS_HAVE_CUDA)
namespace Genten {
void mttkrp_cuda(const Genten::Sptensor_perm    & X,
                 const Genten::Ktensor   & u,
                       ttb_indx                n,
                       Genten::FacMatrix & v);
}
#endif
namespace Genten {
void mttkrp_general(const Genten::Sptensor_perm  & X,
                    const Genten::Ktensor & u,
                    ttb_indx                    n,
                    Genten::FacMatrix     & v);
}

// Version of mttkrp using a permutation array to improve locality of writes,
// and reduce atomic throughput needs.  This version is geared towards CPUs
// and processes a large block of nonzeros before performing a segmented
// reduction across the corresponding rows.
void Genten::mttkrp(const Genten::Sptensor_perm  & X,
                 const Genten::Ktensor & u,
                 ttb_indx                    n,
                 Genten::FacMatrix     & v)
{
#if defined(KOKKOS_HAVE_CUDA)
  typedef Kokkos::DefaultExecutionSpace ExecSpace;
  if (std::is_same<ExecSpace,Kokkos::Cuda>::value)
    mttkrp_cuda(X,u,n,v);
  else
#endif
    mttkrp_general(X,u,n,v);
}

void Genten::mttkrp_general(const Genten::Sptensor_perm  & X,
                            const Genten::Ktensor & u,
                            ttb_indx                    n,
                            Genten::FacMatrix     & v)
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
  typedef Kokkos::View< ttb_real**, Kokkos::LayoutRight, ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > ValScratchSpace;
  typedef Kokkos::View< ttb_indx*, Kokkos::LayoutRight, ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > RowScratchSpace;

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

  // For a FacBlockSize=16 and RowBlockSize=16, the kernel
  // needs 8*16*16 = 2K shared memory per CUDA block.  Most modern GPUs have
  // between 48K and 64K shared memory per SM, allowing between 24 and 32
  // blocks per SM, which is typically enough for 100% occupancy.
  const size_type FacBlockSize = std::min(size_type(16),size_type(nc));
  const size_type RowBlockSize =
    std::max(is_cuda ? size_type(16) : size_type(128),TeamSize);

  // compute how much scratch memory (in bytes) is needed
  const size_t bytes =
    ValScratchSpace::shmem_size(RowBlockSize,FacBlockSize) +
    RowScratchSpace::shmem_size(RowBlockSize);

  ttb_indx nnz = X.nnz();
  ttb_indx N = (nnz+RowBlockSize-1)/RowBlockSize;
  Policy policy(N,TeamSize,VectorSize);

  const ttb_indx invalid_row = ttb_indx(-1);

  Kokkos::parallel_for(policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
                       KOKKOS_LAMBDA(Policy::member_type team)
  {
    const size_type team_size = team.team_size();
    const size_type team_index = team.team_rank();
    const ttb_indx i_block = team.league_rank()*RowBlockSize;

    // Store local product in scratch array of length nc
    ValScratchSpace val(team.team_scratch(0), RowBlockSize, FacBlockSize);
    RowScratchSpace row(team.team_scratch(0), RowBlockSize);

    for (ttb_indx j_block=0; j_block<nc; j_block+=FacBlockSize) {

      team.team_barrier();

      for (ttb_indx ii=team_index; ii<RowBlockSize; ii+=team_size) {
        const ttb_indx i = i_block+ii;

        if (i >= nnz) {
          row(ii) = invalid_row;
        }
        else {
          const ttb_indx p = X.getPerm(i,n);
          const ttb_real x_val = X.value(p);
          const ttb_indx k = X.subscript(p,n);

          row(ii) = k;

          // Start val equal to the weights.
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,FacBlockSize),
                               [&] (const size_type& jj)
          {
            const ttb_indx j = j_block+jj;
            if (j<nc) {
              val(ii,jj) = x_val * u.weights(j);
            }
          });

          for (size_type m=0; m<nd; ++m) {
            if (m != n) {
              // Update tmp array with elementwise product of row i
              // from the m-th factor matrix.  Length of the row is nc.
              const ttb_real *rowptr = u[m].rowptr(X.subscript(p,m));
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,FacBlockSize),
                                   [&] (const size_type& jj)
              {
                const ttb_indx j = j_block+jj;
                if (j<nc) {
                  val(ii,jj) *= rowptr[j];
                }
              });
            }
          }
        }

      }

      team.team_barrier();

      // Perform segmented reduction of val for the same row indices.
      // Reduction is done is serial by thread 0 in the team.  Parallel version
      // is slower.
      if (team_index == 0) {
        ttb_indx ii = 0;
        while (ii < RowBlockSize) {
          ttb_indx kk=ii+1;
          while ( (kk < RowBlockSize) && (row[kk] == row[kk-1]) ) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,FacBlockSize),
                                 [&] (const size_type& jj)
            {
              val(kk,jj) += val(kk-1,jj); // Don't need to check j_block+jj < nc
            });
            ++kk;
          }
          ii = kk;
        }
      }

      team.team_barrier();

      // Update output by adding val array.
      for (ttb_indx ii=team_index; ii<RowBlockSize; ii+=team_size) {
        if ( (row[ii] != invalid_row) &&
             ((ii == RowBlockSize-1) || (row[ii] != row[ii+1])) ) {
           Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,FacBlockSize),
                               [&] (const size_type& jj)
          {
            const ttb_indx j = j_block+jj;
            if (j<nc)
              Kokkos::atomic_add(&v.entry(row[ii],j), val(ii,jj));
          });
        }
      }

    }

  });

  return;
}

#if defined(KOKKOS_HAVE_CUDA)
// Version of mttkrp using a permutation array to improve locality of writes,
// and reduce atomic throughput needs.  This version is geared towards GPUs
// and performs a team-size segmented reduction while processing a large block
// of nonzeros.  This is a pure-Cuda implementation of the same kernel above,
// and it appears to therefore be somewhat faster.
void Genten::mttkrp_cuda(const Genten::Sptensor_perm  & X,
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
  typedef Kokkos::View< ttb_real**, Kokkos::LayoutRight, ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > ValScratchSpace;
  typedef Kokkos::View< ttb_indx*, Kokkos::LayoutRight, ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > RowScratchSpace;

  // Use the largest power of 2 <= nc, with a maximum of 16 for the vector size.
  const size_type VectorSize =
    nc == 1 ? 1 : std::min(16,2 << int(std::log2(nc))-1);
  const size_type TeamSize = 128/VectorSize;

  // For a maximum TeamSize=16 and VectorSize=16, the kernel
  // needs 8*16*16 = 2K shared memory per CUDA block.  Most modern GPUs have
  // between 48K and 64K shared memory per SM, allowing between 24 and 32
  // blocks per SM, which is typically enough for 100% occupancy.

  // compute how much scratch memory (in bytes) is needed
  const size_t bytes =
    ValScratchSpace::shmem_size(TeamSize,VectorSize) +
    RowScratchSpace::shmem_size(TeamSize);

  const size_type RowBlockSize = std::max(size_type(128),TeamSize);
  ttb_indx nnz = X.nnz();
  ttb_indx N = (nnz+RowBlockSize-1)/RowBlockSize;
  Policy policy(N,TeamSize,VectorSize);

  const ttb_indx invalid_row = ttb_indx(-1);

  Kokkos::parallel_for(policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
                       [=]__device__(Policy::member_type team)
  {
    const ttb_indx i_block = blockIdx.x*RowBlockSize;

    // Store local product in scratch array of length nc
    ValScratchSpace val(team.team_scratch(0), blockDim.y, blockDim.x);
    RowScratchSpace row(team.team_scratch(0), blockDim.y);

    for (ttb_indx ii=threadIdx.y; ii<RowBlockSize; ii+=blockDim.y) {
      const ttb_indx i = i_block+ii;

      if (i >= nnz) {
        row(threadIdx.y) = invalid_row;
      }
      else {
        const ttb_indx p = X.getPerm(i,n);
        const ttb_real x_val = X.value(p);
        const ttb_indx k = X.subscript(p,n);

        row(threadIdx.y) = k;

        for (ttb_indx j_block=0; j_block<nc; j_block+=blockDim.x) {
          const ttb_indx j = j_block+threadIdx.x;
          if (j >= nc)
            break;

          // Finish the previous block before starting the next
          team.team_barrier();

          // Start val equal to the weights.
          val(threadIdx.y,threadIdx.x) = x_val * u.weights(j);

          for (size_type m=0; m<nd; ++m) {
            if (m != n) {
              // Update tmp array with elementwise product of row i
              // from the m-th factor matrix.  Length of the row is nc.
              const ttb_real *rowptr = u[m].rowptr(X.subscript(p,m));
              val(threadIdx.y,threadIdx.x) *= rowptr[j];
            }
          }

          // Finish writing to row before starting reduction
          team.team_barrier();

          // Perform segmented reduction of val for the same row indices
#if 1
          // Serial version seems faster
          if (threadIdx.y == 0) {
            ttb_indx ii = 0;
            while (ii < blockDim.y) {
              ttb_indx kk=ii+1;
              while ( (kk < blockDim.y) && (row(kk) == row(kk-1)) ) {
                val(kk,threadIdx.x) += val(kk-1,threadIdx.x);
                ++kk;
              }
              ii = kk;
            }
          }
#else
          // Parallel version relying on blockDim.x >= blockDim.y
          if (threadIdx.x < blockDim.y) {
            for (ttb_indx col=threadIdx.y; col<blockDim.x; col+=blockDim.y) {
              for (ttb_indx w=1; w<blockDim.y; w*=2) {
                if (threadIdx.x >= w && row[threadIdx.x] == row[threadIdx.x-w])
                  val(threadIdx.x,col) += val(threadIdx.x-w,col);
              }
            }
          }
#endif

          // Finish reduction before updating results
          team.team_barrier();

          // Update output by adding val array.
          if ((row(threadIdx.y) != invalid_row) &&
              ((threadIdx.y == blockDim.y-1) ||
               (row(threadIdx.y) != row(threadIdx.y+1)))) {
            if (threadIdx.y == blockDim.y-1 || row(threadIdx.y) == row(0))
              Kokkos::atomic_add(&v.entry(row(threadIdx.y),j),
                                 val(threadIdx.y,threadIdx.x));
            else
              v.entry(row(threadIdx.y),j) += val(threadIdx.y,threadIdx.x);
          }

        }

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
