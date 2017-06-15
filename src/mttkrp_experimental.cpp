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
// ************************************************************************
//@HEADER

/*!
 * This file contains various experimental implementations of mttkrp with
 * Kokkos.  It should not be compiled and linked in directly.  This file
 * is meant for archival purposes to save several failed implementations
 * that didn't improve performance.
 */

#if defined(KOKKOS_HAVE_CUDA)
namespace Genten {
void mttkrp_cuda(const Genten::Sptensor  & X,
                 const Genten::Ktensor   & u,
                       ttb_indx                n,
                       Genten::FacMatrix & v);
}
#endif

#if defined(KOKKOS_HAVE_CUDA)
// Pure cuda implementation of the basic mttrkp kernel.  Useful for comparison.
void Genten::mttkrp_cuda(const Genten::Sptensor  & X,
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

  // Use the largest power of 2 <= nc, with a maximum of 8 for the vector size.
  const size_type VectorSize =
    nc == 1 ? 1 : std::min(8,2 << int(std::log2(nc))-1);
  const size_type TeamSize = 128/VectorSize;

  ttb_indx nnz = X.nnz();
  ttb_indx N = (nnz+TeamSize-1)/TeamSize;
  Policy policy(N,TeamSize,VectorSize);

  Kokkos::parallel_for(policy,
                       [=]__device__(Policy::member_type team)
  {
    const ttb_indx i = blockIdx.x*blockDim.y+threadIdx.y;
    if (i >= nnz)
      return;

    ttb_real x_val = X.value(i);
    ttb_indx k = X.subscript(i,n);

    for (ttb_indx j=threadIdx.x; j<nc; j+=blockDim.x) {

      ttb_real tmp = x_val * u.weights(j);

      for (size_type m=0; m<nd; ++m) {
        if (m != n) {
          // Update tmp array with elementwise product of row i
          // from the m-th factor matrix.  Length of the row is nc.
          const ttb_real *row = u[m].rowptr(X.subscript(i,m));
          tmp *= row[j];
        }
      }

      // Update output by adding tmp array.
      Kokkos::atomic_add(&v.entry(k,j), tmp);
    }
  });

  return;
}
#endif

#if 0

template <typename ExecSpace>
struct AtomicAddRow {
  template <typename TeamMember>
  KOKKOS_INLINE_FUNCTION
  static void eval(const TeamMember& team,
                   volatile ttb_real * const dest,
                   const ttb_real * const val,
                   const ttb_indx nc)
  {
    Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(team,nc),
      [&] (const ttb_indx& j) {
        Kokkos::atomic_add(dest+j, val[j]);
      });
  }
};

#if defined(KOKKOS_HAVE_CUDA)
// This code attempts to implement an atomic update for a whole row
// of the output matrix, by having thread 0 lock and unlock the starting
// address of that row.  However it appears to dead-lock for reasons I don't
// understand.
template <>
struct AtomicAddRow<Kokkos::Cuda> {
  template <typename TeamMember>
  inline __device__
  static void eval(const TeamMember& team,
                   volatile ttb_real * const dest,
                   const ttb_real * const val,
                   const ttb_indx nc)
  {
    int done = 1;
    while ( done>0 ) {
      done++;
      int locked = 0;
      if (threadIdx.x == 0)
        locked = Kokkos::Impl::lock_address_cuda_space( (void*) dest );
      __shfl(locked, threadIdx.y*blockDim.x, blockDim.x);
      if( locked  ) {
        for (ttb_indx j=threadIdx.x; j<nc; j+=blockDim.x) {
          ttb_real return_val = dest[j];
          dest[j] = return_val + val[j];
        };
        if (threadIdx.x == 0)
          Kokkos::Impl::unlock_address_cuda_space( (void*) dest );
        done = 0;
      }
    }
  }
};
#endif

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
  typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > SubScratchSpace;

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
  const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,BlockSize); // + SubScratchSpace::shmem_size(TeamSize,nd);

  ttb_indx nnz = X.nnz();
  ttb_indx N = (nnz+TeamSize-1)/TeamSize;
  Policy policy(N,TeamSize,VectorSize);

  // typedef Kokkos::View<const ttb_real**,Kokkos::LayoutRight,Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> > RandomAccessView;
  // Kokkos::View<RandomAccessView*> u_view("u_view", nd);
  // for (ttb_indx m=0; m<nd; ++m) {
  //   u_view(m) = u[m].view();
  // }

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
    // SubScratchSpace sub(team.team_scratch(0), team_size, nd);

    // Read subs
    // Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,nd),
    //                      [&] (const size_type& m)
    // {
    //   sub(team_index,m) = X.subscript(i,m);
    // });

    ttb_real x_val = X.value(i);
    ttb_indx k = X.subscript(i,n);
    //ttb_indx k = sub(team_index,n);

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
          //const ttb_real *row = u[m].rowptr(sub(team_index,m));
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,BlockSize),
                               [&] (const size_type& jj)
          {
            const ttb_indx j = j_block+jj;
            if (j<nc)
              tmp(team_index,jj) *= row[j];
          });
          // const ttb_indx l = X.subscript(i,m);
          // RandomAccessView view = u_view(m);
          // Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,BlockSize),
          //                      [&] (const size_type& jj)
          // {
          //   const ttb_indx j = j_block+jj;
          //   if (j<nc)
          //     tmp(team_index,jj) *= view(l,j);
          // });
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

// Version of mttkrp using a permutation array to improve locality of writes,
// and reduce atomic throughput needs.  This version is geared towards CPUs
// and processes a large block of nonzeros before performing a segmented
// reduction across the corresponding rows.
void Genten::mttkrp(const Genten::Sptensor_perm  & X,
                 const Genten::Ktensor & u,
                 ttb_indx                    n,
                 Genten::FacMatrix     & v)
{
  typedef Kokkos::DefaultExecutionSpace ExecSpace;
  typedef typename ExecSpace::size_type size_type;

#if defined(KOKKOS_HAVE_CUDA)
  if (std::is_same<ExecSpace,Kokkos::Cuda>::value) {
    mttkrp_cuda(X,u,n,v);
    return;
  }
#endif

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

      // Perform segmented reduction of val for the same row indices
      if (team_index == 0) {
        ttb_indx ii = 0;
        while (ii < RowBlockSize) {
          ttb_indx kk=ii+1;
          while ( (kk < RowBlockSize) && (row[kk] == row[kk-1]) ) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,FacBlockSize),
                                 [&] (const size_type& jj)
            {
              // const ttb_indx j = j_block+jj;
              // if (j<nc)
                val(kk,jj) += val(kk-1,jj);
            });
            ++kk;
          }
          ii = kk;
        }
      }

      // Parallel version of the segmented reduction, which is appears to be
      // generally slower
      // {
      //   ttb_indx width = RowBlockSize/team_size;
      //   ttb_indx offset=width*team_index;
      //   ttb_indx ii = offset;
      //   while (ii < offset+width) {
      //     ttb_indx kk=ii+1;
      //     while ( (kk < offset+width) && (row[kk] == row[kk-1]) ) {
      //       Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,FacBlockSize),
      //                            [&] (const size_type& jj)
      //       {
      //         const ttb_indx j = j_block+jj;
      //         if (j<nc)
      //           val(kk,jj) += val(kk-1,jj);
      //       });
      //       ++kk;
      //     }
      //     ii = kk;
      //   }

      //   ttb_indx l = 2;
      //   while (width < RowBlockSize) {
      //     team.team_barrier();
      //     if (team_index % l == 0) {
      //       const ttb_indx e = offset+width-1;
      //       ii = offset+width;
      //       while (ii<offset+2*width && row[ii] == row[e]) {
      //         Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,FacBlockSize),
      //                              [&] (const size_type& jj)
      //         {
      //           const ttb_indx j = j_block+jj;
      //           if (j<nc)
      //             val(ii,jj) += val(e,jj);
      //         });
      //         ++ii;
      //       }
      //     }
      //     l *= 2;
      //     width *= 2;
      //   }
      // }

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
#if 0
// Version of mttkrp using a permutation array to improve locality of writes,
// and reduce atomic throughput needs.  This version is geared towards GPUs
// and performs a team-size segmented reduction while processing a large block
// of nonzeros.
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

  // For a maximum TeamSize=16, and FacBlockSize=16, the kernel
  // needs 8*16*16 = 2K shared memory per CUDA block.  Most modern GPUs have
  // between 48K and 64K shared memory per SM, allowing between 24 and 32
  // blocks per SM, which is typically enough for 100% occupancy.
  const size_type FacBlockSize = std::min(size_type(16),size_type(nc));
  const size_type RowBlockSize =
    std::max(is_cuda ? size_type(128) : size_type(128),TeamSize);

  // compute how much scratch memory (in bytes) is needed
  const size_t bytes =
    ValScratchSpace::shmem_size(TeamSize,FacBlockSize) +
    RowScratchSpace::shmem_size(TeamSize);

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
    ValScratchSpace val(team.team_scratch(0), team_size, FacBlockSize);
    RowScratchSpace row(team.team_scratch(0), team_size);

    for (ttb_indx j_block=0; j_block<nc; j_block+=FacBlockSize) {

      // Finish the previous block before starting the next
      team.team_barrier();

      for (ttb_indx ii=team_index; ii<RowBlockSize; ii+=team_size) {
        const ttb_indx i = i_block+ii;

        if (i >= nnz) {
          row(team_index) = invalid_row;
        }
        else {
          const ttb_indx p = X.getPerm(i,n);
          const ttb_real x_val = X.value(p);
          const ttb_indx k = X.subscript(p,n);

          row(team_index) = k;

          // Start val equal to the weights.
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,FacBlockSize),
                               [&] (const size_type& jj)
          {
            const ttb_indx j = j_block+jj;
            if (j<nc) {
              val(team_index,jj) = x_val * u.weights(j);
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
                  val(team_index,jj) *= rowptr[j];
                }
              });
            }
          }
        }

        // Finish writing to row before starting reduction
        team.team_barrier();

        // Perform segmented reduction of val for the same row indices
        if (team_index == 0) {
          ttb_indx ii = 0;
          while (ii < team_size) {
            ttb_indx kk=ii+1;
            while ( (kk < team_size) && (row(kk) == row(kk-1)) ) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,FacBlockSize),
                                   [&] (const size_type& jj)
              {
                // const ttb_indx j = j_block+jj;
                // if (j<nc)
                  val(kk,jj) += val(kk-1,jj);
              });
              ++kk;
            }
            ii = kk;
          }
        }

        // Finish reduction before updating results
        team.team_barrier();

        // Update output by adding val array.
        if ((row(team_index) != invalid_row) &&
            ((team_index == team_size-1) ||
             (row(team_index) != row(team_index+1)))) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,FacBlockSize),
                               [&] (const size_type& jj)
          {
            const ttb_indx j = j_block+jj;
            if (j<nc)
              Kokkos::atomic_add(&v.entry(row(team_index),j),
                                 val(team_index,jj));
          });
        }

      }

    }

  });

  return;
}
#elif 0
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
#else
// Version of mttkrp using a permutation array to improve locality of writes,
// and reduce atomic throughput needs.  This version is geared towards GPUs
// and performs a team-size segmented reduction while processing a large block
// of nonzeros.  This is a pure-Cuda implementation of the same kernel above,
// and it appears to therefore be somewhat faster.  This version also implements
// carry across the team loop to reduce atomic writes
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
    ValScratchSpace::shmem_size(TeamSize+1,VectorSize) +
    RowScratchSpace::shmem_size(TeamSize+1);

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
    ValScratchSpace val(team.team_scratch(0), blockDim.y+1, blockDim.x);
    RowScratchSpace row(team.team_scratch(0), blockDim.y+1);

    for (ttb_indx j_block=0; j_block<nc; j_block+=blockDim.x) {
      const ttb_indx j = j_block+threadIdx.x;
      if (j >= nc)
        break;

      ttb_indx first_row = invalid_row;

      if (threadIdx.y == 0)
        row(blockDim.y) = invalid_row;

      for (ttb_indx ii=threadIdx.y; ii<RowBlockSize; ii+=blockDim.y) {
        const ttb_indx i = i_block+ii;

        // Finish previous iteration before starting the next
        team.team_barrier();

        if (i >= nnz) {
          row(threadIdx.y) = invalid_row;

          // Scatter carry from previous iteration if rows differ
          if (threadIdx.y == 0 && row(blockDim.y) != invalid_row) {
            if (row(blockDim.y) == first_row)
              Kokkos::atomic_add(&v.entry(row(blockDim.y),j),
                                 val(blockDim.y,threadIdx.x));
            else
              v.entry(row(blockDim.y),j) += val(blockDim.y,threadIdx.x);
            row(blockDim.y) = invalid_row;
          }
        }
        else {
          const ttb_indx p = X.getPerm(i,n);
          const ttb_real x_val = X.value(p);
          const ttb_indx k = X.subscript(p,n);

          row(threadIdx.y) = k;

          if (ii == threadIdx.y)
            first_row = k;

          // Scatter carry from previous iteration if rows differ
          if (threadIdx.y == 0 && row(blockDim.y) != invalid_row &&
              row(blockDim.y) != k) {
            if (row(blockDim.y) == first_row)
              Kokkos::atomic_add(&v.entry(row(blockDim.y),j),
                                 val(blockDim.y,threadIdx.x));
            else
              v.entry(row(blockDim.y),j) += val(blockDim.y,threadIdx.x);
            row(blockDim.y) = invalid_row;
          }

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
          if (threadIdx.y == 0) {
            // Add in carry from previous iteration
            if (row(blockDim.y) != invalid_row && row(blockDim.y) == k)
              val(0,threadIdx.x) += val(blockDim.y,threadIdx.x);

            ttb_indx ii = 0;
            while (ii < blockDim.y) {
              ttb_indx kk=ii+1;
              while ( (kk < blockDim.y) && (row(kk) == row(kk-1)) ) {
                val(kk,threadIdx.x) += val(kk-1,threadIdx.x);
                ++kk;
              }
              ii = kk;
            }

            // Copy last entry to carry position
            row(blockDim.y) = row(blockDim.y-1);
            val(blockDim.y,threadIdx.x) = val(blockDim.y-1,threadIdx.x);
          }

          // Finish reduction before updating results
          team.team_barrier();

          // Update output by adding val array.
          if ((row(threadIdx.y) != invalid_row) &&
              (threadIdx.y < blockDim.y-1) &&
              (row(threadIdx.y) != row(threadIdx.y+1))) {
            if (row(threadIdx.y) == first_row)
              Kokkos::atomic_add(&v.entry(row(threadIdx.y),j),
                                 val(threadIdx.y,threadIdx.x));
            else
              v.entry(row(threadIdx.y),j) += val(threadIdx.y,threadIdx.x);
          }

        }

      }

      team.team_barrier();

      // Add in carry from last iteration
      if (threadIdx.y == 0 && row(blockDim.y) != invalid_row) {
        Kokkos::atomic_add(&v.entry(row(blockDim.y),j),
                           val(blockDim.y,threadIdx.x));
        row(blockDim.y) = invalid_row;
      }

    }

  });

  return;
}
#endif
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

#endif
