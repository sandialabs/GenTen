//@HEADER
// ************************************************************************
//     Genten Tensor Toolbox
//     Software package for tensor math by Sandia National Laboratories
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
  @file Genten_FacMatrix.cpp
  @brief Implementation of Genten::FacMatrix.
*/

#include "Genten_IndxArray.h"
#include "Genten_FacMatrix.h"
#include "Genten_FacMatArray.h"
#include "Genten_MathLibs_Wpr.h"
#include "Genten_portability.h"
#include "CMakeInclude.h"
#include <algorithm>     // for std::max with MSVC compiler
#include <assert.h>
#include <cstring>

#if defined(KOKKOS_HAVE_CUDA) && defined(HAVE_CUSOLVER)
#include "cusolverDn.h"
#endif

using namespace std;

Genten::FacMatrix::
FacMatrix(ttb_indx m, ttb_indx n, const ttb_real * cvec):
  data("Genten::FacMatrix::data",m,n)
{
  this->convertFromCol(m,n,cvec);
}


void Genten::FacMatrix::
operator=(ttb_real val)
{
  Kokkos::deep_copy(data, val);
}

void Genten::FacMatrix::
rand()
{
  auto data_1d = make_data_1d();
  data_1d.rand();
}

void Genten::FacMatrix::
scatter (const bool bUseMatlabRNG,
         RandomMT &  cRMT)
{
  auto data_1d = make_data_1d();
  data_1d.scatter (bUseMatlabRNG, cRMT);
}

void Genten::FacMatrix::
convertFromCol(ttb_indx nr, ttb_indx nc, const ttb_real * cvec)
{
  const ttb_indx nrows = data.dimension_0();
  const ttb_indx ncols = data.dimension_1();
  for (ttb_indx  i = 0; i < nrows; i++)
  {
    for (ttb_indx  j = 0; j < ncols; j++)
    {
      data(i,j) = cvec[i + j * nrows];
    }
  }
}


bool Genten::FacMatrix::
isEqual(const Genten::FacMatrix & b, ttb_real tol) const
{
  // Check for equal sizes first
  if ((data.dimension_0() != b.data.dimension_0()) ||
      (data.dimension_1() != b.data.dimension_1()))
  {
    return false;
  }

  // Check for equal data (within tolerance)
  auto data_1d = make_data_1d();
  auto b_data_1d = b.make_data_1d();
  return (data_1d.isEqual(b_data_1d, tol));

}

void Genten::FacMatrix::
times(const Genten::FacMatrix & v)
{
  if ((v.data.dimension_0() != data.dimension_0()) ||
      (v.data.dimension_1() != data.dimension_1()))
  {
    error("Genten::FacMatrix::hadamard - size mismatch");
  }
  auto data_1d = make_data_1d();
  auto v_data_1d = v.make_data_1d();
  data_1d.times(v_data_1d);
}

void Genten::FacMatrix::
plus(const Genten::FacMatrix & y)
{
  // TODO: check size compatibility, parallelize
  auto data_1d = make_data_1d();
  auto y_data_1d = y.make_data_1d();
  data_1d.plus(y_data_1d);
}

void Genten::FacMatrix::
plusAll(const Genten::FacMatArray & ya)
{
  // TODO: check size compatibility
  auto data_1d = make_data_1d();
  for (ttb_indx i =0;i< ya.size(); i++)
  {
    auto ya_data_1d = ya[i].make_data_1d();
    data_1d.plus(ya_data_1d);
  }
}

void Genten::FacMatrix::
transpose(const Genten::FacMatrix & y)
{
  // TODO: Replace with call to BLAS3: DGEMM (?)
  const ttb_indx nrows = data.dimension_0();
  const ttb_indx ncols = data.dimension_1();

  for (ttb_indx i = 0; i < nrows; i ++)
  {
    for (ttb_indx j = 0; j < ncols; j ++)
    {
      this->entry(i,j) = y.entry(j,i);
    }
  }
}

void Genten::FacMatrix::
times(ttb_real a)
{
  auto data_1d = make_data_1d();
  data_1d.times(a);
}

void Genten::FacMatrix::
gramian(const Genten::FacMatrix & v)
{
  typedef Kokkos::DefaultExecutionSpace ExecSpace;
  typedef typename ExecSpace::size_type size_type;

  // Get the size of v
  const size_type m = v.data.dimension_0();
  const size_type n = v.data.dimension_1();

  assert(data.dimension_0() == n);
  assert(data.dimension_1() == n);

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

  // Use the largest power of 2 <= n, with a maximum of 16 for the vector size.
  const size_type VectorSize =
    n == 1 ? 1 : std::min(16,2 << int(std::log2(n))-1);
  const size_type TeamSize = is_cuda ? 256/VectorSize : thread_pool_size;

  // To do:  optimize TeamSize for small n (right now we are wasting threads
  // if n < 16).
  const size_type BlockSize = std::min(size_type(16), n);

  // compute how much scratch memory (in bytes) is needed
  const size_t bytes = TmpScratchSpace::shmem_size(BlockSize,BlockSize);

  const size_type NumRow = 16;
  size_type M = (m+NumRow-1)/NumRow;
  Policy policy(M,TeamSize,VectorSize);

  // Can't capture "this" pointer by value
  view_type my_data = data;
  Kokkos::deep_copy( my_data, 0.0 );

  Kokkos::parallel_for(policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
                       KOKKOS_LAMBDA(Policy::member_type team)
  {
    const size_type team_size = team.team_size();
    const size_type team_index = team.team_rank();
    const size_type k0 = team.league_rank()*NumRow;

    // Allocate scratch spaces
    TmpScratchSpace tmp(team.team_scratch(0), BlockSize, BlockSize);
    if (tmp.data() == 0)
      Kokkos::abort("Allocation of temp space failed.");

    // Compute upper-triangular blocks of v(k,i)*v(k,j)
    // in blocks of size (NumRow,BlockSize)
    for (size_type i_block=0; i_block<n; i_block+=BlockSize) {
      for (size_type j_block=i_block; j_block<n; j_block+=BlockSize) {

        // Initialize tmp scratch space
        for (size_type ii=team_index; ii<BlockSize; ii+=team_size) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,BlockSize),
                               [&] (const size_type& jj)
          {
            tmp(ii,jj) = 0.0;
          });
        }

        // Loop over rows in block
        for (size_type kk=0; kk<NumRow; ++kk) {
          const size_type k = k0+kk;
          if (k >= m)
            break;

          // Compute v(k,i)*v(k,j) for (i,j) in this block
          for (size_type ii=team_index; ii<BlockSize; ii+=team_size) {
            size_type i = i_block+ii;
            if (i >= n)
              break;

            const ttb_real v_ki = v.data(k,i);
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,BlockSize),
                                 [&] (const size_type& jj)
            {
              const size_type j = j_block+jj;
              if (j<n)
                tmp(ii,jj) += v_ki*v.data(k,j);
            });

          }

        }

        // Accumulate inner products into global result using atomics
        for (size_type ii=team_index; ii<BlockSize; ii+=team_size) {
          size_type i = i_block+ii;
          if (i >= n)
            break;

          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,BlockSize),
                               [&] (const size_type& jj)
          {
            const size_type j = j_block+jj;
            if (j<n) {
              Kokkos::atomic_add(&my_data(i,j), tmp(ii,jj));
              if (j_block != i_block)
                Kokkos::atomic_add(&my_data(j,i), tmp(ii,jj));
            }
          });
        }
      }
    }
  });

  // Copy upper triangle into lower triangle
  // This seems to be the most efficient over adding the lower triangle in
  // the above kernel or launching a new one below.  However that may change
  // when n is large.
  // for (ttb_indx i=0; i<n; ++i)
  //   for (ttb_indx j=i+1; j<n; j++)
  //     my_data(j,i) = my_data(i,j);

  // const ttb_indx N = (n+TeamSize-1)/TeamSize;
  // Policy policy2(N,TeamSize,VectorSize);
  //  Kokkos::parallel_for(policy2, KOKKOS_LAMBDA(Policy::member_type team)
  // {
  //   const ttb_indx team_size = team.team_size();
  //   const ttb_indx team_index = team.team_rank();
  //   const ttb_indx i = team.league_rank()*team_size+team_index;

  //   if (i >= n)
  //     return;

  //   Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,n-i-1),
  //                        [&] (const ttb_indx& j)
  //   {
  //     my_data(i+j,i) = my_data(i,i+j);
  //   });

  // });

  return;
}

// return the index of the first entry, s, such that entry(s,c) > r.
// assumes/requires the values entry(s,c) are nondecreasing as s increases.
// if the assumption fails, result is undefined but <= nrows.
ttb_indx Genten::FacMatrix::
firstGreaterSortedIncreasing(ttb_real r, ttb_indx c) const
{
  const ttb_indx nrows = data.dimension_0();
  const ttb_indx ncols = data.dimension_1();

  ttb_indx least = 0;
  const ttb_real *base = ptr();
  base += c;
  ttb_indx trips=0;
  if (nrows > 2) {
    ttb_indx mini=0, maxi=nrows-1, midi = 0;
    do {
      trips++;
      midi = (mini+maxi) /2;
      if (r > base[ncols*(midi)]) {
        mini = midi +1; // overflow never a problem if ttb_indx is unsigned
      } else {
        maxi = midi -1; // must be guarded against underflow elsewhere
      }
      if (trips > nrows) {
        Genten::error("Genten::FacMatrix::firstGreaterSortedIncreasing - trip limit exceeded");
        break;
      }
    } while (base[ncols*midi] != r && mini < maxi && midi >0);
    // cleanup from all cases:
    least = (maxi>= mini) ? mini : maxi;
  }
  while (least < nrows-1 && base[ncols*least] < r) {
    least++;
  }
  return least;
}

void Genten::FacMatrix::
oprod(const Genten::Array & v)
{
  // TODO: Replace with call to BLAS2:DGER

  ttb_indx n = v.size();
  for (ttb_indx j = 0; j < n; j ++)
  {
    for (ttb_indx i = 0; i < n; i ++)
    {
      this->entry(i,j) = v[i] * v[j];
    }
  }
}

void Genten::FacMatrix::
colNorms(Genten::NormType normtype, Genten::Array & norms, ttb_real minval)
{
  typedef Kokkos::DefaultExecutionSpace ExecSpace;
  typedef typename ExecSpace::size_type size_type;

  const size_type m = data.dimension_0();
  const size_type n = data.dimension_1();

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

  // Use the largest power of 2 <= n, with a maximum of 16 for the vector size.
  const size_type VectorSize =
    n == 1 ? 1 : std::min(16,2 << int(std::log2(n))-1);
  const size_type TeamSize = is_cuda ? 256/VectorSize : thread_pool_size;

  // compute how much scratch memory (in bytes) is needed
  const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,VectorSize);

  size_type M = (m+TeamSize-1)/TeamSize;
  Policy policy(M,TeamSize,VectorSize);

  // Can't capture "this" pointer by value
  view_type my_data = data;

  // Initialize norms to 0
  norms = 0.0;

  // Compute norms
  switch(normtype)
  {
  case Genten::NormInf:
  {
    /*
    for (size_type j = 0; j < n; j ++)
    {
      // Note that STRIDE is being used here.
      // *** Using data.ptr() friend function from Genten::Array ***
      ttb_indx i = Genten::imax(m, data.data() + j, n);
      norms[j] = fabs(this->entry(i,j));
    }
    */
    Kokkos::parallel_for(policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
                         KOKKOS_LAMBDA(Policy::member_type team)
    {
      const size_type team_size = team.team_size();
      const size_type team_index = team.team_rank();
      const size_type i_block = team.league_rank()*team_size;
      const size_type i = i_block+team_index;

      // Allocate scratch spaces
      TmpScratchSpace tmp(team.team_scratch(0), team_size, VectorSize);
      if (tmp.data() == 0)
        Kokkos::abort("Allocation of temp space failed.");

      for (size_type j_block=0; j_block<n; j_block+=VectorSize) {

        team.team_barrier();

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,VectorSize),
                             [&] (const size_type& jj)
        {
          const size_type j = j_block+jj;
          if (i < m && j < n)
            tmp(team_index,jj) = std::fabs(my_data(i,j));
          else
            tmp(team_index,jj) = 0.0;
        });

        team.team_barrier();

        // Team-parallel fan-in reduction...assumes team-size is a power of 2
        for (size_type k=1; k<team_size; k*=2) {
          const size_type ii=2*k*team_index;
          if (ii+k<team_size && i_block+ii+k < m) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,VectorSize),
                                 [&] (const size_type& jj)
            {
              if (tmp(ii+k,jj) > tmp(ii,jj))
                tmp(ii,jj) = tmp(ii+k,jj);
            });
          }
          team.team_barrier();
        }

        if (team_index == 0) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,VectorSize),
                               [&] (const size_type& jj)
          {
            const size_type j = j_block+jj;
            if (j < n) {

              // Unfortunately there is no "atomic_max()", so we have to
              // implement it using compare_and_exchange
              ttb_real prev_val = norms[j];
              ttb_real val = tmp(0,jj);
              bool keep_going = true;
              while (prev_val < val && keep_going) {

                // Replace norms[j] with val when norms[j] == prev_val
                ttb_real next_val =
                  Kokkos::atomic_compare_exchange(&norms[j], prev_val, val);

                // When prev_val == next_val, the exchange happened and we
                // can stop.  This saves an extra iteration
                keep_going = !(prev_val == next_val);
                prev_val = next_val;
              }
            }
          });
        }
      }
    });
    break;
  }
  case Genten::NormOne:
  {
    /*
    for (size_type j = 0; j < n; j ++)
    {
      // Note that STRIDE is being used here.
      // *** Using data.ptr() friend function from Genten::Array ***
      norms[j] = Genten::nrm1(m, data.data() + j, n);
    }
    */
    Kokkos::parallel_for(policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
                         KOKKOS_LAMBDA(Policy::member_type team)
    {
      const size_type team_size = team.team_size();
      const size_type team_index = team.team_rank();
      const size_type i_block = team.league_rank()*team_size;
      const size_type i = i_block+team_index;

      // Allocate scratch spaces
      TmpScratchSpace tmp(team.team_scratch(0), team_size, VectorSize);
      if (tmp.data() == 0)
        Kokkos::abort("Allocation of temp space failed.");

      for (size_type j_block=0; j_block<n; j_block+=VectorSize) {

        team.team_barrier();

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,VectorSize),
                             [&] (const size_type& jj)
        {
          const size_type j = j_block+jj;
          if (i < m && j < n)
            tmp(team_index,jj) = std::fabs(my_data(i,j));
          else
            tmp(team_index,jj) = 0.0;
        });

        team.team_barrier();

        // Team-parallel fan-in reduction...assumes team-size is a power of 2
        for (size_type k=1; k<team_size; k*=2) {
          const size_type ii=2*k*team_index;
          if (ii+k<team_size && i_block+ii+k < m) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,VectorSize),
                                 [&] (const size_type& jj)
            {
              tmp(ii,jj) += tmp(ii+k,jj);
            });
          }
          team.team_barrier();
        }

        if (team_index == 0) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,VectorSize),
                               [&] (const size_type& jj)
          {
            const size_type j = j_block+jj;
            if (j < n)
              Kokkos::atomic_add(&norms[j], tmp(0,jj));
          });
        }
      }
    });
    break;
  }
  case Genten::NormTwo:
  {
    /*
    for (size_type j = 0; j < n; j ++)
    {
      // Note that STRIDE is being used here.
      // *** Using data.ptr() friend function from Genten::Array ***
      norms[j] = Genten::nrm2(m, data.data() + j, n);
    }
    */
    Kokkos::parallel_for(policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
                         KOKKOS_LAMBDA(Policy::member_type team)
    {
      const size_type team_size = team.team_size();
      const size_type team_index = team.team_rank();
      const size_type i_block = team.league_rank()*team_size;
      const size_type i = i_block+team_index;

      // Allocate scratch spaces
      TmpScratchSpace tmp(team.team_scratch(0), team_size, VectorSize);
      if (tmp.data() == 0)
        Kokkos::abort("Allocation of temp space failed.");

      for (size_type j_block=0; j_block<n; j_block+=VectorSize) {

        team.team_barrier();

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,VectorSize),
                             [&] (const size_type& jj)
        {
          const size_type j = j_block+jj;
          if (i < m && j < n)
            tmp(team_index,jj) = my_data(i,j)*my_data(i,j);
          else
            tmp(team_index,jj) = 0.0;
        });

        team.team_barrier();

        // Team-parallel fan-in reduction...assumes team-size is a power of 2
        for (size_type k=1; k<team_size; k*=2) {
          const size_type ii=2*k*team_index;
          if (ii+k<team_size && i_block+ii+k < m) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,VectorSize),
                                 [&] (const size_type& jj)
            {
              tmp(ii,jj) += tmp(ii+k,jj);
            });
          }
          team.team_barrier();
        }

        if (team_index == 0) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,VectorSize),
                               [&] (const size_type& jj)
          {
            const size_type j = j_block+jj;
            if (j < n)
              Kokkos::atomic_add(&norms[j], tmp(0,jj));
          });
        }
      }
    });
    for (size_type j=0; j<n; ++j)
      norms[j] = std::sqrt(norms[j]);
    break;
  }
  default:
  {
    error("Genten::FacMatrix::colNorms - unimplemented norm type");
  }
  }

  // Check for min value
  if (minval > 0)
  {
    for (size_type j=0; j<n; ++j)
    {
      if (norms[j] < minval)
      {
        norms[j] = minval;
      }
    }
  }

}

//TBD could be much better vectorized instead of ncols trips through data.
void Genten::FacMatrix::
colScale(const Genten::Array & v, bool inverse)
{
  typedef Kokkos::DefaultExecutionSpace ExecSpace;
  typedef typename ExecSpace::size_type size_type;

  const size_type m = data.dimension_0();
  const size_type n = data.dimension_1();
  assert(v.size() == n);

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

  // Use the largest power of 2 <= n, with a maximum of 16 for the vector size.
  const size_type VectorSize =
    n == 1 ? 1 : std::min(16,2 << int(std::log2(n))-1);
  const size_type TeamSize = is_cuda ? 256/VectorSize : thread_pool_size;

  size_type M = (m+TeamSize-1)/TeamSize;
  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  Policy policy(M,TeamSize,VectorSize);

  // Can't capture "this" pointer by value
  view_type my_data = data;

  if (inverse) {
    for (ttb_indx j = 0; j < n; j ++)
      if (v[j] == 0)
        Genten::error("Genten::FacMatrix::colScale - divide-by-zero error");

    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(Policy::member_type team)
    {
      const size_type team_size = team.team_size();
      const size_type team_index = team.team_rank();
      const size_type i = team.league_rank()*team_size + team_index;

      if (i >= m)
        return;

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,n),
                           [&] (const size_type& j)
      {
        my_data(i,j) /= v[j];
      });
    });
  }
  else {
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(Policy::member_type team)
    {
      const size_type team_size = team.team_size();
      const size_type team_index = team.team_rank();
      const size_type i = team.league_rank()*team_size + team_index;

      if (i >= m)
        return;

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,n),
                           [&] (const size_type& j)
      {
        my_data(i,j) *= v[j];
      });
    });
  }
}

// Only called by Ben Allan's parallel test code.  It appears he uses the Linux
// random number generator in a special way.
#if !defined(_WIN32)
void Genten::FacMatrix::
scaleRandomElements(ttb_real fraction, ttb_real scale, bool columnwise)
{
  const ttb_indx nrows = data.dimension_0();
  const ttb_indx ncols = data.dimension_1();
  auto data_1d = make_data_1d();
  if (fraction < 0.0 || fraction > 1.0) {
    Genten::error("Genten::FacMatrix::scaleRandomElements - input fraction invalid");
  }
  if (columnwise) {
    // loop over nr
    // do the below process, % ncols
    Genten::error("Genten::FacMatrix::scaleRandomElements - columnwise not yet coded");
  } else {
    ttb_indx tot = nrows * ncols;
    ttb_indx n = (ttb_indx) fraction * tot;
    ttb_indx i=0,k=0;
    if (n < 1) { n =1; }
    // flags array so we don't count repeats twice toward fraction target
    Genten::IndxArray marked(tot,(ttb_indx)0);
    int misses = 0;
    while (i < n ) {
      k = ::random() % tot;
      if (!marked[k]) {
        marked[k] = 1;
        i++;
      } else {
        misses++;
        if (misses > 2*tot) { break; }
        // 2*tot is heuristic. there is a repeat probability in
        // the low end bits of linux random()
        // there is also a cycling possibility that will hang here eternally if not trapped.
      }
    }
    if (i < n) {
      Genten::error("Genten::FacMatrix::scaleRandomElements - ran out of random numbers");
    }
    for (k=0;k<tot;k++)
    {
      if (marked[k]) {
        data_1d[k] *= scale;
      }
    }
  }
}
#endif

// TODO: This function really should be removed and replaced with a ktensor norm function, because that's kind of how it's used.
ttb_real Genten::FacMatrix::
sum() const
{
  const ttb_indx nrows = data.dimension_0();
  const ttb_indx ncols = data.dimension_1();
  auto data_1d = make_data_1d();
  ttb_indx n = nrows * ncols;
  ttb_real sum = 0;
  for (ttb_indx i = 0; i < n; i ++)
  {
    sum += data_1d[i];
  }
  return(sum);
}

bool Genten::FacMatrix::
hasNonFinite(ttb_indx &retval) const
{
  const ttb_real * mptr = ptr();
  ttb_indx imax = data.size();
  retval = 0;
  for (ttb_indx i = 0; i < imax; i ++) {
    if (isRealValid(mptr[i]) == false) {
      retval = i;
      return true;
    }
  }
  return false;
}

void Genten::FacMatrix::
permute(const Genten::IndxArray &perm_indices)
{
  // The current implementation using a single column of temporary storage
  // (i.e., an array of size ncols) and requires at most 2*nrows*(ncols-1)
  // data moves (n-1 column swaps).  The number is less if a swap results
  // in placing both columns in the desired permutation order).

  const ttb_indx nrows = data.dimension_0();
  const ttb_indx ncols = data.dimension_1();
  const ttb_indx invalid = ttb_indx(-1);

  // check that the length of indices equals the number of data columns
  if (perm_indices.size() != ncols)
  {
    error("Number of indices in permutation does not equal the number of columns in the FacMatrix");
  }

  // keep track of columns during permutation
  Genten::IndxArray curr_indices(ncols);
  for (ttb_indx j=0; j<ncols; j++)
    curr_indices[j] = j;

  // storage for moving data aside during column permutation
  Genten::Array temp(nrows);

  for (ttb_indx j=0; j<ncols; j++)
  {
    // Find the current location of the column to be permuted.
    ttb_indx  loc = invalid;
    for (ttb_indx i=0; i<ncols; i++)
    {
      if (curr_indices[i] == perm_indices[j])
      {
        loc = i;
        break;
      }
    }
    if (loc == invalid)
      error("*** TBD");

    // Swap j with the location, or do nothing if already in order.
    if (j != loc) {

      // Move data in column loc to temp column.
      for (ttb_indx i=0; i<nrows; i++)
        temp[i] = this->entry(i,loc);

      // Move data in column j to column loc.
      for (ttb_indx i=0; i<nrows; i++)
        this->entry(i,loc) = this->entry(i,j);

      // Move temp column to column loc.
      for (ttb_indx i=0; i<nrows; i++)
        this->entry(i,j) = temp[i];

      // Swap curr_indices to mark where data was moved.
      ttb_indx  k = curr_indices[j];
      curr_indices[j] = curr_indices[loc];
      curr_indices[loc] = k;
    }
  }
}

void Genten::FacMatrix::
multByVector(bool bTranspose,
             const Genten::Array &  x,
             Genten::Array &  y) const
{
  const ttb_indx nrows = data.dimension_0();
  const ttb_indx ncols = data.dimension_1();

  if (bTranspose == false)
  {
    assert(x.size() == ncols);
    assert(y.size() == nrows);
    // Data for the matrix is stored in row-major order but gemv expects
    // column-major, so tell it transpose dimensions.
    Genten::gemv('T', ncols, nrows, 1.0, ptr(), ncols,
              x.ptr(), 1, 0.0, y.ptr(), 1);
  }
  else
  {
    assert(x.size() == nrows);
    assert(y.size() == ncols);
    // Data for the matrix is stored in row-major order but gemv expects
    // column-major, so tell it transpose dimensions.
    Genten::gemv('N', ncols, nrows, 1.0, ptr(), ncols,
              x.ptr(), 1, 0.0, y.ptr(), 1);
  }
  return;
}

namespace Genten {
  namespace Impl {

    template <typename ViewA, typename ViewB>
    void solveTransposeRHSImpl(const ViewA& A, const ViewB& B) {
      const ttb_indx nrows = B.dimension_0();
      const ttb_indx ncols = B.dimension_1();

      // Throws an exception if Atmp is (exactly?) singular.
      //TBD...consider LAPACK sysv instead of gesv since A is sym indef
      Genten::gesv (ncols, nrows, A.data(), ncols, B.data(), ncols);
    }

#if defined(KOKKOS_HAVE_CUDA) && defined(HAVE_CUSOLVER)

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    typename std::enable_if<
      ( std::is_same<typename Kokkos::View<AT,AP...>::execution_space,
                     Kokkos::Cuda>::value &&
        std::is_same<typename Kokkos::View<BT,BP...>::execution_space,
                     Kokkos::Cuda>::value &&
        std::is_same<typename Kokkos::View<AT,BP...>::non_const_value_type,
                     double>::value )
      >::type
    solveTransposeRHSImpl(const Kokkos::View<AT,AP...>& A,
                          const Kokkos::View<BT,BP...>& B) {
      const int m = B.dimension_0();
      const int n = B.dimension_1();
      const int lda = A.stride_0();
      const int ldb = B.stride_0();
      cusolverStatus_t status;

      assert(A.dimension_0() == n);
      assert(A.dimension_1() == n);

      static cusolverDnHandle_t handle = 0;
      if (handle == 0) {
        status = cusolverDnCreate(&handle);
        if (status != CUSOLVER_STATUS_SUCCESS) {
          std::stringstream ss;
          ss << "Error!  cusolverDnCreate() failed with status "
             << status;
          std::cerr << ss.str() << std::endl;
          throw ss.str();
        }
      }

      Kokkos::View<int> lwork("lwork");
      status = cusolverDnDgetrf_bufferSize(handle, n, n, A.data(), lda,
                                           lwork.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDgetrf_bufferSize() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      Kokkos::View<double*> work("work",lwork());
      Kokkos::View<int*> piv("piv",n);
      Kokkos::View<int> info("info");
      status = cusolverDnDgetrf(handle, n, n, A.data(), lda, work.data(),
                                piv.data(), info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDgetrf() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      if (info() != 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDngetrf() info =  " << info();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      status = cusolverDnDgetrs(handle, CUBLAS_OP_N, n, m, A.data(), lda,
                                piv.data(), B.data(), ldb, info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  ccusolverDnDgetrs() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      if (info() != 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDngetrf() info =  " << info();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
    }

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    typename std::enable_if<
      ( std::is_same<typename Kokkos::View<AT,AP...>::execution_space,
                     Kokkos::Cuda>::value &&
        std::is_same<typename Kokkos::View<BT,BP...>::execution_space,
                     Kokkos::Cuda>::value &&
        std::is_same<typename Kokkos::View<AT,BP...>::non_const_value_type,
                     float>::value )
      >::type
    solveTransposeRHSImpl(const Kokkos::View<AT,AP...>& A,
                          const Kokkos::View<BT,BP...>& B) {
      const int m = B.dimension_0();
      const int n = B.dimension_1();
      const int lda = A.stride_0();
      const int ldb = B.stride_0();
      cusolverStatus_t status;

      assert(A.dimension_0() == n);
      assert(A.dimension_1() == n);

      static cusolverDnHandle_t handle = 0;
      if (handle == 0) {
        status = cusolverDnCreate(&handle);
        if (status != CUSOLVER_STATUS_SUCCESS) {
          std::stringstream ss;
          ss << "Error!  cusolverDnCreate() failed with status "
             << status;
          std::cerr << ss.str() << std::endl;
          throw ss.str();
        }
      }

      Kokkos::View<int> lwork("lwork");
      status = cusolverDnSgetrf_bufferSize(handle, n, n, A.data(), lda,
                                           lwork.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDgetrf_bufferSize() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      Kokkos::View<float*> work("work",lwork());
      Kokkos::View<int*> piv("piv",n);
      Kokkos::View<int> info("info");
      status = cusolverDnSgetrf(handle, n, n, A.data(), lda, work.data(),
                                piv.data(), info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDgetrf() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      if (info() != 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDngetrf() info =  " << info();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      status = cusolverDnSgetrs(handle, CUBLAS_OP_N, n, m, A.data(), lda,
                                piv.data(), B.data(), ldb, info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  ccusolverDnDgetrs() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      if (info() != 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDngetrf() info =  " << info();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
    }
#endif

  }
}


void Genten::FacMatrix::
solveTransposeRHS (const Genten::FacMatrix &  A)
{
  const ttb_indx nrows = data.dimension_0();
  const ttb_indx ncols = data.dimension_1();

  assert(A.nRows() == A.nCols());
  assert(nCols() == A.nRows());

  // Copy A because LAPACK needs to overwrite the invertible square matrix.
  // Since the matrix is assumed symmetric, no need to worry about row major
  // versus column major storage.
  view_type Atmp("Atmp", A.nRows(), A.nCols());
  Kokkos::deep_copy(Atmp, A.data);

  Genten::Impl::solveTransposeRHSImpl(Atmp, data);

  return;
}


void Genten::FacMatrix::
rowTimes(Genten::Array & x,
         const ttb_indx nRow) const
{
  assert(x.size() == data.dimension_1());

  const ttb_real * rptr = this->rowptr(nRow);
  ttb_real * xptr = x.ptr();

  vmul(data.dimension_1(),xptr,rptr);

  return;
}

void Genten::FacMatrix::
rowTimes(const ttb_indx         nRow,
         const Genten::FacMatrix & other,
         const ttb_indx         nRowOther)
{
  assert(other.nCols() == data.dimension_1());

  ttb_real * rowPtr1 = this->rowptr(nRow);
  const ttb_real * rowPtr2 = other.rowptr(nRowOther);

  vmul(data.dimension_1(), rowPtr1, rowPtr2);

  return;
}

ttb_real Genten::FacMatrix::
rowDot(const ttb_indx         nRow,
       const Genten::FacMatrix & other,
       const ttb_indx         nRowOther) const
{
  const ttb_indx ncols = data.dimension_1();
  assert(other.nCols() == ncols);

  // Using LAPACK ddot is slower on perf_CpAprRandomKtensor.
  //   ttb_real  result = dot(ncols, this->rowptr(nRow), 1,
  //                          other.rowptr(nRowOther), 1);

  const ttb_real * rowPtr1 = this->rowptr(nRow);
  const ttb_real * rowPtr2 = other.rowptr(nRowOther);

  ttb_real  result = 0.0;
  for (ttb_indx  i = 0; i < ncols; i++)
    result += rowPtr1[i] * rowPtr2[i];

  return( result );
}

void Genten::FacMatrix::
rowDScale(const ttb_indx         nRow,
          Genten::FacMatrix & other,
          const ttb_indx         nRowOther,
          const ttb_real         dScalar) const
{
  const ttb_indx ncols = data.dimension_1();
  assert(other.nCols() == ncols);

  // Using LAPACK daxpy is slower on perf_CpAprRandomKtensor.
  //   axpy(ncols, dScalar, this->rowptr(nRow), 1, other.rowptr(nRowOther), 1);

  const ttb_real * rowPtr1 = this->rowptr(nRow);
  ttb_real * rowPtr2 = other.rowptr(nRowOther);

  for (ttb_indx  i = 0; i < ncols; i++)
    rowPtr2[i] = rowPtr2[i] + (dScalar * rowPtr1[i]);

  return;
}
