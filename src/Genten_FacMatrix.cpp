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
  @file Genten_FacMatrix.cpp
  @brief Implementation of Genten::FacMatrix.
*/

#include "Genten_IndxArray.hpp"
#include "Genten_FacMatrix.hpp"
#include "Genten_FacMatArray.hpp"
#include "Genten_MathLibs_Wpr.hpp"
#include "Genten_portability.hpp"
#include "CMakeInclude.h"
#include <algorithm>     // for std::max with MSVC compiler
#include <assert.h>
#include <cstring>

#if defined(KOKKOS_HAVE_CUDA) && defined(HAVE_CUSOLVER)
#include "cusolverDn.h"
#endif
#if defined(KOKKOS_HAVE_CUDA) && defined(HAVE_CUBLAS)
#include "cublas_v2.h"
#endif

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

#define USE_NEW_GRAMIAN 1
#define USE_NEW_COLNORMS 1
#define USE_NEW_COLSCALE 1

template <typename ExecSpace>
Genten::FacMatrixT<ExecSpace>::
FacMatrixT(ttb_indx m, ttb_indx n)
{
  // Don't use padding if Cuda is the default execution space, so factor
  // matrices allocated on the host have the same shape.  We really need a
  // better way to do this.
#if defined(KOKKOS_HAVE_CUDA)
  const bool have_cuda =
    std::is_same<DefaultExecutionSpace,Kokkos::Cuda>::value;
#else
  const bool have_cuda = false;
#endif
  if (have_cuda)
    data = view_type("Genten::FacMatrix::data",m,n);
  else
    data = view_type(Kokkos::view_alloc("Genten::FacMatrix::data",
                                        Kokkos::AllowPadding),m,n);
}

template <typename ExecSpace>
Genten::FacMatrixT<ExecSpace>::
FacMatrixT(ttb_indx m, ttb_indx n, const ttb_real * cvec) :
  FacMatrixT(m,n)
{
  this->convertFromCol(m,n,cvec);
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
operator=(ttb_real val) const
{
  deep_copy(data, val);
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
rand() const
{
  auto data_1d = make_data_1d();
  data_1d.rand();
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
scatter (const bool bUseMatlabRNG,
         RandomMT &  cRMT) const
{
  auto data_1d = make_data_1d();
  data_1d.scatter (bUseMatlabRNG, cRMT);
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
convertFromCol(ttb_indx nr, ttb_indx nc, const ttb_real * cvec) const
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

template <typename ExecSpace>
bool Genten::FacMatrixT<ExecSpace>::
isEqual(const Genten::FacMatrixT<ExecSpace> & b, ttb_real tol) const
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

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
times(const Genten::FacMatrixT<ExecSpace> & v) const
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

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
plus(const Genten::FacMatrixT<ExecSpace> & y) const
{
  // TODO: check size compatibility, parallelize
  auto data_1d = make_data_1d();
  auto y_data_1d = y.make_data_1d();
  data_1d.plus(y_data_1d);
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
plusAll(const Genten::FacMatArrayT<ExecSpace> & ya) const
{
  // TODO: check size compatibility
  auto data_1d = make_data_1d();
  for (ttb_indx i =0;i< ya.size(); i++)
  {
    auto ya_data_1d = ya[i].make_data_1d();
    data_1d.plus(ya_data_1d);
  }
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
transpose(const Genten::FacMatrixT<ExecSpace> & y) const
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

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
times(ttb_real a) const
{
  auto data_1d = make_data_1d();
  data_1d.times(a);
}

namespace Genten {
namespace Impl {

#if defined(LAPACK_FOUND)
// Gramian implementation using gemm()
template <typename ExecSpace, typename ViewC, typename ViewA>
void gramianImpl(const ViewC& C, const ViewA& A)
{
  const ttb_indx m = A.dimension_0();
  const ttb_indx n = A.dimension_1();
  const ttb_indx lda = A.stride_0();
  const ttb_indx ldc = C.stride_0();

  // We compute C = A'*A.  But since A is LayoutRight, and GEMM/SYRK
  // assumes layout left we compute this as C' = A*A'.

  // SYRK seems faster than GEMM on the CPU but slower on KNL using MKL.
  Genten::gemm('N','T',n,n,m,1.0,A.data(),lda,A.data(),lda,0.0,C.data(),ldc);
  // Genten::syrk('L','N',n,m,1.0,A.data(),lda,0.0,C.data(),ldc);
  // for (ttb_indx i=0; i<n; ++i)
  //   for (ttb_indx j=i+1; j<n; j++)
  //     C(j,i) = C(i,j);
}
#else
#if USE_NEW_GRAMIAN
template <typename ExecSpace, typename ViewC, typename ViewA,
          unsigned ColBlockSize, unsigned RowBlockSize,
          unsigned TeamSize, unsigned VectorSize>
struct GramianKernel {
  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  typedef Kokkos::View< ttb_real**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

  const ViewC& C;
  const ViewA& A;
  const TeamMember& team;
  const unsigned team_index;
  TmpScratchSpace tmp;
  const unsigned k_block;
  const unsigned m;
  const unsigned n;

  static inline Policy policy(const ttb_indx m_) {
    const unsigned M = (m_+RowBlockSize-1)/RowBlockSize;
    Policy policy(M,TeamSize,VectorSize);
    const size_t bytes = TmpScratchSpace::shmem_size(ColBlockSize,ColBlockSize);
    return policy.set_scratch_size(0,Kokkos::PerTeam(bytes));
  }

  KOKKOS_INLINE_FUNCTION
  GramianKernel(const ViewC& C_,
                const ViewA& A_,
                const TeamMember& team_) :
    C(C_),
    A(A_),
    team(team_),
    team_index(team.team_rank()),
    tmp(team.team_scratch(0), ColBlockSize, ColBlockSize),
    k_block(team.league_rank()*RowBlockSize),
    m(A.dimension_0()),
    n(A.dimension_1())
    {
      if (tmp.data() == 0)
        Kokkos::abort("GramianKernel:  Allocation of temp space failed.");
    }

  template <unsigned Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const unsigned i_block, const unsigned j_block, const unsigned nj_)
  {
    // nj.value == Nj_ if Nj_ > 0 and nj_ otherwise
    Kokkos::Impl::integral_nonzero_constant<unsigned, Nj> nj(nj_);

    for (unsigned ii=team_index; ii<ColBlockSize; ii+=TeamSize) {
      ttb_real *tmp_ii = &(tmp(ii,0));
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,unsigned(nj.value)),
                           [&] (const unsigned& jj)
      {
        tmp_ii[jj] = 0.0;
      });
    }

    // Loop over rows in block
    for (unsigned kk=0; kk<RowBlockSize; ++kk) {
      const unsigned k = k_block+kk;
      if (k >= m)
        break;

      // Compute A(k,i)*A(k,j) for (i,j) in this block
      for (unsigned ii=team_index; ii<ColBlockSize; ii+=TeamSize) {
        const unsigned i = i_block+ii;
        if (i < n) {
          const ttb_real A_ki = A(k,i);
          ttb_real *tmp_ii = &(tmp(ii,0));
          const ttb_real *A_kj = &(A(k,j_block));
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,unsigned(nj.value)),
                               [&] (const unsigned& jj)
          {
            tmp_ii[jj] += A_ki*A_kj[jj];
          });
        }
      }
    }

    // Accumulate inner products into global result using atomics
    for (unsigned ii=team_index; ii<ColBlockSize; ii+=TeamSize) {
      const unsigned i = i_block+ii;
      if (i < n) {
        const ttb_real *tmp_ii = &(tmp(ii,0));
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,unsigned(nj.value)),
                             [&] (const unsigned& jj)
        {
          const unsigned j = j_block+jj;
          Kokkos::atomic_add(&C(i,j), tmp_ii[jj]);
          if (j_block != i_block)
            Kokkos::atomic_add(&C(j,i), tmp_ii[jj]);
        });
      }
    }
  }
};

template <typename ExecSpace, unsigned ColBlockSize,
          typename ViewC, typename ViewA>
void gramian_kernel(const ViewC& C, const ViewA& A)
{
  // Compute team and vector sizes, depending on the architecture
#if defined(KOKKOS_HAVE_CUDA)
  const bool is_cuda = std::is_same<ExecSpace,Kokkos::Cuda>::value;
#else
  const bool is_cuda = false;
#endif
  const unsigned VectorSize =
    is_cuda ? (ColBlockSize <= 16 ? ColBlockSize : 16) : 1;
  const unsigned TeamSize = is_cuda ? 256/VectorSize : 1;
  const unsigned RowBlockSize = 128;
  const unsigned m = A.dimension_0();
  const unsigned n = A.dimension_1();

  typedef GramianKernel<ExecSpace,ViewC,ViewA,ColBlockSize,RowBlockSize,TeamSize,VectorSize> Kernel;
  typedef typename Kernel::TeamMember TeamMember;

  deep_copy( C, 0.0 );
  Kokkos::parallel_for(Kernel::policy(m),
                       KOKKOS_LAMBDA(TeamMember team)
  {
    GramianKernel<ExecSpace,ViewC,ViewA,ColBlockSize,RowBlockSize,TeamSize,VectorSize> kernel(C, A, team);
    for (unsigned i_block=0; i_block<n; i_block+=ColBlockSize) {
      for (unsigned j_block=i_block; j_block<n; j_block+=ColBlockSize) {
        if (j_block+ColBlockSize <= n)
          kernel.template run<ColBlockSize>(i_block, j_block, ColBlockSize);
        else
          kernel.template run<0>(i_block, j_block, n-j_block);
      }
    }
  }, "Genten::FacMatrix::gramian_kernel");
}

template <typename ExecSpace, typename ViewC, typename ViewA>
void gramianImpl(const ViewC& C, const ViewA& A)
{
  const ttb_indx n = A.dimension_1();
  if (n < 2)
    gramian_kernel<ExecSpace,1>(C,A);
  else if (n < 4)
    gramian_kernel<ExecSpace,2>(C,A);
  else if (n < 8)
    gramian_kernel<ExecSpace,4>(C,A);
  else if (n < 16)
    gramian_kernel<ExecSpace,8>(C,A);
  else if (n < 32)
    gramian_kernel<ExecSpace,16>(C,A);
  else
    gramian_kernel<ExecSpace,32>(C,A);
}
#else
  // Gramian implementation using Kokkos
  template <typename ExecSpace, typename ViewC, typename ViewA>
  void gramianImpl(const ViewC& C, const ViewA& A)
  {
    typedef typename ExecSpace::size_type size_type;

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

    // Get the size of A
    const size_type m = A.dimension_0();
    const size_type n = A.dimension_1();

    // Use the largest power of 2 <= n, with a maximum of 16 for the
    // vector size.
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

    deep_copy( C, 0.0 );
    Kokkos::parallel_for(policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
                         KOKKOS_LAMBDA(typename Policy::member_type team)
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

            // Compute A(k,i)*A(k,j) for (i,j) in this block
            for (size_type ii=team_index; ii<BlockSize; ii+=team_size) {
              size_type i = i_block+ii;
              if (i >= n)
                break;

              const ttb_real A_ki = A(k,i);
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,BlockSize),
                                   [&] (const size_type& jj)
              {
                const size_type j = j_block+jj;
                if (j<n)
                  tmp(ii,jj) += A_ki*A(k,j);
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
                Kokkos::atomic_add(&C(i,j), tmp(ii,jj));
                if (j_block != i_block)
                  Kokkos::atomic_add(&C(j,i), tmp(ii,jj));
              }
            });
          }
        }
      }
    }, "Genten::FacMatrix::gramian_kernel");

    // Copy upper triangle into lower triangle
    // This seems to be the most efficient over adding the lower triangle in
    // the above kernel or launching a new one below.  However that may change
    // when n is large.
    // for (ttb_indx i=0; i<n; ++i)
    //   for (ttb_indx j=i+1; j<n; j++)
    //     C(j,i) = C(i,j);

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
    //     C(i+j,i) = C(i,i+j);
    //   });
    // });
  }
#endif
#endif

#if defined(KOKKOS_HAVE_CUDA) && defined(HAVE_CUBLAS)
  // Gramian implementation for CUDA and double precision using cuBLAS
  template <typename ExecSpace,
            typename CT, typename ... CP,
            typename AT, typename ... AP>
  typename std::enable_if<
    ( std::is_same<ExecSpace,Kokkos::Cuda>::value &&
      std::is_same<typename Kokkos::View<AT,AP...>::non_const_value_type,
                   double>::value )
    >::type
  gramianImpl(const Kokkos::View<CT,CP...>& C, const Kokkos::View<AT,AP...>& A)
  {
    const int m = A.dimension_0();
    const int n = A.dimension_1();
    const int lda = A.stride_0();
    const int ldc = C.stride_0();
    const double alpha = 1.0;
    const double beta = 0.0;
    cublasStatus_t status;

    // We compute C = A'*A.  But since A is LayoutRight, and GEMM
    // assumes layout left we compute this as C = A*A'

    static cublasHandle_t handle = 0;
    if (handle == 0) {
      status = cublasCreate(&handle);
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cublasCreate() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
    }

    // GEMM appears to be quite a bit faster than SYRK on the GPU
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, m,
                         &alpha, A.data(), lda, A.data(), lda,
                         &beta, C.data(), ldc);
    // status = cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, m,
    //                      &alpha, A.data(), lda, &beta, C.data(), ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::stringstream ss;
      ss << "Error!  cublasDgemm() failed with status "
         << status;
      std::cerr << ss.str() << std::endl;
      throw ss.str();
    }

    // Copy upper triangle into lower triangle when using SYRK
    // for (int i=0; i<n; ++i)
    //   for (int j=i+1; j<n; j++)
    //     C(j,i) = C(i,j);
  }

  // Gramian implementation for CUDA and single precision using cuBLAS
  template <typename ExecSpace,
            typename CT, typename ... CP,
            typename AT, typename ... AP>
  typename std::enable_if<
    ( std::is_same<ExecSpace,Kokkos::Cuda>::value &&
      std::is_same<typename Kokkos::View<AT,AP...>::non_const_value_type,
                   float>::value )
    >::type
  gramianImpl(const Kokkos::View<CT,CP...>& C, const Kokkos::View<AT,AP...>& A)
  {
    const int m = A.dimension_0();
    const int n = A.dimension_1();
    const int lda = A.stride_0();
    const int ldc = C.stride_0();
    const float alpha = 1.0;
    const float beta = 0.0;
    cublasStatus_t status;

    // We compute C = A'*A.  But since A is LayoutRight, and GEMM
    // assumes layout left we compute this as C = A*A'

    static cublasHandle_t handle = 0;
    if (handle == 0) {
      status = cublasCreate(&handle);
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cublasCreate() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
    }

    // GEMM appears to be quite a bit faster than SYRK on the GPU
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, m,
                         &alpha, A.data(), lda, A.data(), lda,
                         &beta, C.data(), ldc);
    // status = cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, m,
    //                      &alpha, A.data(), lda, &beta, C.data(), ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::stringstream ss;
      ss << "Error!  cublasDgemm() failed with status "
         << status;
      std::cerr << ss.str() << std::endl;
      throw ss.str();
    }

    // Copy upper triangle into lower triangle when using SYRK
    // for (int i=0; i<n; ++i)
    //   for (int j=i+1; j<n; j++)
    //     C(j,i) = C(i,j);
  }
#endif

}
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
gramian(const Genten::FacMatrixT<ExecSpace> & v) const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::FacMatrix::gramian");
#endif

  const ttb_indx m = v.data.dimension_0();
  const ttb_indx n = v.data.dimension_1();

  assert(data.dimension_0() == n);
  assert(data.dimension_1() == n);

  Genten::Impl::gramianImpl<ExecSpace>(data,v.data);
}

// return the index of the first entry, s, such that entry(s,c) > r.
// assumes/requires the values entry(s,c) are nondecreasing as s increases.
// if the assumption fails, result is undefined but <= nrows.
template <typename ExecSpace>
ttb_indx Genten::FacMatrixT<ExecSpace>::
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

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
oprod(const Genten::ArrayT<ExecSpace> & v) const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::FacMatrix::oprod");
#endif

  // TODO: Replace with call to BLAS2:DGER

  const ttb_indx n = v.size();
  // for (ttb_indx j = 0; j < n; j ++)
  // {
  //   for (ttb_indx i = 0; i < n; i ++)
  //   {
  //     this->entry(i,j) = v[i] * v[j];
  //   }
  // }
  view_type d = data;
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,n),
                       KOKKOS_LAMBDA(const ttb_indx i)
  {
    const ttb_real vi = v[i];
    for (ttb_indx j = 0; j < n; j ++)
      d(i,j) = vi * v[j];
  }, "Genten::FacMatrix::oprod_kernel");
}

#if USE_NEW_COLNORMS

namespace Genten {
namespace Impl {

template <typename ExecSpace, typename View, typename Norms,
          unsigned RowBlockSize, unsigned ColBlockSize,
          unsigned TeamSize, unsigned VectorSize>
struct ColNormsKernel {
  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  typedef Kokkos::View< ttb_real**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

  const View& data;
  const Norms& norms;
  const TeamMember team;
  const unsigned team_index;
  const unsigned team_size;
  TmpScratchSpace tmp;
  const unsigned i_block;
  const unsigned m;

  static inline Policy policy(const ttb_indx m) {
    const ttb_indx M = (m+RowBlockSize-1)/RowBlockSize;
    Policy policy(M,TeamSize,VectorSize);
    const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,ColBlockSize);
    return policy.set_scratch_size(0,Kokkos::PerTeam(bytes));
  }

  KOKKOS_INLINE_FUNCTION
  ColNormsKernel(const View& data_,
                 const Norms& norms_,
                 const TeamMember& team_) :
    data(data_),
    norms(norms_),
    team(team_), team_index(team.team_rank()), team_size(team.team_size()),
    tmp(team.team_scratch(0), TeamSize, ColBlockSize),
    i_block(team.league_rank()*RowBlockSize),
    m(data.dimension_0())
    {
      if (tmp.data() == 0)
        Kokkos::abort("ColNormsKernel:  Allocation of temp space failed.");
    }

};

template <typename ExecSpace, typename View, typename Norms,
          unsigned RowBlockSize, unsigned ColBlockSize,
          unsigned TeamSize, unsigned VectorSize>
struct ColNormsKernel_Inf
  : public ColNormsKernel<ExecSpace,View,Norms,RowBlockSize,ColBlockSize,
                          TeamSize,VectorSize>
{
  typedef ColNormsKernel<ExecSpace,View,Norms,RowBlockSize,ColBlockSize,
                         TeamSize,VectorSize> Base;

  using Base::ColNormsKernel;

  template <unsigned Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const unsigned j_block, const unsigned nj_)
  {
    // nj.value == Nj_ if Nj_ > 0 and nj_ otherwise
    Kokkos::Impl::integral_nonzero_constant<unsigned, Nj> nj(nj_);
    ttb_real *norms_j = &(this->norms[j_block]);
    ttb_real *tmp_i = &(this->tmp(this->team_index,0));

    this->team.team_barrier();

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(this->team,unsigned(nj.value)),
                         [&] (const unsigned& jj)
    {
      tmp_i[jj] = 0.0;
    });

    for (unsigned ii=this->team_index; ii<RowBlockSize; ii+=TeamSize) {
      const unsigned i = this->i_block+ii;
      if (i < this->m) {
        const ttb_real *data_ij = &(this->data(i,j_block));
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(this->team,unsigned(nj.value)),
                             [&] (const unsigned& jj)
        {
          auto d = std::fabs(data_ij[jj]);
          if (d > tmp_i[jj]) tmp_i[jj] = d;
        });
      }
    }

    this->team.team_barrier();

    if (this->team_index == 0) {
      for (unsigned ii=1; ii<TeamSize; ++ii) {
        ttb_real *tmp_ii = &(this->tmp(ii,0));
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(this->team,unsigned(nj.value)),
                             [&] (const unsigned& jj)
        {
          if (tmp_ii[jj] > tmp_i[jj]) tmp_i[jj] = tmp_ii[jj];
        });
      }

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(this->team,unsigned(nj.value)),
                           [&] (const unsigned& jj)
      {
        // Unfortunately there is no "atomic_max()", so we have to
        // implement it using compare_and_exchange
        ttb_real prev_val = norms_j[jj];
        ttb_real val = tmp_i[jj];
        bool keep_going = true;
        while (prev_val < val && keep_going) {

          // Replace norms[j] with val when norms[j] == prev_val
          ttb_real next_val =
            Kokkos::atomic_compare_exchange(&norms_j[jj], prev_val, val);

          // When prev_val == next_val, the exchange happened and we
          // can stop.  This saves an extra iteration
          keep_going = !(prev_val == next_val);
          prev_val = next_val;
        }
      });
    }

  }

};

template <typename ExecSpace, typename View, typename Norms,
          unsigned RowBlockSize, unsigned ColBlockSize,
          unsigned TeamSize, unsigned VectorSize>
struct ColNormsKernel_1
  : public ColNormsKernel<ExecSpace,View,Norms,RowBlockSize,ColBlockSize,
                          TeamSize,VectorSize>
{
  typedef ColNormsKernel<ExecSpace,View,Norms,RowBlockSize,ColBlockSize,
                         TeamSize,VectorSize> Base;

  using Base::ColNormsKernel;

  template <unsigned Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const unsigned j_block, const unsigned nj_)
  {
    // nj.value == Nj_ if Nj_ > 0 and nj_ otherwise
    Kokkos::Impl::integral_nonzero_constant<unsigned, Nj> nj(nj_);
    ttb_real *norms_j = &(this->norms[j_block]);
    ttb_real *tmp_i = &(this->tmp(this->team_index,0));

    this->team.team_barrier();

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(this->team,unsigned(nj.value)),
                         [&] (const unsigned& jj)
    {
      tmp_i[jj] = 0.0;
    });

    for (unsigned ii=this->team_index; ii<RowBlockSize; ii+=TeamSize) {
      const unsigned i = this->i_block+ii;
      if (i < this->m) {
        const ttb_real *data_ij = &(this->data(i,j_block));
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(this->team,unsigned(nj.value)),
                             [&] (const unsigned& jj)
        {
          tmp_i[jj] += std::fabs(data_ij[jj]);
        });
      }
    }

    this->team.team_barrier();

    if (this->team_index == 0) {
      for (unsigned ii=1; ii<TeamSize; ++ii) {
        ttb_real *tmp_ii = &(this->tmp(ii,0));
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(this->team,unsigned(nj.value)),
                             [&] (const unsigned& jj)
        {
          tmp_i[jj] += tmp_ii[jj];
        });
      }

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(this->team,unsigned(nj.value)),
                           [&] (const unsigned& jj)
      {
        Kokkos::atomic_add(norms_j+jj, tmp_i[jj]);
      });
    }

  }

};

template <typename ExecSpace, typename View, typename Norms,
          unsigned RowBlockSize, unsigned ColBlockSize,
          unsigned TeamSize, unsigned VectorSize>
struct ColNormsKernel_2
  : public ColNormsKernel<ExecSpace,View,Norms,RowBlockSize,ColBlockSize,
                          TeamSize,VectorSize>
{
  typedef ColNormsKernel<ExecSpace,View,Norms,RowBlockSize,ColBlockSize,
                         TeamSize,VectorSize> Base;

  using Base::ColNormsKernel;

  template <unsigned Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const unsigned j_block, const unsigned nj_)
  {
    // nj.value == Nj_ if Nj_ > 0 and nj_ otherwise
    Kokkos::Impl::integral_nonzero_constant<unsigned, Nj> nj(nj_);
    ttb_real *norms_j = &(this->norms[j_block]);
    ttb_real *tmp_i = &(this->tmp(this->team_index,0));

    this->team.team_barrier();

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(this->team,unsigned(nj.value)),
                         [&] (const unsigned& jj)
    {
      tmp_i[jj] = 0.0;
    });

    for (unsigned ii=this->team_index; ii<RowBlockSize; ii+=TeamSize) {
      const unsigned i = this->i_block+ii;
      if (i < this->m) {
        const ttb_real *data_ij = &(this->data(i,j_block));
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(this->team,unsigned(nj.value)),
                             [&] (const unsigned& jj)
        {
          auto d = data_ij[jj];
          tmp_i[jj] += d*d;
        });
      }
    }

    this->team.team_barrier();

    if (this->team_index == 0) {
      for (unsigned ii=1; ii<TeamSize; ++ii) {
        ttb_real *tmp_ii = &(this->tmp(ii,0));
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(this->team,unsigned(nj.value)),
                             [&] (const unsigned& jj)
        {
          tmp_i[jj] += tmp_ii[jj];
        });
      }

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(this->team,unsigned(nj.value)),
                           [&] (const unsigned& jj)
      {
        Kokkos::atomic_add(norms_j+jj, tmp_i[jj]);
      });
    }

  }

};

template <typename ExecSpace, unsigned ColBlockSize,
          typename ViewType, typename NormT>
void colNorms_kernel(
  const ViewType& data, Genten::NormType normtype,
  const NormT& norms, ttb_real minval)
{
  // Compute team and vector sizes, depending on the architecture
#if defined(KOKKOS_HAVE_CUDA)
  const bool is_cuda = std::is_same<ExecSpace,Kokkos::Cuda>::value;
#else
  const bool is_cuda = false;
#endif
  const unsigned VectorSize =
    is_cuda ? (ColBlockSize <= 32 ? ColBlockSize : 32) : 1;
  const unsigned TeamSize = is_cuda ? 256/VectorSize : 1;
  const unsigned RowBlockSize = 128;
  const unsigned m = data.dimension_0();
  const unsigned n = data.dimension_1();

  // Initialize norms to 0
  deep_copy(norms, 0.0);
  auto norms_host = create_mirror_view(norms);

  // Compute norms
  switch(normtype)
  {
  case Genten::NormInf:
  {
    typedef ColNormsKernel_Inf<ExecSpace,ViewType,NormT,RowBlockSize,ColBlockSize,TeamSize,VectorSize> Kernel;
    typedef typename Kernel::TeamMember TeamMember;
    Kokkos::parallel_for(Kernel::policy(m),
                         KOKKOS_LAMBDA(TeamMember team)
    {
      ColNormsKernel_Inf<ExecSpace,ViewType,NormT,RowBlockSize,ColBlockSize,TeamSize,VectorSize> kernel(data, norms, team);
      for (unsigned j_block=0; j_block<n; j_block+=ColBlockSize) {
        if (j_block+ColBlockSize <= n)
          kernel.template run<ColBlockSize>(j_block, ColBlockSize);
        else
          kernel.template run<0>(j_block, n-j_block);
      }
    }, "Genten::FacMatrix::colNorms_inf_kernel");
    deep_copy(norms_host, norms);
    break;
  }

  case Genten::NormOne:
  {
    typedef ColNormsKernel_1<ExecSpace,ViewType,NormT,RowBlockSize,ColBlockSize,TeamSize,VectorSize> Kernel;
    typedef typename Kernel::TeamMember TeamMember;
    Kokkos::parallel_for(Kernel::policy(m),
                         KOKKOS_LAMBDA(TeamMember team)
    {
      ColNormsKernel_1<ExecSpace,ViewType,NormT,RowBlockSize,ColBlockSize,TeamSize,VectorSize> kernel(data, norms, team);
      for (unsigned j_block=0; j_block<n; j_block+=ColBlockSize) {
        if (j_block+ColBlockSize <= n)
          kernel.template run<ColBlockSize>(j_block, ColBlockSize);
        else
          kernel.template run<0>(j_block, n-j_block);
      }
    }, "Genten::FacMatrix::colNorms_1_kernel");
    deep_copy(norms_host, norms);
    break;
  }
  case Genten::NormTwo:
  {
    typedef ColNormsKernel_2<ExecSpace,ViewType,NormT,RowBlockSize,ColBlockSize,TeamSize,VectorSize> Kernel;
    typedef typename Kernel::TeamMember TeamMember;
    Kokkos::parallel_for(Kernel::policy(m),
                         KOKKOS_LAMBDA(TeamMember team)
    {
      ColNormsKernel_2<ExecSpace,ViewType,NormT,RowBlockSize,ColBlockSize,TeamSize,VectorSize> kernel(data, norms, team);
      for (unsigned j_block=0; j_block<n; j_block+=ColBlockSize) {
        if (j_block+ColBlockSize <= n)
          kernel.template run<ColBlockSize>(j_block, ColBlockSize);
        else
          kernel.template run<0>(j_block, n-j_block);
      }
    }, "Genten::FacMatrix::colNorms_2_kernel");
    deep_copy(norms_host, norms);
    for (unsigned j=0; j<n; ++j)
      norms_host[j] = std::sqrt(norms_host[j]);

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
    for (unsigned j=0; j<n; ++j)
    {
      if (norms_host[j] < minval)
      {
        norms_host[j] = minval;
      }
    }
  }
  deep_copy(norms, norms_host);
}

}
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
colNorms(Genten::NormType normtype, Genten::ArrayT<ExecSpace> & norms, ttb_real minval) const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::FacMatrix::colNorms");
#endif

  const ttb_indx nc = data.dimension_1();
  if (nc < 2)
    Impl::colNorms_kernel<ExecSpace,1>(data, normtype, norms.values(), minval);
  else if (nc < 4)
    Impl::colNorms_kernel<ExecSpace,2>(data, normtype, norms.values(), minval);
  else if (nc < 8)
    Impl::colNorms_kernel<ExecSpace,4>(data, normtype, norms.values(), minval);
  else if (nc < 16)
    Impl::colNorms_kernel<ExecSpace,8>(data, normtype, norms.values(), minval);
  else if (nc < 32)
    Impl::colNorms_kernel<ExecSpace,16>(data, normtype, norms.values(), minval);
  else
    Impl::colNorms_kernel<ExecSpace,32>(data, normtype, norms.values(), minval);
}

#else

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
colNorms(Genten::NormType normtype, Genten::ArrayT<ExecSpace> & norms, ttb_real minval) const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::FacMatrix::colNorms");
#endif

  typedef typename ExecSpace::size_type size_type;

  const size_type m = data.dimension_0();
  const size_type n = data.dimension_1();

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
                         KOKKOS_LAMBDA(typename Policy::member_type team)
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
    }, "Genten::FacMatrix::colNorms_inf_kernel");
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
                         KOKKOS_LAMBDA(typename Policy::member_type team)
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
    }, "Genten::FacMatrix::colNorms_1_kernel");
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
                         KOKKOS_LAMBDA(typename Policy::member_type team)
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
    }, "Genten::FacMatrix::colNorms_2_kernel");
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

#endif

#if USE_NEW_COLSCALE
namespace Genten {
namespace Impl {

template <typename ExecSpace, typename View,
          unsigned ColBlockSize, unsigned RowBlockSize,
          unsigned TeamSize, unsigned VectorSize>
struct ColScaleKernel {
  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;

  const View& data;
  const Genten::ArrayT<ExecSpace>& v;
  const TeamMember& team;
  const unsigned team_index;
  const unsigned i_block;
  const unsigned m;

  static inline Policy policy(const ttb_indx m_) {
    const unsigned M = (m_+RowBlockSize-1)/RowBlockSize;
    Policy policy(M,TeamSize,VectorSize);
    return policy;
  }

  KOKKOS_INLINE_FUNCTION
  ColScaleKernel(const View& data_,
                 const Genten::ArrayT<ExecSpace>& v_,
                 const TeamMember& team_) :
    data(data_),
    v(v_),
    team(team_),
    team_index(team.team_rank()),
    i_block(team.league_rank()*RowBlockSize),
    m(data.dimension_0())
    {}

  template <unsigned Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const unsigned j, const unsigned nj_)
  {
    // nj.value == Nj_ if Nj_ > 0 and nj_ otherwise
    Kokkos::Impl::integral_nonzero_constant<unsigned, Nj> nj(nj_);

    const ttb_real *v_j = &v[j];

    for (unsigned ii=team_index; ii<RowBlockSize; ii+=TeamSize) {
      const unsigned i = i_block+ii;
      if (i < m) {
        ttb_real* data_i = &data(i,j);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,unsigned(nj.value)),
                             [&] (const unsigned& jj)
        {
          data_i[jj] *= v_j[jj];
        });
      }
    }
  }
};

template <typename ExecSpace, unsigned ColBlockSize, typename ViewType>
void colScale_kernel(const ViewType& data, const Genten::ArrayT<ExecSpace>& v)
{
  // Compute team and vector sizes, depending on the architecture
#if defined(KOKKOS_HAVE_CUDA)
  const bool is_cuda = std::is_same<ExecSpace,Kokkos::Cuda>::value;
#else
  const bool is_cuda = false;
#endif
  const unsigned VectorSize =
     is_cuda ? (ColBlockSize <= 32 ? ColBlockSize : 32) : 1;
  const unsigned TeamSize = is_cuda ? 256/VectorSize : 1;
  const unsigned RowBlockSize = 128;
  const unsigned m = data.dimension_0();
  const unsigned n = data.dimension_1();

  typedef ColScaleKernel<ExecSpace,ViewType,ColBlockSize,RowBlockSize,TeamSize,VectorSize> Kernel;
  typedef typename Kernel::TeamMember TeamMember;

  Kokkos::parallel_for(Kernel::policy(m), KOKKOS_LAMBDA(TeamMember team)
  {
    ColScaleKernel<ExecSpace,ViewType,ColBlockSize,RowBlockSize,TeamSize,VectorSize> kernel(data, v, team);
    for (unsigned j_block=0; j_block<n; j_block+=ColBlockSize) {
      if (j_block+ColBlockSize <= n)
        kernel.template run<ColBlockSize>(j_block, ColBlockSize);
      else
        kernel.template run<0>(j_block, n-j_block);
    }
  }, "Genten::FacMatrix::colScale_kernel");
}

}
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
colScale(const Genten::ArrayT<ExecSpace> & v, bool inverse) const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::FacMatrix::colScale");
#endif

  const ttb_indx n = data.dimension_1();
  assert(v.size() == n);

  Genten::ArrayT<ExecSpace> w;
  if (inverse) {
    w = Genten::ArrayT<ExecSpace>(n);
    auto v_host = create_mirror_view(v);
    auto w_host = create_mirror_view(w);
    deep_copy(v_host, v);
    for (ttb_indx j = 0; j < n; j ++) {
      if (v_host[j] == 0)
        Genten::error("Genten::FacMatrix::colScale - divide-by-zero error");
      w_host[j] = 1.0 / v_host[j];
    }
    deep_copy(w, w_host);
  }
  else
    w = v;

  if (n < 2)
    Impl::colScale_kernel<ExecSpace,1>(data, w);
  else if (n < 4)
    Impl::colScale_kernel<ExecSpace,2>(data, w);
  else if (n < 8)
    Impl::colScale_kernel<ExecSpace,4>(data, w);
  else if (n < 16)
    Impl::colScale_kernel<ExecSpace,8>(data, w);
  else if (n < 32)
    Impl::colScale_kernel<ExecSpace,16>(data, w);
  else if (n < 64)
    Impl::colScale_kernel<ExecSpace,32>(data, w);
  else
    Impl::colScale_kernel<ExecSpace,64>(data, w);
}
#else
template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
colScale(const Genten::ArrayT<ExecSpace> & v, bool inverse) const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::FacMatrix::colScale");
#endif

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
  const size_type RowBlockSize = 128;

  size_type M = (m+RowBlockSize-1)/RowBlockSize;
  typedef Kokkos::TeamPolicy <ExecSpace> Policy;
  Policy policy(M,TeamSize,VectorSize);

  // Can't capture "this" pointer by value
  view_type my_data = data;

  if (inverse) {
    for (ttb_indx j = 0; j < n; j ++)
      if (v[j] == 0)
        Genten::error("Genten::FacMatrix::colScale - divide-by-zero error");

    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(typename Policy::member_type team)
    {
      const size_type team_size = team.team_size();
      const size_type team_index = team.team_rank();
      const size_type i_block = team.league_rank()*RowBlockSize;

      for (size_type ii=team_index; ii<RowBlockSize; ii+=team_size) {
        const size_type i = i_block+ii;
        if (i < m) {
          ttb_real *my_data_i = &(my_data(i,0));
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,n),
                               [&] (const size_type& j)
          {
            my_data_i[j] /= v[j];
          });
        }
      }
    }, "Genten::FacMatrix::colScale_kernel");
  }
  else {
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(typename Policy::member_type team)
    {
      const size_type team_size = team.team_size();
      const size_type team_index = team.team_rank();
      const size_type i_block = team.league_rank()*RowBlockSize;

      for (size_type ii=team_index; ii<RowBlockSize; ii+=team_size) {
        const size_type i = i_block+ii;
        if (i < m) {
          ttb_real *my_data_i = &(my_data(i,0));
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,n),
                               [&] (const size_type& j)
          {
            my_data_i[j] *= v[j];
          });
        }
      }
    }, "Genten::FacMatrix::colScale_kernel");
  }
}
#endif

// Only called by Ben Allan's parallel test code.  It appears he uses the Linux
// random number generator in a special way.
#if !defined(_WIN32)
template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
scaleRandomElements(ttb_real fraction, ttb_real scale, bool columnwise) const
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
    Genten::IndxArrayT<ExecSpace> marked(tot,(ttb_indx)0);
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
template <typename ExecSpace>
ttb_real Genten::FacMatrixT<ExecSpace>::
sum() const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::FacMatrix::sum");
#endif

  const ttb_indx nrows = data.dimension_0();
  const ttb_indx ncols = data.dimension_1();

  ttb_real sum = 0;
  // for (ttb_indx i=0; i<nrows; ++i)
  //   for (ttb_indx j=0; j<ncols; ++j)
  //     sum += data(i,j);
  view_type d = data;
  Kokkos::parallel_reduce("Genten::FacMatrix::sum_kernel",
                          Kokkos::RangePolicy<ExecSpace>(0,nrows),
                          KOKKOS_LAMBDA(const ttb_indx i, ttb_real& s)
  {
    for (ttb_indx j=0; j<ncols; ++j)
      s += d(i,j);
  }, sum);

  return sum;
}

template <typename ExecSpace>
bool Genten::FacMatrixT<ExecSpace>::
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

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
permute(const Genten::IndxArray& perm_indices) const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::FacMatrix::permute");
#endif

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
  //Genten::ArrayT<ExecSpace> temp(nrows);

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

      /*
      // Move data in column loc to temp column.
      for (ttb_indx i=0; i<nrows; i++)
        temp[i] = this->entry(i,loc);

      // Move data in column j to column loc.
      for (ttb_indx i=0; i<nrows; i++)
        this->entry(i,loc) = this->entry(i,j);

      // Move temp column to column loc.
      for (ttb_indx i=0; i<nrows; i++)
        this->entry(i,j) = temp[i];
      */
      // ETP 10/25/17:  Note this yields strided loads/stores, so it would
      // be much more efficient to move the j-loop inside the parallel_for
      // and use vector parallelism
      view_type d = data;
      Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,nrows),
                           KOKKOS_LAMBDA(const ttb_indx i)
      {
        const ttb_real temp = d(i,loc);
        d(i,loc) = d(i,j);
        d(i,j) = temp;
      }, "Genten::FacMatrix::permute_kernel");

      // Swap curr_indices to mark where data was moved.
      ttb_indx  k = curr_indices[j];
      curr_indices[j] = curr_indices[loc];
      curr_indices[loc] = k;
    }
  }
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
multByVector(bool bTranspose,
             const Genten::ArrayT<ExecSpace> &  x,
             Genten::ArrayT<ExecSpace> &  y) const
{
  const ttb_indx nrows = data.dimension_0();
  const ttb_indx ncols = data.dimension_1();

  if (bTranspose == false)
  {
    assert(x.size() == ncols);
    assert(y.size() == nrows);
    // Data for the matrix is stored in row-major order but gemv expects
    // column-major, so tell it transpose dimensions.
    Genten::gemv('T', ncols, nrows, 1.0, ptr(), data.stride_0(),
              x.ptr(), 1, 0.0, y.ptr(), 1);
  }
  else
  {
    assert(x.size() == nrows);
    assert(y.size() == ncols);
    // Data for the matrix is stored in row-major order but gemv expects
    // column-major, so tell it transpose dimensions.
    Genten::gemv('N', ncols, nrows, 1.0, ptr(), data.stride_0(),
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
      Genten::gesv (ncols, nrows, A.data(), A.stride_0(), B.data(), B.stride_0());
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

      // lwork is a host pointer, but info is a device pointer.  Go figure.
      int lwork = 0;
      status = cusolverDnDgetrf_bufferSize(handle, n, n, A.data(), lda, &lwork);
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDgetrf_bufferSize() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      Kokkos::View<double*,Kokkos::LayoutRight,Kokkos::Cuda> work("work",lwork);
      Kokkos::View<int*,Kokkos::LayoutRight,Kokkos::Cuda> piv("piv",n);
      Kokkos::View<int,Kokkos::LayoutRight,Kokkos::Cuda> info("info");
      status = cusolverDnDgetrf(handle, n, n, A.data(), lda, work.data(),
                                piv.data(), info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDgetrf() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      auto info_host = create_mirror_view(info);
      deep_copy(info_host, info);
      if (info_host() != 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDngetrf() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      status = cusolverDnDgetrs(handle, CUBLAS_OP_N, n, m, A.data(), lda,
                                piv.data(), B.data(), ldb, info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDgetrs() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      deep_copy(info_host, info);
      if (info_host() != 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDngetrf() info =  " << info_host();
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

      // lwork is a host pointer, but info is a device pointer.  Go figure.
      int lwork = 0;
      status = cusolverDnSgetrf_bufferSize(handle, n, n, A.data(), lda, &lwork);
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDgetrf_bufferSize() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      Kokkos::View<float*,Kokkos::LayoutRight,Kokkos::Cuda> work("work",lwork);
      Kokkos::View<int*,Kokkos::LayoutRight,Kokkos::Cuda> piv("piv",n);
      Kokkos::View<int,Kokkos::LayoutRight,Kokkos::Cuda> info("info");
      status = cusolverDnSgetrf(handle, n, n, A.data(), lda, work.data(),
                                piv.data(), info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDgetrf() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      auto info_host = create_mirror_view(info);
      deep_copy(info_host, info);
      if (info_host() != 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDngetrf() info =  " << info_host();
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
      deep_copy(info_host, info);
      if (info_host() != 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDngetrf() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
    }
#endif

  }
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
solveTransposeRHS (const Genten::FacMatrixT<ExecSpace> &  A) const
{
  const ttb_indx nrows = data.dimension_0();
  const ttb_indx ncols = data.dimension_1();

  assert(A.nRows() == A.nCols());
  assert(nCols() == A.nRows());

  // Copy A because LAPACK needs to overwrite the invertible square matrix.
  // Since the matrix is assumed symmetric, no need to worry about row major
  // versus column major storage.
  view_type Atmp("Atmp", A.nRows(), A.nCols());
  deep_copy(Atmp, A.data);

  Genten::Impl::solveTransposeRHSImpl(Atmp, data);

  return;
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
rowTimes(Genten::ArrayT<ExecSpace> & x,
         const ttb_indx nRow) const
{
  assert(x.size() == data.dimension_1());

  const ttb_real * rptr = this->rowptr(nRow);
  ttb_real * xptr = x.ptr();

  vmul(data.dimension_1(),xptr,rptr);

  return;
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
rowTimes(const ttb_indx         nRow,
         const Genten::FacMatrixT<ExecSpace> & other,
         const ttb_indx         nRowOther) const
{
  assert(other.nCols() == data.dimension_1());

  ttb_real * rowPtr1 = this->rowptr(nRow);
  const ttb_real * rowPtr2 = other.rowptr(nRowOther);

  vmul(data.dimension_1(), rowPtr1, rowPtr2);

  return;
}

template <typename ExecSpace>
ttb_real Genten::FacMatrixT<ExecSpace>::
rowDot(const ttb_indx         nRow,
       const Genten::FacMatrixT<ExecSpace> & other,
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

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
rowDScale(const ttb_indx         nRow,
          Genten::FacMatrixT<ExecSpace> & other,
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

#define INST_MACRO(SPACE) template class Genten::FacMatrixT<SPACE>;
GENTEN_INST(INST_MACRO)
