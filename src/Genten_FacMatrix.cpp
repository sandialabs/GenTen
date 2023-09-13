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
#include <cstring>

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

template <typename ExecSpace>
Genten::FacMatrixT<ExecSpace>::
FacMatrixT(ttb_indx m, ttb_indx n, const ProcessorMap::FacMap* pmap_,
           const bool zero) :
  pmap(pmap_)
{
  // Don't use padding if Cuda, HIP or SYCL is the default execution space, so factor
  // matrices allocated on the host have the same shape.  We really need a
  // better way to do this.
  if (Genten::is_gpu_space<DefaultExecutionSpace>::value) {
    if (zero)
      data = view_type("Genten::FacMatrix::data",m,n);
    else
      data = view_type(Kokkos::view_alloc("Genten::FacMatrix::data",
                                          Kokkos::WithoutInitializing),m,n);
  }
  else {
    if (zero)
      data = view_type(Kokkos::view_alloc("Genten::FacMatrix::data",
                                          Kokkos::AllowPadding),m,n);
    else
      data = view_type(Kokkos::view_alloc("Genten::FacMatrix::data",
                                          Kokkos::WithoutInitializing,
                                          Kokkos::AllowPadding),m,n);
  }
}

template <typename ExecSpace>
Genten::FacMatrixT<ExecSpace>::
FacMatrixT(ttb_indx m, ttb_indx n, const ttb_real * cvec,
           const ProcessorMap::FacMap* pmap_) : pmap(pmap_)
{
  // Don't use padding if Cuda, HIP or SYCL is the default execution space, so factor
  // matrices allocated on the host have the same shape.  We really need a
  // better way to do this.
  if (Genten::is_gpu_space<DefaultExecutionSpace>::value)
    data = view_type(Kokkos::view_alloc("Genten::FacMatrix::data",
                                        Kokkos::WithoutInitializing),m,n);
  else
    data = view_type(Kokkos::view_alloc("Genten::FacMatrix::data",
                                        Kokkos::WithoutInitializing,
                                        Kokkos::AllowPadding),m,n);
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
         const bool bUseParallelRNG,
         RandomMT &  cRMT) const
{
  auto data_1d = make_data_1d();
  data_1d.scatter (bUseMatlabRNG, bUseParallelRNG, cRMT);
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
convertFromCol(ttb_indx nr, ttb_indx nc, const ttb_real * cvec) const
{
  const ttb_indx nrows = data.extent(0);
  const ttb_indx ncols = data.extent(1);
  view_type my_data = data;
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,nrows),
                       KOKKOS_LAMBDA(const ttb_indx i)
  {
    for (ttb_indx  j = 0; j < ncols; j++)
    {
      my_data(i,j) = cvec[i + j * nrows];
    }
  });
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
convertToCol(ttb_indx nr, ttb_indx nc, ttb_real * cvec) const
{
  const ttb_indx nrows = data.extent(0);
  const ttb_indx ncols = data.extent(1);
  view_type my_data = data;
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,nrows),
                       KOKKOS_LAMBDA(const ttb_indx i)
  {
    for (ttb_indx  j = 0; j < ncols; j++)
    {
      cvec[i + j * nrows] = my_data(i,j);
    }
  });
}

template <typename ExecSpace>
bool Genten::FacMatrixT<ExecSpace>::
isEqual(const Genten::FacMatrixT<ExecSpace> & b, ttb_real tol) const
{
  // Check for equal sizes first
  if ((data.extent(0) != b.data.extent(0)) ||
      (data.extent(1) != b.data.extent(1)))
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
  if ((v.data.extent(0) != data.extent(0)) ||
      (v.data.extent(1) != data.extent(1)))
  {
    error("Genten::FacMatrix::hadamard - size mismatch");
  }
  auto data_1d = make_data_1d();
  auto v_data_1d = v.make_data_1d();
  data_1d.times(v_data_1d);
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
plus(const Genten::FacMatrixT<ExecSpace> & y, const ttb_real s) const
{
  if (data.span() == y.data.span()) { // matrices have the same padding
    auto data_1d = make_data_1d();
    auto y_data_1d = y.make_data_1d();
    data_1d.plus(y_data_1d, s);
  }
  else { // matrices might not have the same padding
    gt_assert(data.extent(0) == y.data.extent(0));
    gt_assert(data.extent(1) == y.data.extent(1));
    auto d = data;
    Kokkos::parallel_for("FacMatrix::plus",
                         Kokkos::RangePolicy<ExecSpace>(0,d.extent(0)),
                         KOKKOS_LAMBDA(const ttb_indx i)
    {
      for (ttb_indx j=0; j<d.extent(1); ++j)
        d(i,j) += s*y.data(i,j);
    });
  }
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
update(const ttb_real a, const Genten::FacMatrixT<ExecSpace> & y,
       const ttb_real b) const
{
  if (data.span() == y.data.span()) { // matrices have the same padding
    auto data_1d = make_data_1d();
    auto y_data_1d = y.make_data_1d();
    data_1d.update(a, y_data_1d, b);
  }
  else { // matrices might not have the same padding
    gt_assert(data.extent(0) == y.data.extent(0));
    gt_assert(data.extent(1) == y.data.extent(1));
    auto d = data;
    Kokkos::parallel_for("FacMatrix::update",
                         Kokkos::RangePolicy<ExecSpace>(0,d.extent(0)),
                         KOKKOS_LAMBDA(const ttb_indx i)
    {
      for (ttb_indx j=0; j<d.extent(1); ++j)
        d(i,j) = a*y.data(i,j) + b*d(i,j);
    });
  }
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
  const ttb_indx nrows = data.extent(0);
  const ttb_indx ncols = data.extent(1);

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

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
plus(ttb_real a) const
{
  auto data_1d = make_data_1d();
  data_1d.shift(a);
}

namespace Genten {
namespace Impl {

#if defined(LAPACK_FOUND)

// Gramian implementation using gemm()
template <typename ExecSpace, typename ViewC, typename ViewA>
void gramianImpl(const ViewC& C, const ViewA& A, const bool full,
                 const UploType uplo)
{
  const ttb_indx m = A.extent(0);
  const ttb_indx n = A.extent(1);
  const ttb_indx lda = A.stride_0();
  const ttb_indx ldc = C.stride_0();

  // We compute C = A'*A.  But since A is LayoutRight, and GEMM/SYRK
  // assumes layout left we compute this as C' = A*A'.  Since SYRK writes
  // C', uplo == Upper means we call SYRK with 'L', and vice versa.

  //Genten::gemm('N','T',n,n,m,1.0,A.data(),lda,A.data(),lda,0.0,C.data(),ldc);
  if (uplo == Upper) {
    Genten::syrk('L','N',n,m,1.0,A.data(),lda,0.0,C.data(),ldc);
    if (full) {
      for (ttb_indx i=0; i<n; ++i)
        for (ttb_indx j=i+1; j<n; j++)
          C(j,i) = C(i,j);
    }
  }
  else {
    Genten::syrk('U','N',n,m,1.0,A.data(),lda,0.0,C.data(),ldc);
    if (full) {
      for (ttb_indx i=0; i<n; ++i)
        for (ttb_indx j=0; j<i; j++)
          C(j,i) = C(i,j);
    }
  }
}

#else

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
    m(A.extent(0)),
    n(A.extent(1))
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
  const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
  const unsigned VectorSize =
    is_gpu ? (ColBlockSize <= 16 ? ColBlockSize : 16) : 1;
  const unsigned TeamSize = is_gpu ? 256/VectorSize : 1;
  const unsigned RowBlockSize = 128;
  const unsigned m = A.extent(0);
  const unsigned n = A.extent(1);

  typedef GramianKernel<ExecSpace,ViewC,ViewA,ColBlockSize,RowBlockSize,TeamSize,VectorSize> Kernel;
  typedef typename Kernel::TeamMember TeamMember;

  deep_copy( C, 0.0 );
  Kokkos::parallel_for("Genten::FacMatrix::gramian_kernel",
                       Kernel::policy(m),
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
  });
}

template <typename ExecSpace, typename ViewC, typename ViewA>
void gramianImpl(const ViewC& C, const ViewA& A,
                 const bool full, const UploType uplo)
{
  // Currently ignoring full/uplo since this kernel is just a backup anyway
  const ttb_indx n = A.extent(1);
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

#endif

#if defined(KOKKOS_ENABLE_CUDA)

  // Gramian implementation for CUDA and double precision using cuBLAS
  template <typename ExecSpace,
            typename CT, typename ... CP,
            typename AT, typename ... AP>
  std::enable_if_t<
    (is_cuda_space<ExecSpace>::value || is_sycl_space<ExecSpace>::value) &&
    std::is_same<typename Kokkos::View<AT, AP...>::non_const_value_type, double>::value
  >
  gramianImpl(const Kokkos::View<CT,CP...>& C, const Kokkos::View<AT,AP...>& A,
              const bool full, const UploType uplo)
  {
    const int m = A.extent(0);
    const int n = A.extent(1);
    const int lda = A.stride_0();
    const int ldc = C.stride_0();
    const double alpha = 1.0;
    const double beta = 0.0;
    cublasStatus_t status;

    // We compute C = A'*A.  But since A is LayoutRight, and GEMM/SYRK
    // assumes layout left we compute this as C = A*A'.  Since SYRK writes
    // C', uplo == Upper means we call SYRK with 'L', and vice versa.

    if (full) {
      status = cublasDgemm(CublasHandle::get(), CUBLAS_OP_N, CUBLAS_OP_T, n, n, m,
                           &alpha, A.data(), lda, A.data(), lda,
                           &beta, C.data(), ldc);
    }
    else {
      cublasFillMode_t cu_uplo =
        uplo == Upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
      status = cublasDsyrk(CublasHandle::get(), cu_uplo, CUBLAS_OP_N, n, m,
                           &alpha, A.data(), lda, &beta, C.data(), ldc);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::stringstream ss;
      ss << "Error!  cublasDgemm()/cublasDsyrk() failed with status "
         << status;
      std::cerr << ss.str() << std::endl;
      throw ss.str();
    }
  }

  // Gramian implementation for CUDA and single precision using cuBLAS
  template <typename ExecSpace,
            typename CT, typename ... CP,
            typename AT, typename ... AP>
  std::enable_if_t<
    (is_cuda_space<ExecSpace>::value || is_sycl_space<ExecSpace>::value) &&
    std::is_same<typename Kokkos::View<AT, AP...>::non_const_value_type, float>::value
  >
  gramianImpl(const Kokkos::View<CT,CP...>& C, const Kokkos::View<AT,AP...>& A,
              const bool full, const UploType uplo)
  {
    const int m = A.extent(0);
    const int n = A.extent(1);
    const int lda = A.stride_0();
    const int ldc = C.stride_0();
    const float alpha = 1.0;
    const float beta = 0.0;
    cublasStatus_t status;

    // We compute C = A'*A.  But since A is LayoutRight, and GEMM/SYRK
    // assumes layout left we compute this as C = A*A'.  Since SYRK writes
    // C', uplo == Upper means we call SYRK with 'L', and vice versa.

    if (full) {
      status = cublasSgemm(CublasHandle::get(), CUBLAS_OP_N, CUBLAS_OP_T, n, n, m,
                           &alpha, A.data(), lda, A.data(), lda,
                           &beta, C.data(), ldc);
    }
    else {
      cublasFillMode_t cu_uplo =
        uplo == Upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
      status = cublasSsyrk(CublasHandle::get(), cu_uplo, CUBLAS_OP_N, n, m,
                           &alpha, A.data(), lda, &beta, C.data(), ldc);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::stringstream ss;
      ss << "Error!  cublasSgemm()/cublasSsyrk() failed with status "
         << status;
      std::cerr << ss.str() << std::endl;
      throw ss.str();
    }
  }

#endif

#if defined(KOKKOS_ENABLE_HIP)

  // Gramian implementation for HIP and double precision using rocBLAS
  template <typename ExecSpace,
            typename CT, typename ... CP,
            typename AT, typename ... AP>
  std::enable_if_t<
    is_hip_space<ExecSpace>::value &&
    std::is_same<typename Kokkos::View<AT, AP...>::non_const_value_type, double>::value
  >
  gramianImpl(const Kokkos::View<CT,CP...>& C, const Kokkos::View<AT,AP...>& A,
              const bool full, const UploType uplo)
  {
    const int m = A.extent(0);
    const int n = A.extent(1);
    const int lda = A.stride_0();
    const int ldc = C.stride_0();
    const double alpha = 1.0;
    const double beta = 0.0;
    rocblas_status status;

    // We compute C = A'*A.  But since A is LayoutRight, and GEMM/SYRK
    // assumes layout left we compute this as C = A*A'.  Since SYRK writes
    // C', uplo == Upper means we call SYRK with 'L', and vice versa.

    if (full) {
      status = rocblas_dgemm(RocblasHandle::get(), rocblas_operation_none, rocblas_operation_transpose, n, n, m,
                           &alpha, A.data(), lda, A.data(), lda,
                           &beta, C.data(), ldc);
    }
    else {
      rocblas_fill roc_uplo =
        uplo == Upper ? rocblas_fill_lower : rocblas_fill_upper;
      status = rocblas_dsyrk(RocblasHandle::get(), roc_uplo, rocblas_operation_none, n, m,
                           &alpha, A.data(), lda, &beta, C.data(), ldc);
    }

    if (status != rocblas_status_success) {
      std::stringstream ss;
      ss << "Error!  rocblas_dgemm()/rocblas_dsyrk() failed with status "
         << status;
      std::cerr << ss.str() << std::endl;
      throw ss.str();
    }
  }

  // Gramian implementation for HIP and single precision using rocBLAS
  template <typename ExecSpace,
            typename CT, typename ... CP,
            typename AT, typename ... AP>
  typename std::enable_if<
    is_hip_space<ExecSpace>::value &&
    std::is_same<typename Kokkos::View<AT, AP...>::non_const_value_type, float>::value
  >::type
  gramianImpl(const Kokkos::View<CT,CP...>& C, const Kokkos::View<AT,AP...>& A,
              const bool full, const UploType uplo)
  {
    const int m = A.extent(0);
    const int n = A.extent(1);
    const int lda = A.stride_0();
    const int ldc = C.stride_0();
    const float alpha = 1.0;
    const float beta = 0.0;
    rocblas_status status;

    // We compute C = A'*A.  But since A is LayoutRight, and GEMM/SYRK
    // assumes layout left we compute this as C = A*A'.  Since SYRK writes
    // C', uplo == Upper means we call SYRK with 'L', and vice versa.

    if (full) {
      status = rocblas_sgemm(RocblasHandle::get(), rocblas_operation_none, rocblas_operation_transpose, n, n, m,
                           &alpha, A.data(), lda, A.data(), lda,
                           &beta, C.data(), ldc);
    }
    else {
      rocblas_fill roc_uplo =
        uplo == Upper ? rocblas_fill_lower : rocblas_fill_upper;
      status = rocblas_ssyrk(RocblasHandle::get(), roc_uplo, rocblas_operation_none, n, m,
                           &alpha, A.data(), lda, &beta, C.data(), ldc);
    }

    if (status != rocblas_status_success) {
      std::stringstream ss;
      ss << "Error!  rocblas_sgemm()/rocblas_ssyrk() failed with status "
         << status;
      std::cerr << ss.str() << std::endl;
      throw ss.str();
    }
  }

#endif

#if defined(KOKKOS_ENABLE_SYCL)

  // Gramian implementation for CUDA and double precision using cuBLAS
  template <typename ExecSpace,
            typename CT, typename ... CP,
            typename AT, typename ... AP>
  std::enable_if_t< is_sycl_space<ExecSpace>::value >
  gramianImpl(const Kokkos::View<CT,CP...>& C, const Kokkos::View<AT,AP...>& A,
              const bool full, const UploType uplo)
  {
    if (full)
      Genten::gemm(true, false, 1.0, A, A, 0.0, C);
    else {
      // We compute C = A'*A.  But since A is LayoutRight, and GEMM/SYRK
      // assumes layout left we compute this as C = A*A'.  Since SYRK writes
      // C', uplo == Upper means we call SYRK with 'L', and vice versa.
      const std::int64_t m = A.extent(0);
      const std::int64_t n = A.extent(1);
      const std::int64_t lda = A.stride_0();
      const std::int64_t ldc = C.stride_0();
      const ttb_real alpha = 1.0;
      const ttb_real beta = 0.0;
      oneapi::mkl::uplo mkl_uplo =
        uplo == Upper ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper;
      ExecSpace space;
      sycl::queue& q = space.sycl_queue();
      oneapi::mkl::blas::column_major::syrk(
        q, mkl_uplo, oneapi::mkl::transpose::nontrans, n, m, alpha,
        A.data(), lda, beta, C.data(), ldc);
    }
  }

#endif

}
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
gramian(const Genten::FacMatrixT<ExecSpace> & v, const bool full,
        const UploType uplo) const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::FacMatrix::gramian");
#endif

  gt_assert(data.extent(0) == v.data.extent(1));
  gt_assert(data.extent(1) == v.data.extent(1));

  Genten::Impl::gramianImpl<ExecSpace>(data,v.data,full,uplo);

  // We check if v is distributed, not *this, because the result of the gramian
  // should be replicated, not distributed
  if (v.pmap != nullptr) {
    Kokkos::fence();
    v.pmap->allReduce(data.data(), data.span());
  }
}

// return the index of the first entry, s, such that entry(s,c) > r.
// assumes/requires the values entry(s,c) are nondecreasing as s increases.
// if the assumption fails, result is undefined but <= nrows.
template <typename ExecSpace>
ttb_indx Genten::FacMatrixT<ExecSpace>::
firstGreaterSortedIncreasing(ttb_real r, ttb_indx c) const
{
  const ttb_indx nrows = data.extent(0);
  const ttb_indx ncols = data.extent(1);

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

  const ttb_indx n = v.size();
  gt_assert(data.extent(0) == n);
  gt_assert(data.extent(1) == n);

  view_type d = data;
  Kokkos::parallel_for("Genten::FacMatrix::oprod_kernel",
                       Kokkos::RangePolicy<ExecSpace>(0,n),
                       KOKKOS_LAMBDA(const ttb_indx i)
  {
    const ttb_real vi = v[i];
    for (ttb_indx j=0; j<n; ++j)
      d(i,j) = vi * v[j];
  });
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
oprod(const Genten::ArrayT<ExecSpace> & v1,
      const Genten::ArrayT<ExecSpace> & v2) const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::FacMatrix::oprod");
#endif

  const ttb_indx m = v1.size();
  const ttb_indx n = v2.size();
  gt_assert(data.extent(0) == m);
  gt_assert(data.extent(1) == n);

  view_type d = data;
  Kokkos::parallel_for("Genten::FacMatrix::oprod_kernel",
                       Kokkos::RangePolicy<ExecSpace>(0,m),
                       KOKKOS_LAMBDA(const ttb_indx i)
  {
    const ttb_real vi = v1[i];
    for (ttb_indx j=0; j<n; ++j)
      d(i,j) = vi * v2[j];
  });
}

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
    m(data.extent(0))
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
  const NormT& norms, ttb_real minval,
  const ProcessorMap::FacMap* pmap = nullptr)
{
  // Compute team and vector sizes, depending on the architecture
  const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
  const unsigned VectorSize =
    is_gpu ? (ColBlockSize <= 32 ? ColBlockSize : 32) : 1;
  const unsigned TeamSize = is_gpu ? 256/VectorSize : 1;
  const unsigned RowBlockSize = 128;
  const unsigned m = data.extent(0);
  const unsigned n = data.extent(1);

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
    Kokkos::parallel_for("Genten::FacMatrix::colNorms_inf_kernel",
                         Kernel::policy(m),
                         KOKKOS_LAMBDA(TeamMember team)
    {
      ColNormsKernel_Inf<ExecSpace,ViewType,NormT,RowBlockSize,ColBlockSize,TeamSize,VectorSize> kernel(data, norms, team);
      for (unsigned j_block=0; j_block<n; j_block+=ColBlockSize) {
        if (j_block+ColBlockSize <= n)
          kernel.template run<ColBlockSize>(j_block, ColBlockSize);
        else
          kernel.template run<0>(j_block, n-j_block);
      }
    });
    if (pmap != nullptr) {
      Kokkos::fence();
      pmap->allReduce(norms.data(), norms.size(), ProcessorMap::Max);
    }
    deep_copy(norms_host, norms);
    break;
  }

  case Genten::NormOne:
  {
    typedef ColNormsKernel_1<ExecSpace,ViewType,NormT,RowBlockSize,ColBlockSize,TeamSize,VectorSize> Kernel;
    typedef typename Kernel::TeamMember TeamMember;
    Kokkos::parallel_for("Genten::FacMatrix::colNorms_1_kernel",
                         Kernel::policy(m),
                         KOKKOS_LAMBDA(TeamMember team)
    {
      ColNormsKernel_1<ExecSpace,ViewType,NormT,RowBlockSize,ColBlockSize,TeamSize,VectorSize> kernel(data, norms, team);
      for (unsigned j_block=0; j_block<n; j_block+=ColBlockSize) {
        if (j_block+ColBlockSize <= n)
          kernel.template run<ColBlockSize>(j_block, ColBlockSize);
        else
          kernel.template run<0>(j_block, n-j_block);
      }
    });
    if (pmap != nullptr) {
      Kokkos::fence();
      pmap->allReduce(norms.data(), norms.size());
    }
    deep_copy(norms_host, norms);
    break;
  }
  case Genten::NormTwo:
  {
    typedef ColNormsKernel_2<ExecSpace,ViewType,NormT,RowBlockSize,ColBlockSize,TeamSize,VectorSize> Kernel;
    typedef typename Kernel::TeamMember TeamMember;
    Kokkos::parallel_for("Genten::FacMatrix::colNorms_2_kernel",
                         Kernel::policy(m),
                         KOKKOS_LAMBDA(TeamMember team)
    {
      ColNormsKernel_2<ExecSpace,ViewType,NormT,RowBlockSize,ColBlockSize,TeamSize,VectorSize> kernel(data, norms, team);
      for (unsigned j_block=0; j_block<n; j_block+=ColBlockSize) {
        if (j_block+ColBlockSize <= n)
          kernel.template run<ColBlockSize>(j_block, ColBlockSize);
        else
          kernel.template run<0>(j_block, n-j_block);
      }
    });
    if (pmap != nullptr) {
      Kokkos::fence();
      pmap->allReduce(norms.data(), norms.size());
    }
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

  const ttb_indx nc = data.extent(1);
  if (nc < 2)
    Impl::colNorms_kernel<ExecSpace,1>(data, normtype, norms.values(), minval,
                                       pmap);
  else if (nc < 4)
    Impl::colNorms_kernel<ExecSpace,2>(data, normtype, norms.values(), minval,
                                       pmap);
  else if (nc < 8)
    Impl::colNorms_kernel<ExecSpace,4>(data, normtype, norms.values(), minval,
                                       pmap);
  else if (nc < 16)
    Impl::colNorms_kernel<ExecSpace,8>(data, normtype, norms.values(), minval,
                                       pmap);
  else if (nc < 32)
    Impl::colNorms_kernel<ExecSpace,16>(data, normtype, norms.values(), minval,
                                        pmap);
  else
    Impl::colNorms_kernel<ExecSpace,32>(data, normtype, norms.values(), minval,
                                        pmap);
}

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
    m(data.extent(0))
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
  const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
  const unsigned VectorSize =
     is_gpu ? (ColBlockSize <= 32 ? ColBlockSize : 32) : 1;
  const unsigned TeamSize = is_gpu ? 256/VectorSize : 1;
  const unsigned RowBlockSize = 128;
  const unsigned m = data.extent(0);
  const unsigned n = data.extent(1);

  typedef ColScaleKernel<ExecSpace,ViewType,ColBlockSize,RowBlockSize,TeamSize,VectorSize> Kernel;
  typedef typename Kernel::TeamMember TeamMember;

  Kokkos::parallel_for("Genten::FacMatrix::colScale_kernel",
                       Kernel::policy(m), KOKKOS_LAMBDA(TeamMember team)
  {
    ColScaleKernel<ExecSpace,ViewType,ColBlockSize,RowBlockSize,TeamSize,VectorSize> kernel(data, v, team);
    for (unsigned j_block=0; j_block<n; j_block+=ColBlockSize) {
      if (j_block+ColBlockSize <= n)
        kernel.template run<ColBlockSize>(j_block, ColBlockSize);
      else
        kernel.template run<0>(j_block, n-j_block);
    }
  });
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

  const ttb_indx n = data.extent(1);
  gt_assert(v.size() == n);

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

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
rowScale(const Genten::ArrayT<ExecSpace> & v, bool inverse) const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::FacMatrix::rowScale");
#endif

  const ttb_indx m = data.extent(0);
  const ttb_indx n = data.extent(1);
  gt_assert(v.size() == m);

  auto d = data;

  const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
  // t = largest power of 2 <= n at most 256
  const unsigned t = std::min(
    256u, static_cast<unsigned>(std::pow(2,static_cast<unsigned>(std::log2(n)))));
  const unsigned VectorSize = is_gpu ? t : 1;
  const unsigned TeamSize = is_gpu ? 256/t : 1;
  const ttb_indx LeagueSize = (m+TeamSize-1)/TeamSize;
  typedef Kokkos::TeamPolicy<ExecSpace> Policy;
  Policy policy(LeagueSize, TeamSize, VectorSize);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(typename Policy::member_type team)
  {
    const ttb_indx i = team.league_rank()*team.team_size() + team.team_rank();
    if (i >= m) return;
    const ttb_real w = inverse ? ttb_real(1.0)/v[i] : v[i];
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,n),
                         [&](const ttb_indx j)
    {
      d(i,j) *= w;
    });
  });
}

// Only called by Ben Allan's parallel test code.  It appears he uses the Linux
// random number generator in a special way.
#if !defined(_WIN32)
template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
scaleRandomElements(ttb_real fraction, ttb_real scale, bool columnwise) const
{
  const ttb_indx nrows = data.extent(0);
  const ttb_indx ncols = data.extent(1);
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
    ttb_indx misses = 0;
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

  const ttb_indx nrows = data.extent(0);
  const ttb_indx ncols = data.extent(1);

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
  Kokkos::fence();
  // if (pmap != nullptr) {
  //   sum = pmap->gridAllReduce(sum);
  // }

  return sum;
}

template <typename ExecSpace>
ttb_real Genten::FacMatrixT<ExecSpace>::
norm() const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::FacMatrix::sum");
#endif

  const ttb_indx nrows = data.extent(0);
  const ttb_indx ncols = data.extent(1);

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
      s += d(i,j) * d(i,j);
  }, sum);
  Kokkos::fence();
  if (pmap != nullptr) {
    sum = pmap->allReduce(sum);
  }

  return std::sqrt(sum);
}

// TODO: This function really should be removed and replaced with a ktensor norm function, because that's kind of how it's used.
template <typename ExecSpace>
ttb_real Genten::FacMatrixT<ExecSpace>::
sum(const UploType uplo) const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::FacMatrix::sym_sum");
#endif

  const ttb_indx nrows = data.extent(0);
  const ttb_indx ncols = data.extent(1);

  ttb_real sum = 0;
  view_type d = data;
  if (uplo == Upper) {
    Kokkos::parallel_reduce("Genten::FacMatrix::sum_kernel",
                            Kokkos::RangePolicy<ExecSpace>(0,nrows),
                            KOKKOS_LAMBDA(const ttb_indx i, ttb_real& s)
    {
      s += d(i,i);
      for (ttb_indx j=i+1; j<ncols; ++j)
        s += ttb_real(2.0)*d(i,j);
    }, sum);
  }
  else {
    Kokkos::parallel_reduce("Genten::FacMatrix::sum_kernel",
                            Kokkos::RangePolicy<ExecSpace>(0,nrows),
                            KOKKOS_LAMBDA(const ttb_indx i, ttb_real& s)
    {
      s += d(i,i);
      for (ttb_indx j=0; j<i; ++j)
        s += ttb_real(2.0)*d(i,j);
    }, sum);
  }
  Kokkos::fence();
  // if (pmap != nullptr) 
  //   sum = pmap->gridAllReduce(sum);
  // }

  return sum;
}

template <typename ExecSpace>
ttb_real Genten::FacMatrixT<ExecSpace>::
normFsq() const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::FacMatrix::normFsq");
#endif

  const ttb_indx nrows = data.extent(0);
  const ttb_indx ncols = data.extent(1);

  ttb_real nrm = 0;
  // for (ttb_indx i=0; i<nrows; ++i)
  //   for (ttb_indx j=0; j<ncols; ++j)
  //     nrm += data(i,j)*data(i,j);
  view_type d = data;
  Kokkos::parallel_reduce("Genten::FacMatrix::normFsq_kernel",
                          Kokkos::RangePolicy<ExecSpace>(0,nrows),
                          KOKKOS_LAMBDA(const ttb_indx i, ttb_real& s)
  {
    for (ttb_indx j=0; j<ncols; ++j)
      s += d(i,j)*d(i,j);
  }, nrm);
  Kokkos::fence();

  if (pmap != nullptr)
    nrm = pmap->allReduce(nrm);

  return nrm;
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

  const ttb_indx nrows = data.extent(0);
  const ttb_indx ncols = data.extent(1);
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
      Kokkos::parallel_for("Genten::FacMatrix::permute_kernel",
                           Kokkos::RangePolicy<ExecSpace>(0,nrows),
                           KOKKOS_LAMBDA(const ttb_indx i)
      {
        const ttb_real temp = d(i,loc);
        d(i,loc) = d(i,j);
        d(i,j) = temp;
      });

      // Swap curr_indices to mark where data was moved.
      ttb_indx  k = curr_indices[j];
      curr_indices[j] = curr_indices[loc];
      curr_indices[loc] = k;
    }
  }
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
gemm(const bool trans_a, const bool trans_b, const ttb_real alpha,
     const FacMatrixT<ExecSpace>& A, const FacMatrixT<ExecSpace>& B,
     const ttb_real beta) const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::FacMatrix::gemm");
#endif

  const ttb_indx m = data.extent(0);
  const ttb_indx n = data.extent(1);
  const ttb_indx rows_op_a = trans_a ? A.data.extent(1) : A.data.extent(0);
  const ttb_indx cols_op_a = trans_a ? A.data.extent(0) : A.data.extent(1);
  const ttb_indx rows_op_b = trans_b ? B.data.extent(1) : B.data.extent(0);
  const ttb_indx cols_op_b = trans_b ? B.data.extent(0) : B.data.extent(1);

  gt_assert(rows_op_a == m);
  gt_assert(cols_op_a == rows_op_b);
  gt_assert(cols_op_b == n);

  Genten::gemm(trans_a,trans_b,alpha,A.data,B.data,beta,data);
}

namespace Genten {
  namespace Impl {

    template <typename ViewA, typename ViewB>
    bool solveTransposeRHSImpl_SPD(const ViewA& A, const ViewB& B,
                                   const UploType ul)
    {
      // On CPU, call Cholesky SPD solver
      const ttb_indx nrows = B.extent(0);
      const ttb_indx ncols = B.extent(1);

      // Switch Upper/Lower because we store row-wise and lapack assumes column
      char uplo = ul == Upper ? 'L' : 'U';

      return Genten::posv(uplo, ncols, nrows, A.data(), A.stride_0(), B.data(), B.stride_0());
    }

    template <typename ViewA, typename ViewB>
    void solveTransposeRHSImpl_SID(const ViewA& A, const ViewB& B,
                                   const UploType ul)
    {
      // On CPU, call symmetric, indefinite solver
      const ttb_indx nrows = B.extent(0);
      const ttb_indx ncols = B.extent(1);

      // Switch Upper/Lower because we store row-wise and lapack assumes column
      char uplo = ul == Upper ? 'L' : 'U';

      Genten::sysv(uplo, ncols, nrows, A.data(), A.stride_0(), B.data(), B.stride_0());
    }

    template <typename ViewA, typename ViewB>
    void solveTransposeRHSImpl(const ViewA& A, const ViewB& B,
                               const UploType ul,
                               const AlgParams& algParams) {
      // On CPU we use symmetric, indefinite solver instead of non-symmetric
      // solver because it should be faster.
      const ttb_indx nrows = B.extent(0);
      const ttb_indx ncols = B.extent(1);

      // Switch Upper/Lower because we store row-wise and lapack assumes column
      char uplo = ul == Upper ? 'L' : 'U';

      if (algParams.rank_def_solver) {
        /*ttb_indx rank =*/ Genten::gelsy(ncols, ncols, nrows, A.data(), A.stride_0(), B.data(), B.stride_0(), algParams.rcond);
        // if (rank < ncols) {
        //   std::cout << "Matrix is not full rank!  Numerical rank = " << rank
        //             << ", matrix order is " << ncols << std::endl;
        // }
      }
      else {
        Genten::sysv(uplo, ncols, nrows, A.data(), A.stride_0(), B.data(), B.stride_0());
      }
    }

#if defined(KOKKOS_ENABLE_CUDA)

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    std::enable_if_t<
      (is_cuda_space<typename Kokkos::View<AT, AP...>::execution_space>::value ||
       is_sycl_space<typename Kokkos::View<AT, AP...>::execution_space>::value) &&
      (is_cuda_space<typename Kokkos::View<BT, BP...>::execution_space>::value ||
       is_sycl_space<typename Kokkos::View<BT, BP...>::execution_space>::value) &&
      std::is_same<
        typename Kokkos::View<AT, BP...>::non_const_value_type, double
      >::value,
      bool
    >
    solveTransposeRHSImpl_SPD(const Kokkos::View<AT,AP...>& A,
                              const Kokkos::View<BT,BP...>& B,
                              const UploType ul)
    {
      using exec_space = typename Kokkos::View<AT, AP...>::execution_space;

      const int m = B.extent(0);
      const int n = B.extent(1);
      const int lda = A.stride_0();
      const int ldb = B.stride_0();
      const cublasFillMode_t uplo = ul == Upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
      cusolverStatus_t status;

      gt_assert(int(A.extent(0)) == n);
      gt_assert(int(A.extent(1)) == n);

      // lwork is a host pointer, but info is a device pointer.  Go figure.
      int lwork = 0;
      status = cusolverDnDpotrf_bufferSize(CusolverHandle::get(), uplo, n, A.data(), lda, &lwork);
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDpotrf_bufferSize() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      Kokkos::View<double *, Kokkos::LayoutRight, exec_space> work("work", lwork);
      Kokkos::View<int, Kokkos::LayoutRight, exec_space> info("info");
      status = cusolverDnDpotrf(CusolverHandle::get(), uplo, n, A.data(), lda, work.data(),
                                lwork, info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDpotrf() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      auto info_host = create_mirror_view(info);
      deep_copy(info_host, info);
      if (info_host() < 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDpotrf() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      if (info_host() > 0)
        return false;  // Matrix is not SPD

      status = cusolverDnDpotrs(CusolverHandle::get(), uplo, n, m, A.data(), lda,
                                B.data(), ldb, info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDpotrs() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      deep_copy(info_host, info);
      if (info_host() != 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDpotrs() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      return true;
    }

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    std::enable_if_t<
      (is_cuda_space<typename Kokkos::View<AT, AP...>::execution_space>::value ||
       is_sycl_space<typename Kokkos::View<AT, AP...>::execution_space>::value) &&
      (is_cuda_space<typename Kokkos::View<BT, BP...>::execution_space>::value ||
       is_sycl_space<typename Kokkos::View<BT, BP...>::execution_space>::value) &&
      std::is_same<
        typename Kokkos::View<AT, BP...>::non_const_value_type, double
      >::value
    >
    solveTransposeRHSImpl_SID(const Kokkos::View<AT,AP...>& A,
                              const Kokkos::View<BT,BP...>& B,
                              const UploType ul)
    {
      // cusolverDnDsytrs does not appear to work yet, so error out
      Genten::error("Symmetric, indefinite solve with Cuda is not fully implemented in cuSOLVER.  Instead you must use the option '--full-gram' to enable the non-symmetric solver.");
#if 0
      using exec_space = typename Kokkos::View<AT, AP...>::execution_space;

      const int m = B.extent(0);
      const int n = B.extent(1);
      const int lda = A.stride_0();
      const int ldb = B.stride_0();
      const cublasFillMode_t uplo = ul == Upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
      cusolverStatus_t status;

      gt_assert(int(A.extent(0)) == n);
      gt_assert(int(A.extent(1)) == n);

      // lwork is a host pointer, but info is a device pointer.  Go figure.
      int lwork = 0;
      status = cusolverDnDsytrf_bufferSize(CusolverHandle::get(), n, A.data(), lda, &lwork);
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDsytrf_bufferSize() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      Kokkos::View<double*,Kokkos::LayoutRight,exec_space> work("work",lwork);
      Kokkos::View<int*,Kokkos::LayoutRight,exec_space> piv("piv",n);
      Kokkos::View<int,Kokkos::LayoutRight,exec_space> info("info");
      status = cusolverDnDsytrf(CusolverHandle::get(), uplo, n, A.data(), lda, piv.data(),
                                work.data(), lwork, info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDsytrf() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      auto info_host = create_mirror_view(info);
      deep_copy(info_host, info);
      if (info_host() != 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDsytrf() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      status = cusolverDnDsytrs_bufferSize(CusolverHandle::get(), uplo, n, m, A.data(), lda,
                                           piv.data(), B.data(), ldb, &lwork);
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDsytrs_bufferSize() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      Kokkos::View<double*,Kokkos::LayoutRight,exec_space> work2("work2",lwork);
      status = cusolverDnDsytrs(CusolverHandle::get(), uplo, n, m, A.data(), lda,
                                piv.data(), B.data(), ldb, work2.data(), lwork,
                                info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDsytrs() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      deep_copy(info_host, info);
      if (info_host() != 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDsytrs() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
#endif
    }

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    std::enable_if_t<
      (is_cuda_space<typename Kokkos::View<AT, AP...>::execution_space>::value ||
       is_sycl_space<typename Kokkos::View<AT, AP...>::execution_space>::value) &&
      (is_cuda_space<typename Kokkos::View<BT, BP...>::execution_space>::value ||
       is_sycl_space<typename Kokkos::View<BT, BP...>::execution_space>::value) &&
      std::is_same<
        typename Kokkos::View<AT, BP...>::non_const_value_type, double
      >::value
    >
    solveTransposeRHSImpl(const Kokkos::View<AT,AP...>& A,
                          const Kokkos::View<BT,BP...>& B,
                          const UploType uplo,
                          const AlgParams& algParams)
    {
      if (algParams.rank_def_solver)
        throw std::string("Rank-deficient solver not supported on the GPU!");

      using exec_space = typename Kokkos::View<AT, AP...>::execution_space;

      const int m = B.extent(0);
      const int n = B.extent(1);
      const int lda = A.stride_0();
      const int ldb = B.stride_0();
      cusolverStatus_t status;

      gt_assert(int(A.extent(0)) == n);
      gt_assert(int(A.extent(1)) == n);

      // lwork is a host pointer, but info is a device pointer.  Go figure.
      int lwork = 0;
      status = cusolverDnDgetrf_bufferSize(CusolverHandle::get(), n, n, A.data(), lda, &lwork);
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnDgetrf_bufferSize() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      Kokkos::View<double *, Kokkos::LayoutRight, exec_space> work("work", lwork);
      Kokkos::View<int *, Kokkos::LayoutRight, exec_space> piv("piv", n);
      Kokkos::View<int, Kokkos::LayoutRight, exec_space> info("info");
      status = cusolverDnDgetrf(CusolverHandle::get(), n, n, A.data(), lda, work.data(),
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
        ss << "Error!  cusolverDnDgetrf() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      status = cusolverDnDgetrs(CusolverHandle::get(), CUBLAS_OP_N, n, m, A.data(), lda,
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
        ss << "Error!  cusolverDnDgetrs() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
    }

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    std::enable_if_t<
      (is_cuda_space<typename Kokkos::View<AT, AP...>::execution_space>::value ||
       is_sycl_space<typename Kokkos::View<AT, AP...>::execution_space>::value) &&
      (is_cuda_space<typename Kokkos::View<BT, BP...>::execution_space>::value ||
       is_sycl_space<typename Kokkos::View<BT, BP...>::execution_space>::value) &&
      std::is_same<
        typename Kokkos::View<AT, BP...>::non_const_value_type, float
      >::value,
      bool
    >
    solveTransposeRHSImpl_SPD(const Kokkos::View<AT,AP...>& A,
                              const Kokkos::View<BT,BP...>& B,
                              const UploType ul)
    {
      using exec_space = typename Kokkos::View<AT, AP...>::execution_space;

      const int m = B.extent(0);
      const int n = B.extent(1);
      const int lda = A.stride_0();
      const int ldb = B.stride_0();
      const cublasFillMode_t uplo = ul == Upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
      cusolverStatus_t status;

      gt_assert(int(A.extent(0)) == n);
      gt_assert(int(A.extent(1)) == n);

      // lwork is a host pointer, but info is a device pointer.  Go figure.
      int lwork = 0;
      status = cusolverDnSpotrf_bufferSize(CusolverHandle::get(), uplo, n, A.data(), lda, &lwork);
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnSpotrf_bufferSize() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      Kokkos::View<float *, Kokkos::LayoutRight, exec_space> work("work", lwork);
      Kokkos::View<int, Kokkos::LayoutRight, exec_space> info("info");
      status = cusolverDnSpotrf(CusolverHandle::get(), uplo, n, A.data(), lda, work.data(),
                                lwork, info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnSpotrf() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      auto info_host = create_mirror_view(info);
      deep_copy(info_host, info);
      if (info_host() < 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDnSpotrf() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      if (info_host() > 0)
        return false;  // Matrix is not SPD

      status = cusolverDnSpotrs(CusolverHandle::get(), uplo, n, m, A.data(), lda,
                                B.data(), ldb, info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnSpotrs() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      deep_copy(info_host, info);
      if (info_host() != 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDnSpotrs() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      return true;
    }

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    std::enable_if_t<
      (is_cuda_space<typename Kokkos::View<AT, AP...>::execution_space>::value ||
       is_sycl_space<typename Kokkos::View<AT, AP...>::execution_space>::value) &&
      (is_cuda_space<typename Kokkos::View<BT, BP...>::execution_space>::value ||
       is_sycl_space<typename Kokkos::View<BT, BP...>::execution_space>::value) &&
      std::is_same<
        typename Kokkos::View<AT, BP...>::non_const_value_type, float
      >::value
    >
    solveTransposeRHSImpl_SID(const Kokkos::View<AT,AP...>& A,
                              const Kokkos::View<BT,BP...>& B,
                              const UploType ul)
    {
      // cusolverDnSsytrs does not appear to work yet, so error out
      Genten::error("Symmetric, indefinite solve with Cuda is not fully implemented in cuSOLVER.  Instead you must use the option '--full-gram' to enable the non-symmetric solver.");
#if 0
      const int m = B.extent(0);
      const int n = B.extent(1);
      const int lda = A.stride_0();
      const int ldb = B.stride_0();
      const cublasFillMode_t uplo = ul == Upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
      cusolverStatus_t status;

      gt_assert(int(A.extent(0)) == n);
      gt_assert(int(A.extent(1)) == n);

      // lwork is a host pointer, but info is a device pointer.  Go figure.
      int lwork = 0;
      status = cusolverDnSsytrf_bufferSize(CusolverHandle::get(), n, A.data(), lda, &lwork);
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnSsytrf_bufferSize() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      Kokkos::View<float*,Kokkos::LayoutRight,exec_space> work("work",lwork);
      Kokkos::View<int*,Kokkos::LayoutRight,exec_space> piv("piv",n);
      Kokkos::View<int,Kokkos::LayoutRight,exec_space> info("info");
      status = cusolverDnSsytrf(CusolverHandle::get(), uplo, n, A.data(), lda, piv.data(),
                                work.data(), lwork, info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnSsytrf() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      auto info_host = create_mirror_view(info);
      deep_copy(info_host, info);
      if (info_host() != 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDnSsytrf() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      status = cusolverDnSsytrs_bufferSize(CusolverHandle::get(), uplo, n, m, A.data(), lda,
                                           piv.data(), B.data(), ldb, &lwork);
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnSsytrs_bufferSize() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      Kokkos::View<float*,Kokkos::LayoutRight,exec_space> work2("work2",lwork);
      status = cusolverDnSsytrs(CusolverHandle::get(), uplo, n, m, A.data(), lda,
                                piv.data(), B.data(), ldb, work2.data(), lwork,
                                info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnSsytrs() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      deep_copy(info_host, info);
      if (info_host() != 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDnSsytrs() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
#endif
    }

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    std::enable_if_t<
      (is_cuda_space<typename Kokkos::View<AT, AP...>::execution_space>::value ||
       is_sycl_space<typename Kokkos::View<AT, AP...>::execution_space>::value) &&
      (is_cuda_space<typename Kokkos::View<BT, BP...>::execution_space>::value ||
       is_sycl_space<typename Kokkos::View<BT, BP...>::execution_space>::value) &&
      std::is_same<
        typename Kokkos::View<AT, BP...>::non_const_value_type, float
      >::value
    >
    solveTransposeRHSImpl(const Kokkos::View<AT,AP...>& A,
                          const Kokkos::View<BT,BP...>& B,
                          const UploType uplo,
                          const AlgParams& algParams)
    {
      using exec_space = typename Kokkos::View<AT, AP...>::execution_space;

      if (algParams.rank_def_solver)
        throw std::string("Rank-deficient solver not supported on the GPU!");

      const int m = B.extent(0);
      const int n = B.extent(1);
      const int lda = A.stride_0();
      const int ldb = B.stride_0();
      cusolverStatus_t status;

      gt_assert(int(A.extent(0)) == n);
      gt_assert(int(A.extent(1)) == n);

      // lwork is a host pointer, but info is a device pointer.  Go figure.
      int lwork = 0;
      status = cusolverDnSgetrf_bufferSize(CusolverHandle::get(), n, n, A.data(), lda, &lwork);
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnSgetrf_bufferSize() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      Kokkos::View<float *, Kokkos::LayoutRight, exec_space> work("work", lwork);
      Kokkos::View<int *, Kokkos::LayoutRight, exec_space> piv("piv", n);
      Kokkos::View<int, Kokkos::LayoutRight, exec_space> info("info");
      status = cusolverDnSgetrf(CusolverHandle::get(), n, n, A.data(), lda, work.data(),
                                piv.data(), info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  cusolverDnSgetrf() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      auto info_host = create_mirror_view(info);
      deep_copy(info_host, info);
      if (info_host() != 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDnSgetrf() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      status = cusolverDnSgetrs(CusolverHandle::get(), CUBLAS_OP_N, n, m, A.data(), lda,
                                piv.data(), B.data(), ldb, info.data());
      if (status != CUSOLVER_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Error!  ccusolverDnSgetrs() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      deep_copy(info_host, info);
      if (info_host() != 0) {
        std::stringstream ss;
        ss << "Error!  cusolverDnSgetrs() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
    }

#endif

#if defined(KOKKOS_ENABLE_HIP)

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    std::enable_if_t<
      is_hip_space<typename Kokkos::View<AT, AP...>::execution_space>::value &&
      is_hip_space<typename Kokkos::View<BT, BP...>::execution_space>::value &&
      std::is_same<
        typename Kokkos::View<AT, BP...>::non_const_value_type, double
      >::value,
      bool
    >
    solveTransposeRHSImpl_SPD(const Kokkos::View<AT,AP...>& A,
                              const Kokkos::View<BT,BP...>& B,
                              const UploType ul)
    {
      const int m = B.extent(0);
      const int n = B.extent(1);
      const int lda = A.stride_0();
      const int ldb = B.stride_0();
      const rocblas_fill uplo = ul == Upper ? rocblas_fill_lower : rocblas_fill_upper;
      rocblas_status status;

      gt_assert(int(A.extent(0)) == n);
      gt_assert(int(A.extent(1)) == n);

      Kokkos::View<int,Kokkos::LayoutRight,Kokkos::Experimental::HIP> info("info");
      status = rocsolver_dpotrf(RocblasHandle::get(), uplo, n, A.data(), lda, info.data());
      if (status != rocblas_status_success) {
        std::stringstream ss;
        ss << "Error!  rocsolver_dpotrf() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      auto info_host = create_mirror_view(info);
      deep_copy(info_host, info);
      if (info_host() < 0) {
        std::stringstream ss;
        ss << "Error!  rocsolver_dpotrf() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      if (info_host() > 0)
        return false;  // Matrix is not SPD

      status = rocsolver_dpotrs(RocblasHandle::get(), uplo, n, m, A.data(), lda,
                                B.data(), ldb);
      if (status != rocblas_status_success) {
        std::stringstream ss;
        ss << "Error!  rocsolver_dpotrf() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      return true;
    }

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    std::enable_if_t<
      is_hip_space<typename Kokkos::View<AT, AP...>::execution_space>::value &&
      is_hip_space<typename Kokkos::View<BT, BP...>::execution_space>::value &&
      std::is_same<
        typename Kokkos::View<AT, BP...>::non_const_value_type, double
      >::value
    >
    solveTransposeRHSImpl_SID(const Kokkos::View<AT,AP...>& A,
                              const Kokkos::View<BT,BP...>& B,
                              const UploType ul)
    {
      // rocsolver_dsytrs is not yet implemented in rocSOLVER
      Genten::error("Symmetric, indefinite solve with HIP is not fully implemented in rocSOLVER.  Instead you must use the option '--full-gram' to enable the non-symmetric solver.");
    }

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    std::enable_if_t<
      is_hip_space<typename Kokkos::View<AT,AP...>::execution_space>::value &&
      is_hip_space<typename Kokkos::View<BT,BP...>::execution_space>::value &&
      std::is_same<
        typename Kokkos::View<AT, BP...>::non_const_value_type, double
      >::value
    >
    solveTransposeRHSImpl(const Kokkos::View<AT,AP...>& A,
                          const Kokkos::View<BT,BP...>& B,
                          const UploType uplo,
                          const AlgParams& algParams)
    {
      if (algParams.rank_def_solver)
        throw std::string("Rank-deficient solver not supported on the GPU!");

      const int m = B.extent(0);
      const int n = B.extent(1);
      const int lda = A.stride_0();
      const int ldb = B.stride_0();
      rocblas_status status;

      gt_assert(int(A.extent(0)) == n);
      gt_assert(int(A.extent(1)) == n);

      Kokkos::View<int*,Kokkos::LayoutRight,Kokkos::Experimental::HIP> piv("piv",n);
      Kokkos::View<int,Kokkos::LayoutRight,Kokkos::Experimental::HIP> info("info");
      status = rocsolver_dgetrf(RocblasHandle::get(), n, n, A.data(), lda,
                                piv.data(), info.data());
      if (status != rocblas_status_success) {
        std::stringstream ss;
        ss << "Error!  rocsolver_dgetrf() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      auto info_host = create_mirror_view(info);
      deep_copy(info_host, info);
      if (info_host() != 0) {
        std::stringstream ss;
        ss << "Error!  rocsolver_dgetrf() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      status = rocsolver_dgetrs(RocblasHandle::get(), rocblas_operation_none, n, m, A.data(), lda,
                                piv.data(), B.data(), ldb);
      if (status != rocblas_status_success) {
        std::stringstream ss;
        ss << "Error!  rocsolver_dgetrs() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
    }

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    std::enable_if_t<
      is_hip_space<typename Kokkos::View<AT,AP...>::execution_space>::value &&
      is_hip_space<typename Kokkos::View<BT,BP...>::execution_space>::value &&
      std::is_same<
        typename Kokkos::View<AT, BP...>::non_const_value_type, float
      >::value,
      bool
    >
    solveTransposeRHSImpl_SPD(const Kokkos::View<AT,AP...>& A,
                              const Kokkos::View<BT,BP...>& B,
                              const UploType ul)
    {
      const int m = B.extent(0);
      const int n = B.extent(1);
      const int lda = A.stride_0();
      const int ldb = B.stride_0();
      const rocblas_fill uplo = ul == Upper ? rocblas_fill_lower : rocblas_fill_upper;
      rocblas_status status;

      gt_assert(int(A.extent(0)) == n);
      gt_assert(int(A.extent(1)) == n);

      Kokkos::View<int,Kokkos::LayoutRight,Kokkos::Experimental::HIP> info("info");
      status = rocsolver_spotrf(RocblasHandle::get(), uplo, n, A.data(), lda, info.data());
      if (status != rocblas_status_success) {
        std::stringstream ss;
        ss << "Error!  rocsolver_spotrf() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      auto info_host = create_mirror_view(info);
      deep_copy(info_host, info);
      if (info_host() < 0) {
        std::stringstream ss;
        ss << "Error!  rocsolver_spotrf() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      if (info_host() > 0)
        return false;  // Matrix is not SPD

      status = rocsolver_spotrs(RocblasHandle::get(), uplo, n, m, A.data(), lda,
                                B.data(), ldb);
      if (status != rocblas_status_success) {
        std::stringstream ss;
        ss << "Error!  rocsolver_spotrs() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      return true;
    }

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    std::enable_if_t<
      is_hip_space<typename Kokkos::View<AT,AP...>::execution_space>::value &&
      is_hip_space<typename Kokkos::View<BT,BP...>::execution_space>::value &&
      std::is_same<
        typename Kokkos::View<AT, BP...>::non_const_value_type, float
      >::value
    >
    solveTransposeRHSImpl_SID(const Kokkos::View<AT,AP...>& A,
                              const Kokkos::View<BT,BP...>& B,
                              const UploType ul)
    {
      // rocsolver_ssytrf is not yet implemented in rocSOLVER
      Genten::error("Symmetric, indefinite solve with HIP is not fully implemented in rocSOLVER.  Instead you must use the option '--full-gram' to enable the non-symmetric solver.");
    }

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    std::enable_if_t<
      is_hip_space<typename Kokkos::View<AT,AP...>::execution_space>::value &&
      is_hip_space<typename Kokkos::View<BT,BP...>::execution_space>::value &&
      std::is_same<
        typename Kokkos::View<AT,BP...>::non_const_value_type, float
      >::value
    >
    solveTransposeRHSImpl(const Kokkos::View<AT,AP...>& A,
                          const Kokkos::View<BT,BP...>& B,
                          const UploType uplo,
                          const AlgParams& algParams)
    {
      if (algParams.rank_def_solver)
        throw std::string("Rank-deficient solver not supported on the GPU!");

      const int m = B.extent(0);
      const int n = B.extent(1);
      const int lda = A.stride_0();
      const int ldb = B.stride_0();
      rocblas_status status;

      gt_assert(int(A.extent(0)) == n);
      gt_assert(int(A.extent(1)) == n);

      Kokkos::View<int*,Kokkos::LayoutRight,Kokkos::Experimental::HIP> piv("piv",n);
      Kokkos::View<int,Kokkos::LayoutRight,Kokkos::Experimental::HIP> info("info");
      status = rocsolver_sgetrf(RocblasHandle::get(), n, n, A.data(), lda,
                                piv.data(), info.data());
      if (status != rocblas_status_success) {
        std::stringstream ss;
        ss << "Error!  rocsolver_sgetrf() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
      auto info_host = create_mirror_view(info);
      deep_copy(info_host, info);
      if (info_host() != 0) {
        std::stringstream ss;
        ss << "Error!  rocsolver_sgetrf() info =  " << info_host();
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }

      status = rocsolver_sgetrs(RocblasHandle::get(), rocblas_operation_none, n, m, A.data(), lda,
                                piv.data(), B.data(), ldb);
      if (status != rocblas_status_success) {
        std::stringstream ss;
        ss << "Error!  rocsolver_sgetrs() failed with status "
           << status;
        std::cerr << ss.str() << std::endl;
        throw ss.str();
      }
    }

#endif

#if defined(KOKKOS_ENABLE_SYCL)

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    std::enable_if_t<
      is_sycl_space<typename Kokkos::View<AT, AP...>::execution_space>::value &&
      is_sycl_space<typename Kokkos::View<BT, BP...>::execution_space>::value,
      bool
    >
    solveTransposeRHSImpl_SPD(const Kokkos::View<AT,AP...>& A,
                              const Kokkos::View<BT,BP...>& B,
                              const UploType ul)
    {
      using view_type_A = Kokkos::View<AT, AP...>;
      using view_type_B = Kokkos::View<BT, BP...>;
      static_assert(std::is_same_v<typename view_type_A::value_type,
                                   typename view_type_B::value_type>);
      using value_type = typename view_type_A::value_type;

      const std::int64_t m = B.extent(0);
      const std::int64_t n = B.extent(1);
      const std::int64_t lda = A.stride_0();
      const std::int64_t ldb = B.stride_0();
      oneapi::mkl::uplo mkl_uplo =
        ul == Upper ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper;

      using ExecSpace = typename view_type_A::execution_space;
      ExecSpace space;
      sycl::queue& q = space.sycl_queue();

      gt_assert(int(A.extent(0)) == n);
      gt_assert(int(A.extent(1)) == n);

      // Allocate potrf scratchpad
      std::int64_t potrf_lwork =
        oneapi::mkl::lapack::potrf_scratchpad_size<value_type>(
          q, mkl_uplo, n, lda);
      Kokkos::View<value_type*, ExecSpace> potrf_scratch("potrf_scratchpad",
                                                         potrf_lwork);

      // Perform Cholesky factorization
      try {
        oneapi::mkl::lapack::potrf(
          q, mkl_uplo, n, A.data(), lda, potrf_scratch.data(), potrf_lwork);
      }
      catch(oneapi::mkl::lapack::exception& e) {
        if (e.info() > 0 && e.detail() == 0)
          return false; // Matrix is not SPD
        else if (e.info() < 0)
          Genten::error("Error in call to potrf.  Entry " + std::to_string(e.info()) + " had an illegal value!");
        else if (e.info() == potrf_lwork && e.detail() != 0)
          Genten::error("Error in call to potrf.  Scratchpad size (" + std::to_string(potrf_lwork) + ") < required size (" + std::to_string(e.detail()) + ")!");
        else
          Genten::error ("Unknown error in potrf.");
      }

      // Allocate potrs scratchpad
      std::int64_t potrs_lwork =
        oneapi::mkl::lapack::potrs_scratchpad_size<value_type>(
          q, mkl_uplo, n, m, lda, ldb);
      Kokkos::View<value_type*, ExecSpace> potrs_scratch("potrs_scratchpad",
                                                         potrs_lwork);

      // Perform triangular factorization
      try {
        oneapi::mkl::lapack::potrs(
          q, mkl_uplo, n, m, A.data(), lda, B.data(), ldb, potrs_scratch.data(),
          potrs_lwork);
      }
      catch(oneapi::mkl::lapack::exception& e) {
        if (e.info() > 0 && e.detail() == 0)
          return false; // Matrix is not SPD
        else if (e.info() < 0)
          Genten::error("Error in call to potrs.  Entry " + std::to_string(e.info()) + " had an illegal value!");
        else if (e.info() == potrs_lwork && e.detail() != 0)
          Genten::error("Error in call to potrs.  Scratchpad size (" + std::to_string(potrs_lwork) + ") < required size (" + std::to_string(e.detail()) + ")!");
        else
          Genten::error ("Unknown error in potrs.");
      }

      return true;
    }

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    std::enable_if_t<
      is_sycl_space<typename Kokkos::View<AT, AP...>::execution_space>::value &&
      is_sycl_space<typename Kokkos::View<BT, BP...>::execution_space>::value
      >
    solveTransposeRHSImpl_SID(const Kokkos::View<AT,AP...>& A,
                              const Kokkos::View<BT,BP...>& B,
                              const UploType ul)
    {
      Genten::error("Symmetric, indefinite solve with SYCL is not fully implemented in DPC++ MKL.  Instead you must use the option '--full-gram' to enable the non-symmetric solver.");

      using view_type_A = Kokkos::View<AT, AP...>;
      using view_type_B = Kokkos::View<BT, BP...>;
      static_assert(std::is_same_v<typename view_type_A::value_type,
                                   typename view_type_B::value_type>);
      using value_type = typename view_type_A::value_type;

      const std::int64_t m = B.extent(0);
      const std::int64_t n = B.extent(1);
      const std::int64_t lda = A.stride_0();
      const std::int64_t ldb = B.stride_0();
      oneapi::mkl::uplo mkl_uplo =
        ul == Upper ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper;

      using ExecSpace = typename view_type_A::execution_space;
      ExecSpace space;
      sycl::queue& q = space.sycl_queue();

      gt_assert(int(A.extent(0)) == n);
      gt_assert(int(A.extent(1)) == n);

      // Allocate sytrf scratchpad
      std::int64_t sytrf_lwork =
        oneapi::mkl::lapack::sytrf_scratchpad_size<value_type>(
          q, mkl_uplo, n, lda);
      Kokkos::View<value_type*, ExecSpace> sytrf_scratch("sytrf_scratchpad",
                                                         sytrf_lwork);

      // Perform factorization
      Kokkos::View<std::int64_t*,ExecSpace> ipiv("sytrf_ipiv", n);
      try {
        oneapi::mkl::lapack::sytrf(
          q, mkl_uplo, n, A.data(), lda, ipiv.data(), sytrf_scratch.data(),
          sytrf_lwork);
      }
      catch(oneapi::mkl::lapack::exception& e) {
        if (e.info() > 0)
          Genten::error("Error in call to sytrf.  Gram matrix is singular.");
        else if (e.info() < 0)
          Genten::error("Error in call to sytrf.  Entry " + std::to_string(e.info()) + " had an illegal value!");
        else if (e.info() == sytrf_lwork && e.detail() != 0)
          Genten::error("Error in call to sytrf.  Scratchpad size (" + std::to_string(sytrf_lwork) + ") < required size (" + std::to_string(e.detail()) + ")!");
        else
          Genten::error ("Unknown error in potrf.");
      }

      // There appears to not be sytrs in dpc++ mkl
#if 0
      // Allocate sytrs scratchpad
      std::int64_t sytrs_lwork =
        oneapi::mkl::lapack::sytrs_scratchpad_size<value_type>(
          q, mkl_uplo, n, m, lda, ldb);
      Kokkos::View<value_type*, ExecSpace> sytrs_scratch("sytrs_scratchpad",
                                                         sytrs_lwork);

      // Perform triangular solve
      try {
        oneapi::mkl::lapack::sytrs(
          q, mkl_uplo, n, m, A.data(), lda, ipiv.data(), B.data(), ldb,
          sytrs_scratch.data(), sytrs_lwork);
      }
      catch(oneapi::mkl::lapack::exception& e) {
        if (e.info() > 0)
          Genten::error("Error in call to sytrs.  Gram matrix is singular.");
        else if (e.info() < 0)
          Genten::error("Error in call to sytrs.  Entry " + std::to_string(e.info()) + " had an illegal value!");
        else if (e.info() == sytrs_lwork && e.detail() != 0)
          Genten::error("Error in call to sytrs.  Scratchpad size (" + std::to_string(sytrfs_lwork) + ") < required size (" + std::to_string(e.detail()) + ")!");
        else
          Genten::error ("Unknown error in potrs.");
      }
#endif
    }

    template <typename AT, typename ... AP,
              typename BT, typename ... BP>
    std::enable_if_t<
      is_sycl_space<typename Kokkos::View<AT,AP...>::execution_space>::value &&
      is_sycl_space<typename Kokkos::View<BT,BP...>::execution_space>::value
      >
    solveTransposeRHSImpl(const Kokkos::View<AT,AP...>& A,
                          const Kokkos::View<BT,BP...>& B,
                          const UploType uplo,
                          const AlgParams& algParams)
    {
      if (algParams.rank_def_solver)
        throw std::string("Rank-deficient solver not supported on the GPU!");

      using view_type_A = Kokkos::View<AT, AP...>;
      using view_type_B = Kokkos::View<BT, BP...>;
      static_assert(std::is_same_v<typename view_type_A::value_type,
                                   typename view_type_B::value_type>);
      using value_type = typename view_type_A::value_type;

      const std::int64_t m = B.extent(0);
      const std::int64_t n = B.extent(1);
      const std::int64_t lda = A.stride_0();
      const std::int64_t ldb = B.stride_0();

      using ExecSpace = typename view_type_A::execution_space;
      ExecSpace space;
      sycl::queue& q = space.sycl_queue();

      gt_assert(int(A.extent(0)) == n);
      gt_assert(int(A.extent(1)) == n);

      // Allocate getrf scratchpad
      std::int64_t getrf_lwork =
        oneapi::mkl::lapack::getrf_scratchpad_size<value_type>(
          q, n, n, lda);
      Kokkos::View<value_type*, ExecSpace> getrf_scratch("getrf_scratchpad",
                                                         getrf_lwork);

      // Perform LU factorization
      Kokkos::View<std::int64_t*,ExecSpace> ipiv("getrf_ipiv", n);
      try {
        oneapi::mkl::lapack::getrf(
          q, n, n, A.data(), lda, ipiv.data(), getrf_scratch.data(),
          getrf_lwork);
      }
      catch(oneapi::mkl::lapack::exception& e) {
        if (e.info() > 0)
          Genten::error("Error in call to getrf.  Gram matrix is singular.");
        else if (e.info() < 0)
          Genten::error("Error in call to getrf.  Entry " + std::to_string(e.info()) + " had an illegal value!");
        else if (e.info() == getrf_lwork && e.detail() != 0)
          Genten::error("Error in call to getrf.  Scratchpad size (" + std::to_string(getrf_lwork) + ") < required size (" + std::to_string(e.detail()) + ")!");
        else
          Genten::error ("Unknown error in getrf.");
      }

      // Allocate getrs scratchpad
      std::int64_t getrs_lwork =
        oneapi::mkl::lapack::getrs_scratchpad_size<value_type>(
          q, oneapi::mkl::transpose::nontrans, n, m, lda, ldb);
      Kokkos::View<value_type*, ExecSpace> getrs_scratch("getrs_scratchpad",
                                                         getrs_lwork);

      // Perform triangular solve
      try {
        oneapi::mkl::lapack::getrs(
          q, oneapi::mkl::transpose::nontrans, n, m, A.data(), lda, ipiv.data(),
          B.data(), ldb, getrs_scratch.data(), getrs_lwork);
      }
      catch(oneapi::mkl::lapack::exception& e) {
        if (e.info() > 0 && e.detail() == 0)
          Genten::error("Error in call to getrf.  Gram matrix is singular.");
        else if (e.info() < 0)
          Genten::error("Error in call to getrs.  Entry " + std::to_string(e.info()) + " had an illegal value!");
        else if (e.info() == getrs_lwork && e.detail() != 0)
          Genten::error("Error in call to getrs.  Scratchpad size (" + std::to_string(getrs_lwork) + ") < required size (" + std::to_string(e.detail()) + ")!");
        else
          Genten::error ("Unknown error in getrs.");
      }
    }

#endif

  }
}

template <typename ExecSpace>
bool Genten::FacMatrixT<ExecSpace>::
solveTransposeRHS (const Genten::FacMatrixT<ExecSpace> & A,
                   const bool full,
                   const UploType uplo,
                   const bool spd,
                   const AlgParams& algParams) const
{
  gt_assert(A.nRows() == A.nCols());
  gt_assert(nCols() == A.nRows());

  // Copy A because LAPACK needs to overwrite the invertible square matrix.
  view_type Atmp("Atmp", A.nRows(), A.nCols());
  deep_copy(Atmp, A.data);

  bool is_spd = spd;
  if (full) {
    Genten::Impl::solveTransposeRHSImpl(Atmp, data, uplo, algParams);
    is_spd = false;
  }
  else if (spd) {
    is_spd = Genten::Impl::solveTransposeRHSImpl_SPD(Atmp, data, uplo);
    if (!is_spd) {
      std::cout << "Matrix is not SPD.  Switching to indefinite solver"
                << std::endl;
      deep_copy(Atmp, A.data);
      Genten::Impl::solveTransposeRHSImpl_SID(Atmp, data, uplo);
    }
  }
  else {
     Genten::Impl::solveTransposeRHSImpl_SID(Atmp, data, uplo);
     is_spd = false;
  }
  return is_spd;
}

template <typename ExecSpace>
ttb_real Genten::FacMatrixT<ExecSpace>::
rowDot(const ttb_indx         nRow,
       const Genten::FacMatrixT<ExecSpace> & other,
       const ttb_indx         nRowOther) const
{
  const ttb_indx ncols = data.extent(1);
  gt_assert(other.nCols() == ncols);

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
  const ttb_indx ncols = data.extent(1);
  gt_assert(other.nCols() == ncols);

  // Using LAPACK daxpy is slower on perf_CpAprRandomKtensor.
  //   axpy(ncols, dScalar, this->rowptr(nRow), 1, other.rowptr(nRowOther), 1);

  const ttb_real * rowPtr1 = this->rowptr(nRow);
  ttb_real * rowPtr2 = other.rowptr(nRowOther);

  for (ttb_indx  i = 0; i < ncols; i++)
    rowPtr2[i] = rowPtr2[i] + (dScalar * rowPtr1[i]);

  return;
}

namespace Genten {
namespace Impl {

template <typename ExecSpace, typename MatViewType, typename WeightViewType,
          unsigned RowBlockSize, unsigned ColBlockSize,
          unsigned TeamSize, unsigned VectorSize>
struct MatInnerProdKernel {

  typedef Kokkos::TeamPolicy<ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  typedef Kokkos::View< ttb_real**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

  const MatViewType& x;
  const MatViewType& y;
  const WeightViewType& w;

  const TeamMember& team;
  const unsigned team_index;
  const unsigned team_size;
  TmpScratchSpace tmp;
  const unsigned m, n, i_block;

  static inline Policy policy(const unsigned m) {
    const ttb_indx M = (m+RowBlockSize-1)/RowBlockSize;
    Policy policy(M,TeamSize,VectorSize);
    size_t bytes = TmpScratchSpace::shmem_size(RowBlockSize,ColBlockSize);
    return policy.set_scratch_size(0,Kokkos::PerTeam(bytes));
  }

  KOKKOS_INLINE_FUNCTION
  MatInnerProdKernel(const MatViewType& x_,
                     const MatViewType& y_,
                     const WeightViewType& w_,
                     const TeamMember& team_) :
    x(x_), y(y_), w(w_),
    team(team_), team_index(team.team_rank()), team_size(team.team_size()),
    tmp(team.team_scratch(0), RowBlockSize, ColBlockSize),
    m(x.extent(0)), n(x.extent(1)),
    i_block(team.league_rank()*RowBlockSize)
    {}

  template <unsigned Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const unsigned j, const unsigned nj_, ttb_real& d)
  {
    // nj.value == Nj_ if Nj_ > 0 and nj_ otherwise
    Kokkos::Impl::integral_nonzero_constant<unsigned, Nj> nj(nj_);

    const ttb_real *ww = &w[j];

    for (unsigned ii=team_index; ii<RowBlockSize; ii+=team_size) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,unsigned(nj.value)),
                           [&] (const unsigned& jj)
      {
        tmp(ii,jj) = 0.0;
      });

      const unsigned i = i_block + ii;
      if (i < m) {
        const ttb_real *xx = &x(i,j);
        const ttb_real *yy = &y(i,j);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,unsigned(nj.value)),
                             [&] (const unsigned& jj)
        {
          tmp(ii,jj) += ww[jj]*xx[jj]*yy[jj];
        });
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

template <typename ExecSpace, unsigned ColBlockSize,
          typename MatViewType, typename WeightViewType>
ttb_real mat_innerprod_kernel(const MatViewType& x, const MatViewType& y,
                              const WeightViewType& w)
{
  // Compute team and vector sizes, depending on the architecture
  const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
  const unsigned VectorSize =
    is_gpu ? (ColBlockSize <= 32 ? ColBlockSize : 32) : 1;
  const unsigned TeamSize = is_gpu ? 256/VectorSize : 1;
  const unsigned RowBlockSize = 128;
  const unsigned m = x.extent(0);
  const unsigned n = x.extent(1);

  typedef MatInnerProdKernel<ExecSpace,MatViewType,WeightViewType,RowBlockSize,ColBlockSize,TeamSize,VectorSize> Kernel;
  typedef typename Kernel::TeamMember TeamMember;

  ttb_real dTotal = 0.0;
  Kokkos::parallel_reduce("Genten::FacMatrix::innerprod_kernel",
                          Kernel::policy(m),
                          KOKKOS_LAMBDA(TeamMember team, ttb_real& d)
  {
    MatInnerProdKernel<ExecSpace,MatViewType,WeightViewType,RowBlockSize,ColBlockSize,TeamSize,VectorSize> kernel(x, y, w, team);
    for (unsigned j_block=0; j_block<n; j_block+=ColBlockSize) {
      if (j_block+ColBlockSize <= n)
        kernel.template run<ColBlockSize>(j_block, ColBlockSize, d);
      else
        kernel.template run<0>(j_block, n-j_block, d);
    }
  }, dTotal);
  Kokkos::fence();

  return dTotal;
}

}
}

template <typename ExecSpace>
ttb_real Genten::FacMatrixT<ExecSpace>::
innerprod(const Genten::FacMatrixT<ExecSpace>& A,
          const Genten::ArrayT<ExecSpace>& lambda) const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::FacMatrix::innerprod");
#endif

  const ttb_indx nc = data.extent(1);
  ttb_real ret = 0.0;
  if (nc < 2)
    ret = Impl::mat_innerprod_kernel<ExecSpace,1>(data, A.data, lambda.values());
  else if (nc < 4)
    ret = Impl::mat_innerprod_kernel<ExecSpace,2>(data, A.data, lambda.values());
  else if (nc < 8)
    ret = Impl::mat_innerprod_kernel<ExecSpace,4>(data, A.data, lambda.values());
  else if (nc < 16)
    ret = Impl::mat_innerprod_kernel<ExecSpace,8>(data, A.data, lambda.values());
  else if (nc < 32)
    ret = Impl::mat_innerprod_kernel<ExecSpace,16>(data, A.data, lambda.values());
  else
    ret = Impl::mat_innerprod_kernel<ExecSpace,32>(data, A.data, lambda.values());

  if (pmap != nullptr) {
    Kokkos::fence();
    ret = pmap->allReduce(ret);
  }

  return ret;
}

template <typename ExecSpace>
void Genten::FacMatrixT<ExecSpace>::
diagonalShift(const ttb_real shift) const
{
  auto d = data;
  const unsigned n =
    data.extent(0) <= data.extent(1) ? data.extent(0) : data.extent(1);
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,n),
                       KOKKOS_LAMBDA(const unsigned i)
  {
    d(i,i) += shift;
  });
}

template <typename ExecSpace>
Genten::FacMatrixT<ExecSpace> Genten::FacMatrixT<ExecSpace>::
khatrirao(const FacMatrixT& B) const
{
  const ttb_indx A_ncols = data.extent(1);
  const ttb_indx B_ncols = B.data.extent(1);
  gt_assert(A_ncols == B_ncols);

  const ttb_indx A_nrows = data.extent(0);
  const ttb_indx B_nrows = B.data.extent(0);
  const ttb_indx P_nrows = A_nrows*B_nrows;

  FacMatrixT P(P_nrows, A_ncols, nullptr, false);
  auto A_data = data;
  auto B_data = B.data;
  auto P_data = P.data;
  using PolicyType = Kokkos::TeamPolicy<ExecSpace>;
  Kokkos::parallel_for(
    PolicyType(P_nrows, 1, Kokkos::AUTO),
    KOKKOS_LAMBDA(const typename PolicyType::member_type& team)
  {
    const ttb_indx P_row = team.league_rank();
    const ttb_indx A_row = P_row / B_nrows;
    const ttb_indx B_row = P_row - B_nrows*A_row;
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,A_ncols),
                         [&](const ttb_indx col)
    {
      P_data(P_row,col) = A_data(A_row,col)*B_data(B_row,col);
    });
  });

  return P;
}

#define INST_MACRO(SPACE) template class Genten::FacMatrixT<SPACE>;
GENTEN_INST(INST_MACRO)
