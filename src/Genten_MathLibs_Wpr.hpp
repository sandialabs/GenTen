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
 * @file Genten_MathLibs_Wpr.h
 * @brief Wrapper functions for calls to the BLAS/LAPACK libraries.
 *
 * Wrappers separate calls in Genten from the actual implementations.
 * The Genten package provides a "default" implementation of most BLAS/LAPACK
 * functions, but users should install an appropriately tuned library for their
 * hardware and link with it instead.
 */

#pragma once

#include "Genten_Util.hpp"

#if defined(KOKKOS_ENABLE_CUDA) || defined(ENABLE_SYCL_FOR_CUDA)
#include "Genten_CublasHandle.hpp"
#include "Genten_CusolverHandle.hpp"
#endif

#if defined(KOKKOS_ENABLE_HIP)
#if defined(HAVE_ROCBLAS)
#include "Genten_RocblasHandle.hpp"
#include "rocblas.h"
#endif
#if defined(HAVE_ROCSOLVER)
#include "Genten_RocblasHandle.hpp"
#include "rocsolver.h"
#endif
#endif


namespace Genten
{

  //
  // Double precision
  //
  // COPY Y = X
  void copy(ttb_indx n, const double * x, ttb_indx incx, double * y, ttb_indx incy);

  // SCAL - Scale the vector x by alpha
  void scal(ttb_indx n, double alpha, double * x, ttb_indx incx);

  // NRM2 - Compute the 2-norm of a vector.
  double nrm2(ttb_indx len, const double * vec, ttb_indx stride);

  // NRM1 - Compute the 1-norm of a vector.
  double nrm1(ttb_indx len, const double * vec, ttb_indx stride);

  // IMAX - Compute index of first entry with the largest absolute value
  ttb_indx imax(ttb_indx len, const double * vec, ttb_indx stride);

  // DOT - Compute the dot product of 2 vectors.
  double dot(ttb_indx len, const double * x, ttb_indx incx, const double * y, ttb_indx incy);

  // AXPY y = a*x + y
  void axpy(ttb_indx n, double a, const double * x, ttb_indx incx, double * y, ttb_indx incy);

  /*
   * GEMV - Compute y = alpha A*x  + beta y , if trans='N'
   *                y = alpha A'*x + beta y , if trans='T'
   *
   * Where A is m x n and is stored in column major form.
   */
  void gemv(char trans, ttb_indx m, ttb_indx n, double alpha, const double * A, ttb_indx LDA,
            const double * x, ttb_indx incx, double beta, double *y, ttb_indx incy);

  // GER - Compute A = alpha * x * y' + A
  void ger(ttb_indx m, ttb_indx n, double alpha, const double * x, ttb_indx incx,
           const double * y, ttb_indx incy, double * a, ttb_indx lda);

  // GEMM - Compute C = alpha * A*B + beta * C
  /* C is always m x n.
     C = A*B  : transa = 'N', transb = 'N', A is m x k, B is k x n.
     C = A'*B : transa = 'T', transb = 'N', A is k x m, B is k x n.
     C = A*B' : transa = 'N', transb = 'T', A is m x k, B is n x k.
     C = A'*B': transa = 'T', transb = 'T', A is k x m, B is n x k.

     *** NOTE: The code has not been implemented in the default library for transb='T'. ***
     */
  void gemm(char transa, char transb, ttb_indx m, ttb_indx n, ttb_indx k, double alpha,
            const double * a, ttb_indx ldaa, const double * b, ttb_indx ldab,
            double beta, double * c, ttb_indx ldac);

  // SYRK - Compute C = alpha * A*A' + beta * C
  /* C is always n x n.
     C = A*A' : trans = 'N', A is n x k.
     C = A'*A : trans = 'T', A is k x n.
     Only the upper/lower triangular portion is computed as determined up uplo.

     *** NOTE: The default implementation just uses gemm() ***
     */
  void syrk(char uplo, char trans, ttb_indx n, ttb_indx k, double alpha,
            const double * a, ttb_indx ldaa,
            double beta, double * c, ttb_indx ldac);

  // VMUL - Compute A = A .* B (Hadamard or elementwise product)
  void vmul(const ttb_indx n, double * a, const double * b);

  // GESV - Compute solution to linear system AX = B.
  /*
    Solve AX = B where A is an n x n matrix, B is an n x nrhs matrix,
    and X is the n x nrhs matrix to be computed. It is assumed that
    the matrices are stored columnwise so that the (i,j) entry is in
    position (i + n*j) in the one-dimensional array.

    a    - On entry, the n x n matrix A stored columnwise.
    On exit, the factors L and U stored in the lower and upper halves
    of the matrix (the unit diagonal of L is not stored).
    b    - On entry, the n x nrhs matrix B (of right-hand-sides) stored
    columnwise.
    On exit, the n x nrhs solution matrix X stored columnwise.

    throws string exception if the solve failed.
  */
  void gesv(ttb_indx n, ttb_indx nrhs, double * a, ttb_indx lda, double * b, ttb_indx ldb);

  // POSV - Compute solution to linear system AX = B for SPD A.
  /*
    Solve AX = B where A is an n x n matrix, B is an n x nrhs matrix,
    and X is the n x nrhs matrix to be computed. It is assumed that
    the matrices are stored columnwise so that the (i,j) entry is in
    position (i + n*j) in the one-dimensional array.

    uplo - On entry, whether upper or lower triangle of A is accessed
    a    - On entry, the n x n matrix A stored columnwise.
    On exit, the factors L and U stored in the lower and upper halves
    of the matrix (the unit diagonal of L is not stored).
    b    - On entry, the n x nrhs matrix B (of right-hand-sides) stored
    columnwise.
    On exit, the n x nrhs solution matrix X stored columnwise.

    Returns false if the solve failed (doesn't throw).
  */
  bool posv (char uplo, ttb_indx n, ttb_indx nrhs, double * a, ttb_indx lda, double * b, ttb_indx ldb);

  // SYSV - Compute solution to linear system AX = B for symmetric, indefinite A
  /*
    Solve AX = B where A is an n x n matrix, B is an n x nrhs matrix,
    and X is the n x nrhs matrix to be computed. It is assumed that
    the matrices are stored columnwise so that the (i,j) entry is in
    position (i + n*j) in the one-dimensional array.

    uplo - On entry, whether upper or lower triangle of A is accessed
    a    - On entry, the n x n matrix A stored columnwise.
    On exit, the factors L and U stored in the lower and upper halves
    of the matrix (the unit diagonal of L is not stored).
    b    - On entry, the n x nrhs matrix B (of right-hand-sides) stored
    columnwise.
    On exit, the n x nrhs solution matrix X stored columnwise.

    throws string exception if the solve failed.
  */
  void sysv(char uplo, ttb_indx n, ttb_indx nrhs, double * a, ttb_indx lda, double * b, ttb_indx ldb);

  // GELSY - Compute minimum norm solution to linear, least squares problem
  /*
    Computes minumum norm solution to ||AX-B||_2 where A is an m x n matrix,
    B is an m x nrhs matrix, and X is the n x nrhs matrix to be computed, where
    A may be rank deficient.  It is assumed the matrices are stored in column
    major order.  The algorithm uses a complete orthogonal factorization via
    column-pivoted QR.  The parameter rcond is used to determine where to
    truncate R, determining the effective rank of A.

    a    - On entry, the m x n matrix A stored columnwise.
    On exit, a has been overwritten by details of the QR factoriztion.
    b    - On entry, the m x nrhs matrix B (of right-hand-sides) stored
    columnwise.
    On exit, the n x nrhs solution matrix X stored columnwise.
    rcond- 1/rcond determines the maximum condition number of R used in the
    least-squares solve.

    throws string exception if the solve failed.
    returns the effective rank of A.
  */
  ttb_indx gelsy(ttb_indx m, ttb_indx n, ttb_indx nrhs, double * a, ttb_indx lda, double * b, ttb_indx ldb, double rcond);

  //
  // Single precision
  //

  // COPY Y = X
  void copy(ttb_indx n, const float * x, ttb_indx incx, float * y, ttb_indx incy);

  // SCAL - Scale the vector x by alpha
  void scal(ttb_indx n, float alpha, float * x, ttb_indx incx);

  // NRM2 - Compute the 2-norm of a vector.
  float nrm2(ttb_indx len, const float * vec, ttb_indx stride);

  // NRM1 - Compute the 1-norm of a vector.
  float nrm1(ttb_indx len, const float * vec, ttb_indx stride);

  // IMAX - Compute index of first entry with the largest absolute value
  ttb_indx imax(ttb_indx len, const float * vec, ttb_indx stride);

  // DOT - Compute the dot product of 2 vectors.
  float dot(ttb_indx len, const float * x, ttb_indx incx, const float * y, ttb_indx incy);

  // AXPY y = a*x + y
  void axpy(ttb_indx n, float a, const float * x, ttb_indx incx, float * y, ttb_indx incy);

  /*
   * GEMV - Compute y = alpha A*x  + beta y , if trans='N'
   *                y = alpha A'*x + beta y , if trans='T'
   *
   * Where A is m x n and is stored in column major form.
   */
  void gemv(char trans, ttb_indx m, ttb_indx n, float alpha, const float * A, ttb_indx LDA,
            const float * x, ttb_indx incx, float beta, float *y, ttb_indx incy);

  // GER - Compute A = alpha * x * y' + A
  void ger(ttb_indx m, ttb_indx n, float alpha, const float * x, ttb_indx incx,
           const float * y, ttb_indx incy, float * a, ttb_indx lda);

  // GEMM - Compute C = alpha * A*B + beta * C
  /* C is always m x n.
     C = A*B  : transa = 'N', transb = 'N', A is m x k, B is k x n.
     C = A'*B : transa = 'T', transb = 'N', A is k x m, B is k x n.
     C = A*B' : transa = 'N', transb = 'T', A is m x k, B is n x k.
     C = A'*B': transa = 'T', transb = 'T', A is k x m, B is n x k.

     *** NOTE: The code has not been implemented in the default library for transb='T'. ***
     */
  void gemm(char transa, char transb, ttb_indx m, ttb_indx n, ttb_indx k, float alpha,
            const float * a, ttb_indx ldaa, const float * b, ttb_indx ldab,
            float beta, float * c, ttb_indx ldac);

  // SYRK - Compute C = alpha * A*A' + beta * C
  /* C is always n x n.
     C = A*A' : trans = 'N', A is n x k.
     C = A'*A : trans = 'T', A is k x n.
     Only the upper/lower triangular portion is computed as determined up uplo.

     *** NOTE: The default implementation just uses gemm() ***
     */
  void syrk(char uplo, char trans, ttb_indx n, ttb_indx k, float alpha,
            const float * a, ttb_indx ldaa,
            float beta, float * c, ttb_indx ldac);

  // VMUL - Compute A = A .* B (Hadamard or elementwise product)
  void vmul(const ttb_indx n, float * a, const float * b);

  // GESV - Compute solution to linear system AX = B.
  /*
    Solve AX = B where A is an n x n matrix, B is an n x nrhs matrix,
    and X is the n x nrhs matrix to be computed. It is assumed that
    the matrices are stored columnwise so that the (i,j) entry is in
    position (i + n*j) in the one-dimensional array.

    a    - On entry, the n x n matrix A stored columnwise.
    On exit, the factors L and U stored in the lower and upper halves
    of the matrix (the unit diagonal of L is not stored).
    b    - On entry, the n x nrhs matrix B (of right-hand-sides) stored
    columnwise.
    On exit, the n x nrhs solution matrix X stored columnwise.

    throws string exception if the solve failed
  */
  void gesv(ttb_indx n, ttb_indx nrhs, float * a, ttb_indx lda, float * b, ttb_indx ldb);

  // POSV - Compute solution to linear system AX = B for SPD A.
  /*
    Solve AX = B where A is an n x n matrix, B is an n x nrhs matrix,
    and X is the n x nrhs matrix to be computed. It is assumed that
    the matrices are stored columnwise so that the (i,j) entry is in
    position (i + n*j) in the one-dimensional array.

    uplo - On entry, whether upper or lower triangle of A is accessed
    a    - On entry, the n x n matrix A stored columnwise.
    On exit, the factors L and U stored in the lower and upper halves
    of the matrix (the unit diagonal of L is not stored).
    b    - On entry, the n x nrhs matrix B (of right-hand-sides) stored
    columnwise.
    On exit, the n x nrhs solution matrix X stored columnwise.

    Returns false if the solve failed (doesn't throw).
  */
  bool posv (char uplo, ttb_indx n, ttb_indx nrhs, float * a, ttb_indx lda, float * b, ttb_indx ldb);

  // SYSV - Compute solution to linear system AX = B for symmetric, indefinite A
  /*
    Solve AX = B where A is an n x n matrix, B is an n x nrhs matrix,
    and X is the n x nrhs matrix to be computed. It is assumed that
    the matrices are stored columnwise so that the (i,j) entry is in
    position (i + n*j) in the one-dimensional array.

    uplo - On entry, whether upper or lower triangle of A is accessed
    a    - On entry, the n x n matrix A stored columnwise.
    On exit, the factors L and U stored in the lower and upper halves
    of the matrix (the unit diagonal of L is not stored).
    b    - On entry, the n x nrhs matrix B (of right-hand-sides) stored
    columnwise.
    On exit, the n x nrhs solution matrix X stored columnwise.

    throws string exception if the solve failed.
  */
  void sysv(char uplo, ttb_indx n, ttb_indx nrhs, float * a, ttb_indx lda, float * b, ttb_indx ldb);

  // GELSY - Compute minimum norm solution to linear, least squares problem
  /*
    Computes minumum norm solution to ||AX-B||_2 where A is an m x n matrix,
    B is an m x nrhs matrix, and X is the n x nrhs matrix to be computed, where
    A may be rank deficient.  It is assumed the matrices are stored in column
    major order.  The algorithm uses a complete orthogonal factorization via
    column-pivoted QR.  The parameter rcond is used to determine where to
    truncate R, determining the effective rank of A.

    a    - On entry, the m x n matrix A stored columnwise.
    On exit, a has been overwritten by details of the QR factoriztion.
    b    - On entry, the m x nrhs matrix B (of right-hand-sides) stored
    columnwise.
    On exit, the n x nrhs solution matrix X stored columnwise.
    rcond- 1/rcond determines the maximum condition number of R used in the
    least-squares solve.

    throws string exception if the solve failed.
    returns the effective rank of A.
  */
  ttb_indx gelsy(ttb_indx m, ttb_indx n, ttb_indx nrhs, float * a, ttb_indx lda, float * b, ttb_indx ldb, float rcond);
}

// Implementations of GEMM for various architectures

namespace Genten {
namespace Impl {

template <typename ExecSpace, typename Scalar, typename Enabled = void>
struct GemmImpl {};

template <typename ExecSpace, typename Scalar>
struct GemmImpl<ExecSpace, Scalar,
                std::enable_if_t< is_host_space<ExecSpace>::value > >
{
  static void apply(const bool trans_a, const bool trans_b,
                    const ttb_indx m, const ttb_indx n, const ttb_indx k,
                    const Scalar alpha, const Scalar *A, const ttb_indx lda,
                    const Scalar *B, const ttb_indx ldb,
                    const Scalar beta, Scalar *C, const ttb_indx ldc)
  {
    const char ta = trans_a ? 'T' : 'N';
    const char tb = trans_b ? 'T' : 'N';
    Genten::gemm(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
};

#if (defined(KOKKOS_ENABLE_CUDA) || defined(ENABLE_SYCL_FOR_CUDA)) && defined(HAVE_CUBLAS)

template <typename ExecSpace>
struct GemmImpl<
  ExecSpace, double,
  std::enable_if_t<
    (is_cuda_space<ExecSpace>::value || is_sycl_space<ExecSpace>::value) > >
{
  using Scalar = double;
  static void apply(const bool trans_a, const bool trans_b,
                    const ttb_indx m, const ttb_indx n, const ttb_indx k,
                    const Scalar alpha, const Scalar *A, const ttb_indx lda,
                    const Scalar *B, const ttb_indx ldb,
                    const Scalar beta, Scalar *C, const ttb_indx ldc)
  {
    const cublasOperation_t ta = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t tb = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasStatus_t status = cublasDgemm(
      CublasHandle::get(), ta, tb, m, n, k, &alpha, A, lda, B, ldb, &beta,
      C, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::stringstream ss;
      ss << "Error!  cublasDgemm() failed with status "
         << status;
      Genten::error(ss.str());
    }
  }
};

template <typename ExecSpace>
struct GemmImpl<
  ExecSpace, float,
  std::enable_if_t<
    (is_cuda_space<ExecSpace>::value || is_sycl_space<ExecSpace>::value) > >
{
  using Scalar = float;
  static void apply(const bool trans_a, const bool trans_b,
                    const ttb_indx m, const ttb_indx n, const ttb_indx k,
                    const Scalar alpha, const Scalar *A, const ttb_indx lda,
                    const Scalar *B, const ttb_indx ldb,
                    const Scalar beta, Scalar *C, const ttb_indx ldc)
  {
    const cublasOperation_t ta = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t tb = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasStatus_t status = cublasSgemm(
      CublasHandle::get(), ta, tb, m, n, k, &alpha, A, lda, B, ldb, &beta,
      C, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::stringstream ss;
      ss << "Error!  cublasSgemm() failed with status "
         << status;
      Genten::error(ss.str());
    }
  }
};

#endif

#if defined(KOKKOS_ENABLE_HIP) && defined(HAVE_ROCBLAS)

template <typename ExecSpace>
struct GemmImpl<
  ExecSpace, double,
  std::enable_if_t< is_hip_space<ExecSpace>::value > >
{
  using Scalar = double;
  static void apply(const bool trans_a, const bool trans_b,
                    const ttb_indx m, const ttb_indx n, const ttb_indx k,
                    const Scalar alpha, const Scalar *A, const ttb_indx lda,
                    const Scalar *B, const ttb_indx ldb,
                    const Scalar beta, Scalar *C, const ttb_indx ldc)
  {
    const rocblas_operation ta = trans_a ? rocblas_operation_transpose : rocblas_operation_none;
    const rocblas_operation tb = trans_b ? rocblas_operation_transpose : rocblas_operation_none;
    rocblas_status status = rocblas_dgemm(
      RocblasHandle::get(), ta, tb, m, n, k, &alpha, A, lda, B, ldb, &beta,
      C, ldc);
     if (status != rocblas_status_success) {
       std::stringstream ss;
       ss << "Error!  rocblas_dgemm() failed with status "
          << status;
       Genten::error(ss.str());
    }
  }
};

template <typename ExecSpace>
struct GemmImpl<
  ExecSpace, float,
  std::enable_if_t< is_hip_space<ExecSpace>::value > >
{
  using Scalar = float;
  static void apply(const bool trans_a, const bool trans_b,
                    const ttb_indx m, const ttb_indx n, const ttb_indx k,
                    const Scalar alpha, const Scalar *A, const ttb_indx lda,
                    const Scalar *B, const ttb_indx ldb,
                    const Scalar beta, Scalar *C, const ttb_indx ldc)
  {
    const rocblas_operation ta = trans_a ? rocblas_operation_transpose : rocblas_operation_none;
    const rocblas_operation tb = trans_b ? rocblas_operation_transpose : rocblas_operation_none;
    rocblas_status status = rocblas_sgemm(
      RocblasHandle::get(), ta, tb, m, n, k, &alpha, A, lda, B, ldb, &beta,
      C, ldc);
     if (status != rocblas_status_success) {
       std::stringstream ss;
       ss << "Error!  rocblas_sgemm() failed with status "
          << status;
       Genten::error(ss.str());
    }
  }
};

#endif

}

template <typename ViewA, typename ViewB, typename ViewC>
void
gemm(const bool trans_a, const bool trans_b, const ttb_real alpha,
     const ViewA& A, const ViewB& B,
     const ttb_real beta, const ViewC& C)
{
  GENTEN_TIME_MONITOR("GEMM");

  // Ensure execution spaces match
  static_assert(std::is_same_v<typename ViewA::execution_space,
                               typename ViewB::execution_space>);
  static_assert(std::is_same_v<typename ViewA::execution_space,
                               typename ViewC::execution_space>);

  // Ensure all views are rank-2
  static_assert(unsigned(ViewA::rank) == 2u);
  static_assert(unsigned(ViewB::rank) == 2u);
  static_assert(unsigned(ViewC::rank) == 2u);

  using ExecSpace = typename ViewA::execution_space;
  using LayoutA = typename ViewA::array_layout;
  using LayoutB = typename ViewB::array_layout;
  using LayoutC = typename ViewC::array_layout;

  if constexpr(std::is_same_v<LayoutA, Kokkos::LayoutLeft> &&
               std::is_same_v<LayoutB, Kokkos::LayoutLeft> &&
               std::is_same_v<LayoutC, Kokkos::LayoutLeft>) {
    // C = alpha * op(A) * op(B) + beta * C
    const ttb_indx m = C.extent(0);
    const ttb_indx n = C.extent(1);
    const ttb_indx k = trans_a ? A.extent(0) : A.extent(1);
    const ttb_indx lda = A.stride(1);
    const ttb_indx ldb = B.stride(1);
    const ttb_indx ldc = C.stride(1);
    Impl::GemmImpl<ExecSpace,ttb_real>::apply(
      trans_a, trans_b, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta,
      C.data(), ldc);
  }
  else if constexpr(std::is_same_v<LayoutA, Kokkos::LayoutRight> &&
                    std::is_same_v<LayoutB, Kokkos::LayoutLeft> &&
                    std::is_same_v<LayoutC, Kokkos::LayoutLeft>) {
    // C = alpha * op'(A) * op(B) + beta * C
    const ttb_indx m = C.extent(0);
    const ttb_indx n = C.extent(1);
    const ttb_indx k = trans_a ? A.extent(1) : A.extent(0);
    const ttb_indx lda = A.stride(0);
    const ttb_indx ldb = B.stride(1);
    const ttb_indx ldc = C.stride(1);
    Impl::GemmImpl<ExecSpace,ttb_real>::apply(
      !trans_a, trans_b, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta,
      C.data(), ldc);
  }
  else if constexpr(std::is_same_v<LayoutA, Kokkos::LayoutLeft> &&
                    std::is_same_v<LayoutB, Kokkos::LayoutRight> &&
                    std::is_same_v<LayoutC, Kokkos::LayoutLeft>) {
    // C = alpha * op(A) * op'(B) + beta * C
    const ttb_indx m = C.extent(0);
    const ttb_indx n = C.extent(1);
    const ttb_indx k = trans_a ? A.extent(0) : A.extent(1);
    const ttb_indx lda = A.stride(1);
    const ttb_indx ldb = B.stride(0);
    const ttb_indx ldc = C.stride(1);
    Impl::GemmImpl<ExecSpace,ttb_real>::apply(
      trans_a, !trans_b, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta,
      C.data(), ldc);
  }
  else if constexpr(std::is_same_v<LayoutA, Kokkos::LayoutRight> &&
                    std::is_same_v<LayoutB, Kokkos::LayoutRight> &&
                    std::is_same_v<LayoutC, Kokkos::LayoutLeft>) {
    // C = alpha * op'(A) * op'(B) + beta * C
    const ttb_indx m = C.extent(0);
    const ttb_indx n = C.extent(1);
    const ttb_indx k = trans_a ? A.extent(1) : A.extent(0);
    const ttb_indx lda = A.stride(0);
    const ttb_indx ldb = B.stride(0);
    const ttb_indx ldc = C.stride(1);
    Impl::GemmImpl<ExecSpace,ttb_real>::apply(
      !trans_a, !trans_b, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta,
      C.data(), ldc);
  }
  else if constexpr(std::is_same_v<LayoutA, Kokkos::LayoutRight> &&
                    std::is_same_v<LayoutB, Kokkos::LayoutRight> &&
                    std::is_same_v<LayoutC, Kokkos::LayoutRight>) {
    // C' = alpha * op(B') * op(A') + beta * C'
    const ttb_indx m = C.extent(1);
    const ttb_indx n = C.extent(0);
    const ttb_indx k = trans_b ? B.extent(1) : B.extent(0);
    const ttb_indx lda = A.stride(0);
    const ttb_indx ldb = B.stride(0);
    const ttb_indx ldc = C.stride(0);
    Impl::GemmImpl<ExecSpace,ttb_real>::apply(
      trans_b, trans_a, m, n, k, alpha, B.data(), ldb, A.data(), lda, beta,
      C.data(), ldc);
  }
  else if constexpr(std::is_same_v<LayoutA, Kokkos::LayoutLeft> &&
                    std::is_same_v<LayoutB, Kokkos::LayoutRight> &&
                    std::is_same_v<LayoutC, Kokkos::LayoutRight>) {
    // C' = alpha * op(B') * op'(A) + beta * C'
    const ttb_indx m = C.extent(1);
    const ttb_indx n = C.extent(0);
    const ttb_indx k = trans_b ? B.extent(1) : B.extent(0);
    const ttb_indx lda = A.stride(1);
    const ttb_indx ldb = B.stride(0);
    const ttb_indx ldc = C.stride(0);
    Impl::GemmImpl<ExecSpace,ttb_real>::apply(
      trans_b, !trans_a, m, n, k, alpha, B.data(), ldb, A.data(), lda, beta,
      C.data(), ldc);
  }
  else if constexpr(std::is_same_v<LayoutA, Kokkos::LayoutLeft> &&
                    std::is_same_v<LayoutB, Kokkos::LayoutLeft> &&
                    std::is_same_v<LayoutC, Kokkos::LayoutRight>) {
    // C' = alpha * op'(B) * op'(A) + beta * C'
    const ttb_indx m = C.extent(1);
    const ttb_indx n = C.extent(0);
    const ttb_indx k = trans_b ? B.extent(0) : B.extent(1);
    const ttb_indx lda = A.stride(1);
    const ttb_indx ldb = B.stride(1);
    const ttb_indx ldc = C.stride(0);
    Impl::GemmImpl<ExecSpace,ttb_real>::apply(
      !trans_b, !trans_a, m, n, k, alpha, B.data(), ldb, A.data(), lda, beta,
      C.data(), ldc);
  }
  else if constexpr(std::is_same_v<LayoutA, Kokkos::LayoutRight> &&
                    std::is_same_v<LayoutB, Kokkos::LayoutLeft> &&
                    std::is_same_v<LayoutC, Kokkos::LayoutRight>) {
    // C' = alpha * op'(B) * op(A') + beta * C'
    const ttb_indx m = C.extent(1);
    const ttb_indx n = C.extent(0);
    const ttb_indx k = trans_b ? B.extent(1) : B.extent(0);
    const ttb_indx lda = A.stride(0);
    const ttb_indx ldb = B.stride(1);
    const ttb_indx ldc = C.stride(0);
    Impl::GemmImpl<ExecSpace,ttb_real>::apply(
      !trans_b, trans_a, m, n, k, alpha, B.data(), ldb, A.data(), lda, beta,
      C.data(), ldc);
  }
  else
    Genten::error("Unhandled layout combination in GEMM");

  Kokkos::fence();
}

}
