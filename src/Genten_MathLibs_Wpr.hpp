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

    throws string exception if the solve failed
  */
  void gesv(ttb_indx n, ttb_indx nrhs, double * a, ttb_indx lda, double * b, ttb_indx ldb);

  // POSV - Solve Ax=b for x assuming A is symmetric positive definite.
  /*
   * Use LAPACK routine dposv to solve for x using dense Cholesky factorization.
   * TTBC++ does not have a default dposv, so the use of this routine requires
   * linking with an LAPACK library.
   *
   *   n - Size of n x n matrix A.
   *   A - Dense n x n matrix A, but only the lower triangle is used
   *       (assuming column major order, where row index varies fastest).
   *   b - On entry, the n-vector on the right-hand side.
   *       On exit, the solution x.
   *
   *   throws string exception if the solve failed
   */
  void posv (char uplo, ttb_indx n, ttb_indx nrhs, double * a, ttb_indx lda, double * b, ttb_indx ldb);

  // SYSV - Compute solution to linear system AX = B for symmetric indefinite A.
  void sysv(char uplo, ttb_indx n, ttb_indx nrhs, double * a, ttb_indx lda, double * b, ttb_indx ldb);

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

  // POSV - Solve Ax=b for x assuming A is symmetric positive definite.
  /*
   * Use LAPACK routine dposv to solve for x using dense Cholesky factorization.
   * TTBC++ does not have a default dposv, so the use of this routine requires
   * linking with an LAPACK library.
   *
   *   n - Size of n x n matrix A.
   *   A - Dense n x n matrix A, but only the lower triangle is used
   *       (assuming column major order, where row index varies fastest).
   *   b - On entry, the n-vector on the right-hand side.
   *       On exit, the solution x.
   *
   *   throws string exception if the solve failed
   */
  void posv (char uplo, ttb_indx n, ttb_indx nrhs, float * a, ttb_indx lda, float * b, ttb_indx ldb);

  // SYSV - Compute solution to linear system AX = B for symmetric indefinite A.
  void sysv(char uplo, ttb_indx n, ttb_indx nrhs, float * a, ttb_indx lda, float * b, ttb_indx ldb);
}
