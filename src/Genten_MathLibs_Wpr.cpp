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


#include "Genten_MathLibs.hpp"
#include "Genten_MathLibs_Wpr.hpp"

//
// Double precision
//

void Genten::copy(ttb_indx n, const double * x, ttb_indx incx, double * y, ttb_indx incy)
{
  // Casting, possibly to different type
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int incx_ml = (ttb_blas_int) incx;
  ttb_blas_int incy_ml = (ttb_blas_int) incy;

  // Just removing const
  double * x_ml = (double *) x;

  ::dcopy(&n_ml, x_ml, &incx_ml, y, &incy_ml);
}

void Genten::scal(ttb_indx n, double alpha, double * x, ttb_indx incx)
{
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int incx_ml = (ttb_blas_int) incx;
  ::dscal(&n_ml, &alpha, x, &incx_ml);
}

double Genten::nrm2(ttb_indx n, const double * x, ttb_indx incx)
{
  ttb_blas_int n_ml = (ttb_blas_int) n;
  double * x_ml = (double *) x; // remove const
  ttb_blas_int incx_ml = (ttb_blas_int) incx;

  return(::dnrm2(&n_ml, x_ml, &incx_ml));
}

double Genten::nrm1(ttb_indx n, const double * x, ttb_indx incx)
{
  ttb_blas_int n_ml = (ttb_blas_int) n;
  double * x_ml = (double *) x; // remove const
  ttb_blas_int incx_ml = (ttb_blas_int) incx;

  return(::dasum(&n_ml, x_ml, &incx_ml));
}

ttb_indx Genten::imax(ttb_indx n, const double * x, ttb_indx incx)
{
  ttb_blas_int n_ml = (ttb_blas_int) n;
  double * x_ml = (double *) x; // remove const
  ttb_blas_int incx_ml = (ttb_blas_int) incx;
  ttb_blas_int idx = ::idamax(&n_ml, x_ml, &incx_ml);

  // idamax returns index between 1 and n (Fortran 1-indexing),
  // so subtract 1 to return index between 0 and n-1 (0-indexing)
  return((ttb_indx) idx-1);
}


double Genten::dot(ttb_indx n, const double * x, ttb_indx incx, const double * y, ttb_indx incy)
{
  ttb_blas_int n_ml = (ttb_blas_int) n;
  double * x_ml = (double *) x; // remove const
  ttb_blas_int incx_ml = (ttb_blas_int) incx;
  double * y_ml = (double *) y; // remove const
  ttb_blas_int incy_ml = (ttb_blas_int) incy;

  return(::ddot(&n_ml, x_ml, &incx_ml, y_ml, &incy_ml));
}

void Genten::axpy(ttb_indx n, double a, const double * x, ttb_indx incx, double * y, ttb_indx incy)
{
  ttb_blas_int n_ml = (ttb_blas_int) n;
  double a_ml = (double) a;
  double * x_ml = (double *) x; // remove const, don't change type
  ttb_blas_int incx_ml = (ttb_blas_int) incx;
  double * y_ml = (double *) y; // remove const, don't change type
  ttb_blas_int incy_ml = (ttb_blas_int) incy;

  ::daxpy(&n_ml, &a_ml, x_ml, &incx_ml, y_ml, &incy_ml);
}

void Genten::gemv(char trans, ttb_indx m, ttb_indx n, double alpha, const double * a, ttb_indx lda,
                  const double * x, ttb_indx incx, double beta, double *y, ttb_indx incy)
{
  // casting
  ttb_blas_int m_ml = (ttb_blas_int) m;
  ttb_blas_int n_ml = (ttb_blas_int) n;
  double alpha_ml = (double) alpha;
  ttb_blas_int lda_ml = (ttb_blas_int) lda;
  ttb_blas_int incx_ml = (ttb_blas_int) incx;
  double beta_ml = (double) beta;
  ttb_blas_int incy_ml = (ttb_blas_int) incy;

  // Just remove const; don't change type
  double * a_ml = (double *) a;
  double * x_ml = (double *) x;

  ::dgemv(&trans, &m_ml, &n_ml, &alpha_ml, a_ml, &lda_ml, x_ml, &incx_ml, &beta_ml, y, &incy_ml);
}

void Genten::ger(ttb_indx m, ttb_indx n, double alpha, const double * x, ttb_indx incx,
                 const double * y, ttb_indx incy, double * a, ttb_indx lda)

{
  // casting
  ttb_blas_int m_ml = (ttb_blas_int) m;
  ttb_blas_int n_ml = (ttb_blas_int) n;
  double alpha_ml = (double) alpha;
  ttb_blas_int incx_ml = (ttb_blas_int) incx;
  ttb_blas_int incy_ml = (ttb_blas_int) incy;
  ttb_blas_int lda_ml = (ttb_blas_int) lda;

  // Just remove const; don't change type
  double * x_ml = (double *) x;
  double * y_ml = (double *) y;

  ::dger(&m_ml, &n_ml, &alpha_ml, x_ml, &incx_ml, y_ml, &incy_ml, a, &lda_ml);
}

void Genten::gemm(char transa, char transb, ttb_indx m, ttb_indx n, ttb_indx k, double alpha,
                  const double * a, ttb_indx ldaa, const double * b, ttb_indx ldab,
                  double beta, double * c, ttb_indx ldac)
{
  ttb_blas_int m_ml = (ttb_blas_int) m;
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int k_ml = (ttb_blas_int) k;
  ttb_blas_int ldaa_ml = (ttb_blas_int) ldaa;
  ttb_blas_int ldab_ml = (ttb_blas_int) ldab;
  ttb_blas_int ldac_ml = (ttb_blas_int) ldac;

  // Just remove const; don't change type
  double * a_ml = (double *) a;
  double * b_ml = (double *) b;

  ::dgemm(&transa, &transb, &m_ml, &n_ml, &k_ml, &alpha, a_ml, &ldaa_ml, b_ml, &ldab_ml, &beta, c, &ldac_ml);
}

void Genten::syrk(char uplo, char trans, ttb_indx n, ttb_indx k, double alpha,
                  const double * a, ttb_indx ldaa,
                  double beta, double * c, ttb_indx ldac)
{
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int k_ml = (ttb_blas_int) k;
  ttb_blas_int ldaa_ml = (ttb_blas_int) ldaa;
  ttb_blas_int ldac_ml = (ttb_blas_int) ldac;

  // Just remove const; don't change type
  double * a_ml = (double *) a;

  ::dsyrk(&uplo, &trans, &n_ml, &k_ml, &alpha, a_ml, &ldaa_ml, &beta, c, &ldac_ml);
}

void Genten::gesv(ttb_indx n, ttb_indx nrhs, double * a, ttb_indx lda, double * b, ttb_indx ldb)
{
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int nrhs_ml = (ttb_blas_int) nrhs;
  ttb_blas_int lda_ml = (ttb_blas_int) lda;
  ttb_blas_int ldb_ml = (ttb_blas_int) ldb;

  // Initialize output info and create pivot array
  ttb_blas_int info_ml = 0;
  ttb_blas_int * ipiv_ml = new ttb_blas_int[ n ];

  ::dgesv(&n_ml, &nrhs_ml, a, &lda_ml, ipiv_ml, b, &ldb_ml, &info_ml);

  delete[] ipiv_ml;

  // Check output info and free pivot array
  if (info_ml < 0)
  {
    Genten::error("Genten::gesv - argument error in call to dgesv");
  }
  if (info_ml > 0)
  {
    Genten::error("Genten::gesv - dgesv failed because matrix is singular");
  }
}

void Genten::sysv(char uplo, ttb_indx n, ttb_indx nrhs, double * a, ttb_indx lda, double * b, ttb_indx ldb)
{
#if !defined(LAPACK_FOUND)
  Genten::error("Genten::sysv - not found, must link with an LAPACK library.");
#else
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int nrhs_ml = (ttb_blas_int) nrhs;
  ttb_blas_int lda_ml = (ttb_blas_int) lda;
  ttb_blas_int ldb_ml = (ttb_blas_int) ldb;

  // Initialize output info and create pivot array
  ttb_blas_int info_ml = 0;
  ttb_blas_int * ipiv_ml = new ttb_blas_int[ n ];

  // Workspace query
  ttb_blas_int lwork = -1;
  double work_tmp = 0;
  ::dsysv(&uplo, &n_ml, &nrhs_ml, a, &lda_ml, ipiv_ml, b, &ldb_ml, &work_tmp, &lwork, &info_ml);

  lwork = ttb_blas_int(work_tmp);
  double * work = new double[lwork];
  ::dsysv(&uplo, &n_ml, &nrhs_ml, a, &lda_ml, ipiv_ml, b, &ldb_ml, work, &lwork, &info_ml);

  delete[] work;
  delete[] ipiv_ml;

  // Check output info and free pivot array
  if (info_ml < 0)
  {
    Genten::error("Genten::sysv - argument error in call to dsysv");
  }
  if (info_ml > 0)
  {
    Genten::error("Genten::sysv - sysv failed because matrix is singular");
  }
#endif
}

void Genten::vmul(const ttb_indx n, double * a, const double * b)
{
  // Casting, possibly to different type
  const ttb_vml_int n_ml = (ttb_vml_int) n;

  ::vdmul(n_ml,a,b);

}


void Genten::posv (char uplo, ttb_indx n, ttb_indx nrhs, double * a, ttb_indx lda, double * b, ttb_indx ldb)
{
#if !defined(LAPACK_FOUND)
  Genten::error("Genten::posv - not found, must link with an LAPACK library.");
#else

  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int nrhs_ml = (ttb_blas_int) nrhs;
  ttb_blas_int lda_ml = (ttb_blas_int) lda;
  ttb_blas_int ldb_ml = (ttb_blas_int) ldb;
  ttb_blas_int info_ml = 0;

  ::dposv(&uplo, &n_ml, &nrhs_ml, a, &lda_ml, b, &ldb_ml, &info_ml);

  // Check output info and free pivot array
  if (info_ml < 0)
  {
    Genten::error("Genten::posv - argument error in call to dposv");
  }
  if (info_ml > 0)
  {
    Genten::error("Genten::posv - dposv failed because matrix is not positive definite");
  }
#endif
}

//
// Single precision
//

void Genten::copy(ttb_indx n, const float * x, ttb_indx incx, float * y, ttb_indx incy)
{
  // Casting, possibly to different type
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int incx_ml = (ttb_blas_int) incx;
  ttb_blas_int incy_ml = (ttb_blas_int) incy;

  // Just removing const
  float * x_ml = (float *) x;

  ::scopy(&n_ml, x_ml, &incx_ml, y, &incy_ml);
}

void Genten::scal(ttb_indx n, float alpha, float * x, ttb_indx incx)
{
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int incx_ml = (ttb_blas_int) incx;
  ::sscal(&n_ml, &alpha, x, &incx_ml);
}

float Genten::nrm2(ttb_indx n, const float * x, ttb_indx incx)
{
  ttb_blas_int n_ml = (ttb_blas_int) n;
  float * x_ml = (float *) x; // remove const
  ttb_blas_int incx_ml = (ttb_blas_int) incx;

  return(::snrm2(&n_ml, x_ml, &incx_ml));
}

float Genten::nrm1(ttb_indx n, const float * x, ttb_indx incx)
{
  ttb_blas_int n_ml = (ttb_blas_int) n;
  float * x_ml = (float *) x; // remove const
  ttb_blas_int incx_ml = (ttb_blas_int) incx;

  return(::sasum(&n_ml, x_ml, &incx_ml));
}

ttb_indx Genten::imax(ttb_indx n, const float * x, ttb_indx incx)
{
  ttb_blas_int n_ml = (ttb_blas_int) n;
  float * x_ml = (float *) x; // remove const
  ttb_blas_int incx_ml = (ttb_blas_int) incx;
  ttb_blas_int idx = ::isamax(&n_ml, x_ml, &incx_ml);

  // idamax returns index between 1 and n (Fortran 1-indexing),
  // so subtract 1 to return index between 0 and n-1 (0-indexing)
  return((ttb_indx) idx-1);
}


float Genten::dot(ttb_indx n, const float * x, ttb_indx incx, const float * y, ttb_indx incy)
{
  ttb_blas_int n_ml = (ttb_blas_int) n;
  float * x_ml = (float *) x; // remove const
  ttb_blas_int incx_ml = (ttb_blas_int) incx;
  float * y_ml = (float *) y; // remove const
  ttb_blas_int incy_ml = (ttb_blas_int) incy;

  return(::sdot(&n_ml, x_ml, &incx_ml, y_ml, &incy_ml));
}

void Genten::axpy(ttb_indx n, float a, const float * x, ttb_indx incx, float * y, ttb_indx incy)
{
  ttb_blas_int n_ml = (ttb_blas_int) n;
  float a_ml = (float) a;
  float * x_ml = (float *) x; // remove const, don't change type
  ttb_blas_int incx_ml = (ttb_blas_int) incx;
  float * y_ml = (float *) y; // remove const, don't change type
  ttb_blas_int incy_ml = (ttb_blas_int) incy;

  ::saxpy(&n_ml, &a_ml, x_ml, &incx_ml, y_ml, &incy_ml);
}

void Genten::gemv(char trans, ttb_indx m, ttb_indx n, float alpha, const float * a, ttb_indx lda,
                  const float * x, ttb_indx incx, float beta, float *y, ttb_indx incy)
{
  // casting
  ttb_blas_int m_ml = (ttb_blas_int) m;
  ttb_blas_int n_ml = (ttb_blas_int) n;
  float alpha_ml = (float) alpha;
  ttb_blas_int lda_ml = (ttb_blas_int) lda;
  ttb_blas_int incx_ml = (ttb_blas_int) incx;
  float beta_ml = (float) beta;
  ttb_blas_int incy_ml = (ttb_blas_int) incy;

  // Just remove const; don't change type
  float * a_ml = (float *) a;
  float * x_ml = (float *) x;

  ::sgemv(&trans, &m_ml, &n_ml, &alpha_ml, a_ml, &lda_ml, x_ml, &incx_ml, &beta_ml, y, &incy_ml);
}

void Genten::ger(ttb_indx m, ttb_indx n, float alpha, const float * x, ttb_indx incx,
                 const float * y, ttb_indx incy, float * a, ttb_indx lda)

{
  // casting
  ttb_blas_int m_ml = (ttb_blas_int) m;
  ttb_blas_int n_ml = (ttb_blas_int) n;
  float alpha_ml = (float) alpha;
  ttb_blas_int incx_ml = (ttb_blas_int) incx;
  ttb_blas_int incy_ml = (ttb_blas_int) incy;
  ttb_blas_int lda_ml = (ttb_blas_int) lda;

  // Just remove const; don't change type
  float * x_ml = (float *) x;
  float * y_ml = (float *) y;

  ::sger(&m_ml, &n_ml, &alpha_ml, x_ml, &incx_ml, y_ml, &incy_ml, a, &lda_ml);
}

void Genten::gemm(char transa, char transb, ttb_indx m, ttb_indx n, ttb_indx k, float alpha,
                  const float * a, ttb_indx ldaa, const float * b, ttb_indx ldab,
                  float beta, float * c, ttb_indx ldac)
{
  ttb_blas_int m_ml = (ttb_blas_int) m;
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int k_ml = (ttb_blas_int) k;
  ttb_blas_int ldaa_ml = (ttb_blas_int) ldaa;
  ttb_blas_int ldab_ml = (ttb_blas_int) ldab;
  ttb_blas_int ldac_ml = (ttb_blas_int) ldac;

  // Just remove const; don't change type
  float * a_ml = (float *) a;
  float * b_ml = (float *) b;

  ::sgemm(&transa, &transb, &m_ml, &n_ml, &k_ml, &alpha, a_ml, &ldaa_ml, b_ml, &ldab_ml, &beta, c, &ldac_ml);
}

void Genten::syrk(char uplo, char trans, ttb_indx n, ttb_indx k, float alpha,
                  const float * a, ttb_indx ldaa,
                  float beta, float * c, ttb_indx ldac)
{
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int k_ml = (ttb_blas_int) k;
  ttb_blas_int ldaa_ml = (ttb_blas_int) ldaa;
  ttb_blas_int ldac_ml = (ttb_blas_int) ldac;

  // Just remove const; don't change type
  float * a_ml = (float *) a;

  ::ssyrk(&uplo, &trans, &n_ml, &k_ml, &alpha, a_ml, &ldaa_ml, &beta, c, &ldac_ml);
}

void Genten::gesv(ttb_indx n, ttb_indx nrhs, float * a, ttb_indx lda, float * b, ttb_indx ldb)
{
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int nrhs_ml = (ttb_blas_int) nrhs;
  ttb_blas_int lda_ml = (ttb_blas_int) lda;
  ttb_blas_int ldb_ml = (ttb_blas_int) ldb;

  // Initialize output info and create pivot array
  ttb_blas_int info_ml = 0;
  ttb_blas_int * ipiv_ml = new ttb_blas_int[ n ];

  ::sgesv(&n_ml, &nrhs_ml, a, &lda_ml, ipiv_ml, b, &ldb_ml, &info_ml);

  delete[] ipiv_ml;

  // Check output info and free pivot array
  if (info_ml < 0)
  {
    Genten::error("Genten::gesv - argument error in call to dgesv");
  }
  if (info_ml > 0)
  {
    Genten::error("Genten::gesv - dgesv failed because matrix is singular");
  }
}

void Genten::sysv(char uplo, ttb_indx n, ttb_indx nrhs, float * a, ttb_indx lda, float * b, ttb_indx ldb)
{
#if !defined(LAPACK_FOUND)
  Genten::error("Genten::sysv - not found, must link with an LAPACK library.");
#else
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int nrhs_ml = (ttb_blas_int) nrhs;
  ttb_blas_int lda_ml = (ttb_blas_int) lda;
  ttb_blas_int ldb_ml = (ttb_blas_int) ldb;

  // Initialize output info and create pivot array
  ttb_blas_int info_ml = 0;
  ttb_blas_int * ipiv_ml = new ttb_blas_int[ n ];

  // Workspace query
  ttb_blas_int lwork = -1;
  float work_tmp = 0;
  ::ssysv(&uplo, &n_ml, &nrhs_ml, a, &lda_ml, ipiv_ml, b, &ldb_ml, &work_tmp, &lwork, &info_ml);

  lwork = ttb_blas_int(work_tmp);
  float * work = new float[lwork];
  ::ssysv(&uplo, &n_ml, &nrhs_ml, a, &lda_ml, ipiv_ml, b, &ldb_ml, work, &lwork, &info_ml);

  delete[] work;
  delete[] ipiv_ml;

  // Check output info and free pivot array
  if (info_ml < 0)
  {
    Genten::error("Genten::sysv - argument error in call to ssysv");
  }
  if (info_ml > 0)
  {
    Genten::error("Genten::sysv - sysv failed because matrix is singular");
  }
#endif
}

void Genten::vmul(const ttb_indx n, float * a, const float * b)
{
  // Casting, possibly to different type
  const ttb_vml_int n_ml = (ttb_vml_int) n;

  ::vsmul(n_ml,a,b);

}


void Genten::posv (char uplo, ttb_indx n, ttb_indx nrhs, float * a, ttb_indx lda, float * b, ttb_indx ldb)
{
#if !defined(LAPACK_FOUND)
  Genten::error("Genten::posv - not found, must link with an LAPACK library.");
#else

  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int nrhs_ml = (ttb_blas_int) nrhs;
  ttb_blas_int lda_ml = (ttb_blas_int) lda;
  ttb_blas_int ldb_ml = (ttb_blas_int) ldb;
  ttb_blas_int info_ml = 0;

  ::sposv(&uplo, &n_ml, &nrhs_ml, a, &lda_ml, b, &ldb_ml, &info_ml);

  // Check output info and free pivot array
  if (info_ml < 0)
  {
    Genten::error("Genten::posv - argument error in call to dposv");
  }
  if (info_ml > 0)
  {
    Genten::error("Genten::posv - dposv failed because matrix is not positive definite");
  }
#endif
}
