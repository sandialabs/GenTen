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

#include "Genten_MathLibs_Wpr.hpp"

#ifdef HAVE_MKL

typedef MKL_INT ttb_blas_int;

#else

typedef ptrdiff_t ttb_blas_int;

#if defined(LAPACK_FOUND)

#if defined(HAVE_BLAS_F2C)
#define dgemm f2c_dgemm
#define dsyrk f2c_dsyrk

#define sgemm f2c_sgemm
#define ssyrk f2c_ssyrk

#define dposv dposv_
#define dsysv dsysv_
#define dgelsy dgelsy_
#define dsyev dsyev_

#define sposv sposv_
#define ssysv ssysv_
#define sgelsy sgelsy_
#define ssyev ssyev_

#elif defined (__IBMCPP__)
#define dgemm dgemm
#define dsyrk dsyrk

#define sgemm sgemm
#define ssyrk ssyrk

#define dposv dposv
#define dsysv dsysv
#define dgelsy dgelsy
#define dsyev dsyev

#define sposv sposv
#define ssysv ssysv
#define sgelsy sgelsy
#define ssyev ssyev

#else
#define dgemm dgemm_
#define dsyrk dsyrk_

#define sgemm sgemm_
#define ssyrk ssyrk_

#define dposv dposv_
#define dsysv dsysv_
#define dgelsy dgelsy_
#define dsyev dsyev_

#define sposv sposv_
#define ssysv ssysv_
#define sgelsy sgelsy_
#define ssyev ssyev_
#endif

// Declare external LAPACK functions supplied by LAPACK libraries.
extern "C"
{

  //
  // Double precision
  //

  void dgemm (char *transaptr,
              char *transbptr,
              ttb_blas_int *mptr,
              ttb_blas_int *nptr,
              ttb_blas_int *kptr,
              double *alphaptr,
              double *a,
              ttb_blas_int *ldaptr,
              double *b,
              ttb_blas_int *ldbptr,
              double *betaptr,
              double *c,
              ttb_blas_int *ldcptr);

  void dposv (char * uplo,
              ttb_blas_int * n,
              ttb_blas_int * nrhs,
              double * a,
              ttb_blas_int * lda,
              double * b,
              ttb_blas_int * ldb,
              ttb_blas_int * info);

  void dsysv (char * uplo,
              ttb_blas_int * n,
              ttb_blas_int * nrhs,
              double * a,
              ttb_blas_int * lda,
              ttb_blas_int * ipiv,
              double * b,
              ttb_blas_int * ldb,
              double * work,
              ttb_blas_int * lwork,
              ttb_blas_int * info);

  void dgelsy (ttb_blas_int * m,
               ttb_blas_int * n,
               ttb_blas_int * nrhs,
               double * a,
               ttb_blas_int * lda,
               double * b,
               ttb_blas_int * ldb,
               ttb_blas_int * ipiv,
               double * rcond,
               ttb_blas_int * rank,
               double * work,
               ttb_blas_int * lwork,
               ttb_blas_int * info);

  void dsyrk (char *uplo,
              char *trans,
              ttb_blas_int *nptr,
              ttb_blas_int *kptr,
              double *alphaptr,
              double *a,
              ttb_blas_int *ldaptr,
              double *betaptr,
              double *c,
              ttb_blas_int *ldcptr);

  void dsyev (char *jobz,
              char *uplo,
              ttb_blas_int *n,
              double *a,
              ttb_blas_int *lda,
              double *w,
              double *work,
              ttb_blas_int *lwork,
              ttb_blas_int *info);

  //
  // Single precision
  //

  void sgemm (char *transaptr,
              char *transbptr,
              ttb_blas_int *mptr,
              ttb_blas_int *nptr,
              ttb_blas_int *kptr,
              float *alphaptr,
              float *a,
              ttb_blas_int *ldaptr,
              float *b,
              ttb_blas_int *ldbptr,
              float *betaptr,
              float *c,
              ttb_blas_int *ldcptr);

  void sposv (char * uplo,
              ttb_blas_int * n,
              ttb_blas_int * nrhs,
              float * a,
              ttb_blas_int * lda,
              float * b,
              ttb_blas_int * ldb,
              ttb_blas_int * info);

  void ssysv (char * uplo,
              ttb_blas_int * n,
              ttb_blas_int * nrhs,
              float * a,
              ttb_blas_int * lda,
              ttb_blas_int * ipiv,
              float * b,
              ttb_blas_int * ldb,
              float * work,
              ttb_blas_int * lwork,
              ttb_blas_int * info);

  void sgelsy (ttb_blas_int * m,
               ttb_blas_int * n,
               ttb_blas_int * nrhs,
               float * a,
               ttb_blas_int * lda,
               float * b,
               ttb_blas_int * ldb,
               ttb_blas_int * ipiv,
               float * rcond,
               ttb_blas_int * rank,
               float * work,
               ttb_blas_int * lwork,
               ttb_blas_int * info);

  void ssyrk (char *uplo,
              char *trans,
              ttb_blas_int *nptr,
              ttb_blas_int *kptr,
              float *alphaptr,
              float *a,
              ttb_blas_int *ldaptr,
              float *betaptr,
              float *c,
              ttb_blas_int *ldcptr);

  void ssyev (char *jobz,
              char *uplo,
              ttb_blas_int *n,
              float *a,
              ttb_blas_int *lda,
              float *w,
              float *work,
              ttb_blas_int *lwork,
              ttb_blas_int *info);

}

#endif
#endif


//
// Double precision
//

void Genten::gemm(char transa, char transb, ttb_indx m, ttb_indx n, ttb_indx k, double alpha,
                  const double * a, ttb_indx ldaa, const double * b, ttb_indx ldab,
                  double beta, double * c, ttb_indx ldac)
{
#if !defined(LAPACK_FOUND)
  Genten::error("Genten::gemm - not found, must link with an LAPACK library.");
#else
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
#endif
}

void Genten::syrk(char uplo, char trans, ttb_indx n, ttb_indx k, double alpha,
                  const double * a, ttb_indx ldaa,
                  double beta, double * c, ttb_indx ldac)
{
#if !defined(LAPACK_FOUND)
  Genten::error("Genten::syrk - not found, must link with an LAPACK library.");
#else
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int k_ml = (ttb_blas_int) k;
  ttb_blas_int ldaa_ml = (ttb_blas_int) ldaa;
  ttb_blas_int ldac_ml = (ttb_blas_int) ldac;

  // Just remove const; don't change type
  double * a_ml = (double *) a;

  ::dsyrk(&uplo, &trans, &n_ml, &k_ml, &alpha, a_ml, &ldaa_ml, &beta, c, &ldac_ml);
#endif
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

bool Genten::posv (char uplo, ttb_indx n, ttb_indx nrhs, double * a, ttb_indx lda, double * b, ttb_indx ldb)
{
#if !defined(LAPACK_FOUND)
  Genten::error("Genten::posv - not found, must link with an LAPACK library.");
  return false;
#else

  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int nrhs_ml = (ttb_blas_int) nrhs;
  ttb_blas_int lda_ml = (ttb_blas_int) lda;
  ttb_blas_int ldb_ml = (ttb_blas_int) ldb;
  ttb_blas_int info_ml = 0;

  ::dposv(&uplo, &n_ml, &nrhs_ml, a, &lda_ml, b, &ldb_ml, &info_ml);

  // Check output info
  if (info_ml < 0)
  {
    Genten::error("Genten::posv - argument error in call to dposv");
  }
  if (info_ml > 0)
    return false;
  return true;
#endif
}

ttb_indx Genten::gelsy(ttb_indx m, ttb_indx n, ttb_indx nrhs, double * a, ttb_indx lda, double * b, ttb_indx ldb, double rcond)
{
#if !defined(LAPACK_FOUND)
  Genten::error("Genten::gelsy - not found, must link with an LAPACK library.");
  return 0;
#else
  ttb_blas_int m_ml = (ttb_blas_int) m;
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int nrhs_ml = (ttb_blas_int) nrhs;
  ttb_blas_int lda_ml = (ttb_blas_int) lda;
  ttb_blas_int ldb_ml = (ttb_blas_int) ldb;

  // Initialize output info and create pivot array
  ttb_blas_int info_ml = 0;
  ttb_blas_int * ipiv_ml = new ttb_blas_int[ n ];
  ttb_blas_int rank_ml = 0;

  // Workspace query
  ttb_blas_int lwork = -1;
  double work_tmp = 0;
  ::dgelsy(&m_ml, &n_ml, &nrhs_ml, a, &lda_ml, b, &ldb_ml, ipiv_ml, &rcond, &rank_ml, &work_tmp, &lwork, &info_ml);

  lwork = ttb_blas_int(work_tmp);
  double * work = new double[lwork];
  ::dgelsy(&m_ml, &n_ml, &nrhs_ml, a, &lda_ml, b, &ldb_ml, ipiv_ml, &rcond, &rank_ml, work, &lwork, &info_ml);

  delete[] work;
  delete[] ipiv_ml;

  // Check output info and free pivot array
  if (info_ml < 0)
  {
    Genten::error("Genten::gelsy - argument error in call to dgelsy");
  }

  ttb_indx rank = (ttb_indx) rank_ml;
  return rank;
#endif
}

void Genten::syev(char jobz, char uplo, ttb_indx n, double *a, ttb_indx lda,
                  double *w)
{
#if !defined(LAPACK_FOUND)
  Genten::error("Genten::syev - not found, must link with an LAPACK library.");
#else
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int lda_ml = (ttb_blas_int) lda;

  // Workspace query
  ttb_blas_int lwork = -1;
  double work_tmp = 0;
  ttb_blas_int info_ml = 0;
  ::dsyev(&jobz, &uplo, &n_ml, a, &lda_ml, w, &work_tmp, &lwork, &info_ml);

  lwork = ttb_blas_int(work_tmp);
  double * work = new double[lwork];
  ::dsyev(&jobz, &uplo, &n_ml, a, &lda_ml, w, &work_tmp, &lwork, &info_ml);

  delete[] work;

  // Check output info and free pivot array
  if (info_ml < 0)
  {
    Genten::error("Genten::syev - argument error in call to dsyev");
  }
#endif
}

//
// Single precision
//

void Genten::gemm(char transa, char transb, ttb_indx m, ttb_indx n, ttb_indx k, float alpha,
                  const float * a, ttb_indx ldaa, const float * b, ttb_indx ldab,
                  float beta, float * c, ttb_indx ldac)
{
#if !defined(LAPACK_FOUND)
  Genten::error("Genten::gemm - not found, must link with an LAPACK library.");
#else
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
#endif
}

void Genten::syrk(char uplo, char trans, ttb_indx n, ttb_indx k, float alpha,
                  const float * a, ttb_indx ldaa,
                  float beta, float * c, ttb_indx ldac)
{
#if !defined(LAPACK_FOUND)
  Genten::error("Genten::syrk - not found, must link with an LAPACK library.");
#else
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int k_ml = (ttb_blas_int) k;
  ttb_blas_int ldaa_ml = (ttb_blas_int) ldaa;
  ttb_blas_int ldac_ml = (ttb_blas_int) ldac;

  // Just remove const; don't change type
  float * a_ml = (float *) a;

  ::ssyrk(&uplo, &trans, &n_ml, &k_ml, &alpha, a_ml, &ldaa_ml, &beta, c, &ldac_ml);
#endif
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

bool Genten::posv (char uplo, ttb_indx n, ttb_indx nrhs, float * a, ttb_indx lda, float * b, ttb_indx ldb)
{
#if !defined(LAPACK_FOUND)
  Genten::error("Genten::posv - not found, must link with an LAPACK library.");
  return false;
#else

  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int nrhs_ml = (ttb_blas_int) nrhs;
  ttb_blas_int lda_ml = (ttb_blas_int) lda;
  ttb_blas_int ldb_ml = (ttb_blas_int) ldb;
  ttb_blas_int info_ml = 0;

  ::sposv(&uplo, &n_ml, &nrhs_ml, a, &lda_ml, b, &ldb_ml, &info_ml);

  // Check output info
  if (info_ml < 0)
  {
    Genten::error("Genten::posv - argument error in call to sposv");
  }
  if (info_ml > 0)
    return false;
  return true;
#endif
}

ttb_indx Genten::gelsy(ttb_indx m, ttb_indx n, ttb_indx nrhs, float * a, ttb_indx lda, float * b, ttb_indx ldb, float rcond)
{
#if !defined(LAPACK_FOUND)
  Genten::error("Genten::gelsy - not found, must link with an LAPACK library.");
  return 0;
#else
  ttb_blas_int m_ml = (ttb_blas_int) m;
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int nrhs_ml = (ttb_blas_int) nrhs;
  ttb_blas_int lda_ml = (ttb_blas_int) lda;
  ttb_blas_int ldb_ml = (ttb_blas_int) ldb;

  // Initialize output info and create pivot array
  ttb_blas_int info_ml = 0;
  ttb_blas_int * ipiv_ml = new ttb_blas_int[ n ];
  ttb_blas_int rank_ml = 0;

  // Workspace query
  ttb_blas_int lwork = -1;
  float work_tmp = 0;
  ::sgelsy(&m_ml, &n_ml, &nrhs_ml, a, &lda_ml, b, &ldb_ml, ipiv_ml, &rcond, &rank_ml, &work_tmp, &lwork, &info_ml);

  lwork = ttb_blas_int(work_tmp);
  float * work = new float[lwork];
  ::sgelsy(&m_ml, &n_ml, &nrhs_ml, a, &lda_ml, b, &ldb_ml, ipiv_ml, &rcond, &rank_ml, work, &lwork, &info_ml);

  delete[] work;
  delete[] ipiv_ml;

  // Check output info and free pivot array
  if (info_ml < 0)
  {
    Genten::error("Genten::gelsy - argument error in call to sgelsy");
  }

  ttb_indx rank = (ttb_indx) rank_ml;
  return rank;
#endif
}

void Genten::syev(char jobz, char uplo, ttb_indx n, float *a, ttb_indx lda,
                  float *w)
{
#if !defined(LAPACK_FOUND)
  Genten::error("Genten::syev - not found, must link with an LAPACK library.");
#else
  ttb_blas_int n_ml = (ttb_blas_int) n;
  ttb_blas_int lda_ml = (ttb_blas_int) lda;

  // Workspace query
  ttb_blas_int lwork = -1;
  float work_tmp = 0;
  ttb_blas_int info_ml = 0;
  ::ssyev(&jobz, &uplo, &n_ml, a, &lda_ml, w, &work_tmp, &lwork, &info_ml);

  lwork = ttb_blas_int(work_tmp);
  float * work = new float[lwork];
  ::ssyev(&jobz, &uplo, &n_ml, a, &lda_ml, w, &work_tmp, &lwork, &info_ml);

  delete[] work;

  // Check output info and free pivot array
  if (info_ml < 0)
  {
    Genten::error("Genten::syev - argument error in call to ssyev");
  }
#endif
}
