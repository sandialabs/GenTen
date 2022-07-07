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


#pragma once

#include "CMakeInclude.h"
#include "Genten_Util.hpp"

typedef ptrdiff_t ttb_blas_int;
typedef ptrdiff_t ttb_vml_int;

#if defined(LAPACK_FOUND)

  #if defined(HAVE_BLAS_F2C)
     #define dasum f2c_dasum
     #define daxpy f2c_daxpy
     #define dcopy f2c_dcopy
     #define ddot f2c_ddot
     #define dgemm f2c_dgemm
     #define dgemv f2c_dgemv
     #define dger f2c_dger
     #define dnrm2 f2c_dnrm2
     #define dscal f2c_dscal
     #define idamax f2c_idamax
     #define dsyrk f2c_dsyrk

     #define sasum f2c_sasum
     #define saxpy f2c_saxpy
     #define scopy f2c_scopy
     #define sdot f2c_sdot
     #define sgemm f2c_sgemm
     #define sgemv f2c_sgemv
     #define sger f2c_sger
     #define snrm2 f2c_snrm2
     #define sscal f2c_sscal
     #define isamax f2c_isamax
     #define ssyrk f2c_ssyrk

     #define dgesv dgesv_
     #define dposv dposv_
     #define dsysv dsysv_
     #define dsyev dsyev_
     #define dgelsy dgelsy_
     #define sgesv sgesv_
     #define sposv sposv_
     #define ssysv ssysv_
     #define sgelsy sgelsy_

  #elif defined (__IBMCPP__)
     #define dasum dasum
     #define daxpy daxpy
     #define dcopy dcopy
     #define ddot ddot
     #define dgemm dgemm
     #define dgemv dgemv
     #define dger dger
     #define dnrm2 dnrm2
     #define dscal dscal
     #define idamax idamax
     #define dsyrk dsyrk

     #define sasum sasum
     #define saxpy saxpy
     #define scopy scopy
     #define sdot sdot
     #define sgemm sgemm
     #define sgemv sgemv
     #define sger sger
     #define snrm2 snrm2
     #define sscal sscal
     #define isamax isamax
     #define ssyrk ssyrk

     #define dgesv dgesv
     #define dposv dposv
     #define dsysv dsysv
     #define dsyev dsyev
     #define dgelsy dgelsy
     #define sgesv sgesv
     #define sposv sposv
     #define ssysv ssysv
     #define sgelsy sgelsy

  #else
     #define dasum dasum_
     #define daxpy daxpy_
     #define dcopy dcopy_
     #define ddot ddot_
     #define dgemm dgemm_
     #define dgemv dgemv_
     #define dger dger_
     #define dnrm2 dnrm2_
     #define dscal dscal_
     #define idamax idamax_
     #define dsyrk dsyrk_

     #define sasum sasum_
     #define saxpy saxpy_
     #define scopy scopy_
     #define sdot sdot_
     #define sgemm sgemm_
     #define sgemv sgemv_
     #define sger sger_
     #define snrm2 snrm2_
     #define sscal sscal_
     #define isamax isamax_
     #define ssyrk ssyrk_

     #define dgesv dgesv_
     #define dposv dposv_
     #define dsysv dsysv_
     #define dsyev dsyev_
     #define dgelsy dgelsy_
     #define sgesv sgesv_
     #define sposv sposv_
     #define ssysv ssysv_
     #define sgelsy sgelsy_
  #endif

#endif


// Declare external LAPACK functions supplied by LAPACK libraries.
extern "C"
{

    //
    // Double precision
    //

    double dasum (ttb_blas_int * nptr,
                  double * x,
                  ttb_blas_int * incxptr);

    void daxpy (ttb_blas_int *nptr,
                double *aptr,
                double *x,
                ttb_blas_int *incxptr,
                double *y,
                ttb_blas_int *incyptr);

    void dcopy (ttb_blas_int *nptr,
                double *dx,
                ttb_blas_int *incxptr,
                double *dy,
                ttb_blas_int *incyptr);

    double ddot (ttb_blas_int * nptr,
                 double * x,
                 ttb_blas_int * incxptr,
                 double * y,
                 ttb_blas_int * incyptr);

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

    void dgemv (char *transb,
                ttb_blas_int *mb,
                ttb_blas_int *nb,
                double *alphab,
                double *a,
                ttb_blas_int *ldab,
                double *x,
                ttb_blas_int *incxb,
                double *betab,
                double *y,
                ttb_blas_int *incyb);

    void dger (ttb_blas_int *mptr,
               ttb_blas_int *nptr,
               double *alphaptr,
               double *x,
               ttb_blas_int *incxptr,
               double *y,
               ttb_blas_int *incyptr,
               double *a,
               ttb_blas_int *ldaptr);

    void dgesv (ttb_blas_int * n,
                ttb_blas_int * nrhs,
                double * a,
                ttb_blas_int * lda,
                ttb_blas_int * ipiv,
                double * b,
                ttb_blas_int * ldb,
                ttb_blas_int * info);

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

    void dsyev (char const* jobz,
		char const* uplo,
		ttb_blas_int * n,
		double* A,
		ttb_blas_int * lda,
		double* W,
		double* work,
		ttb_blas_int * lwork,
		ttb_blas_int * info );

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

    double dnrm2 (ttb_blas_int * nptr,
                  double * x,
                  ttb_blas_int * incxptr);

    void dscal (ttb_blas_int * nptr,
                double * aptr,
                double * x,
                ttb_blas_int * incxptr);

    ttb_blas_int idamax (ttb_blas_int * nptr,
                         double * x,
                         ttb_blas_int * incxptr);

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

    //
    // Single precision
    //

    float sasum (ttb_blas_int * nptr,
                 float * x,
                 ttb_blas_int * incxptr);

    void saxpy (ttb_blas_int *nptr,
                float *aptr,
                float *x,
                ttb_blas_int *incxptr,
                float *y,
                ttb_blas_int *incyptr);

    void scopy (ttb_blas_int *nptr,
                float *dx,
                ttb_blas_int *incxptr,
                float *dy,
                ttb_blas_int *incyptr);

    float sdot (ttb_blas_int * nptr,
                 float * x,
                 ttb_blas_int * incxptr,
                 float * y,
                 ttb_blas_int * incyptr);

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

    void sgemv (char *transb,
                ttb_blas_int *mb,
                ttb_blas_int *nb,
                float *alphab,
                float *a,
                ttb_blas_int *ldab,
                float *x,
                ttb_blas_int *incxb,
                float *betab,
                float *y,
                ttb_blas_int *incyb);

    void sger (ttb_blas_int *mptr,
               ttb_blas_int *nptr,
               float *alphaptr,
               float *x,
               ttb_blas_int *incxptr,
               float *y,
               ttb_blas_int *incyptr,
               float *a,
               ttb_blas_int *ldaptr);

    void sgesv (ttb_blas_int * n,
                ttb_blas_int * nrhs,
                float * a,
                ttb_blas_int * lda,
                ttb_blas_int * ipiv,
                float * b,
                ttb_blas_int * ldb,
                ttb_blas_int * info);

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

    float snrm2 (ttb_blas_int * nptr,
                  float * x,
                  ttb_blas_int * incxptr);

    void sscal (ttb_blas_int * nptr,
                float * aptr,
                float * x,
                ttb_blas_int * incxptr);

    ttb_blas_int isamax (ttb_blas_int * nptr,
                         float * x,
                         ttb_blas_int * incxptr);

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

}

// Declare external VML functions supplied by MKL.
extern "C"
{
// Declare vdmul function which overwrites first input
// Computes Hadamard/elementwise product: a = a .* b
// Note: function definition differs from MKL's vdMul,
//       which has separate output array y = a .* b
void vdmul (const ttb_vml_int n,
            double * a,
            const double * b);

void vsmul (const ttb_vml_int n,
            float * a,
            const float * b);
}
