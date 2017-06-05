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


//#include "Genten_MathLibs.h"
#include "math.h"
#include "stddef.h"

void dcopy(ptrdiff_t *nptr, double *dx, ptrdiff_t *incxptr, double *dy, ptrdiff_t *incyptr)
{
  // De-referencing
  ptrdiff_t n = *nptr;
  ptrdiff_t incx = *incxptr;
  ptrdiff_t incy = *incyptr;

  // Loop variables
  ptrdiff_t i;

  for (i = 0; i < n; i ++)
  {
    dy[i*incx] = dx[i*incy];
  }
}


void dscal(ptrdiff_t * nptr, double * aptr, double * x, ptrdiff_t * incxptr)
{
  // De-reference pointers
  ptrdiff_t n = *nptr;
  ptrdiff_t incx = *incxptr;
  double alpha = *aptr;

  // Loop variables
  ptrdiff_t nincx = n * incx;
  ptrdiff_t i;

  for (i = 0; i < nincx; i += incx)
  {
    x[i] *= alpha;
  }
}

double dnrm2(ptrdiff_t * nptr, double * x, ptrdiff_t * incxptr)
{
  // De-referencing the pointers
  ptrdiff_t n = *nptr;
  ptrdiff_t incx = *incxptr;

  // Setting up loop variables
  ptrdiff_t nincx = n * incx;
  ptrdiff_t i;
  double nrm = 0;

  // Main loop - summing the squares of all the entries
  for (i = 0; i < nincx; i += incx)
  {
    nrm += (x[i] * x[i]);
  }

  // Take a final square root
  nrm = sqrt(nrm);

  // Return the final answer
  return(nrm);
}

double dasum(ptrdiff_t * nptr, double * x, ptrdiff_t * incxptr)
{
  // De-referencing the pointers
  ptrdiff_t n = *nptr;
  ptrdiff_t incx = *incxptr;

  // Setting up loop variables
  ptrdiff_t nincx = n * incx;
  ptrdiff_t i;
  double nrm = 0;

  // Main loop
  for (i = 0; i < nincx; i += incx)
  {
    nrm += fabs(x[i]);
  }

  // Return the final answer
  return(nrm);
}

ptrdiff_t idamax(ptrdiff_t * nptr, double * x, ptrdiff_t * incxptr)
{
  // De-referencing the pointers
  ptrdiff_t n = *nptr;
  ptrdiff_t incx = *incxptr;

  // Setting up loop variables
  ptrdiff_t nincx = n * incx;
  ptrdiff_t i;
  ptrdiff_t idx;
  double max;

  // Just in case we get an empty array
  if (n == 0)
  {
    return(0);
  }

  idx = 0;
  max = fabs(x[0]);
  // Main loop
  for (i = incx; i < nincx; i += incx)
  {
    if (fabs(x[i]) > max)
    {
      idx = i / incx;
      max = fabs(x[i]);
    }
  }

  // Return the final answer (convert to Fortran 1-indexing to match BLAS)
  return(idx+1);
}

double ddot(ptrdiff_t * nptr, double * x, ptrdiff_t * incxptr, double * y, ptrdiff_t * incyptr)
{
  // De-referencing the pointers
  ptrdiff_t n = *nptr;
  ptrdiff_t incx = *incxptr;
  ptrdiff_t incy = *incyptr;

  // Setting up loop variables
  ptrdiff_t i;
  double result = 0;

  // Main loop - summing the products of the entries
  for (i = 0; i < n; i ++)
  {
    result += x[i*incx] * y[i*incy];
  }

  // Return the result
  return(result);
}


// y = a*x + y
void daxpy(ptrdiff_t *nptr, double *aptr, double *x, ptrdiff_t *incxptr, double *y, ptrdiff_t *incyptr)
{
  // De-referencing the pointers
  ptrdiff_t n = *nptr;
  ptrdiff_t incx = *incxptr;
  ptrdiff_t incy = *incyptr;
  double a = *aptr;

  // Setting up loop variables
  ptrdiff_t i;

  // Main Loop
  for (i = 0; i < n; i ++)
  {
    y[i*incx] = a * x[i*incx] + y[i*incy];
  }
}

void dgemv(char *transb, ptrdiff_t *mb, ptrdiff_t *nb, double *alphab, double *a, ptrdiff_t *ldab,
           double *x, ptrdiff_t *incxb, double *betab, double *y, ptrdiff_t *incyb)
{
  char trans = *transb;
  ptrdiff_t m = *mb;
  ptrdiff_t n = *nb;
  ptrdiff_t lda = *ldab;
  ptrdiff_t incx = *incxb;
  ptrdiff_t incy = *incyb;
  double alpha = *alphab;
  double beta = *betab;

  ptrdiff_t i,j;
  double tmp;

  if (trans == 'N')
  {
    for (i = 0; i < m; i ++)
    {
      tmp = 0;
      for (j = 0; j < n; j ++)
      {
        tmp += a[i + (j * lda)] * x[j * incx];
      }
      y[i * incy] = alpha * tmp + beta * y[i * incy];
    }

  }
  else if (trans == 'T')
  {
    for (i = 0; i < n; i ++)
    {
      tmp = 0;
      for (j = 0; j < m; j ++)
      {
        tmp += a[j + (i * lda)] * x[j * incx];
      }
      y[i * incy] = alpha * tmp + beta * y[i * incy];
    }
  }
}

void dger(ptrdiff_t *mptr, ptrdiff_t *nptr, double *alphaptr, double *x, ptrdiff_t *incxptr,
          double *y, ptrdiff_t *incyptr, double *a, ptrdiff_t *ldaptr)
{
  ptrdiff_t m = *mptr;
  ptrdiff_t n = *nptr;
  double alpha = *alphaptr;
  ptrdiff_t incx = *incxptr;
  ptrdiff_t incy = *incyptr;
  ptrdiff_t lda = *ldaptr;
  ptrdiff_t i, j;
  double * acolptr;

  for (j = 0; j < n; j ++)
  {
    acolptr = a + (j * lda);
    for (i = 0; i < m; i ++)
    {
      acolptr[i] = x[i*incx] * y[j*incy];
    }
  }
}

void dgemm(char *transaptr, char *transbptr, ptrdiff_t *mptr, ptrdiff_t *nptr, ptrdiff_t *kptr,
           double *alphaptr, double *a, ptrdiff_t *ldaptr, double *b, ptrdiff_t *ldbptr,
           double *betaptr, double *c, ptrdiff_t *ldcptr)
{
  char transa = *transaptr;
  char transb = *transbptr;
  ptrdiff_t m = *mptr;
  ptrdiff_t n = *nptr;
  ptrdiff_t k = *kptr;
  double alpha = *alphaptr;
  ptrdiff_t lda = *ldaptr;
  ptrdiff_t ldb = *ldbptr;
  double beta = *betaptr;
  ptrdiff_t ldc = *ldcptr;
  ptrdiff_t j;
  ptrdiff_t one = 1;

  if ((transa == 'N') && (transb == 'N'))
  {
    // Compute c(:,j) = alpha * a * b(:,j) + beta * c(:,j)
    for (j = 0; j < n; j ++)
    {
      dgemv(transaptr, mptr, kptr, alphaptr, a, ldaptr, b + j*ldb, &one, betaptr, c + j*ldc, &one);
    }
  }
  else if ((transa == 'T') && (transb == 'N'))
  {
    // Compute c(:,j) = alpha * a * b(:,j) + beta * c(:,j)
    for (j = 0; j < n; j ++)
    {
      dgemv(transaptr, kptr, mptr, alphaptr, a, ldaptr, b + j*ldb, &one, betaptr, c + j*ldc, &one);
    }
  }
  else
  {
    // Not implemented
  }
}

// -------------------------------------------------------------------------
/* The code that follows below was extracted from Mark Sears' T++ project. */
// -------------------------------------------------------------------------
#define ABS(x) ((x) >= 0 ? (x) : -(x))            ///< absolute value of expression
#define A(i,j) a [(i) + an*(j)]

int dmat_lu(double *a, ptrdiff_t an, ptrdiff_t *pivots)
{
  int i,j,k;
  int npivots;
  int nReturnStatus = 0;

  for(k=0;k<an;k++)
    pivots[k] = k;

  npivots = 0;
  for(k=0;k<an;k++)
  {
    // in column k, find the best pivot
    int ip;
    double apivot;
    apivot = ABS(A(k,k));
    ip = k;
    for(i=k+1;i<an;i++)
    {
      double a1;
      a1 = ABS(A(i,k));
      if(a1 > apivot)
      {
        apivot = a1;
        ip = i;
      }
    }

    if(ip != k)
      npivots++;

    pivots[k] = ip;

    // swap row ip, k. Note that we swap the entire row
    if(ip != k)
    {
      for(j=0;j<an;j++)
      {
        double t = A(k, j);
        A(k, j) = A(ip, j);
        A(ip, j) = t;
      }
    }

    // build the Gauss transform
    apivot = A(k,k);
    if (apivot == 0.0)
      nReturnStatus = k+1;
    for(j=k+1;j<an;j++)
      A(j,k) = A(j,k)/apivot;

    // apply Gauss transform to remainder of A
    for(i=k+1;i<an;i++)
      for(j=k+1;j<an;j++)
        A(j,i) -= A(k,i)*A(j,k);
  }

  // Return >0 if there was a divide by exact zero, similar to dgetf2.
  return( nReturnStatus );
}


// (i should implement a block version of this)
void dmat_lu_solve(double *a, ptrdiff_t an,  ptrdiff_t *pivots, double *x)
{
  int i, j;
  double sum;

  // Apply permutation
  for(i=0;i<an;i++)
  {
    int ip = (int) pivots[i];
    double t = x[ip];
    x[ip] = x[i];
    x[i] = t;
  }

  // Solve Lz = y where L is unit lower triangular
  for(i=0;i<an;i++)
  {
    sum = 0.;
    for(j=0;j<i;j++)
      sum += x[j] * A(i,j);
    x[i] = x[i] - sum;
  }

  // Solve Ux = z where U is upper triangular
  for (i = (int) (an-1); i >= 0; i--)
  {
    sum = 0.;
    for(j=i+1;j<an;j++)
      sum += x[j] * A(i,j);
    x[i] = (x[i] - sum)/A(i,i);
  }

}

// -------------------------------------------------------------------------
/* End of code extracted from Mark Sears' T++ project. */
// -------------------------------------------------------------------------

void dgesv(ptrdiff_t * n, ptrdiff_t * nrhs, double * a, ptrdiff_t * lda, ptrdiff_t * ipiv,
           double * b, ptrdiff_t * ldb, ptrdiff_t * info)
{
  double * bptr;
  int i;

  // Default solver does minimal error checking.
  *info = 0;

  // Compute LU factors of A
  if (dmat_lu(a, *n, ipiv) != 0)
  {
    *info = 1;
  }

  // Use LU factors to solve each system
  bptr = b;
  for (i = 0; i < *nrhs; i ++)
  {
    dmat_lu_solve(a, *n, ipiv, bptr);
    bptr += *n;
  }
}

void scopy(ptrdiff_t *nptr, float *dx, ptrdiff_t *incxptr, float *dy, ptrdiff_t *incyptr)
{
  // De-referencing
  ptrdiff_t n = *nptr;
  ptrdiff_t incx = *incxptr;
  ptrdiff_t incy = *incyptr;

  // Loop variables
  ptrdiff_t i;

  for (i = 0; i < n; i ++)
  {
    dy[i*incx] = dx[i*incy];
  }
}


void sscal(ptrdiff_t * nptr, float * aptr, float * x, ptrdiff_t * incxptr)
{
  // De-reference pointers
  ptrdiff_t n = *nptr;
  ptrdiff_t incx = *incxptr;
  float alpha = *aptr;

  // Loop variables
  ptrdiff_t nincx = n * incx;
  ptrdiff_t i;

  for (i = 0; i < nincx; i += incx)
  {
    x[i] *= alpha;
  }
}

float snrm2(ptrdiff_t * nptr, float * x, ptrdiff_t * incxptr)
{
  // De-referencing the pointers
  ptrdiff_t n = *nptr;
  ptrdiff_t incx = *incxptr;

  // Setting up loop variables
  ptrdiff_t nincx = n * incx;
  ptrdiff_t i;
  float nrm = 0;

  // Main loop - summing the squares of all the entries
  for (i = 0; i < nincx; i += incx)
  {
    nrm += (x[i] * x[i]);
  }

  // Take a final square root
  nrm = sqrt(nrm);

  // Return the final answer
  return(nrm);
}

float sasum(ptrdiff_t * nptr, float * x, ptrdiff_t * incxptr)
{
  // De-referencing the pointers
  ptrdiff_t n = *nptr;
  ptrdiff_t incx = *incxptr;

  // Setting up loop variables
  ptrdiff_t nincx = n * incx;
  ptrdiff_t i;
  float nrm = 0;

  // Main loop
  for (i = 0; i < nincx; i += incx)
  {
    nrm += fabs(x[i]);
  }

  // Return the final answer
  return(nrm);
}

ptrdiff_t isamax(ptrdiff_t * nptr, float * x, ptrdiff_t * incxptr)
{
  // De-referencing the pointers
  ptrdiff_t n = *nptr;
  ptrdiff_t incx = *incxptr;

  // Setting up loop variables
  ptrdiff_t nincx = n * incx;
  ptrdiff_t i;
  ptrdiff_t idx;
  float max;

  // Just in case we get an empty array
  if (n == 0)
  {
    return(0);
  }

  idx = 0;
  max = fabs(x[0]);
  // Main loop
  for (i = incx; i < nincx; i += incx)
  {
    if (fabs(x[i]) > max)
    {
      idx = i / incx;
      max = fabs(x[i]);
    }
  }

  // Return the final answer (convert to Fortran 1-indexing to match BLAS)
  return(idx+1);
}

float sdot(ptrdiff_t * nptr, float * x, ptrdiff_t * incxptr, float * y, ptrdiff_t * incyptr)
{
  // De-referencing the pointers
  ptrdiff_t n = *nptr;
  ptrdiff_t incx = *incxptr;
  ptrdiff_t incy = *incyptr;

  // Setting up loop variables
  ptrdiff_t i;
  float result = 0;

  // Main loop - summing the products of the entries
  for (i = 0; i < n; i ++)
  {
    result += x[i*incx] * y[i*incy];
  }

  // Return the result
  return(result);
}


// y = a*x + y
void saxpy(ptrdiff_t *nptr, float *aptr, float *x, ptrdiff_t *incxptr, float *y, ptrdiff_t *incyptr)
{
  // De-referencing the pointers
  ptrdiff_t n = *nptr;
  ptrdiff_t incx = *incxptr;
  ptrdiff_t incy = *incyptr;
  float a = *aptr;

  // Setting up loop variables
  ptrdiff_t i;

  // Main Loop
  for (i = 0; i < n; i ++)
  {
    y[i*incx] = a * x[i*incx] + y[i*incy];
  }
}

void sgemv(char *transb, ptrdiff_t *mb, ptrdiff_t *nb, float *alphab, float *a, ptrdiff_t *ldab,
           float *x, ptrdiff_t *incxb, float *betab, float *y, ptrdiff_t *incyb)
{
  char trans = *transb;
  ptrdiff_t m = *mb;
  ptrdiff_t n = *nb;
  ptrdiff_t lda = *ldab;
  ptrdiff_t incx = *incxb;
  ptrdiff_t incy = *incyb;
  float alpha = *alphab;
  float beta = *betab;

  ptrdiff_t i,j;
  float tmp;

  if (trans == 'N')
  {
    for (i = 0; i < m; i ++)
    {
      tmp = 0;
      for (j = 0; j < n; j ++)
      {
        tmp += a[i + (j * lda)] * x[j * incx];
      }
      y[i * incy] = alpha * tmp + beta * y[i * incy];
    }

  }
  else if (trans == 'T')
  {
    for (i = 0; i < n; i ++)
    {
      tmp = 0;
      for (j = 0; j < m; j ++)
      {
        tmp += a[j + (i * lda)] * x[j * incx];
      }
      y[i * incy] = alpha * tmp + beta * y[i * incy];
    }
  }
}

void sger(ptrdiff_t *mptr, ptrdiff_t *nptr, float *alphaptr, float *x, ptrdiff_t *incxptr,
          float *y, ptrdiff_t *incyptr, float *a, ptrdiff_t *ldaptr)
{
  ptrdiff_t m = *mptr;
  ptrdiff_t n = *nptr;
  float alpha = *alphaptr;
  ptrdiff_t incx = *incxptr;
  ptrdiff_t incy = *incyptr;
  ptrdiff_t lda = *ldaptr;
  ptrdiff_t i, j;
  float * acolptr;

  for (j = 0; j < n; j ++)
  {
    acolptr = a + (j * lda);
    for (i = 0; i < m; i ++)
    {
      acolptr[i] = x[i*incx] * y[j*incy];
    }
  }
}

void sgemm(char *transaptr, char *transbptr, ptrdiff_t *mptr, ptrdiff_t *nptr, ptrdiff_t *kptr,
           float *alphaptr, float *a, ptrdiff_t *ldaptr, float *b, ptrdiff_t *ldbptr,
           float *betaptr, float *c, ptrdiff_t *ldcptr)
{
  char transa = *transaptr;
  char transb = *transbptr;
  ptrdiff_t m = *mptr;
  ptrdiff_t n = *nptr;
  ptrdiff_t k = *kptr;
  float alpha = *alphaptr;
  ptrdiff_t lda = *ldaptr;
  ptrdiff_t ldb = *ldbptr;
  float beta = *betaptr;
  ptrdiff_t ldc = *ldcptr;
  ptrdiff_t j;
  ptrdiff_t one = 1;

  if ((transa == 'N') && (transb == 'N'))
  {
    // Compute c(:,j) = alpha * a * b(:,j) + beta * c(:,j)
    for (j = 0; j < n; j ++)
    {
      sgemv(transaptr, mptr, kptr, alphaptr, a, ldaptr, b + j*ldb, &one, betaptr, c + j*ldc, &one);
    }
  }
  else if ((transa == 'T') && (transb == 'N'))
  {
    // Compute c(:,j) = alpha * a * b(:,j) + beta * c(:,j)
    for (j = 0; j < n; j ++)
    {
      sgemv(transaptr, kptr, mptr, alphaptr, a, ldaptr, b + j*ldb, &one, betaptr, c + j*ldc, &one);
    }
  }
  else
  {
    // Not implemented
  }
}

// -------------------------------------------------------------------------
/* The code that follows below was extracted from Mark Sears' T++ project. */
// -------------------------------------------------------------------------

int smat_lu(float *a, ptrdiff_t an, ptrdiff_t *pivots)
{
  int i,j,k;
  int npivots;
  int nReturnStatus = 0;

  for(k=0;k<an;k++)
    pivots[k] = k;

  npivots = 0;
  for(k=0;k<an;k++)
  {
    // in column k, find the best pivot
    int ip;
    float apivot;
    apivot = ABS(A(k,k));
    ip = k;
    for(i=k+1;i<an;i++)
    {
      float a1;
      a1 = ABS(A(i,k));
      if(a1 > apivot)
      {
        apivot = a1;
        ip = i;
      }
    }

    if(ip != k)
      npivots++;

    pivots[k] = ip;

    // swap row ip, k. Note that we swap the entire row
    if(ip != k)
    {
      for(j=0;j<an;j++)
      {
        float t = A(k, j);
        A(k, j) = A(ip, j);
        A(ip, j) = t;
      }
    }

    // build the Gauss transform
    apivot = A(k,k);
    if (apivot == 0.0)
      nReturnStatus = k+1;
    for(j=k+1;j<an;j++)
      A(j,k) = A(j,k)/apivot;

    // apply Gauss transform to remainder of A
    for(i=k+1;i<an;i++)
      for(j=k+1;j<an;j++)
        A(j,i) -= A(k,i)*A(j,k);
  }

  // Return >0 if there was a divide by exact zero, similar to dgetf2.
  return( nReturnStatus );
}


// (i should implement a block version of this)
void smat_lu_solve(float *a, ptrdiff_t an,  ptrdiff_t *pivots, float *x)
{
  int i, j;
  float sum;

  // Apply permutation
  for(i=0;i<an;i++)
  {
    int ip = (int) pivots[i];
    float t = x[ip];
    x[ip] = x[i];
    x[i] = t;
  }

  // Solve Lz = y where L is unit lower triangular
  for(i=0;i<an;i++)
  {
    sum = 0.;
    for(j=0;j<i;j++)
      sum += x[j] * A(i,j);
    x[i] = x[i] - sum;
  }

  // Solve Ux = z where U is upper triangular
  for (i = (int) (an-1); i >= 0; i--)
  {
    sum = 0.;
    for(j=i+1;j<an;j++)
      sum += x[j] * A(i,j);
    x[i] = (x[i] - sum)/A(i,i);
  }

}

// -------------------------------------------------------------------------
/* End of code extracted from Mark Sears' T++ project. */
// -------------------------------------------------------------------------

void sgesv(ptrdiff_t * n, ptrdiff_t * nrhs, float * a, ptrdiff_t * lda, ptrdiff_t * ipiv,
           float * b, ptrdiff_t * ldb, ptrdiff_t * info)
{
  float * bptr;
  int i;

  // Default solver does minimal error checking.
  *info = 0;

  // Compute LU factors of A
  if (smat_lu(a, *n, ipiv) != 0)
  {
    *info = 1;
  }

  // Use LU factors to solve each system
  bptr = b;
  for (i = 0; i < *nrhs; i ++)
  {
    smat_lu_solve(a, *n, ipiv, bptr);
    bptr += *n;
  }
}
