#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusolverDn.h"
#endif
#if defined(KOKKOS_ENABLE_HIP)
#include "rocblas.h"
#include "rocsolver.h"
#endif
#include "Genten_MathLibs_Wpr.hpp"

typedef typename Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> gram_view_type;
typedef typename Kokkos::View<double*,  Kokkos::DefaultExecutionSpace> eig_view_type;

void perform_eigen_decomp(int nRows, gram_view_type& gram_matrix, eig_view_type& eig_vals)
{

#if defined(KOKKOS_ENABLE_CUDA)
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat = cudaSuccess;

    // Eigenvalue step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    gt_assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    // Eigenvalue step 2: query working space of syevd
    int lwork = 0;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo,
                                                  nRows,  // M of input matrix
                                                  gram_matrix.data(),   // input matrix
                                                  nRows,  // leading dimension of input matrix
                                                  eig_vals.data(),    // vector of eigenvalues
                                                  &lwork);// on return size of working array
    gt_assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    //Eigenvalue step 3: compute actualy eigen decomposition

    double *d_work = NULL;
    int *devInfo = NULL;

    cudaMalloc((void**)&d_work , sizeof(double)*lwork);
    cudaMalloc ((void**)&devInfo, sizeof(int));

    cusolver_status = cusolverDnDsyevd(cusolverH, jobz, uplo,
                                       nRows,  // M of input matrix
                                       gram_matrix.data(),   // input matrix
                                       nRows,  // leading dimension of input matrix
                                       eig_vals.data(),    // vector of eigenvalues
                                       d_work,
                                       lwork,
                                       devInfo);


    cudaStat = cudaDeviceSynchronize();
    gt_assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    gt_assert(cudaSuccess == cudaStat);
#elif defined (LAPACK_FOUND)
    Genten::syev('V', 'U', nRows, gram_matrix.data(), nRows, eig_vals.data());
#endif
}
