#if defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_CUBLAS)
#include <cuda_runtime.h>
#include "cublas_v2.h"
#endif
#if defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_CUSOLVER)
#include <cusolverDn.h>
#endif
#if defined(KOKKOS_ENABLE_HIP) && defined(HAVE_ROCBLAS)
#include "rocblas.h"
#endif
#if defined(KOKKOS_ENABLE_HIP) && defined(HAVE_ROCSOLVER)
#include "rocsolver.h"
#endif
#include "Genten_MathLibs.hpp"

typedef typename Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> gram_view_type;
typedef typename Kokkos::View<double*,  Kokkos::DefaultExecutionSpace> eig_view_type;

void perform_eigen_decomp(int nRows, gram_view_type& gram_matrix, eig_view_type& eig_vals)
{

#if defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_CUBLAS)
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat = cudaSuccess;

    // Eigenvalue step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

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
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

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
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat);
#elif defined(KOKKOS_ENABLE_HIP) && defined(HAVE_ROCSOLVER)
    rocblas_status status;
    static rocblas_handle handle = 0;
    if (handle == 0) {
        status = rocblas_create_handle(&handle);
    }
        
    //Ask for eigenvectors to be computed
    rocblas_evect jobz = rocblas_evect_original;
    rocblas_fill uplo = rocblas_fill_lower;

    //Allocate a working array on GPU of size nRows
    double *dE;
    hipMalloc(&dE, sizeof(double)*nRows);
    int *devInfo = NULL;
    hipMalloc((void**)&devInfo, sizeof(int));

    //Now call the actual function
    status = rocsolver_dsyevd(handle, jobz, uplo,
                              nRows,  // M of input matrix
                              gram_matrix.data(),   // input matrix
                              nRows,  // leading dimension of input matrix
                              eig_vals.data(),    // vector of eigenvalues
			      dE, //Internal work array on GPU
			      devInfo);

    if (status != rocblas_status_success) {
    std::cout << "rocsolver_dsyevd() exited with status "
         << status << std::endl;
    }
#elif defined (LAPACK_FOUND)

    ttb_blas_int lwork = 0; 
    ttb_blas_int info_ml=0;
    //double *d_work = NULL;

    ttb_blas_int n_ml = (ttb_blas_int) nRows;
    ttb_blas_int lda_ml = (ttb_blas_int) nRows;

    // First perform a workspace query
    lwork = -1;
    double best_lwork_val;
    dsyev("V", "U", &n_ml, gram_matrix.data(), &lda_ml, eig_vals.data(), &best_lwork_val, &lwork, &info_ml);
    lwork = (ttb_blas_int)best_lwork_val;
    double * d_work = new double[lwork];//(double*)malloc(lwork*sizeof(double));

    // Call for actual eigensolve
    dsyev("V", "U", &n_ml, gram_matrix.data(), &lda_ml, eig_vals.data(), d_work, &lwork, &info_ml);
#endif
}
