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

#include "Genten_ComputePrincipalKurtosisVectors.hpp"
#include "Genten_FormCokurtosisSlice.hpp"
#include "Genten_MathLibs_Wpr.hpp"
#include "compute_krp.hpp"
#include <math.h>

#include "perform_eigen_decomp.hpp"

//namespace Genten {

//namespace Impl{

template <typename ExecSpace>
void form_cokurtosis_tensor_naive(const Kokkos::View<ttb_real**, Kokkos::LayoutLeft, ExecSpace>& data_view,
                                  const ttb_indx nsamples, const ttb_indx nvars,
                                  Genten::TensorT<ExecSpace>& moment_tensor) 
{

}

//}// namespace Impl

double * FormRawMomentTensor(double *raw_data_ptr, int nsamples, int nvars, const int order=4) {

  typedef Genten::DefaultExecutionSpace Space;
  typedef Genten::TensorT<Space> Tensor_type;
  typedef Genten::DefaultHostExecutionSpace HostSpace;
  typedef Genten::TensorT<HostSpace> Tensor_host_type;

  //Create the size of moment tensor
  //On host first, then mirror copy to device
  //moment tensor is size nvars^d, where d is order of moment, i.e. nvars*nvars*.... (d times)
  Genten::IndxArrayT<HostSpace> moment_tensor_size_host(order, nvars);

  Genten::IndxArrayT<Space> moment_tensor_size = create_mirror_view( Space(), moment_tensor_size_host);
  deep_copy(moment_tensor_size, moment_tensor_size_host);

  //Now construct the tensor on the device
  Tensor_type X(moment_tensor_size, 0.0);

  //Create a Tensor_type of raw data
  //We will be basically casting the raw_data_ptr to a Kokkos Unmanaged View
  //Not as straightforward as it seems
  //Example: https://github.com/kokkos/kokkos-fortran-interop/blob/master/src/flcl-cxx.hpp/#L157
  //raw data is "viewed" as a 2D-array
  Kokkos::View<ttb_real**,Kokkos::LayoutLeft, HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged> > raw_data_host(raw_data_ptr, nsamples, nvars);

  //Create mirror of raw_data_host on device and copy over
  Kokkos::View<ttb_real**,Kokkos::LayoutLeft, Space> raw_data = Kokkos::create_mirror_view(Space(), raw_data_host);
  deep_copy(raw_data, raw_data_host);


  //---------Call the Kernel to Compute Moment Tensor----------------
  form_cokurtosis_tensor_naive(raw_data, nsamples, nvars, X);


  //Now Mirror the result back from device to host
  Tensor_host_type X_host = create_mirror_view(HostSpace(), X);
  deep_copy(X_host, X);

  return X_host.getValues().ptr();
}


void ComputePrincipalKurtosisVectors(double *raw_data_ptr, int nsamples, int nvars,
                                     double *pvecs, double *pvals)
{

  typedef Genten::DefaultExecutionSpace Space;
  //typedef Genten::TensorT<Space> Tensor_type;
  typedef Genten::DefaultHostExecutionSpace HostSpace;
  //typedef Genten::TensorT<HostSpace> Tensor_host_type;

  //Declare Kokkos Views for principal kurtosis vectors and values
  //These Views are internal, and their content will be memcpied to the pointers
  //passed in by calling code
  Kokkos::View<ttb_real**, Kokkos::LayoutLeft, HostSpace> principal_vecs("principal_vecs", nvars, nvars);
  Kokkos::View<ttb_real*, Kokkos::LayoutLeft, HostSpace> principal_vals("principal_vals", nvars);

  //Create a Tensor_type of raw data
  //We will be basically casting the raw_data_ptr to a Kokkos Unmanaged View
  Kokkos::View<ttb_real**,Kokkos::LayoutLeft, HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged> > raw_data_host(raw_data_ptr, nvars, nsamples);

  //Create mirror of raw_data_host on device and copy over
  Kokkos::View<ttb_real**,Kokkos::LayoutLeft, Space> raw_data = Kokkos::create_mirror_view(Space(), raw_data_host);
  deep_copy(raw_data, raw_data_host);

  Kokkos::View<ttb_real**,Kokkos::LayoutLeft, Space> krp_of_raw_data("krp_of_raw_data", nvars*nvars, nsamples);

  Kokkos::View<ttb_real**,Kokkos::LayoutLeft, Space> cokurtosis_tensor("cokurt_tensor", nvars*nvars, nvars*nvars);

#if defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_CUBLAS)
  //Setup cublas handles once, will be reused repeatedly
  cublasStatus_t status;

  static cublasHandle_t handle = 0;
  if (handle == 0) {
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout<< "Error!  cublasCreate() failed with status "<< status<< std::endl;
    }
  }
#elif defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_ROCBLAS)
  //Setup rocblas handles once, will be reused repeatedly
  rocblas_status status;

  static rocblas_handle handle = 0;
  if (handle == 0) {
    status = rocblas_create_handle(&handle);
    if (status != rocblas_status_success) {
      std::cout<< "Error!  rocblas_create_handle() failed with status "<< status<< std::endl;
    }
  }
#endif //KOKKOS_ENABLE_CUDA


  int nthreads_per_team = 64;
  //HKint nteams_x = nvars*nvars/nthreads_per_team;
  //HKif((nvars*nvars)%nthreads_per_team != 0) nteams_x += 1;
  int nteams_x = (int)( ceil( (double)(nvars*nvars)/((double)(nthreads_per_team)) ) );
  int nteams_y = 32; //hard-coding for now
  int nteams = nteams_x * nteams_y;

  std::cout<<"Launching KRP kernel with "<<nteams_x<<" "<<nteams_y<<" "<<nteams<<std::endl;

  //We're using a 2D block of thread teams (nteams_x*nteams_y in number)
  //The Team Policy can also be specified with a 1D index, so we'll flatten the 2D index
  //Inside the kernel we will re-construct the 2D indices from the flattened index, x-index varying fastest
  ComputeKhatriRaoProduct<Space> compute_krp(nvars, nsamples, nteams_x, nteams_y, raw_data, krp_of_raw_data);
  
  Kokkos::parallel_for(Kokkos::TeamPolicy<Space>(nteams, nthreads_per_team), compute_krp);

  //Now compute cokurtosis tensor = (1/nCols)(krp_of_raw * transp(krp_of_raw) )
  ttb_real alpha = 1.0/ttb_real(nsamples); ttb_real beta = 0.0;
#if defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_CUBLAS)
  status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                       nvars*nvars, nvars*nvars, nsamples,
                       &alpha,
                       krp_of_raw_data.data(), nvars*nvars,
                       krp_of_raw_data.data(), nvars*nvars,
                       &beta,
                       cokurtosis_tensor.data(), nvars*nvars);
#elif defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_ROCBLAS)
  status = rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
		         nvars*nvars, nvars*nvars, nsamples,
                         &alpha,
                         krp_of_raw_data.data(), nvars*nvars,
                         krp_of_raw_data.data(), nvars*nvars,
                         &beta,
                         cokurtosis_tensor.data(), nvars*nvars);
#else
  Genten::gemm('N','T',
               nvars*nvars, nvars*nvars, nsamples,
               alpha,
               krp_of_raw_data.data(), nvars*nvars,
               krp_of_raw_data.data(), nvars*nvars,
               beta,
               cokurtosis_tensor.data(), nvars*nvars); 
#endif

  // Now that the matricised cokurt tensor is done, compute gram_matrix =  cokurt*transp(cokurt)
  //HKKokkos::View<ttb_real**,Kokkos::LayoutLeft, ExecSpace> gram_matrix("result_gram", nvars, nvars);
  Kokkos::View<ttb_real**,Kokkos::LayoutLeft, Space> gram_matrix = Kokkos::create_mirror_view(Space(), principal_vecs);

  // C is implicitly reshaped to (nvars)x(nvars^3) in the call to DGEMM itself
  ttb_real alpha2 = 1.0;
#if defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_CUBLAS)
  status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                       nvars, nvars, nvars*nvars*nvars,
                       &alpha2,
                       cokurtosis_tensor.data(), nvars,
                       cokurtosis_tensor.data(), nvars,
                       &beta,
                       gram_matrix.data(), nvars);
#elif defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_ROCBLAS)
  status = rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
		         nvars, nvars, nvars*nvars*nvars,
                         &alpha2,
                         cokurtosis_tensor.data(), nvars,
                         cokurtosis_tensor.data(), nvars,
                         &beta,
                         gram_matrix.data(), nvars);
#else
  Genten::gemm('N','T',
               nvars, nvars, nvars*nvars*nvars,
               alpha2,
               cokurtosis_tensor.data(), nvars,
               cokurtosis_tensor.data(), nvars,
               beta,
               gram_matrix.data(), nvars);

#endif

  //Now perform the eigen decomposition of the gram matrix
  //Allocate views for eigen values
  //HKKokkos::View<ttb_real*, ExecSpace> eig_vals("gram_eig_vals", nvars);
  Kokkos::View<ttb_real*, Space> eig_vals = Kokkos::create_mirror_view(Space(), principal_vals);
  std::cout<<"Starting eigen decomposition of gram matrix"<<std::endl;

  perform_eigen_decomp(nvars, gram_matrix, eig_vals); 

  std::cout<<"Finished eigen decomposition"<<std::endl;

  //Copy over mirrored views from device to host 
  deep_copy(principal_vecs, gram_matrix);
  deep_copy(principal_vals, eig_vals); 

  //Now that principal_vecs/vals are on host, do a memcpy into ptrs passed in
  memcpy(pvecs, principal_vecs.data(), nvars*nvars*sizeof(double));
  memcpy(pvals, principal_vals.data(), nvars*sizeof(double) );
} 

//}// namespace Genten
