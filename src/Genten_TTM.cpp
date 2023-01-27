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

#include <iostream>
#include <cassert>
#include <utility>
#include <sstream>

#include "Genten_TTM.hpp"
#include "Genten_Util.hpp"
#include "Genten_MathLibs.hpp"

#if (defined(KOKKOS_ENABLE_CUDA) || defined(ENABLE_SYCL_FOR_CUDA)) && defined(HAVE_CUBLAS)
#include "Genten_CublasHandle.hpp"
#endif

#if defined(KOKKOS_ENABLE_HIP) && defined(HAVE_ROCBLAS)
#include "Genten_RocblasHandle.hpp"
#include "rocblas.h"
#endif

//-----------------------------------------------------------------------------
//  Method:  TTM, Tensor Y, Matrix V, output to Tensor Z
//-----------------------------------------------------------------------------

namespace Genten
{
  namespace Impl
  {

    /////////////////////////////////////////
    //Kokkos parfor-loop using dgemm function inside of Genten_MathLibs.hpp
    //If input TensorT not on host, will copy it to host, execute on host
    // then copy back to original space
    template <typename ExecSpace>
    TensorT<ExecSpace> genten_ttm_parfor_dgemm(ttb_indx mode,
                                               const TensorT<ExecSpace>& ten,
                                               const TensorT<ExecSpace>& mat,
                                               TensorT<ExecSpace> &ans)
    {
      //NOTE: if TensorT already resides on host then deep_copy is no-op
      auto ten_host = create_mirror_view(ten);
      deep_copy(ten_host, ten);

      auto mat_host = create_mirror_view(mat);
      deep_copy(mat_host, mat);

      auto ans_host = create_mirror_view(ans);

      if ((mode + 1 > 0) && (mode < ten_host.ndims()))
      {
        if (ten_host.size(mode) != mat_host.size(1))
        {
          std::stringstream dim_error;
          dim_error << "From genten_ttm_parfor_dgemm, tensor dimension " << mode << " of size " << ten_host.size(mode) <<" does not match number of columns, "<< mat_host.size(1) <<", of input matrix";
          std::cerr << dim_error.str() << std::endl;
          throw dim_error.str();
        }
        int mode_dim = ten_host.size(mode);
        int prod = ten_host.size().prod();
        int I_slash = prod / mode_dim;

        int I_Less = ten_host.size().prod(0, mode, 1);
        int I_Greater = ten_host.size().prod(mode + 1, ten_host.ndims(), 1);

        char transaptr = 'N';
        char transbptr = 'N';
        ttb_blas_int mptr = mat_host.size(0);
        ttb_blas_int nptr = I_slash;
        ttb_blas_int kptr = mode_dim;
        double alphaptr = 1;
        double *a = (double *)mat_host.getValues().values().data();
        ttb_blas_int ldaptr = mptr;
        double *b = (double *)ten_host.getValues().values().data();
        ttb_blas_int ldbptr = kptr;
        double betaptr = 0;
        double *c = (double *)ans_host.getValues().values().data();
        ttb_blas_int ldcptr = mptr;

        if (mode == 0)
        {
          dgemm(&transaptr,
                &transbptr,
                &mptr,
                &nptr,
                &kptr,
                &alphaptr,
                a,
                &ldaptr,
                b,
                &ldbptr,
                &betaptr,
                c,
                &ldcptr);
        }
        else if ((mode < ten_host.ndims()) && (mode > 0))
        {
          transbptr = 'T';

          mptr = I_Less;
          nptr = mat_host.size(0);
          kptr = mat_host.size(1);
          ldaptr = mptr;
          ldbptr = mat_host.size(0);
          ldcptr = mptr;

          Kokkos::View<ttb_real **, Kokkos::LayoutLeft, Genten::DefaultHostExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> Y(ten_host.getValues().values().data(), I_Less, I_Greater * mode_dim);

          // Don't use KOKKOS_LAMBDA in this case because this
          // parallel-for only runs on the host (and KOKKOS_LAMBDA
          // adds __host__ __device__)
          Kokkos::parallel_for(
            "genten_ttm_parfor_dgemm_loop",
            Kokkos::RangePolicy<Genten::DefaultHostExecutionSpace>(0,I_Greater), [=](const int i) {
              auto ten_Y = Kokkos::subview(Y, Kokkos::ALL(), std::make_pair((mode_dim * i), (mode_dim * (i + 1))));
              auto ans_sub = Kokkos::subview(ans_host.getValues().values(), std::make_pair((mat_host.size(0) * I_Less * i), (mat_host.size(0) * I_Less * (i + 1))));

              double alphaptr_par = 1;
              double betaptr_par = 0;
              double *a_par = (double *)ten_Y.data();
              double *b_par = (double *)mat_host.getValues().values().data();
              double *c_par = (double *)ans_sub.data();
              char transbptr_par = 'T';
              char transaptr_par = 'N';
              ttb_blas_int mptr_par = I_Less;
              ttb_blas_int nptr_par = mat_host.size(0);
              ttb_blas_int kptr_par = mat_host.size(1);
              ttb_blas_int ldaptr_par = mptr;
              ttb_blas_int ldbptr_par = mat_host.size(0);
              ttb_blas_int ldcptr_par = mptr;

              dgemm(&transaptr_par,
                    &transbptr_par,
                    &mptr_par,
                    &nptr_par,
                    &kptr_par,
                    &alphaptr_par,
                    a_par,
                    &ldaptr_par,
                    b_par,
                    &ldbptr_par,
                    &betaptr_par,
                    c_par,
                    &ldcptr_par);
            });
        }
      }
      else
      {
        std::stringstream mode_error;
        mode_error << "From genten_ttm_parfor_dgemm, mode: " << mode << " is invalid. Please provide valid mode";
        std::cerr << mode_error.str() << std::endl;
        throw mode_error.str();
      }
      deep_copy(ans, ans_host);
      return ans;
    }

    /////////////////////////////////////////
    //Serial for-loop using dgemm function inside of Genten_MathLibs.hpp
    //If input TensorT not on host, will copy it to host, execute on host
    // then copy back to original space
    template <typename ExecSpace>
    TensorT<ExecSpace> genten_ttm_serial_dgemm(ttb_indx mode,
                                               const TensorT<ExecSpace>& ten,
                                               const TensorT<ExecSpace>& mat,
                                               TensorT<ExecSpace> &ans)
    {
      //NOTE: if TensorT already resides on host then deep_copy is no-op
      auto ten_host = create_mirror_view(ten);
      deep_copy(ten_host, ten);

      auto mat_host = create_mirror_view(mat);
      deep_copy(mat_host, mat);

      auto ans_host = create_mirror_view(ans);

      if ((mode + 1 > 0) && (mode < ten_host.ndims()))
      {
        if (ten_host.size(mode) != mat_host.size(1))
        {
          std::stringstream dim_error;
          dim_error << "From genten_ttm_serial_dgemm, tensor dimension " << mode << " of size " << ten_host.size(mode) << " does not match number of columns, " << mat_host.size(1) << ", of input matrix";
          std::cerr << dim_error.str() << std::endl;
          throw dim_error.str();
        }

        int mode_dim = ten_host.size(mode);
        int prod = ten_host.size().prod();
        int I_slash = prod / mode_dim;

        int I_Less = ten_host.size().prod(0, mode, 1);
        int I_Greater = ten_host.size().prod(mode + 1, ten_host.ndims(), 1);

        char transaptr = 'N';
        char transbptr = 'N';
        ttb_blas_int mptr = mat_host.size(0);
        ttb_blas_int nptr = I_slash;
        ttb_blas_int kptr = mode_dim;
        double alphaptr = 1;
        double *a = (double *)mat_host.getValues().values().data();
        ttb_blas_int ldaptr = mptr;
        double *b = (double *)ten_host.getValues().values().data();
        ttb_blas_int ldbptr = kptr;
        double betaptr = 0;
        double *c = (double *)ans_host.getValues().values().data();
        ttb_blas_int ldcptr = mptr;

        if (mode == 0)
        {
          dgemm(&transaptr,
                &transbptr,
                &mptr,
                &nptr,
                &kptr,
                &alphaptr,
                a,
                &ldaptr,
                b,
                &ldbptr,
                &betaptr,
                c,
                &ldcptr);
        }
        else if ((mode < ten_host.ndims()) && (mode > 0))
        {
          transbptr = 'T';

          mptr = I_Less;
          nptr = mat_host.size(0);
          kptr = mat_host.size(1);
          ldaptr = mptr;
          ldbptr = mat_host.size(0);
          ldcptr = mptr;

          Kokkos::View<ttb_real **, Kokkos::LayoutLeft, Genten::DefaultHostExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> Y(ten_host.getValues().values().data(), I_Less, I_Greater * mode_dim);
          for (int i = 0; i < I_Greater; ++i)
          {

            auto ten_Y = Kokkos::subview(Y, Kokkos::ALL(), std::make_pair((mode_dim * i), (mode_dim * (i + 1))));
            auto ans_sub = Kokkos::subview(ans_host.getValues().values(), std::make_pair((mat_host.size(0) * I_Less * i), (mat_host.size(0) * I_Less * (i + 1))));

            a = (double *)ten_Y.data();
            b = (double *)mat_host.getValues().values().data();
            c = (double *)ans_sub.data();

            dgemm(&transaptr,
                  &transbptr,
                  &mptr,
                  &nptr,
                  &kptr,
                  &alphaptr,
                  a,
                  &ldaptr,
                  b,
                  &ldbptr,
                  &betaptr,
                  c,
                  &ldcptr);
          }
        }
      }
      else
      {
        std::stringstream mode_error;
        mode_error << "From genten_ttm_serial_dgemm, mode: " << mode << " is invalid. Please provide valid mode";
        std::cerr << mode_error.str() << std::endl;
        throw mode_error.str();
      }
      deep_copy(ans, ans_host);
      return ans;
    }


    template <typename ExecSpace>
    void genten_ttm_batched_cublas(const TensorT<ExecSpace> &Y,
                                   const TensorT<ExecSpace> &V,
                                   const ttb_indx mode,
                                   TensorT<ExecSpace> &Z)
    {
#if (defined(KOKKOS_ENABLE_CUDA) || defined(ENABLE_SYCL_FOR_CUDA)) && defined(HAVE_CUBLAS)
      if ((mode + 1 > 0) && (mode < Y.ndims()))
      {
        if (Y.size(mode) != V.size(1))
        {
          std::stringstream dim_error;
          dim_error << "From genten_ttm_batched_cublas, tensor dimension " << mode << " of size " << Y.size(mode) << " does not match number of columns, " << V.size(1) << ", of input matrix";
          std::cerr << dim_error.str() << std::endl;
          throw dim_error.str();
        }

        typedef typename Kokkos::View<ttb_real *, ExecSpace> sub_view_type;
        typedef typename sub_view_type::device_type device_type;

        IndxArrayT<DefaultHostExecutionSpace> Y_size_host = create_mirror_view(DefaultHostExecutionSpace(), Y.size());
        deep_copy(Y_size_host, Y.size());

        //Get the numbers (rows, cols, num_matrices) for sub matrices of unfolded tensor
        ttb_indx ncols = Y_size_host.prod(0, mode, 1);             // Y.size().prod(0,mode,1);
        ttb_indx nmats = Y_size_host.prod(mode + 1, Y.ndims(), 1); //Y.size().prod(mode+1, Y.ndims(), 1);
        ttb_indx nrows = Y_size_host[mode];                        //Y.size(mode);


        //Get nrows, ncols for the V_matrix
        IndxArrayT<DefaultHostExecutionSpace> V_size_host = create_mirror_view(DefaultHostExecutionSpace(), V.size());
        deep_copy(V_size_host, V.size());

        ttb_indx Vmat_nrows = V_size_host[0]; //V.size(0);
        ttb_indx Vmat_ncols = V_size_host[1]; //V.size(1);

        //Setting up parameters for cublasDgemmStridedBatched
        const int m = ncols;
        const int k = nrows;
        const int n = Vmat_nrows;
        const int lda = ncols;
        const int strideA = nrows * ncols;
        const int ldb = Vmat_nrows;
        const int strideB = 0;
        const int ldc = ncols;
        const int strideC = Vmat_nrows * ncols;
        const ttb_real alpha = ttb_real(1.0);
        const ttb_real beta = ttb_real(0.0);
        cublasStatus_t status;

        // We need Z = V*Y, where Z/Y are matricised tensors, and V is input matrix.
        // But since Z and Y (matricised) are logically LayoutRight, we instead seek Z'=Y'*V'
        // This way, Y' is LayoutLeft. V, naturally LayoutLeft needs the transpose flag.
        // The result Z' also comes out LayoutLeft, as desired. All LayoutLeft is what Gemm expects
        status = cublasDgemmStridedBatched(CublasHandle::get(), CUBLAS_OP_N, CUBLAS_OP_T,
                                           m, n, k,
                                           &alpha,
                                           Y.getValues().values().data(), lda, strideA,
                                           V.getValues().values().data(), ldb, strideB,
                                           &beta,
                                           Z.getValues().values().data(), ldc, strideC,
                                           nmats);

        if (status != CUBLAS_STATUS_SUCCESS)
        {
          std::stringstream cublasDgemmStridedBatched_error;
          cublasDgemmStridedBatched_error << "Error!  cublasDgemmStridedBatched() failed with status "
                                          << status;
          std::cerr << cublasDgemmStridedBatched_error.str() << std::endl;
          throw cublasDgemmStridedBatched_error.str();
        }
      }
      else
      {
        std::stringstream mode_error;
        mode_error << "From genten_ttm_batched_cublas, mode: " << mode << " is invalid. Please provide valid mode";
        std::cerr << mode_error.str() << std::endl;
        throw mode_error.str();
      }
#else
      std::stringstream no_cuda_error;
      no_cuda_error << "Error!  genten_ttm_batched_cublas function called, but CUDA not enabled";
      std::cerr << no_cuda_error.str() << std::endl;
      throw no_cuda_error.str();
#endif

    }

    template <typename ExecSpace>
    void genten_ttm_last_mode_cublas(const TensorT<ExecSpace> &Y,
                                     const TensorT<ExecSpace> &V,
                                     const ttb_indx mode,
                                     TensorT<ExecSpace> &Z)
    {
#if (defined(KOKKOS_ENABLE_CUDA) || defined(ENABLE_SYCL_FOR_CUDA)) && defined(HAVE_CUBLAS)

      typedef typename Kokkos::View<ttb_real *, ExecSpace> sub_view_type;
      typedef typename sub_view_type::device_type device_type;

      IndxArrayT<DefaultHostExecutionSpace> Y_size_host = create_mirror_view(DefaultHostExecutionSpace(), Y.size());
      deep_copy(Y_size_host, Y.size());

      //Get the numbers (rows, cols, num_matrices) for sub matrices of unfolded tensor
      ttb_indx ncols = Y_size_host.prod(0, mode, 1); // Y.size().prod(0,n,1);
      ttb_indx nrows = Y_size_host[mode];            //Y.size(n);

      //Get nrows, ncols for the V_matrix
      IndxArrayT<DefaultHostExecutionSpace> V_size_host = create_mirror_view(DefaultHostExecutionSpace(), V.size());
      deep_copy(V_size_host, V.size());

      ttb_indx Vmat_nrows = V_size_host[0]; //V.size(0);
      ttb_indx Vmat_ncols = V_size_host[1]; //V.size(1);

      const int m = ncols;
      const int k = nrows;
      const int n = Vmat_nrows;
      const int lda = ncols;
      const int ldb = Vmat_nrows;
      const int ldc = ncols;
      const double alpha = 1.0;
      const double beta = 0.0;
      cublasStatus_t status;

      // We need Z = V*Y, where Z/Y are matricised tensors, and V is input matrix.
      // But since Z and Y (matricised) are logically LayoutRight, we instead seek Z'=Y'*V'
      // This way, Y' is LayoutLeft. V, naturally LayoutLeft needs the transpose flag.
      // The result Z' also comes out LayoutLeft, as desired. All LayoutLeft is what Gemm expects
      status = cublasDgemm(CublasHandle::get(), CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                           &alpha, Y.getValues().values().data(), lda, V.getValues().values().data(), ldb,
                           &beta, Z.getValues().values().data(), ldc);

      if (status != CUBLAS_STATUS_SUCCESS)
      {
        std::stringstream cublasDgemm_error;
        cublasDgemm_error << "Error!  cublasDgemm() failed with status "
                          << status;
        std::cerr << cublasDgemm_error.str() << std::endl;
        throw cublasDgemm_error.str();
      }
#else
      std::stringstream no_cuda_error;
      no_cuda_error << "Error!  genten_ttm_last_mode_cublas function called, but CUDA not enabled";
      std::cerr << no_cuda_error.str() << std::endl;
      throw no_cuda_error.str();
#endif
    }

    template <typename ExecSpace>
    void genten_ttm_batched_rocblas(const TensorT<ExecSpace> &Y,
                                   const TensorT<ExecSpace> &V,
                                   const ttb_indx mode,
                                   TensorT<ExecSpace> &Z)
    {
#if defined(KOKKOS_ENABLE_HIP) && defined(HAVE_ROCBLAS)
      if ((mode + 1 > 0) && (mode < Y.ndims()))
      {
        if (Y.size(mode) != V.size(1))
        {
          std::stringstream dim_error;
          dim_error << "From genten_ttm_batched_cublas, tensor dimension " << mode << " of size " << Y.size(mode) << " does not match number of columns, " << V.size(1) << ", of input matrix";
          std::cerr << dim_error.str() << std::endl;
          throw dim_error.str();
        }

        typedef typename Kokkos::View<ttb_real *, ExecSpace> sub_view_type;
        typedef typename sub_view_type::device_type device_type;

        IndxArrayT<DefaultHostExecutionSpace> Y_size_host = create_mirror_view(DefaultHostExecutionSpace(), Y.size());
        deep_copy(Y_size_host, Y.size());

        //Get the numbers (rows, cols, num_matrices) for sub matrices of unfolded tensor
        ttb_indx ncols = Y_size_host.prod(0, mode, 1);             // Y.size().prod(0,mode,1);
        ttb_indx nmats = Y_size_host.prod(mode + 1, Y.ndims(), 1); //Y.size().prod(mode+1, Y.ndims(), 1);
        ttb_indx nrows = Y_size_host[mode];                        //Y.size(mode);


        //Get nrows, ncols for the V_matrix
        IndxArrayT<DefaultHostExecutionSpace> V_size_host = create_mirror_view(DefaultHostExecutionSpace(), V.size());
        deep_copy(V_size_host, V.size());

        ttb_indx Vmat_nrows = V_size_host[0]; //V.size(0);
        ttb_indx Vmat_ncols = V_size_host[1]; //V.size(1);

        //Setting up parameters for rocblas_dgemm_strided_batched
        const int m = ncols;
        const int k = nrows;
        const int n = Vmat_nrows;
        const int lda = ncols;
        const int strideA = nrows * ncols;
        const int ldb = Vmat_nrows;
        const int strideB = 0;
        const int ldc = ncols;
        const int strideC = Vmat_nrows * ncols;
        const ttb_real alpha = ttb_real(1.0);
        const ttb_real beta = ttb_real(0.0);
        rocblas_status status;

        // We need Z = V*Y, where Z/Y are matricised tensors, and V is input matrix.
        // But since Z and Y (matricised) are logically LayoutRight, we instead seek Z'=Y'*V'
        // This way, Y' is LayoutLeft. V, naturally LayoutLeft needs the transpose flag.
        // The result Z' also comes out LayoutLeft, as desired. All LayoutLeft is what Gemm expects
        status = rocblas_dgemm_strided_batched(RocblasHandle::get(), rocblas_operation_none, rocblas_operation_transpose,
                                           m, n, k,
                                           &alpha,
                                           Y.getValues().values().data(), lda, strideA,
                                           V.getValues().values().data(), ldb, strideB,
                                           &beta,
                                           Z.getValues().values().data(), ldc, strideC,
                                           nmats);

        if (status != rocblas_status_success)
        {
          std::stringstream rocblas_dgemm_strided_batched_error;
          rocblas_dgemm_strided_batched_error << "Error!  rocblas_dgemm_strided_batched() failed with status "
                                          << status;
          std::cerr << rocblas_dgemm_strided_batched_error.str() << std::endl;
          throw rocblas_dgemm_strided_batched_error.str();
        }
      }
      else
      {
        std::stringstream mode_error;
        mode_error << "From genten_ttm_batched_rocblas, mode: " << mode << " is invalid. Please provide valid mode";
        std::cerr << mode_error.str() << std::endl;
        throw mode_error.str();
      }
#else
      std::stringstream no_hip_error;
      no_hip_error << "Error!  genten_ttm_batched_rocblas function called, but HIP not enabled";
      std::cerr << no_hip_error.str() << std::endl;
      throw no_hip_error.str();
#endif

    }

    template <typename ExecSpace>
    void genten_ttm_last_mode_rocblas(const TensorT<ExecSpace> &Y,
                                     const TensorT<ExecSpace> &V,
                                     const ttb_indx mode,
                                     TensorT<ExecSpace> &Z)
    {
#if defined(KOKKOS_ENABLE_HIP) && defined(HAVE_ROCBLAS)

      typedef typename Kokkos::View<ttb_real *, ExecSpace> sub_view_type;
      typedef typename sub_view_type::device_type device_type;

      IndxArrayT<DefaultHostExecutionSpace> Y_size_host = create_mirror_view(DefaultHostExecutionSpace(), Y.size());
      deep_copy(Y_size_host, Y.size());

      //Get the numbers (rows, cols, num_matrices) for sub matrices of unfolded tensor
      ttb_indx ncols = Y_size_host.prod(0, mode, 1); // Y.size().prod(0,n,1);
      ttb_indx nrows = Y_size_host[mode];            //Y.size(n);

      //Get nrows, ncols for the V_matrix
      IndxArrayT<DefaultHostExecutionSpace> V_size_host = create_mirror_view(DefaultHostExecutionSpace(), V.size());
      deep_copy(V_size_host, V.size());

      ttb_indx Vmat_nrows = V_size_host[0]; //V.size(0);
      ttb_indx Vmat_ncols = V_size_host[1]; //V.size(1);

      const int m = ncols;
      const int k = nrows;
      const int n = Vmat_nrows;
      const int lda = ncols;
      const int ldb = Vmat_nrows;
      const int ldc = ncols;
      const double alpha = 1.0;
      const double beta = 0.0;
      rocblas_status status;

      // We need Z = V*Y, where Z/Y are matricised tensors, and V is input matrix.
      // But since Z and Y (matricised) are logically LayoutRight, we instead seek Z'=Y'*V'
      // This way, Y' is LayoutLeft. V, naturally LayoutLeft needs the transpose flag.
      // The result Z' also comes out LayoutLeft, as desired. All LayoutLeft is what Gemm expects
      status = rocblas_dgemm(RocblasHandle::get(), rocblas_operation_none, rocblas_operation_transpose, m, n, k,
                           &alpha, Y.getValues().values().data(), lda, V.getValues().values().data(), ldb,
                           &beta, Z.getValues().values().data(), ldc);

      if (status != rocblas_status_success)
      {
        std::stringstream rocblas_dgemm_error;
        rocblas_dgemm_error << "Error!  rocblas_dgemm() failed with status "
                          << status;
        std::cerr << rocblas_dgemm_error.str() << std::endl;
        throw rocblas_dgemm_error.str();
      }
#else
      std::stringstream no_hip_error;
      no_hip_error << "Error!  genten_ttm_last_mode_rocblas function called, but HIP not enabled";
      std::cerr << no_hip_error.str() << std::endl;
      throw no_hip_error.str();
#endif
    }
  } // namespace Impl

  template <typename ExecSpace>
  void ttm(const TensorT<ExecSpace> &Y,
           const TensorT<ExecSpace> &V,
           const ttb_indx n,
           TensorT<ExecSpace> &Z,
           Genten::AlgParams al)
  {

    const ttb_indx nd = Y.ndims(); // Number of dimensions

    gt_assert(Y.size(n) == V.size(1));
    if (al.ttm_method == Genten::TTM_Method::DGEMM)
    {
      if (Genten::is_cuda_space<ExecSpace>::value ||
          Genten::is_sycl_space<ExecSpace>::value
         )
      {
        if (n == nd - 1)
        {
          Impl::genten_ttm_last_mode_cublas(Y, V, n, Z);
        }
        else
        {
          Impl::genten_ttm_batched_cublas(Y, V, n, Z);
        }
      }
      else if (Genten::is_hip_space<ExecSpace>::value)
      {
        if (n == nd - 1)
        {
          Impl::genten_ttm_last_mode_rocblas(Y, V, n, Z);
        }
        else
        {
          Impl::genten_ttm_batched_rocblas(Y, V, n, Z);
        }
      }
      else
      {
        Impl::genten_ttm_serial_dgemm(n, Y, V, Z);
      }
    }
    else
    {
      if (Genten::is_cuda_space<ExecSpace>::value ||
          Genten::is_sycl_space<ExecSpace>::value
         )
      {
        if (n == nd - 1)
        {
          Impl::genten_ttm_last_mode_cublas(Y, V, n, Z);
        }
        else
        {
          Impl::genten_ttm_batched_cublas(Y, V, n, Z);
        }
      }
      else if (Genten::is_hip_space<ExecSpace>::value)
      {
        if (n == nd - 1)
        {
          Impl::genten_ttm_last_mode_rocblas(Y, V, n, Z);
        }
        else
        {
          Impl::genten_ttm_batched_rocblas(Y, V, n, Z);
        }
      }
      else
      {
        Impl::genten_ttm_parfor_dgemm(n, Y, V, Z);
      }
    }

  }// ttm
} // namespace Genten

#define INST_MACRO(SPACE)                                               \
  template TensorT<SPACE> Impl::genten_ttm_parfor_dgemm(                \
    ttb_indx mode,                                                      \
    const TensorT<SPACE>& ten,                                          \
    const TensorT<SPACE>& mat,                                          \
    TensorT<SPACE> &ans);                                               \
  template TensorT<SPACE> Impl::genten_ttm_serial_dgemm(                \
    ttb_indx mode,                                                      \
    const TensorT<SPACE>& ten,                                          \
    const TensorT<SPACE>& mat,                                          \
    TensorT<SPACE> &ans);                                               \
  template void Impl::genten_ttm_batched_cublas(                        \
    const TensorT<SPACE> &Y,                                            \
    const TensorT<SPACE> &V,                                            \
    const ttb_indx mode,                                                \
    TensorT<SPACE> &Z);                                                 \
  template void Impl::genten_ttm_last_mode_cublas(                      \
    const TensorT<SPACE> &Y,                                            \
    const TensorT<SPACE> &V,                                            \
    const ttb_indx mode,                                                \
    TensorT<SPACE> &Z);                                                 \
  template void ttm(const TensorT<SPACE> &Y,                            \
                    const TensorT<SPACE> &V,                            \
                    const ttb_indx n,                                   \
                    TensorT<SPACE> &Z,                                  \
                    Genten::AlgParams al);
GENTEN_INST(INST_MACRO)
