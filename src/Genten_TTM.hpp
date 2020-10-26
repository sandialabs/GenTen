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

#include <iostream>
#include <assert.h>
#include <utility>
#include <sstream>

#include <Kokkos_Core.hpp>

#include "Genten_Util.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_MathLibs.hpp"
#include "Genten_AlgParams.hpp"

#if defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_CUBLAS)
#include "cublas_v2.h"
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
        //
        template <typename ExecSpace>
        TensorT<ExecSpace> genten_ttm_parfor_dgemm(ttb_indx mode, TensorT<ExecSpace> ten, TensorT<ExecSpace> mat, TensorT<ExecSpace> &ans)
        {
            if ((mode + 1 > 0) && (mode < ten.ndims()))
            {
                if (ten.size(mode) != mat.size(1))
                {
                    std::stringstream dim_error;
                    dim_error << "From genten_ttm_parfor_dgemm, tensor dimension " << mode << " of size " << ten.size(mode) <<" does not match number of columns, "<< mat.size(1) <<", of input matrix";
                    std::cerr << dim_error.str() << std::endl;
                    throw dim_error.str();
                }
                int mode_dim = ten.size(mode);
                int prod = ten.size().prod();
                int I_slash = prod / mode_dim;

                int I_Less = ten.size().prod(0, mode, 1);
                int I_Greater = ten.size().prod(mode + 1, ten.ndims(), 1);

                char transaptr = 'N';
                char transbptr = 'N';
                ttb_blas_int mptr = mat.size(0);
                ttb_blas_int nptr = I_slash;
                ttb_blas_int kptr = mode_dim;
                double alphaptr = 1;
                double *a = (double *)mat.getValues().values().data();
                ttb_blas_int ldaptr = mptr;
                double *b = (double *)ten.getValues().values().data();
                ttb_blas_int ldbptr = kptr;
                double betaptr = 0;
                double *c = (double *)ans.getValues().values().data();
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
                else if ((mode < ten.ndims()) && (mode > 0))
                {
                    transbptr = 'T';

                    mptr = I_Less;
                    nptr = mat.size(0);
                    kptr = mat.size(1);
                    ldaptr = mptr;
                    ldbptr = mat.size(0);
                    ldcptr = mptr;

                    Kokkos::View<ttb_real **, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>> Y(ten.getValues().values().data(), I_Less, I_Greater * mode_dim);
                   
                    Kokkos::parallel_for( "genten_ttm_parfor_dgemm_loop",
                        Kokkos::RangePolicy<ExecSpace>(0,I_Greater), KOKKOS_LAMBDA(const int i) {
                            auto ten_Y = Kokkos::subview(Y, Kokkos::ALL(), std::make_pair((mode_dim * i), (mode_dim * (i + 1))));
                            auto ans_sub = Kokkos::subview(ans.getValues().values(), std::make_pair((mat.size(0) * I_Less * i), (mat.size(0) * I_Less * (i + 1))));

                            double test = 0;
                            double alphaptr_par = 1;
                            double betaptr_par = 0;
                            double *a_par = (double *)ten_Y.data();
                            double *b_par = (double *)mat.getValues().values().data();
                            double *c_par = (double *)ans_sub.data();
                            char transbptr_par = 'T';
                            char transaptr_par = 'N';
                            ttb_blas_int mptr_par = I_Less;
                            ttb_blas_int nptr_par = mat.size(0);
                            ttb_blas_int kptr_par = mat.size(1);
                            ttb_blas_int ldaptr_par = mptr;
                            ttb_blas_int ldbptr_par = mat.size(0);
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

            return ans;
        }

        /////////////////////////////////////////
        //Serial for-loop using dgemm function inside of Genten_MathLibs.hpp
        //
        template <typename ExecSpace>
        TensorT<ExecSpace> genten_ttm_serial_dgemm(ttb_indx mode, TensorT<ExecSpace> ten, TensorT<ExecSpace> mat, TensorT<ExecSpace> &ans)
        {

            if ((mode + 1 > 0) && (mode < ten.ndims()))
            {
                if (ten.size(mode) != mat.size(1))
                {
                    std::stringstream dim_error;
                    dim_error << "From genten_ttm_serial_dgemm, tensor dimension " << mode << " of size " << ten.size(mode) << " does not match number of columns, " << mat.size(1) << ", of input matrix";
                    std::cerr << dim_error.str() << std::endl;
                    throw dim_error.str();
                }

                int mode_dim = ten.size(mode);
                int prod = ten.size().prod();
                int I_slash = prod / mode_dim;

                int I_Less = ten.size().prod(0, mode, 1);
                int I_Greater = ten.size().prod(mode + 1, ten.ndims(), 1);

                char transaptr = 'N';
                char transbptr = 'N';
                ttb_blas_int mptr = mat.size(0);
                ttb_blas_int nptr = I_slash;
                ttb_blas_int kptr = mode_dim;
                double alphaptr = 1;
                double *a = (double *)mat.getValues().values().data();
                ttb_blas_int ldaptr = mptr;
                double *b = (double *)ten.getValues().values().data();
                ttb_blas_int ldbptr = kptr;
                double betaptr = 0;
                double *c = (double *)ans.getValues().values().data();
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
                else if ((mode < ten.ndims()) && (mode > 0))
                {
                    transbptr = 'T';

                    mptr = I_Less;
                    nptr = mat.size(0);
                    kptr = mat.size(1);
                    ldaptr = mptr;
                    ldbptr = mat.size(0);
                    ldcptr = mptr;

                    Kokkos::View<ttb_real **, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>> Y(ten.getValues().values().data(), I_Less, I_Greater * mode_dim);
                    for (int i = 0; i < I_Greater; ++i)
                    {

                        auto ten_Y = Kokkos::subview(Y, Kokkos::ALL(), std::make_pair((mode_dim * i), (mode_dim * (i + 1))));
                        auto ans_sub = Kokkos::subview(ans.getValues().values(), std::make_pair((mat.size(0) * I_Less * i), (mat.size(0) * I_Less * (i + 1))));

                        a = (double *)ten_Y.data();
                        b = (double *)mat.getValues().values().data();
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

            return ans;
        }

#if defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_CUBLAS)
        template <typename ExecSpace>
        void kokkos_ttm_batched_cublas(const TensorT<ExecSpace> &Y,
                                       const TensorT<ExecSpace> &V,
                                       const ttb_indx mode,
                                       TensorT<ExecSpace> &Z)
        {
            if ((mode + 1 > 0) && (mode < Y.ndims()))
            {
                if (Y.size(mode) != V.size(1))
                {
                    std::stringstream dim_error;
                    dim_error << "From kokkos_ttm_batched_cublas, tensor dimension " << mode << " of size " << Y.size(mode) << " does not match number of columns, " << mat.size(1) << ", of input matrix";
                    std::cerr << dim_error.str() << std::endl;
                    throw dim_error.str();
                }

            typedef typename Kokkos::View<ttb_real *, ExecSpace> sub_view_type;
            typedef typename sub_view_type::device_type device_type;

            typedef typename Kokkos::TeamPolicy<ExecSpace>::member_type member_type;

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

            static cublasHandle_t handle = 0;
            if (handle == 0)
            {
                status = cublasCreate(&handle);
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    std::stringstream cublasCreate_error;
                    cublasCreate_error << "Error!  cublasCreate() failed with status "
                       << status;
                    std::cerr << cublasCreate_error.str() << std::endl;
                    throw cublasCreate_error.str();
                }
            }

            // We need Z = V*Y, where Z/Y are matricised tensors, and V is input matrix.
            // But since Z and Y (matricised) are logically LayoutRight, we instead seek Z'=Y'*V'
            // This way, Y' is LayoutLeft. V, naturally LayoutLeft needs the transpose flag.
            // The result Z' also comes out LayoutLeft, as desired. All LayoutLeft is what Gemm expects
            status = cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
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
                mode_error << "From kokkos_ttm_batched_cublas, mode: " << mode << " is invalid. Please provide valid mode";
                std::cerr << mode_error.str() << std::endl;
                throw mode_error.str();
            }
        }

        template <typename ExecSpace>
        void kokkos_ttm_serial_cublas(const TensorT<ExecSpace> &Y,
                                      const TensorT<ExecSpace> &V,
                                      const ttb_indx mode,
                                      TensorT<ExecSpace> &Z)
        {

            typedef typename Kokkos::View<ttb_real *, ExecSpace> sub_view_type;
            typedef typename sub_view_type::device_type device_type;

            typedef typename Kokkos::TeamPolicy<ExecSpace>::member_type member_type;

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

            for (ttb_indx i = 0; i < nmats; i++)
            {
                ttb_indx start = i * nrows * ncols;
                ttb_indx end = start + nrows * ncols;

                //Get the subview of the entries of "this" submatrix
                sub_view_type tensor_slice = Kokkos::subview(Y.getValues().values(), std::make_pair(start, end));

                Kokkos::View<ttb_real **, Kokkos::LayoutLeft, device_type,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    sub_matrix(tensor_slice.data(), ncols, nrows);

                //Now "reshape" the V_matrix from flat 1D-indexed tenspr to matrix of expected shape
                Kokkos::View<ttb_real **, Kokkos::LayoutLeft, device_type,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    V_matrix(V.getValues().values().data(), Vmat_nrows, Vmat_ncols);

                //Now create a subview to store the result
                start = i * Vmat_nrows * ncols;
                end = start + Vmat_nrows * ncols;
                sub_view_type result_slice = Kokkos::subview(Z.getValues().values(), std::make_pair(start, end));

                Kokkos::View<ttb_real **, Kokkos::LayoutLeft, device_type,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    result_sub_matrix(result_slice.data(), ncols, Vmat_nrows);

                const int m = ncols;
                const int k = nrows;
                const int n = Vmat_nrows;
                const int lda = ncols;
                const int ldb = Vmat_nrows;
                const int ldc = ncols;
                const ttb_real alpha = ttb_real(1.0);
                const ttb_real beta = ttb_real(0.0);
                cublasStatus_t status;

                static cublasHandle_t handle = 0;
                if (handle == 0)
                {
                    status = cublasCreate(&handle);
                    if (status != CUBLAS_STATUS_SUCCESS)
                    {
                        std::stringstream cublasCreate_error;
                        cublasCreate_error << "Error!  cublasCreate() failed with status "
                                           << status;
                        std::cerr << cublasCreate_error.str() << std::endl;
                        throw cublasCreate_error.str();
                    }
                }

                // We need Z = V*Y, where Z/Y are matricised tensors, and V is input matrix.
                // But since Z and Y (matricised) are logically LayoutRight, we instead seek Z'=Y'*V'
                // This way, Y' is LayoutLeft. V, naturally LayoutLeft needs the transpose flag.
                // The result Z' also comes out LayoutLeft, as desired. All LayoutLeft is what Gemm expects
                status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                                     &alpha, sub_matrix.data(), lda, V_matrix.data(), ldb,
                                     &beta, result_sub_matrix.data(), ldc);

                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    std::stringstream cublasDgemm_error;
                    cublasDgemm_error << "Error!  cublasDgemm()/cublasDsyrk() failed with status "
                       << status;
                    std::cerr << cublasDgemm_error.str() << std::endl;
                    throw cublasDgemm_error.str();
                }
            }

            return;
        }

        template <typename ExecSpace>
        void kokkos_ttm_last_mode(const TensorT<ExecSpace> &Y,
                                  const TensorT<ExecSpace> &V,
                                  const ttb_indx mode,
                                  TensorT<ExecSpace> &Z)
        {

            typedef typename Kokkos::View<ttb_real *, ExecSpace> sub_view_type;
            typedef typename sub_view_type::device_type device_type;

            typedef typename Kokkos::TeamPolicy<ExecSpace>::member_type member_type;

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

            static cublasHandle_t handle = 0;
            if (handle == 0)
            {
                status = cublasCreate(&handle);
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    std::stringstream cublasCreate_error;
                    cublasCreate_error << "Error!  cublasCreate() failed with status "
                                       << status;
                    std::cerr << cublasCreate_error.str() << std::endl;
                    throw cublasCreate_error.str();
                }
            }

            // We need Z = V*Y, where Z/Y are matricised tensors, and V is input matrix.
            // But since Z and Y (matricised) are logically LayoutRight, we instead seek Z'=Y'*V'
            // This way, Y' is LayoutLeft. V, naturally LayoutLeft needs the transpose flag.
            // The result Z' also comes out LayoutLeft, as desired. All LayoutLeft is what Gemm expects
            status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                                 &alpha, Y.getValues().values().data(), lda, V.getValues().values().data(), ldb,
                                 &beta, Z.getValues().values().data(), ldc);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                std::stringstream cublasDgemm_error;
                cublasDgemm_error << "Error!  cublasDgemm()/cublasDsyrk() failed with status "
                                  << status;
                std::cerr << cublasDgemm_error.str() << std::endl;
                throw cublasDgemm_error.str();
            }
        }
#endif

    } // namespace Impl

    template <typename ExecSpace>
    void ttm(const TensorT<ExecSpace> &Y,
             const TensorT<ExecSpace> &V,
             const ttb_indx n,
             TensorT<ExecSpace> &Z,
             bool all_cublas)
    {

        const ttb_indx nd = Y.ndims(); // Number of dimensions

        assert(Y.size(n) == V.size(1));
        if (all_cublas)
        {
#if defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_CUBLAS)
            if (n == nd - 1)
            {
                Impl::kokkos_ttm_last_mode(Y, V, n, Z);
            }
            else
            {
                Impl::kokkos_ttm_batched_cublas(Y, V, n, Z);
                //Below implementation was universally slow
                //Impl::kokkos_ttm_serial_cublas(Y,V,n,Z);
            }
#else
                    std::stringstream kokkos_cublas_error;
                    kokkos_cublas_error << "TTM is asked to launch all cublas kernels but KOKKOS not built with CUBLAS";
                    std::cerr << kokkos_cublas_error.str() << std::endl;
                    throw kokkos_cublas_error.str();
#endif
        }
        else
        {
            Impl::genten_ttm_serial_dgemm(n, Y, V, Z);
        }
        
    }// ttm
} // namespace Genten
