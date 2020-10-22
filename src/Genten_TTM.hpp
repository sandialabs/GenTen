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

///////////////////////////////////////////////////Kokkos headers/includes...
#include <Kokkos_Core.hpp>
// #include <KokkosBlas3_gemm.hpp>

// #include <KokkosBatched_Gemm_Decl.hpp>
// #include <KokkosBatched_Gemm_Serial_Impl.hpp>

// #include <KokkosBatched_Gemm_Decl.hpp>
// #include <KokkosBatched_Gemm_Team_Impl.hpp>
///////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring> //for memcpy

/////////////////////////////////////////////////// Genten includes/headers...
#include <type_traits>

#include "Genten_Util.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_TinyVec.hpp"
#include "Genten_AlgParams.hpp"
#include "Genten_SimdKernel.hpp"
///////////////////////////////////////////////////

// #include "KokkosBatched_Gemv_Decl.hpp"
// #include "KokkosBatched_Gemv_Serial_Impl.hpp"
// #include "KokkosBatched_Gemv_Team_Impl.hpp"
//////////////////////////////////////////////////

//07/04/2020
#include "Genten_MathLibs.hpp"

#include <thread>         
#include <chrono> 


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

            const double alpha = double(1.0);
            const double beta = double(0.0);
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

                Kokkos::parallel_for(
                    I_Greater, KOKKOS_LAMBDA(const int i) {
                        auto ten_Y = Kokkos::subview(Y, Kokkos::ALL(), std::make_pair((mode_dim * i), (mode_dim * (i + 1))));
                        auto ans_sub = Kokkos::subview(ans.getValues().values(), std::make_pair((mat.size(0) * I_Less * i), (mat.size(0) * I_Less * (i + 1))));

                        //NOTE: as of now we are not using the layout right result_sub. If the answer comes out transposed, we might be able to use it to get it in the correct order
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
                        //NOTE: we may have switched a, b and corresponding ptrs as inputs here
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
            else
            {
                std::cout << "mode: " << mode << " is invalid. Please provide valid mode" << std::endl;
            }

            return ans;
        }

        /////////////////////////////////////////
        //Serial for-loop using dgemm function inside of Genten_MathLibs.hpp
        //
        template <typename ExecSpace>
        TensorT<ExecSpace> genten_ttm_serial_dgemm(ttb_indx mode, TensorT<ExecSpace> ten, TensorT<ExecSpace> mat, TensorT<ExecSpace> &ans)
        {

            const double alpha = double(1.0);
            const double beta = double(0.0);
            int mode_dim = ten.size(mode);
            int prod = ten.size().prod();
            int I_slash = prod / mode_dim;

            int I_Less = ten.size().prod(0, mode, 1);
            int I_Greater = ten.size().prod(mode + 1, ten.ndims(), 1);
            ///////////
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

                    //NOTE: as of now we are not using the layout right result_sub. If the answer comes out transposed, we might be able to use it to get it in the correct order

                    //NOTE: we may have switched a, b and corresponding ptrs as inputs here
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
            else
            {
                std::cout << "mode: " << mode << " is invalid. Please provide valid mode" << std::endl;
            }

            return ans;
        }
    }
}