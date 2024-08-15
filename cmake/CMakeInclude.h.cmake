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


//----------------------------------------------------------------------
//  Platform/build specific symbols
//
//  Lines with a cmakedefine directive are replaced at build time with
//  either "#define symbol" or "#undef symbol".
//
//  Include this in source files where a symbol of interest is present.
//----------------------------------------------------------------------

//---- DEFINED IF REAL TIME SYSTEM UTILITIES ARE FOUND.
#cmakedefine HAVE_REALTIME_CLOCK

//---- DEFINED IF LINKING WITH A BLAS LIBRARY THAT USES F2C WRAPPERS.
#cmakedefine HAVE_BLAS_F2C

#if defined(_WIN32)
  #if (_MSC_VER >= 1400)
    //---- WINDOWS MSVC COMPILER INSISTS THAT SECURE STRING FNS BE USED.
    #define HAVE_MSVC_SECURE_STRING_FNS
  #endif
#endif

//---- DEFINED IF KOKKOS IS ENABLED.
#cmakedefine HAVE_KOKKOS

//---- DEFINED IF BOOST IS ENABLED.
#cmakedefine HAVE_BOOST

//---- DEFINED IF Caliper IS ENABLED.
#cmakedefine HAVE_CALIPER

//---- DEFINED IF Teuchos IS ENABLED.
#cmakedefine HAVE_TEUCHOS

//---- DEFINED IF Tpetra IS ENABLED.
#cmakedefine HAVE_TPETRA

//---- DEFINED IF ROL IS ENABLED.
#cmakedefine HAVE_ROL

//---- DEFINED IF SEACAS IS ENABLED.
#cmakedefine HAVE_SEACAS

//---- DEFINED IF GCP IS ENABLED.
#cmakedefine HAVE_GCP

//---- DEFINED IF LBFGSB IS ENABLED.
#cmakedefine HAVE_LBFGSB

//---- DEFINED IF MPI IS ENABLED.
#cmakedefine HAVE_MPI

//---- DEFINED IF DISTRIBUTED CODE IS ENABLED.
#cmakedefine HAVE_DIST

//---- DEFINED IF WE ARE USING MKL.
#cmakedefine HAVE_MKL

#include <cstddef>
#include <cstdint>

// Floating-point type
typedef @GENTEN_FLOAT_TYPE@ ttb_real;

// Tensor index type
typedef @GENTEN_INDEX_TYPE@ ttb_indx;

// Which execution spaces are enabled in executables
#cmakedefine HAVE_SERIAL
#cmakedefine HAVE_THREADS
#cmakedefine HAVE_OPENMP
#cmakedefine HAVE_CUDA
#cmakedefine HAVE_HIP
#cmakedefine HAVE_SYCL

//---- DEFINED IF PYTHON
#cmakedefine HAVE_PYTHON

//---- DEFINED IF EMBEDDED PYTHON INTERPRETER IS ENABLED
#cmakedefine HAVE_PYTHON_EMBED
