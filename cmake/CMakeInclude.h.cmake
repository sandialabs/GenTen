//@HEADER
// ************************************************************************
//     Genten Tensor Toolbox
//     Software package for tensor math by Sandia National Laboratories
//
// Sandia National Laboratories is a multimission laboratory managed
// and operated by National Technology and Engineering Solutions of Sandia,
// LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
// U.S. Department of Energy’s National Nuclear Security Administration under
// contract DE-NA0003525.
//
// Copyright 20XX, Sandia Corporation.
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

//---- DEFINED IF cuBLAS IS ENABLED.
#cmakedefine HAVE_CUBLAS

//---- DEFINED IF cuSOLVER IS ENABLED.
#cmakedefine HAVE_CUSOLVER

//---- DEFINED IF BOOST IS ENABLED.
#cmakedefine HAVE_BOOST
