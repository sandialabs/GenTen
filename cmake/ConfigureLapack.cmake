#@HEADER
# ************************************************************************
#     Genten: Software for Generalized Tensor Decompositions
#     by Sandia National Laboratories
#
# Sandia National Laboratories is a multimission laboratory managed
# and operated by National Technology and Engineering Solutions of Sandia,
# LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
# U.S. Department of Energy's National Nuclear Security Administration under
# contract DE-NA0003525.
#
# Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ************************************************************************
#@HEADER


#---------------------------------------------------------------------------
# This module finds one or more libraries supplying LAPACK/BLAS functions.
# To link with a known library, set its location with command line options.
# If not set on the command line, then CMake searches in default locations.
# If no LAPACK/BLAS is found, then default to a GenTen implementation; however,
# it is not be tuned for best architecture performance.
#
# Command line options:
#   -DLAPACK_LIBS=
#   -DLAPACK_ADD_LIBS=
# The second option is for any additional libraries required to resolve
# missing symbols in the LAPACK libraries.
#
# Remember that CMake command line options are usually enclosed in
# double quotes, and multiple libraries are separated by a semicolon; eg:
#   -DLAPACK_LIBS="~/tools/liblapack.a;~/tools/libblas.a"    (Unix)
#   -DLAPACK_LIBS="c:\Temp\lapack.a;c:\Temp\blas.a"          (Windows)
#---------------------------------------------------------------------------

# Let CMake search in the usual system default locations.
# If LAPACK_LIBS is already defined from the command line,
# then FIND_LIBRARY accepts it without modification (or validating).
#
# Note that FIND_LIBRARY does not test if all references are resolved.
# Do not use LAPACK_LIBRARIES as this is an internal CMake variable name.
IF(ENABLE_LAPACK)
  IF(NOT LAPACK_LIBS)
    MESSAGE (STATUS "GenTen: Looking for LAPACK/BLAS libraries")
    # FIND_LIBRARY(LAPACK lapack)
    # FIND_LIBRARY(BLAS blas)
    # MARK_AS_ADVANCED(LAPACK BLAS)
    # IF(LAPACK AND BLAS)
    #   SET(LAPACK_LIBS "${LAPACK};${BLAS}" CACHE STRING "Location of LAPACK/BLAS libraries")
    # ENDIF()
    find_package(LAPACK)
    IF(LAPACK_FOUND)
      SET(LAPACK_LIBS "${LAPACK_LIBRARIES}" CACHE STRING "Location of LAPACK/BLAS libraries")
    ENDIF()
  ELSE (NOT LAPACK_LIBS)
    MESSAGE (STATUS "GenTen: Attempting to use user-defined BLAS/LAPACK libraries: ${LAPACK_LIBS}")
    # Check if user-specified libraries exist.
    FOREACH (loopVar  ${LAPACK_LIBS})
      IF (NOT EXISTS ${loopVar})
        MESSAGE (FATAL_ERROR "GenTen: Cannot find file " ${loopVar})
      ENDIF (NOT EXISTS ${loopVar})
    ENDFOREACH (loopVar)
  ENDIF(NOT LAPACK_LIBS)
ENDIF(ENABLE_LAPACK)

# Set the boolean LAPACK_FOUND.
IF (NOT ENABLE_LAPACK OR NOT LAPACK_LIBS)
  SET (LAPACK_FOUND FALSE)
  SET (LAPACK_LIBS "")
ELSE (NOT ENABLE_LAPACK OR NOT LAPACK_LIBS)
  SET (LAPACK_FOUND TRUE)
ENDIF (NOT ENABLE_LAPACK OR NOT LAPACK_LIBS)

# Report which libraries will be used.
IF (LAPACK_FOUND)
  MESSAGE (STATUS "GenTen: Using BLAS/LAPACK libraries:")
  FOREACH (loopVar ${LAPACK_LIBS})
    MESSAGE (STATUS "  " ${loopVar})
  ENDFOREACH (loopVar)
  FOREACH (loopVar ${LAPACK_ADD_LIBS})
    MESSAGE (STATUS "  linking with " ${loopVar})
  ENDFOREACH (loopVar)
ELSE (LAPACK_FOUND)
  IF(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
    MESSAGE (STATUS "GenTen: BLAS/LAPACK not enabled or not found, will compile GenTen BLAS/LAPACK stubs for correct host-side linking")
  ELSE()
    MESSAGE (FATAL_ERROR "GenTen: BLAS/LAPACK not found, but is required when building only for CPU!")
  ENDIF()
ENDIF (LAPACK_FOUND)
