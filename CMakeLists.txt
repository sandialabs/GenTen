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


#---- Specify the CMake version required.  Kokkos requires 3.3
CMAKE_MINIMUM_REQUIRED (VERSION 3.3)

#---- Declaring the project name will define ${Genten_SOURCE_DIR} to be
#---- the path to this source file, and ${Genten_BINARY_DIR} to be the path
#---- to the directory where cmake has been invoked.
PROJECT (Genten)

SET(CMAKE_BUILD_TYPE "DEBUG")


#------------------------------------------------------------
#---- Package OPTIONS
#------------------------------------------------------------
OPTION(BUILD_SHARED_LIBS "Build shared libraries." OFF)
OPTION(KOKKOS_INLINE_BUILD "Build Kokkos inline with Genten.  Kokkos source must be in genten/kokkos." OFF)
OPTION(KOKKOS_PATH "Path to Kokkos installation.  Do not set this for an inline build of Kokkos.")
OPTION(ENABLE_CUBLAS "Flag to enable cuBLAS for Kokkos with Cuda" OFF)
OPTION(ENABLE_CUSOLVER "Flag to enable cuSolver for Kokkos with Cuda" OFF)
OPTION(ENABLE_BOOST "Enable Boost for reading compressed tensor files" OFF)
OPTION(ENABLE_LAPACK "Enable BLAS/LAPACK (if disabled, Genten implementations will be used" ON)

# If we are doing an inline build of Kokkos
IF(KOKKOS_INLINE_BUILD)

  # We must use UVM and enable Lambda support in Kokkos
  IF(KOKKOS_ENABLE_CUDA)
    #SET(KOKKOS_ENABLE_CUDA_UVM ON CACHE BOOL "Enable CUDA unified virtual memory.")
    SET(KOKKOS_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable lambdas for CUDA.")
  ENDIF()

  # Configure kokkos
  ADD_SUBDIRECTORY(kokkos)

ENDIF()

SET(GENTEN_LIBS "")

IF (ENABLE_BOOST)
  OPTION(BOOST_PATH "Path to BOOST installation")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${BOOST_PATH}/include")
  SET(GENTEN_LIBS "${GENTEN_LIBS} -L${BOOST_PATH}/lib -lboost_iostreams -lz")
  SET(HAVE_BOOST ${ENABLE_BOOST})
ENDIF()

# For non-inline build of Kokkos, grep Makefile.kokkos for settings
IF(NOT KOKKOS_INLINE_BUILD)
  SET(KOKKOS_LIBS "")
  SET(KOKKOS_LDFLAGS "")
  SET(KOKKOS_MAKEFILE "${KOKKOS_PATH}/Makefile.kokkos")
  IF (NOT EXISTS ${KOKKOS_MAKEFILE})
    MESSAGE(FATAL_ERROR "Cannot find Makefile.kokkos in ${KOKKOS_PATH}.  Please make KOKKOS_PATH is set correctly.")
  ENDIF()

  FILE(STRINGS ${KOKKOS_MAKEFILE} tmp REGEX KOKKOS_CXXFLAGS)
  STRING(REGEX REPLACE " *KOKKOS_CXXFLAGS *= *(.*)" "\\1" KOKKOS_CXXFLAGS ${tmp})
  FILE(STRINGS ${KOKKOS_MAKEFILE} tmp REGEX KOKKOS_CPPFLAGS)
  STRING(REGEX REPLACE " *KOKKOS_CPPFLAGS *= *(.*)" "\\1" KOKKOS_CPPFLAGS ${tmp})
  FILE(STRINGS ${KOKKOS_MAKEFILE} tmp REGEX KOKKOS_LDFLAGS)
  STRING(REGEX REPLACE " *KOKKOS_LDFLAGS *= *(.*)" "\\1" KOKKOS_LDFLAGS ${tmp})
  FILE(STRINGS ${KOKKOS_MAKEFILE} tmp REGEX KOKKOS_LIBS)
  STRING(REGEX REPLACE " *KOKKOS_LIBS *= *(.*)" "\\1" KOKKOS_LIBS ${tmp})

  # strip -arch=xxx from KOKKOS_LDFLAGS due to Kokkos bug
  # (see https://github.com/kokkos/kokkos/issues/872)
  STRING(REGEX REPLACE "-arch=[a-zA-Z0-9_]+" "" KOKKOS_LDFLAGS ${KOKKOS_LDFLAGS})

  MESSAGE("-- Genten:  Using Kokkos:  ${KOKKOS_PATH}")
  MESSAGE("--   KOKKOS_CXXFLAGS = ${KOKKOS_CXXFLAGS}")
  MESSAGE("--   KOKKOS_CPPFLAGS = ${KOKKOS_CPPFLAGS}")
  MESSAGE("--   KOKKOS_LDFLAGS  = ${KOKKOS_LDFLAGS}")
  MESSAGE("--   KOKKOS_LIBS     = ${KOKKOS_LIBS}")

  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${KOKKOS_CXXFLAGS} ${KOKKOS_CPPFLAGS}")
  SET(GENTEN_LIBS "${GENTEN_LIBS} ${KOKKOS_LDFLAGS} ${KOKKOS_LIBS}")
  MESSAGE("--   CMAKE_CXXFLAGS  = ${CMAKE_CXX_FLAGS}")
ENDIF()
SET(HAVE_KOKKOS ON)

IF (ENABLE_CUBLAS)
  SET(GENTEN_LIBS "${GENTEN_LIBS} -lcublas")
  SET(HAVE_CUBLAS ${ENABLE_CUBLAS})
ENDIF()
IF (ENABLE_CUSOLVER)
  SET(GENTEN_LIBS "${GENTEN_LIBS} -lcusolver")
  SET(HAVE_CUSOLVER ${ENABLE_CUSOLVER})
ENDIF()

# Strip leading/trailing whitespace from GENTEN_LIBS
STRING(STRIP "${GENTEN_LIBS}" GENTEN_LIBS)

#------------------------------------------------------------
#---- Output Directories
#------------------------------------------------------------
#---- Put all the executables in a /bin subdirectory.
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Genten_BINARY_DIR}/bin)

#---- Put all the libraries in a /lib subdirectory.
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${Genten_BINARY_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${Genten_BINARY_DIR}/lib)


#------------------------------------------------------------
#---- INCLUDE Directories
#------------------------------------------------------------
#---- Configure the build type (debug or production).
INCLUDE (cmake/ConfigureBuildType.cmake)

#---- Define the build location as the default install directory.
IF (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET (CMAKE_INSTALL_PREFIX build)
ENDIF (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

#---- Define system-dependent C++ preprocessing macros,
#---- and copy to the build directory.
CONFIGURE_FILE (
  ${Genten_SOURCE_DIR}/cmake/CMakeInclude.h.cmake
  ${Genten_BINARY_DIR}/CMakeInclude/CMakeInclude.h
  )


#------------------------------------------------------------
#---- INCLUDE Directories
#------------------------------------------------------------
INCLUDE_DIRECTORIES (
  ${Kokkos_INCLUDE_DIRS_RET}
  ${Genten_SOURCE_DIR}/src/
  ${Genten_SOURCE_DIR}/src/mathlib/
  ${Genten_BINARY_DIR}/CMakeInclude/
  ${Genten_SOURCE_DIR}/driver/
  )
FILE (GLOB genten_headers
  ${Genten_SOURCE_DIR}/src/*.h
  ${Genten_SOURCE_DIR}/src/*.hpp
  ${Genten_SOURCE_DIR}/src/mathlib/*.h
  ${Genten_SOURCE_DIR}/src/mathlib/*.hpp
  )
INSTALL (
  FILES ${genten_headers}
  DESTINATION include
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  )


#------------------------------------------------------------
#---- Test Data
#------------------------------------------------------------
#---- Copy test data.
ADD_CUSTOM_TARGET (copy_data_dir ALL)
ADD_CUSTOM_COMMAND (
  TARGET copy_data_dir POST_BUILD
  COMMAND cmake -E copy_directory
  ${Genten_SOURCE_DIR}/data
  ${Genten_BINARY_DIR}/data
  )
INSTALL (
  DIRECTORY ${Genten_SOURCE_DIR}/test_data
  DESTINATION .
  DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE
  FILE_PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
  )

#---- The location of test data is unfortunately hard-coded into source code.
#---- The CMake directive below puts the relative location into source, but it
#---- causes strange compiler errors on Windows.
#---- ADD_DEFINITIONS (-DDATADIR=\"${Genten_SOURCE_DIR}/test_data/\")


#------------------------------------------------------------
#---- LAPACK Math Library
# Separate pure c code from c++ code so CMake doesn't add Kokkos CXX flags
# to compile of c code (since the c code can't depend on kokkos)
#------------------------------------------------------------
INCLUDE (cmake/ConfigureLapack.cmake)
IF (LAPACK_FOUND)
  ADD_DEFINITIONS (-DLAPACK_FOUND)
  ADD_LIBRARY (genten_mathlibs_c
    ${Genten_SOURCE_DIR}/src/mathlib/Genten_Default_VML.c
    )
ELSE (LAPACK_FOUND)
  #---- Tell CMake to build with default BLAS implementation.
  ADD_LIBRARY (genten_mathlibs_c
    ${Genten_SOURCE_DIR}/src/mathlib/Genten_Default_Blas.c
    ${Genten_SOURCE_DIR}/src/mathlib/Genten_Default_VML.c
    )
ENDIF (LAPACK_FOUND)
TARGET_LINK_LIBRARIES(
  genten_mathlibs_c
  ${LAPACK_LIBS} ${LAPACK_ADD_LIBS}
  )
INSTALL (
  TARGETS genten_mathlibs_c
  DESTINATION lib
  )

# Mathlibs
ADD_LIBRARY (genten_mathlibs
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_DiscreteCDF.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_FacTestSetGenerator.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_RandomMT.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_SystemTimer.cpp
  )
IF(NOT KOKKOS_INLINE_BUILD)
  SET(GENTEN_MATHLIBS
    genten_mathlibs
    genten_mathlibs_c
    )
ELSE()
  SET(GENTEN_MATHLIBS
    genten_mathlibs
    genten_mathlibs_c
    kokkos
    )
ENDIF()
TARGET_LINK_LIBRARIES(
  ${GENTEN_MATHLIBS} ${GENTEN_LIBS} ${LAPACK_LIBS} ${LAPACK_ADD_LIBS}
  )
INSTALL (
  TARGETS genten_mathlibs
  DESTINATION lib
  )


#------------------------------------------------------------
#---- Genten Library
#------------------------------------------------------------

#---- List the source files to be built.
SET (Genten_SOURCE_LIST
  ${Genten_SOURCE_DIR}/src/Genten_Array.cpp
  ${Genten_SOURCE_DIR}/src/Genten_CpAls.cpp
  ${Genten_SOURCE_DIR}/src/Genten_FacMatArray.cpp
  ${Genten_SOURCE_DIR}/src/Genten_FacMatrix.cpp
  ${Genten_SOURCE_DIR}/src/Genten_IndxArray.cpp
  ${Genten_SOURCE_DIR}/src/Genten_IOtext.cpp
  ${Genten_SOURCE_DIR}/src/Genten_Ktensor.cpp
  ${Genten_SOURCE_DIR}/src/Genten_MixedFormatOps.cpp
  ${Genten_SOURCE_DIR}/src/Genten_MathLibs_Wpr.cpp
  ${Genten_SOURCE_DIR}/src/Genten_portability.cpp
  ${Genten_SOURCE_DIR}/src/Genten_Sptensor.cpp
  ${Genten_SOURCE_DIR}/src/Genten_Sptensor_perm.cpp
  ${Genten_SOURCE_DIR}/src/Genten_Sptensor_row.cpp
  ${Genten_SOURCE_DIR}/src/Genten_Util.cpp
  )

#---- Tell CMake to build the library.
ADD_LIBRARY (genten ${Genten_SOURCE_LIST})
SET(GENTEN_LINK_LIBS genten ${GENTEN_MATHLIBS} ${GENTEN_LIBS} ${LAPACK_LIBS} ${LAPACK_ADD_LIBS} ${OPSYS_LIBRARIES})
TARGET_LINK_LIBRARIES(${GENTEN_LINK_LIBS})
INSTALL (
  TARGETS genten
  DESTINATION lib
  )

#------------------------------------------------------------
#---- Unit Test Executables
#------------------------------------------------------------

#---- Specifications for building unit_tests.

ADD_EXECUTABLE (
  unit_tests
  ${Genten_SOURCE_DIR}/test/Genten_Test_Array.cpp
  ${Genten_SOURCE_DIR}/test/Genten_Test_CpAls.cpp
  ${Genten_SOURCE_DIR}/test/Genten_Test_FacMatrix.cpp
  ${Genten_SOURCE_DIR}/test/Genten_Test_IndxArray.cpp
  ${Genten_SOURCE_DIR}/test/Genten_Test_IOtext.cpp
  ${Genten_SOURCE_DIR}/test/Genten_Test_Ktensor.cpp
  ${Genten_SOURCE_DIR}/test/Genten_Test_MixedFormats.cpp
  ${Genten_SOURCE_DIR}/test/Genten_Test_Sptensor.cpp
  ${Genten_SOURCE_DIR}/test/Genten_Test_UnitTests.cpp
  ${Genten_SOURCE_DIR}/test/Genten_Test_Utils.cpp
  )
TARGET_LINK_LIBRARIES (unit_tests ${GENTEN_LINK_LIBS})
INSTALL (
  TARGETS unit_tests
  DESTINATION test
  )


#------------------------------------------------------------
#---- Performance Test Executables
#------------------------------------------------------------

#---- Specifications for building each performance test.
#---- In each case specify the executable and its link library dependencies.

ADD_EXECUTABLE (
  perf_CpAlsAminoAcid
  ${Genten_SOURCE_DIR}/performance/Genten_CpAlsAminoAcid.cpp
  )
TARGET_LINK_LIBRARIES (perf_CpAlsAminoAcid ${GENTEN_LINK_LIBS})
INSTALL (
  TARGETS perf_CpAlsAminoAcid
  DESTINATION test
  )

ADD_EXECUTABLE (
  perf_CpAlsRandomKtensor
  ${Genten_SOURCE_DIR}/performance/Genten_CpAlsRandomKtensor.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_DiscreteCDF.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_FacTestSetGenerator.cpp
  )
TARGET_LINK_LIBRARIES (perf_CpAlsRandomKtensor ${GENTEN_LINK_LIBS})
INSTALL (
  TARGETS perf_CpAlsRandomKtensor
  DESTINATION test
  )

ADD_EXECUTABLE (
  perf_CpAlsRandomKtensor_Sweep
  ${Genten_SOURCE_DIR}/performance/Genten_CpAlsRandomKtensor_Sweep.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_DiscreteCDF.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_FacTestSetGenerator.cpp
  )
TARGET_LINK_LIBRARIES (perf_CpAlsRandomKtensor_Sweep ${GENTEN_LINK_LIBS})
INSTALL (
  TARGETS perf_CpAlsRandomKtensor_Sweep
  DESTINATION test
  )

#------------------------------------------------------------
#---- Drivers
#------------------------------------------------------------

#---- Specifications for building each performance test.
#---- In each case specify the executable and its link library dependencies.

ADD_EXECUTABLE (
  cpals
  ${Genten_SOURCE_DIR}/driver/Genten_Driver_CpAls.cpp
  )
TARGET_LINK_LIBRARIES (cpals ${GENTEN_LINK_LIBS})
INSTALL (
  TARGETS cpals
  DESTINATION drivers
  )
