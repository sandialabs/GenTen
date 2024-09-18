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


#---- Specify the CMake version required.  Kokkos requires 3.16
CMAKE_MINIMUM_REQUIRED (VERSION 3.16)

#---- Require C++17 without GNU extensions
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_EXTENSIONS OFF)

#------------------------------------------------------------
#---- Package OPTIONS
#------------------------------------------------------------
# Note: OPTION is a binary thing and can't have string in them
OPTION(BUILD_SHARED_LIBS "Build shared libraries." OFF)
OPTION(ENABLE_BOOST "Enable Boost for reading compressed tensor files and distributed memory functionality" OFF)
OPTION(ENABLE_MPI "Enable MPI distributed memory functionality" OFF)
OPTION(ENABLE_CALIPER "Enable Caliper for application profiling" OFF)
OPTION(ENABLE_MATLAB "Enable MATLAB linkage to Tensor Toolbox" OFF)
OPTION(ENABLE_PYTHON "Enable PYTHON bindings" OFF)
OPTION(ENABLE_PYTHON_EMBED "Enable embedded PYTHON interpreter when ENABLE_PYTHON=ON (requires linking to libpython)" ON)
OPTION(ENABLE_PHYSICS_UTILS "Enable PYTHON physics specific tools" OFF)
OPTION(FLOAT_TYPE "C++ data type used for floating-point values" OFF)
OPTION(INDEX_TYPE "C++ data type used for tensor indices" OFF)
OPTION(ENABLE_GCP "Enable experimental GCP code" ON)
OPTION(ENABLE_LAPACK "Enable BLAS/LAPACK.  Only required for CPU (cuBLAS/cuSolver are used on Cuda and rocBLAS/rocSolver on HIP)." ON)
OPTION(ENABLE_TRILINOS "Enable use of Trilinos packages" OFF)
OPTION(ENABLE_TEUCHOS "Enable Teuchos utilities" ON)
OPTION(ENABLE_TPETRA "Enable Tpetra distributed parallelism" ON)
OPTION(ENABLE_ROL "Enable ROL optimization package" ON)
OPTION(ENABLE_SEACAS "Enable SEACAS tools" ON)
OPTION(ENABLE_CMAKE_TIMERS "Enable timers for each compile/link step" OFF)
OPTION(PYGENTEN_PIP "Indicate pyGenTen is being built through pip" OFF)
OPTION(PYGENTEN_MPI "Enable MPI in pyGenTen when being built through pip" OFF)
OPTION(PYGENTEN_SERIAL "Enable SERIAL in pyGenTen when being built through pip" OFF)
OPTION(PYGENTEN_OPENMP "Enable OPENMP in pyGenTen when being built through pip" OFF)
OPTION(PYGENTEN_CUDA "Enable CUDA in pyGenTen when being built through pip" OFF)
OPTION(PYGENTEN_HIP "Enable HIP in pyGenTen when being built through pip" OFF)
OPTION(PYGENTEN_SYCL "Enable SYCL in pyGenTen when being built through pip" OFF)

# Warn about some deprecated variables
IF(DEFINED KOKKOS_PATH)
  MESSAGE(DEPRECATION "KOKKOS_PATH is deprecated.  Use KOKKOS_ROOT instead")
  SET(Kokkos_ROOT ${KOKKOS_PATH})
ENDIF()
IF(DEFINED ROL_PATH)
  MESSAGE(DEPRECATION "ROL_PATH is deprecated.  Use Trilinos_ROOT instead")
  SET(Trilinos_ROOT ${ROL_PATH})
ENDIF()
IF(DEFINED MATLAB_PATH)
  MESSAGE(DEPRECATION "MATLAB_PATH is deprecated.  Use Matlab_ROOT instead")
  SET(Matlab_ROOT ${MATLAB_PATH})
ENDIF()

IF(FLOAT_TYPE)
  SET(GENTEN_FLOAT_TYPE ${FLOAT_TYPE})
ELSE()
  SET(GENTEN_FLOAT_TYPE "double")
ENDIF()

IF(INDEX_TYPE)
  SET(GENTEN_INDEX_TYPE ${INDEX_TYPE})
ELSE()
  SET(GENTEN_INDEX_TYPE "size_t")
ENDIF()

# Advertise that we support shared libs
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

# Setup architectures when building pyGenTen through pip.  We do this
# because pip/scikit-build doesn't have a way to specify meta-options, so
# we try to set things up for the user based on a single meta-cmake option
IF(PYGENTEN_PIP)
  SET(ENABLE_PYTHON ON)
  SET(BUILD_SHARED_LIBS ON)

  # If no device parallelism enabled, use Serial for MPI and OpenMP otherwise
  IF(NOT PYGENTEN_SERIAL AND NOT PYGENTEN_OPENMP AND NOT PYGENTEN_CUDA AND NOT PYGENTEN_HIP AND NOT PYGENTEN_SYCL)
    IF(PYGENTEN_MPI)
      SET(PYGENTEN_SERIAL ON)
    ELSE()
      SET(PYGENTEN_OPENMP ON)
    ENDIF()
  ENDIF()

  IF(PYGENTEN_MPI)
    SET(ENABLE_MPI ON)
    # Use standard compiler wrappers if we aren't told to do something else
    IF(NOT DEFINED CMAKE_CXX_COMPILER)
      SET(CMAKE_CXX_COMPILER "mpicxx")
    ENDIF()
    IF(NOT DEFINED CMAKE_C_COMPILER)
      SET(CMAKE_C_COMPILER "mpicc")
    ENDIF()
  ENDIF()

  IF(PYGENTEN_SERIAL)
    SET(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
    SET(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL "")
  ENDIF()

  IF(PYGENTEN_OPENMP)
    SET(Kokkos_ENABLE_OPENMP ON CACHE BOOL "")
    SET(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL "")
  ENDIF()

  IF(PYGENTEN_CUDA)
    SET(Kokkos_ENABLE_CUDA ON CACHE BOOL "")
    SET(NVCC_WRAPPER "${CMAKE_SOURCE_DIR}/tpls/kokkos/bin/nvcc_wrapper")
    IF(PYGENTEN_MPI)
      # We need to override the internal compilers called by the compiler
      # wrappers, which is done through environment variables.  While CMake
      # allows you to set environment variables for the current CMake process,
      # they are not propagated to the build, so we have to go through a script.
      # The user can override the compiler nvcc_wrapper calls be setting
      # NVCC_WRAPPER_DEFAULT_COMPILER as an environment variable.
      CONFIGURE_FILE (
        ${CMAKE_SOURCE_DIR}/cmake/genten_mpicxx_nvccwrapper.sh.tmpl
        ${CMAKE_BINARY_DIR}/cmake/genten_mpicxx_nvccwrapper.sh
	)
        SET(CMAKE_CXX_COMPILER "${CMAKE_BINARY_DIR}/cmake/genten_mpicxx_nvccwrapper.sh")
    ELSE()
      SET(CMAKE_CXX_COMPILER "${NVCC_WRAPPER}")
    ENDIF()
  ENDIF()

  IF(PYGENTEN_HIP)
    SET(Kokkos_ENABLE_HIP ON CACHE BOOL "")
    # We assume the user has MPI compiler wrappers for AMD GPUs, which are set
    # above, so we only need to set a compiler if none is provided and there is
    # no MPI.
    IF(NOT PYGENTEN_MPI AND NOT DEFINED CMAKE_CXX_COMPILER)
      SET(CMAKE_CXX_COMPILER "hipcc")
      SET(CMAKE_C_COMPILER "hipcc")
    ENDIF()
  ENDIF()

  IF(PYGENTEN_SYCL)
    SET(Kokkos_ENABLE_SYCL ON CACHE BOOL "")
    SET(Kokkos_ENABLE_ONEDPL OFF CACHE BOOL "")
    # We assume the user has MPI compiler wrappers for Intel GPUs, which are set
    # above, so we only need to set a compiler if none is provided and there is
    # no MPI.
    IF(NOT PYGENTEN_MPI AND NOT DEFINED CMAKE_CXX_COMPILER)
      SET(CMAKE_CXX_COMPILER "icpx")
      SET(CMAKE_C_COMPILER "icx")
    ENDIF()
  ENDIF()
ENDIF()

# Parse Trilinos support.  Do this before any compiler checks as we may
# set the compilers to match Trilinos
SET(Trilinos_LIBRARIES "")
SET(Trilinos_TPL_LIBRARIES "")
IF(ENABLE_TRILINOS)
  SET(Kokkos_FIND_QUIETLY ON)
  FIND_PACKAGE(Trilinos REQUIRED)
  MESSAGE(STATUS "GenTen:  Found Trilinos: ")
  MESSAGE(STATUS "  Location: ${Trilinos_DIR}")
  MESSAGE(STATUS "  Version: ${Trilinos_VERSION}")
  MESSAGE(STATUS "  Git Commit ID: ${Trilinos_GIT_COMMIT_ID}")
  MESSAGE(STATUS "  Enabled Packages: ${Trilinos_PACKAGE_LIST}")
  MESSAGE(STATUS "  Enabled  TPLs: ${Trilinos_TPL_LIST}")
  MESSAGE(STATUS "  Build shared libs: ${Trilinos_BUILD_SHARED_LIBS}")
  MESSAGE(STATUS "  CXX compiler flags: ${Trilinos_CXX_COMPILER_FLAGS}")

  # Get list of target so we can figure out what was enabled
  GET_PROPERTY(Trilinos_Target_List DIRECTORY PROPERTY IMPORTED_TARGETS)
  #MESSAGE("Trilinos targets: ${Trilinos_Target_List}")

  # Setup GenTen configuration to match Trilinos, unless user is providing
  # their own settings
  IF(NOT DEFINED CMAKE_CXX_COMPILER)
    SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
  ELSEIF(NOT ${Trilinos_CXX_COMPILER} STREQUAL ${CMAKE_CXX_COMPILER})
    MESSAGE(WARNING "Provided CMAKE_CXX_COMPILER (${CMAKE_CXX_COMPILER}) does not match Trilinos_CXX_COMPILER (${Trilinos_CXX_COMPILER}).  Using provided CMAKE_CXX_COMPILER.")
  ENDIF()
  IF(NOT DEFINED CMAKE_C_COMPILER)
    SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
  ELSEIF(NOT ${Trilinos_C_COMPILER} STREQUAL ${CMAKE_C_COMPILER})
    MESSAGE(WARNING "Provided CMAKE_C_COMPILER (${CMAKE_C_COMPILER}) does not match Trilinos_C_COMPILER (${Trilinos_C_COMPILER}).  Using provided CMAKE_C_COMPILER.")
  ENDIF()
  IF(NOT DEFINED BUILD_SHARED_LIBS)
    SET(BUILD_SHARED_LIBS ${Trilinos_BUILD_SHARED_LIBS})
  ELSEIF((Trilinos_BUILD_SHARED_LIBS AND NOT BUILD_SHARED_LIBS) OR
         (NOT Trilinos_BUILD_SHARED_LIBS AND BUILD_SHARED_LIBS))
    MESSAGE(WARNING "Provided BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} but Trilinos_BUILD_SHARED_LIBS=${Trilinos_BUILD_SHARED_LIBS}.  Using provided BUILD_SHARED_LIBS.")
  ENDIF()
  SET(CMAKE_CXX_FLAGS "${Trilinos_CXX_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
  SET(CMAKE_C_FLAGS "${Trilinos_C_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")

  # Use SYSTEM keyword so that output is not littered with ROL warnings
  INCLUDE_DIRECTORIES(SYSTEM ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})
  LINK_DIRECTORIES(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS} )

  # Look for Kokkos (optional)
  LIST(FIND Trilinos_Target_List Kokkos::all_libs Kokkos_List_ID)
  IF(Kokkos_List_ID GREATER -1)
    MESSAGE(STATUS "  Looking for Kokkos: Found")
    MESSAGE(STATUS "    Enabled Kokkos devices: ${Kokkos_DEVICES}")
    SET(HAVE_KOKKOS ON)
    SET(HAVE_TRILINOS_KOKKOS ON)
  ELSE()
    MESSAGE(STATUS "  Looking for Kokkos: NOT Found")
  ENDIF()

  # Look for Kokkos-Kernels (optional)
  LIST(FIND Trilinos_Target_List KokkosKernels::all_libs KokkosKernels_List_ID)
  IF(KokkosKernels_List_ID GREATER -1)
    MESSAGE(STATUS "  Looking for KokkosKernels: Found")
    SET(HAVE_KOKKOS_KERNELS ON)
    SET(HAVE_TRILINOS_KOKKOSKERNELS ON)
  ELSE()
    MESSAGE(STATUS "  Looking for KokkosKernels: NOT Found")
  ENDIF()

  # Look for Teuchos (optional)
  IF(ENABLE_TEUCHOS)
    LIST(FIND Trilinos_Target_List TeuchosComm::all_libs Teuchos_List_ID)
    IF(Teuchos_List_ID GREATER -1)
      MESSAGE(STATUS "  Looking for Teuchos: Found")
      SET(HAVE_TEUCHOS ON)
    ELSE()
      MESSAGE(STATUS "  Looking for Teuchos: NOT Found")
      SET(ENABLE_TEUCHOS OFF)
    ENDIF()
  ENDIF()

  # Look for Tpetra (optional)
  IF(ENABLE_TPETRA)
    LIST(FIND Trilinos_Target_List TpetraCore::all_libs Tpetra_List_ID)
    IF(Tpetra_List_ID GREATER -1)
      MESSAGE(STATUS "  Looking for Tpetra: Found")
      SET(HAVE_TPETRA ON)
    ELSE()
      MESSAGE(STATUS "  Looking for Tpetra: NOT Found")
      SET(ENABLE_TPETRA OFF)
    ENDIF()
  ENDIF()

  # Look for ROL (optional)
  IF(ENABLE_ROL)
    LIST(FIND Trilinos_Target_List ROL::all_libs ROL_List_ID)
    IF(ROL_List_ID GREATER -1)
      MESSAGE(STATUS "  Looking for ROL: Found")
      SET(HAVE_ROL ON)
    ELSE()
      MESSAGE(STATUS "  Looking for ROL: NOT Found")
      SET(ENABLE_ROL OFF)
    ENDIF()
  ENDIF()

  # Look for SEACAS (optional)
  IF(ENABLE_SEACAS)
    LIST(FIND Trilinos_Target_List SEACASExodus::all_libs SEACAS_List_ID)
    IF(SEACAS_List_ID GREATER -1)
      MESSAGE(STATUS "  Looking for SEACAS: Found")
      SET(HAVE_SEACAS ON)
    ELSE()
      MESSAGE(STATUS "  Looking for SEACAS: NOT Found")
      SET(ENABLE_SEACAS OFF)
    ENDIF()
  ENDIF()

  # Look for BLAS/LAPACK TPL
  LIST(FIND Trilinos_Target_List LAPACK::all_libs LAPACK_List_ID)
  LIST(FIND Trilinos_Target_List BLAS::all_libs BLAS_List_ID)
  IF(LAPACK_List_ID GREATER -1 AND BLAS_List_ID GREATER -1)
    MESSAGE(STATUS "  Looking for BLAS/LAPACK: Found")
    SET(ENABLE_LAPACK ON)
    SET(LAPACK_FOUND ON)
  ELSE()
    MESSAGE(STATUS "  Looking for BLAS/LAPACK:  NOT Found")
  ENDIF()

  # Look for MPI TPL
  IF(ENABLE_MPI)
    LIST(FIND Trilinos_Target_List MPI::all_libs MPI_List_ID)
    IF(MPI_List_ID GREATER -1)
      MESSAGE(STATUS "  Looking for MPI: Found")
      SET(MPI_FOUND ON)
      SET(HAVE_MPI ON)
      SET(MPI_TARGET MPI::all_libs)
      IF(NOT DEFINED MPI_LIBRARIES)
        set(MPI_LIBRARIES ${Trilinos_MPI_LIBRARIES})
      ENDIF()
      IF(NOT DEFINED MPI_LIBRARY_DIRS)
        set(MPI_LIBRARY_DIRS ${Trilinos_MPI_LIBRARY_DIRS})
      ENDIF()
      IF(NOT DEFINED MPI_INCLUDE_DIRS)
        set(MPI_INCLUDE_DIRS ${Trilinos_MPI_INCLUDE_DIRS})
      ENDIF()
      IF(NOT DEFINED MPIEXEC_EXECUTABLE)
        SET(MPIEXEC_EXECUTABLE ${Trilinos_MPI_EXEC})
      ENDIF()
      IF(NOT DEFINED MPIEXEC_PREFLAGS)
        SET(MPIEXEC_PREFLAGS ${Trilinos_MPI_EXEC_PRE_NUMPROCS_FLAGS})
      ENDIF()
      IF(NOT DEFINED MPIEXEC_MAX_NUMPROCS)
        SET(MPIEXEC_MAX_NUMPROCS ${Trilinos_MPI_EXEC_MAX_NUMPROCS})
      ENDIF()
      IF(NOT DEFINED MPIEXEC_POSTFLAGS)
        SET(MPIEXEC_POSTFLAGS ${Trilinos_MPI_EXEC_POST_NUMPROCS_FLAGS})
      ENDIF()
      IF(NOT DEFINED MPIEXEC_NUMPROC_FLAG)
        SET(MPIEXEC_NUMPROC_FLAG ${Trilinos_MPI_EXEC_NUMPROCS_FLAG})
      ENDIF()
    ELSE()
      MESSAGE(STATUS "  Looking for MPI:  NOT Found")
    ENDIF()
  ENDIF()

  SET(HAVE_TRILINOS ON)
ELSE()
  SET(ENABLE_TEUCHOS OFF)
  SET(ENABLE_TPETRA OFF)
  SET(ENABLE_ROL OFF)
  SET(ENABLE_SEACAS OFF)
ENDIF()

# We support 3 options for Kokkos:
#  - Inline build of bundled sources in tpls/kokkos.
#  - Linking to external library determined by Kokkos_ROOT
#  - Linking to external library through Trilinos.
# If Trilinos is enabled and it enables Kokkos, we must use it
IF(Kokkos_ROOT OR HAVE_TRILINOS_KOKKOS)
  SET(KOKKOS_INLINE_BUILD OFF)
ELSE()
  SET(KOKKOS_INLINE_BUILD ON)
ENDIF()
MESSAGE(STATUS "Inline Kokkos build is ${KOKKOS_INLINE_BUILD}")

# We support 3 options for KokkosKernels:
#  - Inline build of bundled sources in tpls/kokkos-kernels.
#  - Linking to external library determined by KokkosKernels_ROOT.
#  - Linking to external library through Trilinos.
# If Trilinos is enabled and it enables KokkosKernels, we must use it
IF(KokkosKernels_ROOT OR HAVE_TRILINOS_KOKKOSKERNELS)
  SET(KOKKOSKERNELS_INLINE_BUILD OFF)
ELSE()
  SET(KOKKOSKERNELS_INLINE_BUILD ON)
ENDIF()
MESSAGE(STATUS "Inline Kokkos-Kernels build is ${KOKKOSKERNELS_INLINE_BUILD}")

# Make sure kokkos and kokkos-kernels builds are consistent
# IF(KOKKOS_INLINE_BUILD OR KOKKOSKERNELS_INLINE_BUILD)
#   IF(NOT (KOKKOS_INLINE_BUILD EQUAL KOKKOSKERNELS_INLINE_BUILD))
#     MESSAGE(FATAL_ERROR "Both Kokkos and KokkosKernels must use the inline build if either does!")
#   ENDIF()
# ELSEIF(HAVE_TRILINOS_KOKKOS OR HAVE_TRILINOS_KOKKOSKERNELS)
#   IF(NOT (HAVE_TRILINOS_KOKKOS EQUAL HAVE_TRILINOS_KOKKOSKERNELS))
#     MESSAGE(FATAL_ERROR "Both Kokkos and KokkosKernels must use the Trilinos build if either does!")
#   ENDIF()
# ENDIF()

#---- Declaring the project name will define ${Genten_SOURCE_DIR} to be
#---- the path to this source file, and ${Genten_BINARY_DIR} to be the path
#---- to the directory where cmake has been invoked.
project(Genten VERSION "0.0.0")

# Check minimum compiler version
INCLUDE (cmake/CheckCompilerVersion.cmake)

# If we are doing an inline build of Kokkos
IF(KOKKOS_INLINE_BUILD)

  # We must enable Lambda support in Kokkos
  IF(Kokkos_ENABLE_CUDA)
    SET(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable lambdas for CUDA.")
  ENDIF()

  # Configure kokkos
  ADD_SUBDIRECTORY(tpls/kokkos)
  SET(HAVE_KOKKOS ON)

ELSEIF(NOT HAVE_TRILINOS_KOKKOS)

  # Link against installed Kokkos
  FIND_PACKAGE(Kokkos REQUIRED)
  MESSAGE("-- Genten:  Using Kokkos:  ${Kokkos_CONFIG}")
  SET(HAVE_KOKKOS ON)

ENDIF()

# Check for compatible enabled execution spaces in Trilinos/Tpetra and Kokkos
# (Trilinos won't let you enable an execution space in Tpetra but not Kokkos).
IF(HAVE_TPETRA)
  IF(Kokkos_ENABLE_SERIAL AND NOT Tpetra_INST_SERIAL)
    MESSAGE(FATAL_ERROR "Kokkos_ENABLE_SERIAL=${Kokkos_ENABLE_SERIAL} must match Tpetra_INST_SERIAL=${Tpetra_INST_SERIAL} to avoid linking errors")
  ENDIF()
  IF(Kokkos_ENABLE_THREADS AND NOT Tpetra_INST_PTHREAD)
    MESSAGE(FATAL_ERROR "Kokkos_ENABLE_THREADS=${Kokkos_ENABLE_THREADS} must match Tpetra_INST_PTHREAD=${Tpetra_INST_PTHREAD} to avoid linking errors")
  ENDIF()
  IF(Kokkos_ENABLE_OPENMP AND NOT Tpetra_INST_OPENMP)
    MESSAGE(FATAL_ERROR "Kokkos_ENABLE_OPENMP=${Kokkos_ENABLE_OPENMP} must match Tpetra_INST_OPENMP=${Tpetra_INST_OPENMP} to avoid linking errors")
  ENDIF()
  IF(Kokkos_ENABLE_CUDA AND NOT Tpetra_INST_CUDA)
    MESSAGE(FATAL_ERROR "Kokkos_ENABLE_CUDA=${Kokkos_ENABLE_CUDA} must match Tpetra_INST_OPENMP=${Tpetra_INST_CUDA} to avoid linking errors")
  ENDIF()
  IF(Kokkos_ENABLE_HIP AND NOT Tpetra_INST_HIP)
    MESSAGE(FATAL_ERROR "Kokkos_ENABLE_HIP=${Kokkos_ENABLE_HIP} must match Tpetra_INST_HIP=${Tpetra_INST_HIP} to avoid linking errors")
  ENDIF()
  IF(Kokkos_ENABLE_SYCL AND NOT Tpetra_INST_SYCL)
    MESSAGE(FATAL_ERROR "Kokkos_ENABLE_SYCL=${Kokkos_ENABLE_SYCL} must match Tpetra_INST_SYCL=${Tpetra_INST_SYCL} to avoid linking errors")
  ENDIF()
ENDIF()

# Embedded "JSON for Modern C++" library
set(JSON_Install ON CACHE INTERNAL "")
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(tpls/json)

# Embedded "JSON schema validator for JSON for Modern C++"
set(JSON_VALIDATOR_INSTALL ON CACHE INTERNAL "")
set(JSON_VALIDATOR_BUILD_TESTS OFF CACHE INTERNAL "")
set(JSON_VALIDATOR_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(JSON_VALIDATOR_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE INTERNAL "")
set(nlohmann_json_schema_validator_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE INTERNAL "")
add_subdirectory(tpls/json-validator)

SET(GENTEN_LIBS "")

if(ENABLE_MPI AND NOT MPI_FOUND)
  find_package(MPI REQUIRED)
  if (MPI_FOUND)
    set(HAVE_MPI ON)
    set(MPI_TARGET MPI::MPI_C)
  endif()
endif()

if(ENABLE_BOOST)
  find_package(Boost REQUIRED COMPONENTS iostreams)
  find_package(ZLIB REQUIRED)
endif()
if (BOOST_FOUND AND ZLIB_FOUND)
  set(HAVE_BOOST ON)
endif()

if(MPI_FOUND)
  set(HAVE_DIST ON)
endif()

IF(Kokkos_ENABLE_SYCL)
  FIND_PACKAGE(MKL CONFIG REQUIRED)
  MESSAGE(STATUS "GenTen: Using SYCL MKL")
  IF (LAPACK_LIBS)
    MESSAGE(WARNING "GenTen:  Enabling the SYCL execution space requires MKL for host-side linking.  Overriding LAPACK_LIBS with MKL.")
    SET(LAPACK_LIBS "")
  ENDIF()
  SET(HAVE_MKL ON)
  SET(LAPACK_FOUND ON)
ENDIF()

IF(NOT LAPACK_FOUND)
  INCLUDE (cmake/ConfigureLapack.cmake)
ENDIF()
IF (LAPACK_FOUND)
  ADD_DEFINITIONS (-DLAPACK_FOUND)
ENDIF()

IF (ENABLE_CALIPER)
  OPTION(CALIPER_PATH "Path to Caliper installation")
  OPTION(LIBPFM_PATH "Path to libpfm installation")
  OPTION(LIBUNWIND_PATH "Path to libunwind installation")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CALIPER_PATH}/include")
  SET(GENTEN_LIBS "${GENTEN_LIBS} -L${CALIPER_PATH}/lib64 -lcaliper -lcaliper-common -lcaliper-reader -lgotcha")
  IF (LIBPFM_PATH)
    SET(GENTEN_LIBS "${GENTEN_LIBS} -L${LIBPFM_PATH}/lib -lpfm")
  ENDIF()
  IF (LIBUNWIND_PATH)
    SET(GENTEN_LIBS "${GENTEN_LIBS} -L${LIBUNWIND_PATH}/lib -lunwind")
  ENDIF()
  SET(HAVE_CALIPER ${ENABLE_CALIPER})
ENDIF()

# L-BFGS-B requires ttb_real == double
IF (GENTEN_FLOAT_TYPE STREQUAL "double")
  SET(ENABLE_LBFGSB ON)
  SET(HAVE_LBFGSB ON)
ENDIF()

IF(ENABLE_MATLAB)
  IF(NOT BUILD_SHARED_LIBS)
    MESSAGE(FATAL_ERROR "Must build shared libraries when enabling Matlab!")
  ENDIF()
  OPTION(Matlab_ROOT "Path to MATLAB installation")
  IF(NOT DEFINED Matlab_ROOT_DIR AND DEFINED Matlab_ROOT)
    SET(Matlab_ROOT_DIR "${Matlab_ROOT}")
  ENDIF()
  FIND_PACKAGE(Matlab COMPONENTS MX_LIBRARY)
  IF(Matlab_FOUND)
    SET(HAVE_MATLAB ON)
  ELSE()
    MESSAGE(FATAL_ERROR "Could not find MATLAB!")
  ENDIF()
ENDIF()

IF(ENABLE_PYTHON)
  IF(NOT BUILD_SHARED_LIBS)
    MESSAGE(FATAL_ERROR "Must build shared libraries when enabling Python!")
  ENDIF()
  MESSAGE("-- PYTHON_EXECUTABLE:")
  IF(NOT DEFINED PYTHON_EXECUTABLE)
    find_program(PYTHON_EXECUTABLE
        NAMES python3 python
        )
    MESSAGE("  -- CMake has set: PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
  ELSE()
    MESSAGE("  -- User has set: PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
  ENDIF()
  # Get the python version
  EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c
                          "import sys; print(sys.version_info.major)"
    OUTPUT_VARIABLE PYTHON_MAJOR_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # Get the python version
  EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c
                          "import sys; print(sys.version_info.minor)"
    OUTPUT_VARIABLE PYTHON_MINOR_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  SET(PYTHON_VERSION ${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION})
  MESSAGE("  -- PYTHON_VERSION = ${PYTHON_VERSION}")
  MESSAGE("  -- PYTHON_MAJOR_VERSION = ${PYTHON_MAJOR_VERSION}")

  SET(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION})

  ADD_SUBDIRECTORY(tpls/pybind11)
  SET(HAVE_PYTHON ON)
  IF(ENABLE_PYTHON_EMBED)
    SET(HAVE_PYTHON_EMBED ON)
  ENDIF()
ENDIF()

IF(ENABLE_PHYSICS_UTILS)
  IF(NOT HAVE_PYTHON)
    MESSAGE(FATAL_ERROR "Must enable Python when enabling Physics Utils!")
  ENDIF()
  IF(NOT HAVE_TPETRA)
    MESSAGE(FATAL_ERROR "Must enable Tpetra when enabling Physics Utils!")
  ENDIF()
  SET(HAVE_PHYS_UTILS ON)
ENDIF()

IF(Kokkos_ENABLE_CUDA)
  IF(LIBCUBLAS_PATH)
    SET(GENTEN_LIBS "${GENTEN_LIBS} -L${LIBCUBLAS_PATH} -lcublas")
  ELSE()
    SET(GENTEN_LIBS "${GENTEN_LIBS} -lcublas")
  ENDIF()
  MESSAGE(STATUS "GenTen: Using cuBLAS")

  IF(LIBCUSOLVER_PATH)
    SET(GENTEN_LIBS "${GENTEN_LIBS} -L${LIBCUSOLVER_PATH} -lcusolver")
  ELSE()
    SET(GENTEN_LIBS "${GENTEN_LIBS} -lcusolver")
  ENDIF()
  MESSAGE (STATUS "GenTen: Using cuSolver")

  # Add useful CXX flags when building with Cuda
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-relaxed-constexpr")
  IF (KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xcudafe --diag_suppress=code_is_unreachable")
    #message("CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
  ENDIF()
ENDIF()

# Determine which execution spaces are enabled in executables (these are not
# use to determine what is enabled in the code itself).  This prevents the use
# of host-side execution spaces when LAPACK was not found (which will fail).
if(Kokkos_ENABLE_SERIAL)
  if(LAPACK_FOUND)
    set(HAVE_SERIAL ON)
  else()
    message(STATUS "GenTen: Kokkos Serial execution space enabled but LAPACK was not found.  Disabling Serial in GenTen.")
  endif()
endif()
if(Kokkos_ENABLE_THREADS)
  if(LAPACK_FOUND)
    set(HAVE_THREADS ON)
  else()
    message(STATUS "GenTen: Kokkos Threads execution space enabled but LAPACK was not found.  Disabling Threads in GenTen.")
  endif()
endif()
if(Kokkos_ENABLE_OPENMP)
  if(LAPACK_FOUND)
    set(HAVE_OPENMP ON)
  else()
    message(STATUS "GenTen: Kokkos OpenMP execution space enabled but LAPACK was not found.  Disabling OpenMP in GenTen.")
  endif()
endif()
if(Kokkos_ENABLE_CUDA)
  set(HAVE_CUDA ON)
endif()
if(Kokkos_ENABLE_HIP)
  set(HAVE_HIP ON)
endif()
if(Kokkos_ENABLE_SYCL)
  set(HAVE_SYCL ON)
endif()

IF(ENABLE_GCP)
  SET(HAVE_GCP ON)
ENDIF()

# If we are doing an inline build of Kokkos-Kernels
IF(KOKKOSKERNELS_INLINE_BUILD)

  # Configure kokkos-kernels
  SET(KokkosKernels_ADD_DEFAULT_ETI OFF CACHE BOOL "")
  #SET(KokkosKernels_INST_LAYOUTLEFT ON CACHE BOOL "")
  #SET(KokkosKernels_INST_LAYOUTRIGHT ON CACHE BOOL "")
  SET(KokkosKernels_ENABLE_ALL_COMPONENTS OFF CACHE BOOL "")
  SET(KokkosKernels_ENABLE_COMPONENT_BLAS ON CACHE BOOL "")
  SET(KokkosKernels_ENABLE_COMPONENT_BATCHED ON CACHE BOOL "")
  IF(LAPACK_FOUND)
    SET(KokkosKernels_ENABLE_TPL_BLAS ON CACHE BOOL "")
    SET(KokkosKernels_ENABLE_TPL_LAPACK ON CACHE BOOL "")
    IF(LAPACK_LIBS AND (NOT BLAS_LIBRARIES OR NOT LAPACK_LIBRARIES))
      # extract list of libraries and paths from LAPACK_LIBS, LAPACK_ADD_LIBS
      # to pass to KokkosKernels
      SET(blas_libs "")
      SET(blas_dirs "")
      FOREACH(libvar ${LAPACK_LIBS})
        GET_FILENAME_COMPONENT(varname ${libvar} NAME)
        GET_FILENAME_COMPONENT(dirname ${libvar} DIRECTORY)
        LIST(APPEND blas_libs ${varname})
        LIST(APPEND blas_dirs ${dirname})
      ENDFOREACH()
      IF(NOT BLAS_LIBRARIES)
        SET(BLAS_LIBRARIES "${blas_libs}" CACHE STRING "")
        SET(BLAS_LIBRARY_DIRS "${blas_dirs}" CACHE STRING "")
      ENDIF()
      IF(NOT LAPACK_LIBRARIES)
        SET(LAPACK_LIBRARIES "${blas_libs}" CACHE STRING "")
        SET(LAPACK_LIBRARY_DIRS "${blas_dirs}" CACHE STRING "")
      ENDIF()
    ENDIF()
  ENDIF()
  IF(Kokkos_ENABLE_CUDA)
    SET(KokkosKernels_ENABLE_TPL_CUBLAS ON CACHE BOOL "")
  ENDIF()
  IF(Kokkos_ENABLE_HIP)
    SET(KokkosKernels_ENABLE_TPL_ROCBLAS ON CACHE BOOL "")
  ENDIF()
  ADD_SUBDIRECTORY(tpls/kokkos-kernels)
  SET(HAVE_KOKKOS_KERNELS ON)

ELSEIF(NOT HAVE_TRILINOS_KOKKOSKERNELS)

  # Link against installed KokkosKernels
  FIND_PACKAGE(KokkosKernels REQUIRED)
  MESSAGE("-- Genten:  Using KokkosKernels:  ${KokkosKernels_CONFIG}")
  SET(HAVE_KOKKOS_KERNELS ON)

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

#--- Use standard GNU installation directories
INCLUDE(GNUInstallDirs)

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
  ${Genten_BINARY_DIR}/cmake/CMakeInclude.h
  )


#------------------------------------------------------------
#---- INCLUDE Directories
#------------------------------------------------------------
INCLUDE_DIRECTORIES (
  ${Kokkos_INCLUDE_DIRS_RET}
  ${Genten_SOURCE_DIR}/src/
  ${Genten_SOURCE_DIR}/src/mathlib/
  ${Genten_SOURCE_DIR}/src/rol/
  ${Genten_SOURCE_DIR}/src/lbfgsb/
  ${Genten_BINARY_DIR}/cmake/
  ${Genten_SOURCE_DIR}/driver/
  ${Genten_SOURCE_DIR}/tpls/lbfgsb/
  ${Genten_SOURCE_DIR}/tpls/json/single_include/nlohmann/
  ${Genten_SOURCE_DIR}/tpls/json-validator/src/
  ${Genten_SOURCE_DIR}/tpls/json-validator/src/nlohmann/
  ${Genten_SOURCE_DIR}/joint_moments/
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
  DIRECTORY ${Genten_SOURCE_DIR}/data
  DESTINATION ${CMAKE_INSTALL_DATADIR}/genten
  DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE
  FILE_PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
  )

#---- The location of test data is unfortunately hard-coded into source code.
#---- The CMake directive below puts the relative location into source, but it
#---- causes strange compiler errors on Windows.
#---- ADD_DEFINITIONS (-DDATADIR=\"${Genten_SOURCE_DIR}/data/\")

#------------------------------------------------------------
#---- L-BFGS-B library
#------------------------------------------------------------
IF (ENABLE_LBFGSB)
  ADD_LIBRARY (genten_lbfgsb_c
    ${Genten_SOURCE_DIR}/tpls/lbfgsb/lbfgsb.c
    ${Genten_SOURCE_DIR}/tpls/lbfgsb/linesearch.c
    ${Genten_SOURCE_DIR}/tpls/lbfgsb/subalgorithms.c
    ${Genten_SOURCE_DIR}/tpls/lbfgsb/print.c
    ${Genten_SOURCE_DIR}/tpls/lbfgsb/linpack.c
    ${Genten_SOURCE_DIR}/tpls/lbfgsb/miniCBLAS.c
    ${Genten_SOURCE_DIR}/tpls/lbfgsb/timer.c
    )
  TARGET_LINK_LIBRARIES(
    genten_lbfgsb_c
    ${LAPACK_LIBS} ${LAPACK_ADD_LIBS}
    )
  INSTALL (
    TARGETS genten_lbfgsb_c
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    EXPORT genten
    )
ENDIF()

#------------------------------------------------------------
#---- Genten Library
#------------------------------------------------------------

# Add timers for each compile/link
IF(ENABLE_CMAKE_TIMERS)
  SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
  SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
ENDIF()

#---- List the source files to be built.
SET (Genten_SOURCE_LIST
  ${Genten_SOURCE_DIR}/src/Genten_DistContext.cpp
  ${Genten_SOURCE_DIR}/src/Genten_Pmap.cpp
  ${Genten_SOURCE_DIR}/src/Genten_DistTensorContext.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_DiscreteCDF.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_FacTestSetGenerator.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_RandomMT.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_SystemTimer.cpp
  ${Genten_SOURCE_DIR}/src/Genten_Util.cpp
  ${Genten_SOURCE_DIR}/src/Genten_AlgParams.cpp
  ${Genten_SOURCE_DIR}/src/Genten_Array.cpp
  ${Genten_SOURCE_DIR}/src/Genten_CpAls.cpp
  ${Genten_SOURCE_DIR}/src/Genten_FacMatArray.cpp
  ${Genten_SOURCE_DIR}/src/Genten_FacMatrix.cpp
  ${Genten_SOURCE_DIR}/src/Genten_IndxArray.cpp
  ${Genten_SOURCE_DIR}/src/Genten_IOtext.cpp
  ${Genten_SOURCE_DIR}/src/Genten_TensorIO.cpp
  ${Genten_SOURCE_DIR}/src/Genten_Ktensor.cpp
  ${Genten_SOURCE_DIR}/src/Genten_MixedFormatOps.cpp
  # Disabling TTM for now.  Will remove later
  #${Genten_SOURCE_DIR}/src/Genten_TTM.cpp
  ${Genten_SOURCE_DIR}/src/Genten_MathLibs_Wpr.cpp
  ${Genten_SOURCE_DIR}/src/Genten_portability.cpp
  ${Genten_SOURCE_DIR}/src/Genten_Sptensor.cpp
  ${Genten_SOURCE_DIR}/src/Genten_Tensor.cpp
  ${Genten_SOURCE_DIR}/src/Genten_HessVec.cpp
  ${Genten_SOURCE_DIR}/src/Genten_DistKtensorUpdate.cpp
  ${Genten_SOURCE_DIR}/joint_moments/Genten_HigherMoments.cpp
  ${Genten_SOURCE_DIR}/src/Genten_Driver.cpp
  ${Genten_SOURCE_DIR}/src/Genten_CublasHandle.cpp
  ${Genten_SOURCE_DIR}/src/Genten_CusolverHandle.cpp
  ${Genten_SOURCE_DIR}/src/Genten_RocblasHandle.cpp
  )

IF (HAVE_MPI)
SET (Genten_SOURCE_LIST ${Genten_SOURCE_LIST}
  ${Genten_SOURCE_DIR}/src/Genten_MPI_IO.cpp
  )
ENDIF()

IF (ENABLE_LBFGSB)
  SET (Genten_SOURCE_LIST ${Genten_SOURCE_LIST}
    ${Genten_SOURCE_DIR}/src/lbfgsb/Genten_CP_Opt_Lbfgsb.cpp
    )
ENDIF()

IF (ENABLE_ROL)
  SET (Genten_SOURCE_LIST ${Genten_SOURCE_LIST}
    ${Genten_SOURCE_DIR}/src/rol/Genten_CP_Opt_Rol.cpp
    )
ENDIF()

IF (ENABLE_GCP)
  # Default loss types if not specified by the user
  IF(NOT GCP_LOSS_TYPES)
    SET(GCP_LOSS_TYPES "Gaussian;Rayleigh;Gamma;Bernoulli;Poisson")
  ENDIF()

  # Ensure Gaussian is included since tests use it
  LIST(FIND GCP_LOSS_TYPES "Gaussian" GAUSSIAN_INDEX)
  IF (GAUSSIAN_INDEX EQUAL -1)
    MESSAGE(ERROR "Gaussian must be an included GCP loss type!")
  ENDIF()
  MESSAGE(STATUS "Genten: Enabled GCP loss functions:  ${GCP_LOSS_TYPES}")

  # Construct Genten_GCP_LossFunctions.hpp from src/Genten_GCP_LossFunctions.tmpl
  SET(GCP_INC "")
  SET(GCP_DIS "")
  SET(GCP_INS "")
  SET(FIRST TRUE)
  FOREACH(LOSS ${GCP_LOSS_TYPES})
    SET(LOSS_FILE_NAME "Genten_GCP_${LOSS}LossFunction.hpp")
    SET(LOSS_INC "#include \"${LOSS_FILE_NAME}\"")
    STRING(TOLOWER "${LOSS}" LOSS_NAME)
    IF (FIRST)
      SET(LOSS_DIS "    if (loss == \"${LOSS_NAME}\")\n      f(${LOSS}LossFunction(algParams));")
      SET(FIRST FALSE)
    ELSE()
      SET(LOSS_DIS "    else if (loss == \"${LOSS_NAME}\")\n      f(${LOSS}LossFunction(algParams));")
    ENDIF()
    SET(LOSS_INS "  LOSS_INST_MACRO(SPACE,Genten::${LOSS}LossFunction) \\")
    STRING(APPEND GCP_INC "${LOSS_INC}\n")
    STRING(APPEND GCP_DIS "${LOSS_DIS}\n")
    STRING(APPEND GCP_INS "${LOSS_INS}\n")
  ENDFOREACH()
  CONFIGURE_FILE(
    ${Genten_SOURCE_DIR}/src/Genten_GCP_LossFunctions.tmpl
    ${Genten_BINARY_DIR}/cmake/Genten_GCP_LossFunctions.hpp)

  # Construct ETI files for those that use a different file for each loss
  FOREACH(LOSS ${GCP_LOSS_TYPES})
    CONFIGURE_FILE(
      ${Genten_SOURCE_DIR}/src/Genten_GCP_ValueKernels.tmpl
      ${Genten_BINARY_DIR}/cmake/Genten_GCP_ValueKernels_${LOSS}.cpp)
    CONFIGURE_FILE(
      ${Genten_SOURCE_DIR}/src/Genten_GCP_SS_Grad.tmpl
      ${Genten_BINARY_DIR}/cmake/Genten_GCP_SS_Grad_${LOSS}.cpp)
    CONFIGURE_FILE(
      ${Genten_SOURCE_DIR}/src/Genten_GCP_SS_Grad_SA.tmpl
      ${Genten_BINARY_DIR}/cmake/Genten_GCP_SS_Grad_SA_${LOSS}.cpp)
    SET(Genten_SOURCE_LIST ${Genten_SOURCE_LIST}
      ${Genten_BINARY_DIR}/cmake/Genten_GCP_ValueKernels_${LOSS}.cpp
      ${Genten_BINARY_DIR}/cmake/Genten_GCP_SS_Grad_${LOSS}.cpp
      ${Genten_BINARY_DIR}/cmake/Genten_GCP_SS_Grad_SA_${LOSS}.cpp)
  ENDFOREACH()

  SET (Genten_SOURCE_LIST ${Genten_SOURCE_LIST}
    ${Genten_SOURCE_DIR}/src/Genten_GCP_SamplingKernels.cpp
    ${Genten_SOURCE_DIR}/src/Genten_GCP_SGD.cpp
    ${Genten_SOURCE_DIR}/src/Genten_GCP_SGD_SA.cpp
    ${Genten_SOURCE_DIR}/src/Genten_GCP_StreamingHistory.cpp
    ${Genten_SOURCE_DIR}/src/Genten_Online_GCP.cpp
    )

  IF (ENABLE_LBFGSB)
    SET (Genten_SOURCE_LIST ${Genten_SOURCE_LIST}
      ${Genten_SOURCE_DIR}/src/lbfgsb/Genten_GCP_Opt_Lbfgsb.cpp
      )
  ENDIF()

  IF (ENABLE_ROL)
    SET (Genten_SOURCE_LIST ${Genten_SOURCE_LIST}
      ${Genten_SOURCE_DIR}/src/rol/Genten_GCP_Opt_Rol.cpp
      ${Genten_SOURCE_DIR}/src/rol/Genten_GCP_RolObjective.cpp
      )
  ENDIF()
ENDIF()

#---- Tell CMake to build the library.
ADD_LIBRARY (gentenlib ${Genten_SOURCE_LIST})

IF(HAVE_TRILINOS)
  TARGET_LINK_LIBRARIES(gentenlib Trilinos::all_libs)
ENDIF()
IF(NOT HAVE_TRILINOS_KOKKOS)
  TARGET_LINK_LIBRARIES(gentenlib Kokkos::kokkos)
ENDIF()
IF(NOT HAVE_TRILINOS_KOKKOSKERNELS)
  TARGET_LINK_LIBRARIES(gentenlib Kokkos::kokkoskernels)
ENDIF()
IF(ENABLE_LBFGSB)
  TARGET_LINK_LIBRARIES (gentenlib genten_lbfgsb_c)
ENDIF()
TARGET_LINK_LIBRARIES(gentenlib nlohmann_json_schema_validator)
TARGET_LINK_LIBRARIES(gentenlib nlohmann_json::nlohmann_json)
IF(HAVE_BOOST)
  TARGET_LINK_LIBRARIES(gentenlib Boost::iostreams ZLIB::ZLIB)
ENDIF()
IF(HAVE_MPI)
  TARGET_LINK_LIBRARIES(gentenlib ${MPI_TARGET})
  IF(HAVE_BOOST)
    TARGET_LINK_LIBRARIES(gentenlib Boost::boost)
  ENDIF()
ENDIF()
IF(HAVE_PYTHON_EMBED)
  TARGET_LINK_LIBRARIES (gentenlib pybind11::embed)
ENDIF()

IF(Kokkos_ENABLE_HIP)
  IF(ROCM_SEARCH_PATH)
    LIST(APPEND CMAKE_PREFIX_PATH ${ROCM_SEARCH_PATH})
  ENDIF()

  FIND_PACKAGE(rocblas REQUIRED)
  TARGET_LINK_LIBRARIES(gentenlib roc::rocblas)
  MESSAGE(STATUS "GenTen: Using rocBLAS")

  FIND_PACKAGE(rocsolver REQUIRED)
  TARGET_LINK_LIBRARIES(gentenlib roc::rocsolver)
  MESSAGE(STATUS "GenTen: Using rocSOLVER")

  FIND_PACKAGE(rocthrust REQUIRED)
  TARGET_LINK_LIBRARIES(gentenlib roc::rocthrust)
  MESSAGE(STATUS "GenTen: Using rocTHRUST")
ENDIF()

IF(Kokkos_ENABLE_SYCL)
  # We require the DPC++ variant of MKL for SYCL, and to avoid all
  # kinds of issues with inconsistent BLAS/LAPACK libraries, we also
  # then require the regular MKL for host-side linking
  TARGET_LINK_LIBRARIES(gentenlib $<LINK_ONLY:MKL::MKL>)
  TARGET_LINK_LIBRARIES(gentenlib $<LINK_ONLY:MKL::MKL_DPCPP>)
ENDIF()

SET(GENTEN_LINK_LIBS gentenlib ${GENTEN_LIBS} ${LAPACK_LIBS} ${LAPACK_ADD_LIBS} ${OPSYS_LIBRARIES})
TARGET_LINK_LIBRARIES(${GENTEN_LINK_LIBS})
INSTALL (
  TARGETS gentenlib
  EXPORT genten
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

#---- Adding the higher moment interface to the Library sources
#---- Must be a better way to do this. Inside an IF??
ADD_SUBDIRECTORY(joint_moments)

#------------------------------------------------------------
#---- Performance Test Executables (not installed)
#------------------------------------------------------------
FILE (GLOB genten_headers
  ${Genten_SOURCE_DIR}/src/*.h
  ${Genten_SOURCE_DIR}/src/*.hpp
  ${Genten_SOURCE_DIR}/src/mathlib/*.h
  ${Genten_SOURCE_DIR}/src/mathlib/*.hpp
  ${Genten_SOURCE_DIR}/src/rol/*.hpp
  ${Genten_SOURCE_DIR}/src/lbfgsb/*.hpp
  ${Genten_BINARY_DIR}/cmake/*.h
  ${Genten_BINARY_DIR}/cmake/*.hpp
  ${Genten_SOURCE_DIR}/tpls/lbfgsb/*.h
  ${Genten_SOURCE_DIR}/joint_moments/Genten_HigherMoments.hpp
  )
INSTALL (
  FILES ${genten_headers}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/genten
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  )

#------------------------------------------------------------
#---- Performance Test Executables (not installed)
#------------------------------------------------------------

ADD_EXECUTABLE (
  perf_CpAlsAminoAcid
  ${Genten_SOURCE_DIR}/performance/Genten_CpAlsAminoAcid.cpp
  )
TARGET_LINK_LIBRARIES (perf_CpAlsAminoAcid ${GENTEN_LINK_LIBS})

ADD_EXECUTABLE (
  perf_CpAlsRandomKtensor
  ${Genten_SOURCE_DIR}/performance/Genten_CpAlsRandomKtensor.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_DiscreteCDF.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_FacTestSetGenerator.cpp
  )
TARGET_LINK_LIBRARIES (perf_CpAlsRandomKtensor ${GENTEN_LINK_LIBS})

ADD_EXECUTABLE (
  perf_CpAlsRandomKtensor_Sweep
  ${Genten_SOURCE_DIR}/performance/Genten_CpAlsRandomKtensor_Sweep.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_DiscreteCDF.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_FacTestSetGenerator.cpp
  )
TARGET_LINK_LIBRARIES (perf_CpAlsRandomKtensor_Sweep ${GENTEN_LINK_LIBS})

ADD_EXECUTABLE (
  perf_MTTKRP
  ${Genten_SOURCE_DIR}/performance/Genten_MTTKRP.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_DiscreteCDF.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_FacTestSetGenerator.cpp
  )
TARGET_LINK_LIBRARIES (perf_MTTKRP ${GENTEN_LINK_LIBS})

ADD_EXECUTABLE (
  perf_MTTKRP_Sweep
  ${Genten_SOURCE_DIR}/performance/Genten_MTTKRP_Sweep.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_DiscreteCDF.cpp
  ${Genten_SOURCE_DIR}/src/mathlib/Genten_FacTestSetGenerator.cpp
  )
TARGET_LINK_LIBRARIES (perf_MTTKRP_Sweep ${GENTEN_LINK_LIBS})

#------------------------------------------------------------
#---- Drivers
#------------------------------------------------------------

#---- Specifications for building each performance test.
#---- In each case specify the executable and its link library dependencies.

ADD_EXECUTABLE (
  genten
  ${Genten_SOURCE_DIR}/driver/Genten_Driver.cpp
  )
TARGET_LINK_LIBRARIES (genten ${GENTEN_LINK_LIBS})
INSTALL (
  TARGETS genten
  EXPORT genten
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

#------------------------------------------------------------
#---- Matlab
#------------------------------------------------------------

# Matlab Tensor Toolbox stores floats as doubles, so the interface will
# only compile if ttb_real == double
IF(HAVE_MATLAB AND (${GENTEN_FLOAT_TYPE} STREQUAL "double"))

  ADD_LIBRARY(
    gt_matlab
    ${Genten_SOURCE_DIR}/matlab/Genten_Matlab.hpp
    ${Genten_SOURCE_DIR}/matlab/Genten_Matlab.cpp)
  TARGET_INCLUDE_DIRECTORIES(gt_matlab PRIVATE ${Matlab_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(gt_matlab ${GENTEN_LINK_LIBS} ${Matlab_MEX_LIBRARY}
    ${Matlab_MX_LIBRARY})
  SET(GENTEN_MATLAB_LINK_LIBS gt_matlab ${GENTEN_LINK_LIBS})

  # Newer versions of CMake have an API version argument to MATLAB_ADD_MEX
  # which needs to be set to the right value since it defaults to an incorrect
  # value
  SET(GENTEN_MATLAB_API_VERSION "")
  IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.14.0")
    SET(GENTEN_MATLAB_API_VERSION "R2018a")
  ENDIF()

  MATLAB_ADD_MEX(
    NAME gt_cp_driver
    SRC ${Genten_SOURCE_DIR}/matlab/Genten_Matlab_Driver.cpp
    ${GENTEN_MATLAB_API_VERSION})
  MATLAB_ADD_MEX(
    NAME gt_initialize
    SRC ${Genten_SOURCE_DIR}/matlab/Genten_Matlab_Initialize.cpp
    ${GENTEN_MATLAB_API_VERSION})
  MATLAB_ADD_MEX(
    NAME gt_finalize
    SRC ${Genten_SOURCE_DIR}/matlab/Genten_Matlab_Finalize.cpp
    ${GENTEN_MATLAB_API_VERSION})
  MATLAB_ADD_MEX(
    NAME gt_compute_perm_driver
    SRC ${Genten_SOURCE_DIR}/matlab/Genten_Matlab_ComputePermutation.cpp
    ${GENTEN_MATLAB_API_VERSION})
  MATLAB_ADD_MEX(
    NAME gt_mttkrp_driver
    SRC ${Genten_SOURCE_DIR}/matlab/Genten_Matlab_MTTKRP.cpp
    ${GENTEN_MATLAB_API_VERSION})
  MATLAB_ADD_MEX(
    NAME gt_innerprod_driver
    SRC ${Genten_SOURCE_DIR}/matlab/Genten_Matlab_Innerprod.cpp
    ${GENTEN_MATLAB_API_VERSION})
  MATLAB_ADD_MEX(
    NAME gt_gcp_sample_driver
    SRC ${Genten_SOURCE_DIR}/matlab/Genten_Matlab_GCP_Sample.cpp
    ${GENTEN_MATLAB_API_VERSION})
  MATLAB_ADD_MEX(
    NAME gt_gcp_value_driver
    SRC ${Genten_SOURCE_DIR}/matlab/Genten_Matlab_GCP_Value.cpp
    ${GENTEN_MATLAB_API_VERSION})
  MATLAB_ADD_MEX(
    NAME gt_gcp_gradient_driver
    SRC ${Genten_SOURCE_DIR}/matlab/Genten_Matlab_GCP_Gradient.cpp
    ${GENTEN_MATLAB_API_VERSION})
  MATLAB_ADD_MEX(
    NAME gt_online_gcp
    SRC ${Genten_SOURCE_DIR}/matlab/Genten_Matlab_Online_GCP.cpp
    ${GENTEN_MATLAB_API_VERSION})
  TARGET_LINK_LIBRARIES (gt_cp_driver ${GENTEN_MATLAB_LINK_LIBS})
  TARGET_LINK_LIBRARIES (gt_initialize ${GENTEN_MATLAB_LINK_LIBS})
  TARGET_LINK_LIBRARIES (gt_finalize ${GENTEN_MATLAB_LINK_LIBS})
  TARGET_LINK_LIBRARIES (gt_compute_perm_driver ${GENTEN_MATLAB_LINK_LIBS})
  TARGET_LINK_LIBRARIES (gt_mttkrp_driver ${GENTEN_MATLAB_LINK_LIBS})
  TARGET_LINK_LIBRARIES (gt_innerprod_driver ${GENTEN_MATLAB_LINK_LIBS})
  TARGET_LINK_LIBRARIES (gt_gcp_sample_driver ${GENTEN_MATLAB_LINK_LIBS})
  TARGET_LINK_LIBRARIES (gt_gcp_value_driver ${GENTEN_MATLAB_LINK_LIBS})
  TARGET_LINK_LIBRARIES (gt_gcp_gradient_driver ${GENTEN_MATLAB_LINK_LIBS})
  TARGET_LINK_LIBRARIES (gt_online_gcp ${GENTEN_MATLAB_LINK_LIBS})
  INSTALL(
    TARGETS gt_matlab gt_cp_driver gt_initialize gt_finalize gt_mttkrp_driver
      gt_innerprod_driver gt_gcp_sample_driver gt_gcp_value_driver
      gt_gcp_gradient_driver gt_online_gcp
    DESTINATION ${CMAKE_INSTALL_DATADIR}/genten/matlab)
  INSTALL(
    TARGETS gt_compute_perm_driver
    DESTINATION ${CMAKE_INSTALL_DATADIR}/genten/matlab/@sptensor_gt/private)
  SET_PROPERTY(
    TARGET gt_matlab gt_cp_driver gt_initialize gt_finalize gt_mttkrp_driver
      gt_innerprod_driver gt_gcp_sample_driver gt_gcp_value_driver
      gt_gcp_gradient_driver gt_online_gcp
    PROPERTY LIBRARY_OUTPUT_DIRECTORY ${Genten_BINARY_DIR}/matlab)
  SET_PROPERTY(
    TARGET gt_compute_perm_driver
    PROPERTY LIBRARY_OUTPUT_DIRECTORY ${Genten_BINARY_DIR}/matlab/@sptensor_gt/private)

  # Remove -DDLL_EXPORT_SYMBOL=... since it causes problems with nvcc_wrapper
  # Instead we define this inside the source
  SET_PROPERTY(
    TARGET gt_matlab gt_cp_driver gt_initialize gt_finalize
      gt_compute_perm_driver gt_mttkrp_driver gt_innerprod_driver
      gt_gcp_sample_driver gt_gcp_value_driver
      gt_gcp_gradient_driver gt_online_gcp
    PROPERTY DEFINE_SYMBOL "")

  # Targets to install matlab source files
  ADD_CUSTOM_TARGET (copy_matlab_dir ALL)
  ADD_CUSTOM_COMMAND (
    TARGET copy_matlab_dir POST_BUILD
    COMMAND cmake -E copy_directory
    ${Genten_SOURCE_DIR}/matlab/matlab_src
    ${Genten_BINARY_DIR}/matlab
    )
  # Hint:  Use trailing slash in directory name to remove it from install tree
  INSTALL (
    DIRECTORY ${Genten_SOURCE_DIR}/matlab/matlab_src/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/genten/matlab)

ENDIF()

IF(HAVE_PYTHON AND (${GENTEN_FLOAT_TYPE} STREQUAL "double"))
  # Determine the build and install directory
  SET(pyGenten_BUILD_DIR
    ${Genten_BINARY_DIR}/python/pygenten
    )
  IF(PYGENTEN_PIP)
    SET(pyGenten_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
  ELSE()
    SET(pyGenten_INSTALL_DIR
      ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}/site-packages/pygenten
      )
  ENDIF()
  MESSAGE("-- pygenten installation path: ${pyGenten_INSTALL_DIR}")
  add_subdirectory( python )
ENDIF()

add_subdirectory(tools)

#------------------------------------------------------------
#---- Testing
#------------------------------------------------------------
enable_testing()

# Sets up a test command to possibly launch with MPI if PARALLEL is defined.
# For serial tests, still use MPI to launch if MPI is enabled, since the
# machine may not be able to run tests without it (e.g., when launching from a
# head node) but don't include MPI pre/post flags so there is no binding.
# usage:
#    genten_add_test(NAME <name>
#                    COMMAND <command>
#                    [ARGS <arg>...]
#                    [WORKING_DIRECTORY <dir>]
#                    [PARALLEL])
function(genten_add_test)
  set(options PARALLEL)
  set(oneValueArgs NAME COMMAND WORKING_DIRECTORY)
  set(multiValueArgs ARGS)
  cmake_parse_arguments(GENTEN_ADD_TEST "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  separate_arguments(GENTEN_ADD_TEST_ARGS) # Replaces spaces with semi-colons
  set(test_cmd "${GENTEN_ADD_TEST_COMMAND}")
  list(APPEND test_cmd ${GENTEN_ADD_TEST_ARGS})
  if (HAVE_MPI)
    separate_arguments(MPIEXEC_PREFLAGS)
    separate_arguments(MPIEXEC_POSTFLAGS)
    set(test_cmd ${MPIEXEC_EXECUTABLE})
    list(APPEND test_cmd ${MPIEXEC_NUMPROC_FLAG})
    if (GENTEN_ADD_TEST_PARALLEL)
      # Some of the unit tests fail if MPIEXEC_MAX_NUMPROCS is too large
      # due to corner cases when the number of processors is greater than the
      # number of nonzeros in the tensor.  Futhermore, MPIEXEC_MAX_NUMPROCS
      # defaults to the number of cores on the machine.  So restrict the 
      # number of processors these tests run on to something more reasonable.
      set(TEST_NUMPROCS ${MPIEXEC_MAX_NUMPROCS})
      if (TEST_NUMPROCS GREATER 8)
        set(TEST_NUMPROCS 8)
      endif()
      list(APPEND test_cmd ${TEST_NUMPROCS})
    else()
      list(APPEND test_cmd 1)
    endif()
    list(APPEND test_cmd ${MPIEXEC_PREFLAGS})
    list(APPEND test_cmd ${GENTEN_ADD_TEST_COMMAND})
    list(APPEND test_cmd ${MPIEXEC_POSTFLAGS})
    list(APPEND test_cmd ${GENTEN_ADD_TEST_ARGS})
  endif()
  add_test(NAME ${GENTEN_ADD_TEST_NAME} COMMAND ${test_cmd} WORKING_DIRECTORY ${GENTEN_ADD_TEST_WORKING_DIRECTORY})
endfunction(genten_add_test)

# Kokkos devices to run tests on.
set(kokkos_devices "")
if (HAVE_SERIAL)
  list(APPEND kokkos_devices "serial")
endif()
if (HAVE_OPENMP)
  list(APPEND kokkos_devices "openmp")
endif()
if (HAVE_THREADS)
  list(APPEND kokkos_devices "threads")
endif()
if (HAVE_CUDA)
  list(APPEND kokkos_devices "cuda")
endif()
if (HAVE_HIP)
  list(APPEND kokkos_devices "hip")
endif()
if (HAVE_SYCL)
  list(APPEND kokkos_devices "sycl")
endif()

# distributed methods to run
set(dist_methods "")
list(APPEND dist_methods "all-reduce" "two-sided")
if (HAVE_TPETRA)
  list(APPEND dist_methods "tpetra")
endif()

add_subdirectory(tpls/googletest)
add_subdirectory(test)

function(add_genten_perf_tests name cmd args spaces)
  foreach(space ${spaces})
    string(CONCAT test_name ${name} "_" ${space})
    string(CONCAT test_args ${args} " --exec-space " ${space})
    genten_add_test(NAME ${test_name} COMMAND ${cmd} ARGS ${test_args})
  endforeach()
endfunction(add_genten_perf_tests)

function(add_genten_mpi_tests name cmd args spaces dists)
  foreach(space ${spaces})
    foreach(dist ${dists})
      string(CONCAT test_name ${name} "_" ${space} "_" ${dist})
      string(CONCAT test_args ${args} " --exec-space " ${space} " --dist-method " ${dist})
      # The tests can run VERY slowly with multiple threads and MPI ranks,
      # so run with a single thread (for all spaces because OpenMP or Threads
      # maybe the default host execution space).
      if (HAVE_MPI)
        string(CONCAT test_args ${test_args} " --kokkos-num-threads=1")
      endif()
      genten_add_test(NAME ${test_name} COMMAND ${cmd} ARGS ${test_args} PARALLEL)
    endforeach()
  endforeach()
endfunction(add_genten_mpi_tests)

add_genten_perf_tests(Genten_MTTKRP_random_atomic "${Genten_BINARY_DIR}/bin/perf_MTTKRP" "--nc 16 --dims [300,400,500] --nnz 1000 --mttkrp-method atomic" "${kokkos_devices}")
add_genten_perf_tests(Genten_MTTKRP_random_dupl "${Genten_BINARY_DIR}/bin/perf_MTTKRP" "--nc 16 --dims [300,400,500] --nnz 1000 --mttkrp-method duplicated" "${kokkos_devices}")
add_genten_perf_tests(Genten_MTTKRP_random_perm "${Genten_BINARY_DIR}/bin/perf_MTTKRP" "--nc 16 --dims [300,400,500] --nnz 1000 --mttkrp-method perm" "${kokkos_devices}")
add_genten_perf_tests(Genten_MTTKRP_aminoacid_atomic "${Genten_BINARY_DIR}/bin/perf_MTTKRP" "--nc 16 --input ${Genten_BINARY_DIR}/data/aminoacid_data.txt --mttkrp-method atomic" "${kokkos_devices}")
add_genten_perf_tests(Genten_MTTKRP_aminoacid_dupl "${Genten_BINARY_DIR}/bin/perf_MTTKRP" "--nc 16 --input ${Genten_BINARY_DIR}/data/aminoacid_data.txt --mttkrp-method duplicated" "${kokkos_devices}")
add_genten_perf_tests(Genten_MTTKRP_aminoacid_perm "${Genten_BINARY_DIR}/bin/perf_MTTKRP" "--nc 16 --input ${Genten_BINARY_DIR}/data/aminoacid_data.txt --mttkrp-method perm" "${kokkos_devices}")
add_genten_perf_tests(Genten_MTTKRP_random_dense "${Genten_BINARY_DIR}/bin/perf_MTTKRP" "--nc 16 --dims [30,40,50] --dense" "${kokkos_devices}" "${dist_methods}")

add_genten_mpi_tests(Genten_driver_aminoacid_cpals "${Genten_BINARY_DIR}/bin/genten" "--json ${Genten_BINARY_DIR}/data/aminoacid-cpals.json" "${kokkos_devices}" "${dist_methods}")
add_genten_mpi_tests(Genten_driver_aminoacid_cpals_dense "${Genten_BINARY_DIR}/bin/genten" "--json ${Genten_BINARY_DIR}/data/aminoacid-cpals-dense.json" "${kokkos_devices}" "${dist_methods}")
if (HAVE_LBFGSB)
  # L-BFGS-B does not work with MPI (implements its own vector operations)
  add_genten_perf_tests(Genten_driver_aminoacid_cpopt_lbfgsb "${Genten_BINARY_DIR}/bin/genten" "--json ${Genten_BINARY_DIR}/data/aminoacid-cpopt-lbfgsb.json" "${kokkos_devices}")
endif()
if (HAVE_ROL)
  add_genten_mpi_tests(Genten_driver_aminoacid_cpopt_rol "${Genten_BINARY_DIR}/bin/genten" "--json ${Genten_BINARY_DIR}/data/aminoacid-cpopt-rol.json" "${kokkos_devices}" "${dist_methods}")
  add_genten_mpi_tests(Genten_driver_aminoacid_cpopt_rol_hess "${Genten_BINARY_DIR}/bin/genten" "--json ${Genten_BINARY_DIR}/data/aminoacid-cpopt-rol-hess.json" "${kokkos_devices}" "${dist_methods}")
endif()
if (HAVE_GCP)
  add_genten_mpi_tests(Genten_driver_aminoacid_gcpsgd "${Genten_BINARY_DIR}/bin/genten" "--json ${Genten_BINARY_DIR}/data/aminoacid-gcpsgd.json" "${kokkos_devices}" "${dist_methods}")
  if (HAVE_MPI)
    # SGD-Dist cannot use Tpetra
    add_genten_mpi_tests(Genten_driver_aminoacid_gcpfed "${Genten_BINARY_DIR}/bin/genten" "--json ${Genten_BINARY_DIR}/data/aminoacid-gcpfed.json" "${kokkos_devices}" "all-reduce")
  endif()
  add_genten_mpi_tests(Genten_driver_aminoacid_gcpsgd_dense "${Genten_BINARY_DIR}/bin/genten" "--json ${Genten_BINARY_DIR}/data/aminoacid-gcpsgd-dense.json" "${kokkos_devices}" "${dist_methods}")
  if (HAVE_ROL)
    add_genten_mpi_tests(Genten_driver_aminoacid_gcpopt_rol "${Genten_BINARY_DIR}/bin/genten" "--json ${Genten_BINARY_DIR}/data/aminoacid-gcpopt-rol.json" "${kokkos_devices}" "${dist_methods}")
  endif()
  if (HAVE_LBFGSB)
    # L-BFGS-B does not work with MPI (implements its own vector operations)
    add_genten_perf_tests(Genten_driver_aminoacid_gcpopt_lbfgsb "${Genten_BINARY_DIR}/bin/genten" "--json ${Genten_BINARY_DIR}/data/aminoacid-gcpopt-lbfgsb.json" "${kokkos_devices}")
  endif()
endif()

# python tests
if (HAVE_PYTHON)
  genten_add_test(NAME Genten_Test_Python COMMAND "pytest" ARGS "-v;--disable-warnings;${CMAKE_SOURCE_DIR}/python/test")
endif()

#------------------------------------------------------------
#---- Config
#------------------------------------------------------------
add_subdirectory(config)
install(EXPORT genten DESTINATION lib64/cmake/${PROJECT_NAME})
