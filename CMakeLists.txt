# ************************************************************************
#     Genten Tensor Toolbox
#     Software package for tensor math by Sandia National Laboratories
#
# Sandia National Laboratories is a multimission laboratory managed
# and operated by National Technology and Engineering Solutions of Sandia,
# LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
# U.S. Department of Energy’s National Nuclear Security Administration under
# contract DE-NA0003525.
#
# Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
# ************************************************************************


#---- Specify the CMake version required.
CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

#---- Declaring the project name will define ${Genten_SOURCE_DIR} to be
#---- the path to this source file, and ${Genten_BINARY_DIR} to be the path
#---- to the directory where cmake has been invoked.
PROJECT (Genten)

SET(CMAKE_BUILD_TYPE "DEBUG")


#------------------------------------------------------------
#---- Package OPTIONS
#------------------------------------------------------------
OPTION(BUILD_SHARED_LIBS "Build shared libraries." OFF)
OPTION(KOKKOS_PATH "Path to Kokkos installation")
OPTION(ENABLE_CUBLAS "Flag to enable cuBLAS for Kokkos with Cuda" OFF)
OPTION(ENABLE_CUSOLVER "Flag to enable cuSolver for Kokkos with Cuda" OFF)
OPTION(ENABLE_BOOST "Enable Boost for reading compressed tensor files" OFF)

SET(BOOST_LIBS "")
IF (ENABLE_BOOST)
  OPTION(BOOST_PATH "Path to BOOST installation")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${BOOST_PATH}/include")
  SET(BOOST_LIBS "-L${BOOST_PATH}/lib -lboost_iostreams -lz")
  SET(HAVE_BOOST ${ENABLE_BOOST})
ENDIF()

SET(KOKKOS_LIBS "")
SET(KOKKOS_LDFLAGS "")
SET(KOKKOS_MAKEFILE "${KOKKOS_PATH}/Makefile.kokkos")
IF (NOT EXISTS ${KOKKOS_MAKEFILE})
  MESSAGE(FATAL_ERROR "Cannot find Makefile.kokkos in ${KOKKOS_PATH}.  Please make KOKKOS_PATH is set correctly.")
ENDIF()

FILE(STRINGS ${KOKKOS_MAKEFILE} tmp REGEX KOKKOS_CXXFLAGS)
STRING(REGEX REPLACE " *KOKKOS_CXXFLAGS *= *(.*)" "\1" KOKKOS_CXXFLAGS ${tmp})
FILE(STRINGS ${KOKKOS_MAKEFILE} tmp REGEX KOKKOS_CPPFLAGS)
STRING(REGEX REPLACE " *KOKKOS_CPPFLAGS *= *(.*)" "\1" KOKKOS_CPPFLAGS ${tmp})
FILE(STRINGS ${KOKKOS_MAKEFILE} tmp REGEX KOKKOS_LDFLAGS)
STRING(REGEX REPLACE " *KOKKOS_LDFLAGS *= *(.*)" "\1" KOKKOS_LDFLAGS ${tmp})
FILE(STRINGS ${KOKKOS_MAKEFILE} tmp REGEX KOKKOS_LIBS)
STRING(REGEX REPLACE " *KOKKOS_LIBS *= *(.*)" "\1" KOKKOS_LIBS ${tmp})

IF (ENABLE_CUBLAS)
  SET(KOKKOS_LIBS "${KOKKOS_LIBS} -lcublas")
  SET(HAVE_CUBLAS ${ENABLE_CUBLAS})
ENDIF()
IF (ENABLE_CUSOLVER)
  SET(KOKKOS_LIBS "${KOKKOS_LIBS} -lcusolver")
  SET(HAVE_CUSOLVER ${ENABLE_CUSOLVER})
ENDIF()
MESSAGE("-- Genten:  Using Kokkos:  ${KOKKOS_PATH}")
MESSAGE("--   KOKKOS_CXXFLAGS = ${KOKKOS_CXXFLAGS}")
MESSAGE("--   KOKKOS_CPPFLAGS = ${KOKKOS_CPPFLAGS}")
MESSAGE("--   KOKKOS_LDFLAGS  = ${KOKKOS_LDFLAGS}")
MESSAGE("--   KOKKOS_LIBS     = ${KOKKOS_LIBS}")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${KOKKOS_CXXFLAGS} ${KOKKOS_CPPFLAGS}")
SET(HAVE_KOKKOS ON)
MESSAGE("--   CMAKE_CXXFLAGS  = ${CMAKE_CXX_FLAGS}")

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
  ${Genten_SOURCE_DIR}/src/
  ${Genten_SOURCE_DIR}/src/mathlib/
  ${Genten_BINARY_DIR}/CMakeInclude/
  ${Genten_SOURCE_DIR}/driver/
  )
FILE (GLOB genten_headers
  ${Genten_SOURCE_DIR}/src/*.h
  ${Genten_SOURCE_DIR}/src/mathlib/*.h
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
#------------------------------------------------------------
INCLUDE (cmake/ConfigureLapack.cmake)

IF (LAPACK_FOUND)
  ADD_DEFINITIONS (-DLAPACK_FOUND)
  ADD_LIBRARY (genten_mathlibs
    ${Genten_SOURCE_DIR}/src/mathlib/Genten_Default_VML.c
    ${Genten_SOURCE_DIR}/src/mathlib/Genten_DiscreteCDF.cpp
    ${Genten_SOURCE_DIR}/src/mathlib/Genten_FacTestSetGenerator.cpp
    ${Genten_SOURCE_DIR}/src/mathlib/Genten_RandomMT.cpp
    ${Genten_SOURCE_DIR}/src/mathlib/Genten_SystemTimer.cpp
    )
ELSE (LAPACK_FOUND)
  #---- Tell CMake to build with default BLAS implementation.
  ADD_LIBRARY (genten_mathlibs
    ${Genten_SOURCE_DIR}/src/mathlib/Genten_Default_Blas.c
    ${Genten_SOURCE_DIR}/src/mathlib/Genten_Default_VML.c
    ${Genten_SOURCE_DIR}/src/mathlib/Genten_DiscreteCDF.cpp
    ${Genten_SOURCE_DIR}/src/mathlib/Genten_FacTestSetGenerator.cpp
    ${Genten_SOURCE_DIR}/src/mathlib/Genten_RandomMT.cpp
    ${Genten_SOURCE_DIR}/src/mathlib/Genten_SystemTimer.cpp
    )
ENDIF (LAPACK_FOUND)


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
TARGET_LINK_LIBRARIES(genten genten_mathlibs ${LAPACK_LIBS} ${BOOST_LIBS} ${KOKKOS_LDFLAGS} ${KOKKOS_LIBS})
INSTALL (
  TARGETS genten genten_mathlibs
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
TARGET_LINK_LIBRARIES (
  unit_tests
  genten
  genten_mathlibs
  ${OPSYS_LIBRARIES}
  ${BOOST_LIBS} ${LAPACK_LIBS} ${LAPACK_ADD_LIBS}
  ${KOKKOS_LDFLAGS} ${KOKKOS_LIBS}
  )
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
TARGET_LINK_LIBRARIES (
  perf_CpAlsAminoAcid
  genten
  genten_mathlibs
  ${OPSYS_LIBRARIES}
  ${BOOST_LIBS} ${LAPACK_LIBS} ${LAPACK_ADD_LIBS}
  ${KOKKOS_LDFLAGS} ${KOKKOS_LIBS}
  )
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
TARGET_LINK_LIBRARIES (
  perf_CpAlsRandomKtensor
  genten
  genten_mathlibs
  ${OPSYS_LIBRARIES}
  ${BOOST_LIBS} ${LAPACK_LIBS} ${LAPACK_ADD_LIBS}
  ${KOKKOS_LDFLAGS} ${KOKKOS_LIBS}
  )
INSTALL (
  TARGETS perf_CpAlsRandomKtensor
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
TARGET_LINK_LIBRARIES (
  cpals
  genten
  genten_mathlibs
  ${OPSYS_LIBRARIES}
  ${BOOST_LIBS} ${LAPACK_LIBS} ${LAPACK_ADD_LIBS}
  ${KOKKOS_LDFLAGS} ${KOKKOS_LIBS}
  )
INSTALL (
  TARGETS cpals
  DESTINATION drivers
  )
