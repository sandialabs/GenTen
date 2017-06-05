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


#---------------------------------------------------------------------------
# This module finds one or more libraries supplying LAPACK/BLAS functions.
# To link with a known library, set its location with command line options.
# If not set on the command line, then CMake searches in default locations.
# If no LAPACK/BLAS is found, then default to a TTB implementation; however,
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
IF(NOT LAPACK_LIBS)
  MESSAGE (STATUS "TTB: Looking for LAPACK/BLAS libraries")
  
  FIND_LIBRARY(LAPACK lapack)
  FIND_LIBRARY(BLAS blas)
  MARK_AS_ADVANCED(LAPACK BLAS)
  IF(LAPACK AND BLAS)
    SET(LAPACK_LIBS "${LAPACK};${BLAS}" CACHE STRING "Location of LAPACK/BLAS libraries")
  ENDIF(LAPACK AND BLAS)
ELSE (NOT LAPACK_LIBS)
  MESSAGE (STATUS "TTB: Attempting to use user-defined LAPACK/BLAS libraries: ${LAPACK_LIBS}")
ENDIF(NOT LAPACK_LIBS)

# The following did not work on MacOSX, because the BLAS library was missing
#FIND_LIBRARY (LAPACK_LIBS NAMES lapack
#              DOC "Location of LAPACK library")

# Set the boolean LAPACK_FOUND.
IF (NOT LAPACK_LIBS)
    SET (LAPACK_FOUND FALSE)
    SET (LAPACK_LIBS "")
ELSE (NOT LAPACK_LIBS)
    SET (LAPACK_FOUND TRUE)

    # Check if user-specified libraries exist.
    FOREACH (loopVar  ${LAPACK_LIBS})
        IF (NOT EXISTS ${loopVar})
            MESSAGE (FATAL_ERROR "TTB: Cannot find file " ${loopVar})
        ENDIF (NOT EXISTS ${loopVar})
    ENDFOREACH (loopVar)
ENDIF (NOT LAPACK_LIBS)

# Report which libraries will be used.
IF (LAPACK_FOUND)
    MESSAGE (STATUS "TTB: Using LAPACK/BLAS libraries:")
    FOREACH (loopVar ${LAPACK_LIBS})
  	MESSAGE (STATUS "  " ${loopVar})
    ENDFOREACH (loopVar)
    FOREACH (loopVar ${LAPACK_ADD_LIBS})
  	MESSAGE (STATUS "  linking with " ${loopVar})
    ENDFOREACH (loopVar)
ELSE (LAPACK_FOUND)
    MESSAGE (STATUS "TTB: LAPACK not found, will compile TTB LAPACK code.")
ENDIF (LAPACK_FOUND)
