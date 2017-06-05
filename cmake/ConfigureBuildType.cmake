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


#---- Define a boolean user option "debug" that can be set from the
#---- command line ("-Ddebug=true" or "=on" or "=yes"),
#---- or from the CMake cache.

OPTION (debug "TRUE means compile as debug objects" FALSE)

IF (debug)
    #---- TGK: Windows seems to ignore this command.
    SET (CMAKE_BUILD_TYPE DEBUG)
    MESSAGE (STATUS "TTB: Makefiles will compile for debug (may run slower).")
ELSE (debug)
    SET (CMAKE_BUILD_TYPE RELEASE)
    MESSAGE (STATUS "TTB: Makefiles will compile with production flags.")
ENDIF (debug)


#-------------------------------------------------------------------------


#---- Compile with strict ANSI checking on g++.
#---- Microsoft compiler does not understand "pedantic".
IF (CMAKE_COMPILER_IS_GNUCXX)
    #SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ansi")
    #-- Uncomment next two lines to build executables that use gprof:
    #ADD_DEFINITIONS (-pg)
    #SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
ENDIF (CMAKE_COMPILER_IS_GNUCXX)


#-------------------------------------------------------------------------


#---- Compile shared libraries on MACOSX
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  SET(CMAKE_MACOSX_RPATH 1)
ENDIF()


#---- Compile with clang, needs to not resolve symbols until runtime
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -undefined dynamic_lookup")
endif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")


#-------------------------------------------------------------------------


#---- Uncomment the first line to make the native compiler calls visible.
#---- For Windows MSVC also uncomment the the next 2 lines; however, even
#---- then not all make operations are visible.
#SET (CMAKE_VERBOSE_MAKEFILE ON)
#SET (CMAKE_START_TEMP_FILE "")
#SET (CMAKE_END_TEMP_FILE "")

IF (CMAKE_VERBOSE_MAKEFILE)
    MESSAGE (STATUS "TTB: Makefiles requested to display compile commands.")
ENDIF (CMAKE_VERBOSE_MAKEFILE)


#-------------------------------------------------------------------------


#---- Real time timers are enabled if system libraries can be found.
#---- The TTB_SystemTimer class is informed thru HAVE_REALTIME_CLOCK.
SET (HAVE_REALTIME_CLOCK TRUE)
IF (WIN32)
    FIND_LIBRARY (WINMM_LIBRARY winmm ${WIN32_LIBRARY_SEARCHPATHS})
    IF (NOT WINMM_LIBRARY)
        MESSAGE (STATUS "TTB: Could not find WinMM.lib, timers disabled.")
        SET (HAVE_REALTIME_CLOCK FALSE)
    ELSE (NOT WINMM_LIBRARY)
#TBD...have to test this; probably need it linked with executables
        SET (OPSYS_LIBRARIES ${OPSYS_LIBRARIES} ${WINMM_LIBRARY})
    ENDIF (NOT WINMM_LIBRARY)
ENDIF (WIN32)
