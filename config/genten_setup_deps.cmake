if(GENTEN_KOKKOS_DIR)
  if(NOT EXISTS ${GENTEN_KOKKOS_DIR}/lib64/cmake/Kokkos/KokkosConfig.cmake)
    MESSAGE(FATAL_ERROR "Could not find kokkos CMake include file (${GENTEN_KOKKOS_DIR}/lib64/cmake/Kokkos/KokkosConfig.cmake)")
    endif()

    ###############################################################################
    # Import Kokkos CMake targets
    ###############################################################################
    find_package(Kokkos REQUIRED
                 NO_DEFAULT_PATH
                 PATHS ${GENTEN_KOKKOS_DIR}/lib64/cmake/Kokkos)
endif()
