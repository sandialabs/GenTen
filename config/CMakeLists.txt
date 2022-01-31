###############################################################################
# export everything for a cmake build to be able to import with find_package
###############################################################################

if (NOT DEFINED GENTEN_INSTALL_INCLUDE_DIR)
  set(GENTEN_INSTALL_INCLUDE_DIR "include")
endif()

if (NOT DEFINED GENTEN_INSTALL_CONFIG_DIR)
  set(GENTEN_INSTALL_CONFIG_DIR "lib64")
endif()

if (NOT DEFINED GENTEN_INSTALL_LIB_DIR)
  set(GENTEN_INSTALL_LIB_DIR "lib64")
endif()

if (NOT DEFINED GENTEN_INSTALL_BIN_DIR)
  set(GENTEN_INSTALL_BIN_DIR "bin")
endif()

if (NOT DEFINED GENTEN_INSTALL_CMAKE_MODULE_DIR)
  if(WIN32)
    set(GENTEN_INSTALL_CMAKE_MODULE_DIR "cmake")
  else()
    set(GENTEN_INSTALL_CMAKE_MODULE_DIR "${GENTEN_INSTALL_CONFIG_DIR}/cmake/Genten")
  endif()
endif()

include(CMakePackageConfigHelpers)
# write version heler
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/GentenConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# write cmake package config file
configure_package_config_file(
  GentenConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/GentenConfig.cmake
  INSTALL_DESTINATION ${GENTEN_INSTALL_CMAKE_MODULE_DIR}
  PATH_VARS
    GENTEN_INSTALL_INCLUDE_DIR
    GENTEN_INSTALL_LIB_DIR
    GENTEN_INSTALL_BIN_DIR
    GENTEN_INSTALL_CMAKE_MODULE_DIR
    )

# install everything needed
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/GentenConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/GentenConfigVersion.cmake
    genten_setup_deps.cmake
    genten_setup_targets.cmake
    DESTINATION ${GENTEN_INSTALL_CMAKE_MODULE_DIR})

