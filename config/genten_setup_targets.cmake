set(GENTEN_INCLUDE_DIRS "${GENTEN_INSTALL_PREFIX}/include/genten")

add_library(genten::genten INTERFACE IMPORTED)

set_property(TARGET genten::genten
             APPEND PROPERTY
             INTERFACE_INCLUDE_DIRECTORIES "${GENTEN_INSTALL_PREFIX}/include/")

set_property(TARGET genten::genten
             APPEND PROPERTY
             INTERFACE_INCLUDE_DIRECTORIES "${GENTEN_INSTALL_PREFIX}/include/genten/")

set_property(TARGET genten::genten
             PROPERTY INTERFACE_LINK_LIBRARIES
             genten_mathlibs_c gentenlib gt_higher_moments)
