#--https://stackoverflow.com/questions/49857596/cmake-simple-config-file-example/49858236
#--https://stackoverflow.com/questions/20746936/what-use-is-find-package-if-you-need-to-specify-cmake-module-path-anyway
set(joint_moment_headers Genten_HigherMoments.hpp)

ADD_LIBRARY (
  gt_higher_moments
  ${Genten_SOURCE_DIR}/joint_moments/Genten_HigherMoments.cpp
  ${Genten_SOURCE_DIR}/joint_moments/Genten_ComputePrincipalKurtosisVectors.cpp
  )
TARGET_LINK_LIBRARIES(gt_higher_moments PUBLIC ${GENTEN_LINK_LIBS})


INSTALL (
  FILES ${joint_moment_headers}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/genten
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  )

INSTALL(
  TARGETS gt_higher_moments
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  EXPORT genten
  )

# the configure files will be installed by the config/cmake files
#
#INSTALL(
#  EXPORT gt_higher_momentsConfig
#  DESTINATION cmake
#  )
#
