#--https://stackoverflow.com/questions/49857596/cmake-simple-config-file-example/49858236
#--https://stackoverflow.com/questions/20746936/what-use-is-find-package-if-you-need-to-specify-cmake-module-path-anyway
ADD_LIBRARY (
  gt_higher_moments SHARED
  ${Genten_SOURCE_DIR}/joint_moments/Genten_HigherMoments.cpp
  )
TARGET_LINK_LIBRARIES(gt_higher_moments PUBLIC ${GENTEN_LINK_LIBS})


INSTALL(
  TARGETS genten_mathlibs_c gentenlib gt_higher_moments
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  EXPORT gt_higher_momentsConfig
  )

INSTALL(
  EXPORT gt_higher_momentsConfig
  DESTINATION cmake
  )

#HK3
#HK3EXPORT(
#HK3  EXPORT gt_higher_momentsConfig 
#HK3  NAMESPACE gt_higher_moments::
#HK3  FILE gt_higher_momentsConfig.cmake
#HK3  )
#HK3
#HK3EXPORT( PACKAGE gt_higher_moments)

#HKINSTALL (
#HK  TARGETS gt_higher_moments
#HK  DESTINATION ${CMAKE_INSTALL_LIBDIR}
#HK  )
#HK
#HKinclude(CMakePackageConfigHelpers)
#HKconfigure_package_config_file(gt_higher_momentsConfig.cmake.in
#HK                              ${CMAKE_CURRENT_BINARY_DIR}/gt_higher_momentsConfig.cmake
#HK                              INSTALL_DESTINATION cmake)
#HK
#HKINSTALL(
#HK  FILES
#HK  ${CMAKE_CURRENT_BINARY_DIR}/gt_higher_momentsConfig.cmake
#HK  DESTINATION cmake
#HK  )


