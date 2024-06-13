set(PybindGenten_SRCS
  Genten_Pybind11.cpp
  Genten_Pybind11_classes.cpp
)

pybind11_add_module(_pygenten ${PybindGenten_SRCS})
target_include_directories(_pygenten PUBLIC ${Genten_SOURCE_DIR}/src)
target_compile_features(_pygenten PUBLIC cxx_std_17)
target_link_libraries(_pygenten PRIVATE gentenlib)
set_target_properties(_pygenten PROPERTIES SUFFIX ".so")
set_target_properties(_pygenten PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${pyGenten_BUILD_DIR})

install(TARGETS _pygenten
    LIBRARY DESTINATION ${pyGenten_INSTALL_DIR}
    ARCHIVE DESTINATION ${pyGenten_INSTALL_DIR})

IF(HAVE_PHYS_UTILS)
set(PybindGenten_SRCS
  ${PybindGenten_SRCS}
  Genten_Pybind11_phys_utils.cpp
)
pybind11_add_module(_phys_utils ${PybindGenten_SRCS})
target_include_directories(_phys_utils PUBLIC ${Genten_SOURCE_DIR}/src)
target_compile_features(_phys_utils PUBLIC cxx_std_17)
target_link_libraries(_phys_utils PRIVATE gentenlib Trilinos::all_libs)
set_target_properties(_phys_utils PROPERTIES SUFFIX ".so")
set_target_properties(_phys_utils PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${pyGenten_BUILD_DIR})

install(TARGETS _phys_utils
    LIBRARY DESTINATION ${pyGenten_INSTALL_DIR}
    ARCHIVE DESTINATION ${pyGenten_INSTALL_DIR})
ENDIF()
