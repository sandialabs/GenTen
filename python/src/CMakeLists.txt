set(PybindGenten_SRCS
  Genten_Pybind11.cpp
  Genten_Pybind11_classes.cpp
)

pybind11_add_module(_pygenten ${PybindGenten_SRCS})
target_include_directories(_pygenten PUBLIC ${Genten_SOURCE_DIR}/src)
target_compile_features(_pygenten PUBLIC cxx_std_17)
target_link_libraries(_pygenten PRIVATE gentenlib)
set_target_properties(_pygenten PROPERTIES SUFFIX ".so")

install(TARGETS _pygenten
    LIBRARY DESTINATION ${pyGenten_INSTALL_DIR}
    ARCHIVE DESTINATION ${pyGenten_INSTALL_DIR})