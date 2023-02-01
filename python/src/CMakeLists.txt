set(PybindGenten_SRCS
  Genten_Pybind11.cpp
  Genten_Pybind11_classes.cpp
)

pybind11_add_module(pyGenten ${PybindGenten_SRCS})
target_include_directories(pyGenten PUBLIC ${Genten_SOURCE_DIR}/src)
target_compile_features(pyGenten PUBLIC cxx_std_17)
target_link_libraries(pyGenten PRIVATE gentenlib)
set_target_properties(pyGenten PROPERTIES SUFFIX ".so")

install(TARGETS pyGenten
    LIBRARY DESTINATION ${pyGenten_INSTALL_DIR}
    ARCHIVE DESTINATION ${pyGenten_INSTALL_DIR})