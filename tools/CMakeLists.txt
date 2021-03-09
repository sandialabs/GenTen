ADD_EXECUTABLE(sptn_to_binary 
  ${Genten_SOURCE_DIR}/tools/sparse_tensor_to_binary.cpp)
INSTALL(TARGETS sptn_to_binary)

ADD_EXECUTABLE(kokkos_test 
  ${Genten_SOURCE_DIR}/tools/kokkos_test.cpp)

TARGET_LINK_LIBRARIES (kokkos_test ${GENTEN_LINK_LIBS} DistContextLib
  MPI::MPI_C Boost::boost Boost::serialization)

INSTALL(TARGETS kokkos_test)
