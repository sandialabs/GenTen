ADD_EXECUTABLE(sptn_to_binary 
  ${Genten_SOURCE_DIR}/tools/sparse_tensor_to_binary.cpp)
INSTALL(TARGETS sptn_to_binary)

ADD_EXECUTABLE(read_binary_tensor 
  ${Genten_SOURCE_DIR}/tools/read_binary_tensor.cpp)
INSTALL(TARGETS read_binary_tensor)

if(TARGET MPI::MPI_C)
ADD_EXECUTABLE(mpi_read_binary_tensor 
  ${Genten_SOURCE_DIR}/tools/mpi_read_binary_tensor.cpp)
TARGET_LINK_LIBRARIES(mpi_read_binary_tensor GentenMpiIoLib)
INSTALL(TARGETS mpi_read_binary_tensor)
endif(TARGET MPI::MPI_C)

find_package(Eigen3 3.3)

if(TARGET Eigen3::Eigen)
ADD_EXECUTABLE(k-comp 
  ${Genten_SOURCE_DIR}/tools/KTensor_compare.cpp)
TARGET_LINK_LIBRARIES (k-comp ${GENTEN_LINK_LIBS} Boost::boost Eigen3::Eigen)
INSTALL(TARGETS k-comp)
endif(TARGET Eigen3::Eigen)
