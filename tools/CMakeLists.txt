ADD_EXECUTABLE(sptn_to_binary 
  ${Genten_SOURCE_DIR}/tools/sparse_tensor_to_binary.cpp)
INSTALL(TARGETS sptn_to_binary)

find_package(Eigen3 3.3)

if(TARGET Eigen3::Eigen)
ADD_EXECUTABLE(k-comp 
  ${Genten_SOURCE_DIR}/tools/KTensor_compare.cpp)
TARGET_LINK_LIBRARIES (k-comp ${GENTEN_LINK_LIBS} Boost::boost Eigen3::Eigen)
INSTALL(TARGETS k-comp)
endif(TARGET Eigen3::Eigen)
