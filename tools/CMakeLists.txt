ADD_EXECUTABLE(sptn_to_binary 
  ${Genten_SOURCE_DIR}/tools/sparse_tensor_to_binary.cpp)
INSTALL(TARGETS sptn_to_binary)

ADD_EXECUTABLE(read_binary_tensor 
  ${Genten_SOURCE_DIR}/tools/read_binary_tensor.cpp)
INSTALL(TARGETS read_binary_tensor)
