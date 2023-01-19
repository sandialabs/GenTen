ADD_EXECUTABLE(convert_tensor
  ${Genten_SOURCE_DIR}/tools/convert_tensor.cpp)
TARGET_LINK_LIBRARIES (convert_tensor ${GENTEN_LINK_LIBS})
INSTALL(TARGETS convert_tensor)

ADD_EXECUTABLE(read_binary_header
  ${Genten_SOURCE_DIR}/tools/read_binary_header.cpp)
TARGET_LINK_LIBRARIES (read_binary_header ${GENTEN_LINK_LIBS})
INSTALL(TARGETS read_binary_header)
