ADD_SUBDIRECTORY(CG)
ADD_SUBDIRECTORY(cusolver)
ADD_SUBDIRECTORY(GMRES)
ADD_SUBDIRECTORY(SPMV)

FILE(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/scripts/binary_dir.txt
"${CMAKE_CURRENT_BINARY_DIR}"
)