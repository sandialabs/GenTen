# Currently this is a GPU-only build

rm -f CMakeCache.txt;
rm -rf CMakeFiles

cmake \
 -D CMAKE_INSTALL_PREFIX=${GENTEN_INSTALL_PATH} \
 -D CMAKE_CXX_COMPILER=mpicxx \
 -D CMAKE_C_COMPILER=mpicc \
 -D CMAKE_CXX_FLAGS="-g -lineinfo -Wno-psabi" \
 -D CMAKE_C_FLAGS="-g -Wno-psabi" \
 -D BUILD_SHARED_LIBS=ON \
 -D Kokkos_ENABLE_CUDA=ON \
 -D Kokkos_ENABLE_CUDA_UVM=ON \
 -D Kokkos_ENABLE_SERIAL=ON \
 -D Kokkos_ARCH_VOLTA70=ON \
 -D Kokkos_ARCH_POWER9=ON \
 -D debug=OFF \
 -D ENABLE_MPI=ON \
  ${GENTEN_SRC_PATH}
