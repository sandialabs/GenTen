#!/bin/bash -xe
rm -f CMakeCache.txt
rm -rf CMakeFiles

cmake \
    -D CMAKE_VERBOSE_MAKEFILE=OFF \
    -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -D CMAKE_CXX_COMPILER=hipcc \
    -D CMAKE_C_COMPILER=hipcc \
    -D CMAKE_BUILD_TYPE=Debug \
    -D Kokkos_ENABLE_HIP=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D Kokkos_ARCH_VEGA906=ON \
    -D ROCM_SEARCH_PATH="/opt/rocm" \
    ../genten
