#!/bin/bash -xe
rm -f CMakeCache.txt
rm -rf CMakeFiles

KOKKOS=${PWD}/../genten/tpls/kokkos

cmake \
 -D CMAKE_CXX_COMPILER=${KOKKOS}/bin/nvcc_wrapper \
 -D CMAKE_C_COMPILER=gcc \
 -D CMAKE_CXX_FLAGS="-g -lineinfo -Wno-deprecated-gpu-targets" \
 -D CMAKE_C_FLAGS="-g" \
 -D BUILD_SHARED_LIBS=ON \
 -D Kokkos_ENABLE_CUDA=ON \
 -D Kokkos_ENABLE_OPENMP=OFF \
 -D Kokkos_ARCH_SNB=ON \
 -D Kokkos_ARCH_KEPLER35=ON \
 -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
 -D ENABLE_BOOST=ON \
 -D BOOST_PATH=${BOOST_ROOT} \
 ../genten
