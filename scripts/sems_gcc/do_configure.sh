#!/bin/bash -xe
rm -f CMakeCache.txt
rm -rf CMakeFiles

cmake \
 -D CMAKE_CXX_COMPILER=g++ \
 -D CMAKE_C_COMPILER=gcc \
 -D CMAKE_CXX_FLAGS="-g" \
 -D CMAKE_C_FLAGS="-g" \
 -D BUILD_SHARED_LIBS=ON \
 -D Kokkos_ENABLE_OPENMP=ON \
 -D Kokkos_ARCH_SNB=ON \
 -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
 -D ENABLE_BOOST=ON \
 -D BOOST_PATH=${BOOST_ROOT} \
 ../genten
