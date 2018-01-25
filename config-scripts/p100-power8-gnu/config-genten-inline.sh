#!/bin/bash

# For building Genten with an inline build of Kokkos along with genten,
# in which case the Kokkos source must be unpacked in the toplevel genten
# directory.

rm -f CMakeCache.txt;
rm -rf CMakeFiles

EXTRA_ARGS=$@
KOKKOS=${PWD}/../../genten/kokkos

cmake \
 -D CMAKE_CXX_COMPILER=${KOKKOS}/config/nvcc_wrapper \
 -D CMAKE_C_COMPILER=gcc \
 -D CMAKE_CXX_FLAGS="-g  -lineinfo" \
 -D CMAKE_C_FLAGS="-g" \
 -D KOKKOS_INLINE_BUILD=ON \
 -D KOKKOS_ENABLE_OPENMP=ON \
 -D KOKKOS_ENABLE_CUDA=ON \
 -D KOKKOS_ARCH="Power8;Pascal60" \
 -D ENABLE_CUBLAS=ON \
 -D ENABLE_CUSOLVER=ON \
 -D ENABLE_BOOST=ON \
 -D BOOST_PATH=${BOOST_ROOT} \
 -D debug=OFF \
 ${EXTRA_ARGS} \
 ../../genten
