#!/bin/bash

# For building Genten with an inline build of Kokkos along with genten,
# in which case the Kokkos source must be unpacked in the toplevel genten
# directory.

rm -f CMakeCache.txt;
rm -rf CMakeFiles

EXTRA_ARGS=$@

cmake \
 -D CMAKE_CXX_COMPILER=icpc \
 -D CMAKE_C_COMPILER=icc \
 -D CMAKE_CXX_FLAGS="-g -restrict" \
 -D CMAKE_C_FLAGS="-g -restrict" \
 -D KOKKOS_ENABLE_OPENMP=ON \
 -D KOKKOS_HOST_ARCH=KNL \
 -D KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
 -D LAPACK_LIBS=$MKLROOT/lib/intel64/libmkl_rt.so \
 -D LAPACK_ADD_LIBS="-liomp5;-lpthread;-lm;-ldl" \
 -D ENABLE_BOOST=ON \
 -D BOOST_PATH=${BOOST_ROOT} \
 -D debug=OFF \
  ${EXTRA_ARGS} \
 ../../genten

