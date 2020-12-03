#!/bin/bash -xe
rm -f CMakeCache.txt
rm -rf CMakeFiles

cmake \
 -D CMAKE_CXX_COMPILER=icpc \
 -D CMAKE_C_COMPILER=icc \
 -D CMAKE_CXX_FLAGS="-g -restrict" \
 -D CMAKE_C_FLAGS="-g -restrict" \
 -D BUILD_SHARED_LIBS=ON \
 -D Kokkos_ENABLE_OPENMP=ON \
 -D Kokkos_ARCH_SNB=ON \
 -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
 -D LAPACK_LIBS=$MKL_HOME/lib/intel64/libmkl_rt.so \
 -D LAPACK_ADD_LIBS="-liomp5;-lpthread;-lm;-ldl" \
 -D ENABLE_BOOST=ON \
 -D BOOST_PATH=${BOOST_ROOT} \
 ../genten
