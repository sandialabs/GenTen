#!/bin/bash -xe
rm -f CMakeCache.txt
rm -rf CMakeFiles

# Use BDW for Kokkos arch unless specified otherwise
: ${GENTEN_ARCH:=BDW}

# Use ../genten as path to src unless specified otherwise
: ${GENTEN_SRC_PATH:=../genten}

# Use ../install/clang_openmp_${ARCH} (lowercase) unless specified otherwise
: ${GENTEN_INSTALL_PATH:=${PWD}/../install/clang_openmp_${GENTEN_ARCH,,}}

cmake \
 -D CMAKE_INSTALL_PREFIX=${GENTEN_INSTALL_PATH} \
 -D CMAKE_CXX_COMPILER=clang++ \
 -D CMAKE_C_COMPILER=clang \
 -D CMAKE_CXX_FLAGS="-g" \
 -D CMAKE_C_FLAGS="-g" \
 -D BUILD_SHARED_LIBS=ON \
 -D ENABLE_CMAKE_TIMERS=ON \
 -D Kokkos_ENABLE_OPENMP=ON \
 -D Kokkos_ARCH_${GENTEN_ARCH}=ON \
 -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
 ${GENTEN_SRC_PATH}
