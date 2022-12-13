#!/bin/bash -xe
rm -f CMakeCache.txt
rm -rf CMakeFiles

# Use BDW for Kokkos arch unless specified otherwise
: ${GENTEN_ARCH:=BDW}

# Use ../genten as path to src unless specified otherwise
: ${GENTEN_SRC_PATH:=../genten}

# Use ../install/clang_openmp_${ARCH} (lowercase) unless specified otherwise
: ${GENTEN_INSTALL_PATH:=${PWD}/../install/gcc_openmpi_openmp_${GENTEN_ARCH,,}}

cmake \
 -D CMAKE_CXX_COMPILER=mpicxx \
 -D CMAKE_C_COMPILER=mpicc \
 -D CMAKE_CXX_FLAGS="-g" \
 -D CMAKE_C_FLAGS="-g" \
 -D BUILD_SHARED_LIBS=ON \
 -D Kokkos_ENABLE_OPENMP=ON \
 -D Kokkos_ARCH_SNB=ON \
 -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
 -D ENABLE_MPI=ON \
 -D MPIEXEC_MAX_NUMPROCS=4 \
 -D MPIEXEC_PREFLAGS="--bind-to core" \
 ${GENTEN_SRC_PATH}
