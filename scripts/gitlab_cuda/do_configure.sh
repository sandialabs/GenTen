#!/bin/bash -xe
rm -f CMakeCache.txt
rm -rf CMakeFiles

# Use BDW for Kokkos arch unless specified otherwise
: ${GENTEN_ARCH:=BDW}

# Use VOLTA70 for Kokkos GPU arch unless specified otherwise
: ${GENTEN_GPU:=VOLTA70}

# Use ../genten as path to src unless specified otherwise
: ${GENTEN_SRC_PATH:=../genten}

# Use ../install/gcc_cuda_${ARCH}_{GPU} (lowercase) unless specified otherwise
: ${GENTEN_INSTALL_PATH:=${PWD}/../install/gcc_cuda_${GENTEN_ARCH,,}_${GENTEN_GPU,,}}

KOKKOS=${GENTEN_SRC_PATH}/tpls/kokkos/bin

cmake \
 -D CMAKE_INSTALL_PREFIX=${GENTEN_INSTALL_PATH} \
 -D CMAKE_CXX_COMPILER=${KOKKOS}/nvcc_wrapper \
 -D CMAKE_C_COMPILER=gcc \
 -D CMAKE_CXX_FLAGS="-g -lineinfo" \
 -D CMAKE_C_FLAGS="-g" \
 -D BUILD_SHARED_LIBS=ON \
 -D Kokkos_ENABLE_CUDA=ON \
 -D Kokkos_ARCH_${GENTEN_ARCH}=ON \
 -D Kokkos_ARCH_${GENTEN_GPU}=ON \
 -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
 -D Kokkos_ENABLE_CUDA_LDG_INTRINSIC=OFF \
 -D Kokkos_ENABLE_CUDA_UVM=OFF \
 ${GENTEN_SRC_PATH}
