#!/bin/bash -xe
rm -f CMakeCache.txt
rm -rf CMakeFiles

# Use BDW for Kokkos arch unless specified otherwise
: ${GENTEN_ARCH:=BDW}

# Use VOLTA70 for Kokkos GPU arch unless specified otherwise
: ${GENTEN_GPU:=VOLTA70}

# Use ../genten as path to src unless specified otherwise
: ${GENTEN_SRC_PATH:=../genten}

# Use ../install/gcc_sycl_cuda_${ARCH}_{GPU} (lowercase) unless specified otherwise
: ${GENTEN_INSTALL_PATH:=${PWD}/../install/gcc_sycl_cuda_${GENTEN_ARCH,,}_${GENTEN_GPU,,}}

KOKKOS=${GENTEN_SRC_PATH}/tpls/kokkos/bin

# Note, can't add "-g" to C/CXX flags as it will cause seg faults in the
# SYCL compiler

cmake \
 -D CMAKE_INSTALL_PREFIX=${GENTEN_INSTALL_PATH} \
 -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
 -D CMAKE_CXX_COMPILER=clang++ \
 -D CMAKE_C_COMPILER=clang \
 -D CMAKE_CXX_FLAGS="-Wno-unknown-cuda-version -Wno-gnu-zero-variadic-macro-arguments -Wno-deprecated-declarations -Wno-linker-warnings" \
 -D CMAKE_C_FLAGS="" \
 -D BUILD_SHARED_LIBS=ON \
 -D GENTEN_ENABLE_SYCL_FOR_CUDA=ON \
 -D Kokkos_ARCH_${GENTEN_ARCH}=ON \
 -D Kokkos_ARCH_${GENTEN_GPU}=ON \
 -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
 -D Kokkos_ENABLE_CUDA_LDG_INTRINSIC=OFF \
 -D Kokkos_ENABLE_CUDA_UVM=OFF \
 -D Kokkos_ENABLE_UNSUPPORTED_ARCHS=ON \
 -D Kokkos_ENABLE_DEPRECATED_CODE_3=ON \
 -D Kokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
 -D debug=OFF \
 ${GENTEN_SRC_PATH}
