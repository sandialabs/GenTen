#!/usr/bin/env bash

set -ex

rm -f CMakeCache.txt
rm -rf CMakeFiles

cmake \
	-D BUILD_SHARED_LIBS=ON \
	-D CMAKE_BUILD_TYPE=Release \
	-D CMAKE_C_COMPILER=clang \
	-D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
	-D CMAKE_CXX_COMPILER=clang++ \
	-D CMAKE_CXX_FLAGS="-Wno-unknown-cuda-version -Wno-gnu-zero-variadic-macro-arguments -Wno-deprecated-declarations -Wno-linker-warnings" \
	-D CMAKE_VERBOSE_MAKEFILE=OFF \
	-D GENTEN_ENABLE_SYCL_FOR_CUDA=ON \
	-D Kokkos_ARCH_PASCAL61=ON \
	-D Kokkos_ENABLE_DEPRECATED_CODE_3=ON \
	-D Kokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
	-D Kokkos_ENABLE_UNSUPPORTED_ARCHS=ON \
	../genten
