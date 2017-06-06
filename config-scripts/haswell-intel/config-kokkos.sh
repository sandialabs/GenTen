#!/bin/bash

KOKKOS=../../kokkos
INSTALL=`pwd`/../../install/opt_intel_openmp

${KOKKOS}/generate_makefile.bash \
  --kokkos-path=${KOKKOS} \
  --prefix=${INSTALL} \
  --with-openmp \
  --arch=HSW \
  --compiler=icpc
