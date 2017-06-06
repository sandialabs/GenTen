#!/bin/sh

KOKKOS=`pwd`/../../kokkos
KOKKOS=$(cd $KOKKOS; pwd) # for relative path bug in specifying compiler
INSTALL=`pwd`/../../install/opt_gnu_cuda

${KOKKOS}/generate_makefile.bash \
  --kokkos-path=${KOKKOS} \
  --prefix=${INSTALL} \
  --with-openmp \
  --with-cuda \
  --with-cuda-options=force_uvm,enable_lambda \
  --arch=HSW,Kepler37 \
  --compiler=${KOKKOS}/config/nvcc_wrapper
