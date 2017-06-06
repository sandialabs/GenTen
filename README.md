# Genten Tensor Toolbox

Software package for tensor math by Sandia National Laboratories

Sandia National Laboratories is a multimission laboratory managed and operated
by National Technology and Engineering Solutions of Sandia, LLC, a wholly owned
subsidiary of Honeywell International, Inc., for the U.S. Department
of Energy's National Nuclear Security Administration under contract
DE-NA0003525.

Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

## Build Instructions

The instructions below assume a directory structure similar to the following.
For concreteness, assume we will building an optimized version of the code
using GNU compilers and OpenMP parallelism.

```
top-level
| -- kokkos
     | -- kokkos
     | -- build
          | -- opt_gnu_openmp
     | -- install
          | -- opt_gnu_openmp
| -- genten
     | -- genten-kokkos-sandia
     | -- build
          | -- opt_gnu_openmp
```

Of course that structure isn't required, but modifying it will require
adjusting the paths in the scripts below.

### Build Kokkos

Genten requires [Kokkos](github.com/kokkos/kokkos) for on-node thread/GPU
parallelism, and is available from github via

```
git clone https://github.com/kokkos/kokkos.git
```

There are many options for building Kokkos for each architecture, please see
their documentation.  Here is a simple script for building Kokkos using OpenMP
on a SandyBridge CPU architecture, assuming the Kokkos source is
placed in top-level/kokkos/kokkos:

```
#!/bin/bash

KOKKOS=../../kokkos
INSTALL=`pwd`/../../install/opt_gnu_openmp
${KOKKOS}/generate_makefile.bash \
  --kokkos-path=${KOKKOS} \
  --prefix=${INSTALL} \
  --with-openmp \
  --arch=SNB \
  --compiler=g++
```

which goes in the top-level/kokkos/build/opt_gnu_openmp directory above.
After executing this script, do "make" and "make install".

### Build genten:

Genten is built using [CMake](cmake.org), an open-souce build system
that supports multiple operating systems. You must download and install
CMake to build Genten.

Using our example above, the genten source goes in
top-level/genten/genten-kokkos-sandia.  To build the code with CMake,
we create a simple bash script such as the following:

```
#!/bin/bash

rm -f CMakeCache.txt;
rm -rf CMakeFiles

EXTRA_ARGS=$@
KOKKOS=`pwd`/../../../kokkos/install/opt_gnu_openmp

cmake \
 -D CMAKE_CXX_COMPILER=g++ \
 -D CMAKE_C_COMPILER=gcc \
 -D KOKKOS_PATH=${KOKKOS} \
 -D debug=OFF \
 ${EXTRA_ARGS} \
 ../../genten-kokkos-sandia
```

Execute this script to configure genten using CMake.  This will use
Kokkos for setting the necessary CXX flags for your architecture.
Then run "make".  To run the tests, you can run "./bin/unit_tests".

For examples of using genten, look in directories performance, driver,
and tests.

### Build options

#### Boost

Genten can use [Boost](www.boost.org) for reading compressed tensors.
This is enabled by adding the following to your genten configure script:

```
 -D ENABLE_BOOST=ON \
 -D BOOST_PATH=PATH-TO-BOOST \
```

where PATH-TO-BOOST is the path to the top-level of your boost
installation.

#### LAPACK

For best performance, a LAPACK library tuned for your machine should
be used.  Most computations in Genten are implemented directly with
Kokkos, however LAPACK is used for solving linear systems of
equations.  If LAPACK is not enabled, Genten provides its own, serial,
non-performant implementation.  LAPACK is enabled through the LAPACK_LIBS and
LAPACK_ADD_LIBS CMake variables, e.g., for Intel MKL:

```
 -D LAPACK_LIBS=${MKLROOT}/lib/intel64/libmkl_rt.so \
 -D LAPACK_ADD_LIBS="-liomp5;-lpthread;-lm;-ldl" \
```

### Advanced architectures

Through the use of Kokkos, Genten can be compiled and run on a variety
of multi-core and many-core architectures, including multi-core CPUs,
many-core Intel Phi accelerators, and Nvidia GPUs.  Compiling for each
architecutre requires specifying compilers, compiler options, and
support libraries in the Kokkos and Genten configure scripts.
Examples for each supported architecture can be found in
genten-kokkos-sandia/config-scripts, however the necessary steps will
be summarized here.

#### Intel CPU architectures

For Intel CPU architectures, the Intel compilers should be used, along
with Intel MKL.  The configure scripts are similar to the ones above.
For example, a Kokkos configure script for Haswell is

```
#!/bin/bash

KOKKOS=../../kokkos
INSTALL=`pwd`/../../install/opt_intel_openmp
${KOKKOS}/generate_makefile.bash \
  --kokkos-path=${KOKKOS} \
  --prefix=${INSTALL} \
  --with-openmp \
  --arch=HSW \
  --compiler=icpc
```

A corresponding Genten configure script is

```
#!/bin/bash

rm -f CMakeCache.txt;
rm -rf CMakeFiles

EXTRA_ARGS=$@
KOKKOS=`pwd`/../../../kokkos/install/opt_intel_openmp

cmake \
 -D CMAKE_CXX_COMPILER=icpc \
 -D CMAKE_C_COMPILER=icc \
 -D CMAKE_CXX_FLAGS="-g -restrict" \
 -D CMAKE_C_FLAGS="-g -restrict" \
 -D KOKKOS_PATH=${KOKKOS} \
 -D LAPACK_LIBS=$MKLROOT/lib/intel64/libmkl_rt.so \
 -D LAPACK_ADD_LIBS="-liomp5;-lpthread;-lm;-ldl" \
 -D ENABLE_BOOST=ON \
 -D BOOST_PATH=${BOOST_ROOT} \
 -D debug=OFF \
 ${EXTRA_ARGS} \
 ../../genten-kokkos-sandia
```

#### Intel KNL architecture

The configure for Intel KNL is quite similar to CPU architectures
above.  The only changes are to the Kokkos configure script:

```
#!/bin/bash

KOKKOS=../../kokkos
INSTALL=`pwd`/../../install/opt_intel_openmp
${KOKKOS}/generate_makefile.bash \
  --kokkos-path=${KOKKOS} \
  --prefix=${INSTALL} \
  --with-openmp \
  --arch=KNL \
  --compiler=icpc \
  --with-options=aggressive_vectorization
```

#### Nvidia GPU architectures

The build of Kokkos and Genten for GPU architectures is complicated by
the fact that Kokkos requires all source code using Kokkos to be
compiled by nvcc (even code not executed on the GPU).  To facilitate
this, Kokkos provides a script called nvcc_wrapper that makes nvcc act
like a normal compiler in terms of command line arguments, which must
be specified as the compiler.

A Kokkos configure script suitable for Nvida K80 GPUs is then

```
#!/bin/bash

KOKKOS=../../kokkos
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
```

In addition to Cuda, this also enables OpenMP for host-side
computations.  In this case, nvcc_wrapper will use g++ as the host
compiler.  If this is not correct, the compiler can be changed by
setting the NVCC_WRAPPER_DEFAULT_COMPILER environment variable, e.g.,

```
export NVCC_WRAPPER_DEFAULT_COMPILER=/home/software/gcc/4.9.2/bin/g++-4.9
```

Similarly, for the Genten configure script we have


```
#!/bin/bash

rm -f CMakeCache.txt;
rm -rf CMakeFiles

EXTRA_ARGS=$@
KOKKOS=`pwd`/../../../kokkos/install/opt_gnu_cuda

cmake \
 -D CMAKE_CXX_COMPILER=${KOKKOS}/bin/nvcc_wrapper \
 -D CMAKE_C_COMPILER=gcc \
 -D CMAKE_CXX_FLAGS="-g -lineinfo" \
 -D CMAKE_C_FLAGS="-g" \
 -D KOKKOS_PATH=${KOKKOS} \
 -D ENABLE_CUBLAS=ON \
 -D ENABLE_CUSOLVER=ON \
 -D ENABLE_BOOST=ON \
 -D BOOST_PATH=${BOOST_ROOT} \
 -D debug=OFF \
 ${EXTRA_ARGS} \
 ../../genten-kokkos-sandia
```

Note that instead of LAPACK, cuSolver and cuBLAS are used instead,
which are part of the standard Cuda installation.

Currently you must set the following environment variables when running any
tests or examples due to usage of Cuda-UVM:

```
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
```

For the Nvidia Pascal P100 GPUs, the configure is the same, except the
architecture is specified in the Kokkos configure script as Pascal60.
