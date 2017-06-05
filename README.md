# Genten Tensor Toolbox

Software package for tensor math by Sandia National Laboratories

Sandia National Laboratories is a multimission laboratory managed and operated
by National Technology and Engineering Solutions of Sandia, LLC, a wholly owned
subsidiary of Honeywell International, Inc., for the U.S. Department of Energy’s
National Nuclear Security Administration under contract DE-NA0003525.

Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

## Overview

This file contains brief instructions for building the Genten Tensor Toolbox
libraries and examples. Please refer to the user manual in /doc for a
complete explanation. For best performance, we recommend linking with
an LAPACK linear algebra library tuned for your machine; these quick build
notes instead use the source provided in genten/src/mathlibs/Genten_Default_Blas.c.

Genten Tensor Toolbox is built using CMake, an open-souce build system
that supports multiple operating systems. You must download and install
CMake from http://cmake.org .

At the time this documentation was produced, the CMake distribution could
be found by clicking on "Download" to reach
http://cmake.org/cmake/download .
Then select the version that is appropriate for your computer.

Installation of CMake is very simple and explained on the download page.
This README gives instructions for the command line (rather than GUI) usage
of CMake.

You must have a C++ compiler installed on your machine. CMake will find
the compiler and generate appropriate "make" files in a separate directory.
Then you will run "make" to compile the Tensor Toolbox libraries and examples.

## Build Instructions

### Genten requires Kokkos for thread-parallelism:

https://github.com/kokkos/kokkos

Kokkos is available from github via

```
git clone https://github.com/kokkos/kokkos.git
```

There are many options for
building Kokkos for each architecture, please see their documentation.  Here is
a simple script for building Kokkos using OpenMP on a SandyBridge CPU
architecture:

```
KOKKOS=../../kokkos
INSTALL=`pwd`/../../install/opt_gnu_openmp
${KOKKOS}/generate_makefile.bash \
  --kokkos-path=${KOKKOS} \
  --prefix=${INSTALL} \
  --with-openmp \
  --arch=SNB \
  --compiler=g++
```

Then do "make" and "make install".  A similar script for Nvidia GPUs is

```
KOKKOS=`pwd`/../../kokkos
INSTALL=`pwd`/../../install/opt_gnu_cuda
${KOKKOS}/generate_makefile.bash \
  --kokkos-path=${KOKKOS} \
  --prefix=${INSTALL} \
  --with-openmp \
  --with-cuda \
  --with-cuda-options=force_uvm,enable_lambda \
  --arch=SNB,Kepler35 \
  --compiler=${KOKKOS}/config/nvcc_wrapper
```

### Quick Build for Linux:

  Unpack the distribution in any location (root privilege is not required
  to build). The top directory will be named genten.

  Create a build directory at the same level, run CMake, and then run make.
  For example, from the directory above the unpacked distribution:

    ```
    > mkdir genten_build
    > cd genten_build
    > cmake \
       -D KOKKOS_PATH=<PATH TO KOKKOS INSTALL> \
       ../genten
    > make
    > ./bin/unit_tests
    ```

  You might also consider enabling Boost for reading compressed tensors:
    ```
       -D ENABLE_BOOST=ON \
       -D BOOST_PATH=<PATH TO BOOST INSTALL> \
    ```

  For Cuda, you must also set
    ```
       -D CMAKE_CXX_COMPILER=<PATH TO KOKKOS INSTALL>/bin/nvcc_wrapper \
       -D ENABLE_CUBLAS=ON \
       -D ENABLE_CUSOLVER=ON \
    ```
  Currently you must set the following environment variables when running any
  tests or examples due to usage of Cuda-UVM:
    ```
    export CUDA_LAUNCH_BLOCKING=1
    export CUDA_VISIBLE_DEVICES=0
    ```

  You might write some code "myTensorApp.cpp" which uses the Tensor Toolbox
  library. In order to compile and link with g++:

    ```
    > g++ -c myTensorApp.cpp -I<genten>/src
    > g++ myTensorApp.o -L<genten_build> -lgenten -lgenten_mathlibs
    ```

  For examples, look in directories genten/performance and genten/test.