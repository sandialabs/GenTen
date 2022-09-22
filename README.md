# Genten: Software for Generalized Tensor Decompositions by Sandia National Laboratories

Sandia National Laboratories is a multimission laboratory managed and operated
by National Technology and Engineering Solutions of Sandia, LLC, a wholly owned
subsidiary of Honeywell International, Inc., for the U.S. Department
of Energy's National Nuclear Security Administration under contract
DE-NA0003525.

Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

[[_TOC_]]

# Build Instructions

## Required Dependencies

Genten requires the following components in order to build and run:
* Kokkos for perfomance portable shared memory parallelism.  Genten bundles a clone of the Kokkos source using `git subtree` for use with inline builds (see below), which is generally kept up to date with the current Kokkos master branch (which corresponds to Kokkos releases).  This is the only version of Kokkos that is guaranteed to work with Genten (however later versions or the develop branch may work).
* A C++14 standard-compliant compiler.  In principle, any C++14 compiler supported by Kokkos should work, however many older compilers that claim compatability have bugs that are often exposed by Kokkos and/or Genten.  Genten is regularly tested with the following compilers and so these or any later version should work (earlier versions of these compilers *may* work, however it is known that Genten does not compile with GCC 5 and Intel 17 or 18):
  * GCC 7
  * Intel 19
  * Clang 9
* BLAS and LAPACK for CPU (OpenMP and/or pThreads) builds.
* The Cuda toolkit for Nvidia Cuda builds.  In principle any version of Cuda supported by Kokkos should work.  Currently this is Cuda versions 9 and 10.
* CMake for configure/build.  Any version of CMake supported by Kokkos should work.  Currently this is version 3.16 or later.

## Optional Dependencies

Genten can optionally use the following components:
* MATLAB for integration with the Tensor Toolbox.  Version 2018a or later should work.
* Boost for reading compressed sparse tensors.
* Caliper for application profiling.
* Trilinos/ROL for gradient-based GCP optimization approaches (experimental)

## Building Genten

Genten requires [Kokkos](github.com/kokkos/kokkos) for on-node thread/GPU
parallelism.  Genten supports two approaches for building Kokkos for use with Genten:  an
external build of Kokkos that is installed and linked to Genten, or an inline
build of Kokkos along with Genten using the bundled Kokkos source (contained within tpls/kokkos).
The latter is simpler and will be described first.  The former is useful if Genten
must be linked into an application that itself uses Kokkos.  Note however that only
the version of Kokkos that is provided with Genten is regularly tested for either
inline or external builds.

## Inline build with Kokkos

The instructions below assume a directory structure similar to the following.
For concreteness, assume we will building an optimized version of the code using GNU compilers
and OpenMP parallelism.

```
top-level
| -- genten
     | -- genten
          | -- tpls
               | -- kokkos
     | -- build
          | -- opt_gnu_openmp
```

Of course that structure isn't required, but modifying it will require
adjusting the paths in the scripts below.

Genten is built using [CMake](cmake.org), an open-souce build system
that supports multiple operating systems. You must download and install
CMake to build Genten.

Using our example above, the genten source goes in
top-level/genten/genten.  To build the code with CMake,
we create a simple bash script (in top-level/genten/genten/build/opt_gnu_openmp),
such as the following:

```
#!/bin/bash

rm -f CMakeCache.txt;
rm -rf CMakeFiles

EXTRA_ARGS=$@

cmake \
 -D CMAKE_CXX_COMPILER=g++ \
 -D CMAKE_C_COMPILER=gcc \
 -D Kokkos_ENABLE_OPENMP=ON \
 -D Kokkos_ARCH_SKX=ON \
 ${EXTRA_ARGS} \
 ../../genten
```

The script uses Kokkos options to specify the type of parallelism (OpenMP) and
the host architecture (SKX for Intel Skylake CPU).

Execute this script to configure genten and Kokkos using CMake.  This will use
Kokkos for setting the necessary CXX flags for your architecture.
Then run "make".  To run the tests, you can run `./bin/unit_tests`.

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

#### BLAS/LAPACK

Most computations in Genten are implemented directly with Kokkos, however BLAS and LAPACK routines are used when possible.  Therefore these libraries are required for CPU builds (i.e., OpenMP or pThreads).  For Cuda GPU builds, Genten instead uses cuBLAS and cuSolver, which are distributed as part of the Cuda toolkit.  LAPACK and BLAS are enabled through the LAPACK_LIBS and LAPACK_ADD_LIBS CMake variables, e.g., for Intel MKL:

```
 -D LAPACK_LIBS=${MKLROOT}/lib/intel64/libmkl_rt.so \
 -D LAPACK_ADD_LIBS="-liomp5;-lpthread;-lm;-ldl" \
```

#### MATLAB

Genten includes a limited MATLAB interface designed to be integrated with
the [Tensor Toolbox](https://www.tensortoolbox.org/).  To enable it, simply
add the configure options:
```
 -D ENABLE_MATLAB=ON \
 -D MATLAB_PATH=/path/to/MATLAB/R2018b \
```
where `/path/to/MATLAB/R2018b` is the path to your toplevel MATLAB installation
(R2018b in this case).  On recent Mac OS X architectures, you may also need to
add
```
 -D INDEX_TYPE="unsigned long long" \
```
to your configure options.


### Advanced architectures

Through the use of Kokkos, Genten can be compiled and run on a variety
of multi-core and many-core architectures, including multi-core CPUs,
many-core Intel Phi accelerators, and Nvidia GPUs.  Compiling for each
architecture requires specifying compilers and architecture-related Kokkos
options in the Genten configure scripts.
Examples for common cases are summarized here.

#### Intel CPU architectures

For Intel CPU architectures, the Intel compilers should be used, along
with Intel MKL.  The configure scripts are similar to the ones above.
For example, a configure script for Skylake is

```
#!/bin/bash

rm -f CMakeCache.txt;
rm -rf CMakeFiles

EXTRA_ARGS=$@

cmake \
 -D CMAKE_CXX_COMPILER=icpc \
 -D CMAKE_C_COMPILER=icc \
 -D CMAKE_CXX_FLAGS="-g -restrict" \
 -D CMAKE_C_FLAGS="-g -restrict" \
 -D Kokkos_ENABLE_OPENMP=ON \
 -D Kokkos_ARCH_SKX=ON \
 -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
 -D LAPACK_LIBS=$MKLROOT/lib/intel64/libmkl_rt.so \
 -D LAPACK_ADD_LIBS="-liomp5;-lpthread;-lm;-ldl" \
 ${EXTRA_ARGS} \
 ../../genten
```

#### Intel KNL architecture

The configure for Intel KNL is quite similar to CPU architectures
above.  The only change is the host architecture:

```
 -D KOKKOS_ARCH_KNL=ON \
```

#### Nvidia GPU architectures

The build of Kokkos and Genten for GPU architectures is complicated by
the fact that Kokkos requires all source code using Kokkos to be
compiled by nvcc (even code not executed on the GPU).  To facilitate
this, Kokkos provides a script called nvcc_wrapper that makes nvcc act
like a normal compiler in terms of command line arguments, which must
be specified as the compiler.

A configure script suitable for Nvida Volta GPUs is then

```
rm -f CMakeCache.txt;
rm -rf CMakeFiles

EXTRA_ARGS=$@
KOKKOS=${PWD}/../../genten/tpls/kokkos

cmake \
 -D CMAKE_CXX_COMPILER=${KOKKOS}/bin/nvcc_wrapper \
 -D CMAKE_C_COMPILER=gcc \
 -D CMAKE_CXX_FLAGS="-g  -lineinfo" \
 -D CMAKE_C_FLAGS="-g" \
 -D Kokkos_ENABLE_OPENMP=ON \
 -D Kokkos_ENABLE_CUDA=ON \
 -D Kokkos_ARCH_SKX=ON \
 -D Kokkos_ARCH_VOLTA70=ON \
 ${EXTRA_ARGS} \
 ../../genten
```

In addition to Cuda, this also enables OpenMP for host-side
computations.  In this case, nvcc_wrapper will use g++ as the host
compiler.  If this is not correct, the compiler can be changed by
setting the NVCC_WRAPPER_DEFAULT_COMPILER environment variable, e.g.,

```
export NVCC_WRAPPER_DEFAULT_COMPILER=/path/to/my/gnu/compiler/g++
```

Note that instead of LAPACK, cuSolver and cuBLAS are used instead,
which are part of the standard Cuda installation.

Genten does not require the use of Cuda-UVM as all necessary data transfers
between the host and device are implemented through Kokkos.  However one can
enable UVM by adding the configure option

```
-D Kokkos_ENABLE_CUDA_UVM=ON \
```

in which case one should also set the environment variables

```
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
```

For the Nvidia Pascal P100 GPUs, the configure is the same, except the
architecture is specified in the Kokkos configure script as Pascal60.

## External build with Kokkos

For an external build of Kokkos, you must configure, build and install Kokkos
first using their CMake files.  Similar to the inline build above, the
instructions below assume the following directory structure:

```
top-level
| -- kokkos
     | -- kokkos
     | -- build
          | -- opt_gnu_openmp
     | -- install
          | -- opt_gnu_openmp
| -- genten
     | -- genten
     | -- build
          | -- opt_gnu_openmp
```

### Build Kokkos

The CMake script for building Kokkos is similar to the inline build
scripts shown above, just with all of the Genten-specific options
removed.  You must also specify an install directory.  Furthemore,
since Genten requires use of C++14 constructs, the C++ standard
enabled within Kokkos must be set to 14.  For example,
here is a simple script for building Kokkos using OpenMP on a
Skylake CPU architecture, assuming the Kokkos source is placed in
top-level/kokkos/kokkos:

```
rm -f CMakeCache.txt;
rm -rf CMakeFiles

KOKKOS=../../kokkos
INSTALL=`pwd`/../../install/opt_gnu_openmp

cmake \
  -D CMAKE_INSTALL_PREFIX=${INSTALL} \
  -D CMAKE_CXX_COMPILER=g++ \
  -D Kokkos_CXX_STANDARD=14 \
  -D Kokkos_ENABLE_OPENMP=ON \
  -D Kokkos_ARCH_SKX=ON \
  ${KOKKOS}
```

which goes in the top-level/kokkos/build/opt_gnu_openmp directory above.
After executing this script, do "make" and "make install".

### Build Genten

Genten is then built with CMake similar to the inline build discussed above,
however the path to the Kokkos installation is specified instead of any
Kokkos-related build options:

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
 ${EXTRA_ARGS} \
 ../../genten
```

#### Nvidia GPU architectures

Since Kokkos now uses CMake for their standalone build, the configure
scripts for advanced architectures for an external build are similar
to the inline build as shown above.  However for Nvidia GPU
architectures, there are some caveats.  As with the inline build,
nvcc_wrapper must be used as the compiler when compiling Kokkos.
Furthermore you must enable lambda support through the option `-D
Kokkos_ENABLE_CUDA_LAMBDA=ON \`, since Genten makes heavy use of
lambdas.  For example, a Kokkos configure script suitable for Nvidia
Volta GPUs is then

```
rm -f CMakeCache.txt;
rm -rf CMakeFiles

KOKKOS=`pwd`/../../kokkos
INSTALL=`pwd`/../../install/opt_gnu_cuda

cmake \
  -D CMAKE_INSTALL_PREFIX=${INSTALL} \
  -D CMAKE_CXX_COMPILER=${KOKKOS}/bin/nvcc_wrapper \
  -D Kokkos_CXX_STANDARD=14 \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ENABLE_CUDA_UVM=OFF \
  -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
  -D Kokkos_ARCH_SKX=ON \
  -D Kokkos_ARCH_VOLTA70=ON \
  ${KOKKOS}
```

This script additionally turns UVM off, which is optional.  Similarly, for the Genten configure script we have

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
 ${EXTRA_ARGS} \
 ../../genten
```

# Testing Genten

Once Genten has been compiled, it can be tested by executing `ctest`.

# Using Genten

The primary executable for Genten is `bin/genten` in your build tree, which is
a driver for reading in a (sparse) tensor and performing a CP or GCP
decomposition of it.  The driver accepts numerous command line options
controlling various aspects of the computation.  Run `genten --help` for a full
listing.  For example
```
./bin/genten --input data/aminoacid_data.txt --rank 16 --output aa.ktns
```
will perform a rank 16 CP decomposition of the amino-acid tensor data set
included with Genten in the data directory, and save the resulting factors in
`aa.ktns`.  One should see output similar to:
```
./bin/genten --input data/aminoacid_data.txt --rank 16 --output aa.ktns
Read tensor with 61305 nonzeros, dimensions [ 5 201 61 ], and starting index 0
Data import took  0.033 seconds

CP-ALS (perm MTTKRP method, symmetric-gram formulation):
Iter   1: fit =  9.710805e-01 fitdelta =  9.7e-01
Iter   2: fit =  9.865534e-01 fitdelta =  1.5e-02
Iter   3: fit =  9.876203e-01 fitdelta =  1.1e-03
Iter   4: fit =  9.880996e-01 fitdelta =  4.8e-04
Iter   5: fit =  9.883227e-01 fitdelta =  2.2e-04
Final fit =  9.883227e-01
Ktensor export took  0.005 seconds
```

For larger tensor datasets, consider those available from the
[FROSTT](https://frost.io) collection.  Note that Genten *does not* require
a header at the top of the sparse tensor file indicating the number of modes,
their dimensions, and the number of nonzeros.  Any textfile consisting of a list
of nonzeros in coordinate format (i.e., nonzero indices and value) can be
read.  If configured with Boost support, compressed tensors can be read directly
without first decompressing them.

## Using the MATLAB interface

To use Genten within MATLAB, you must first add the Tensor Toolbox and the path
to the `matlab` directory in your Genten build tree to your MATLAB path.
Genten provides a MATLAB class `sptensor_gt` that works similarly to the
Tensor Toolbox sparse tensor class `sptensor` that can be passed to various
MATLAB functions provided by Genten for manipulating the tensor.  A given
tensor `X` in `sptensor` format can be converted to `sptensor_gt` format
by calling the constructor:
```
>> X = sptenrand([10 20 30],100);
>> X_gt = sptensor_gt(X);
```
Genten then provides overloads of several functions in the Tensor Toolbox
that call the corresponding implementation in Genten when passed a tensor
in `sptensor_gt` format, e.g.,
```
>> U = cp_als(X_gt,16,'maxiters',5);

CP-ALS (perm MTTKRP method, symmetric-gram formulation):
Iter   1: fit =  1.265589e-01 fitdelta =  1.3e-01
Iter   2: fit =  1.909217e-01 fitdelta =  6.4e-02
Iter   3: fit =  2.195554e-01 fitdelta =  2.9e-02
Iter   4: fit =  2.465984e-01 fitdelta =  2.7e-02
Iter   5: fit =  2.692030e-01 fitdelta =  2.3e-02
Final fit =  2.692030e-01
```
Note that in addition to the normal options accepted by `cp_als`,
all options accepted by the `genten` command-line driver (without the
leading '--') are also accepted, e.g.,
```
>> U = cp_als(X_gt,16,'maxiters',5,'mttkrp-method','duplicated','timings');
Parsing tensor took 3.740000e-04 seconds

CP-ALS (duplicated MTTKRP method, symmetric-gram formulation):
Iter   1: fit =  1.250860e-01 fitdelta =  1.3e-01
Iter   2: fit =  2.081054e-01 fitdelta =  8.3e-02
Iter   3: fit =  2.570137e-01 fitdelta =  4.9e-02
Iter   4: fit =  2.900351e-01 fitdelta =  3.3e-02
Iter   5: fit =  3.106995e-01 fitdelta =  2.1e-02
Final fit =  3.106995e-01
CpAls completed 6 iterations in 1.03e-02 seconds
	MTTKRP total time = 9.27e-04 seconds, average time = 6.18e-05 seconds
	MTTKRP throughput = 9.64e-02 GFLOP/s, bandwidth factor = 1.48e-01
	Inner product total time = 7.60e-05 seconds, average time = 1.52e-05 seconds
	Gramian total time = 1.22e-04 seconds, average time = 8.13e-06 seconds
	Solve total time = 4.93e-04 seconds, average time = 3.29e-05 seconds
	Scale total time = 6.54e-03 seconds, average time = 4.36e-04 seconds
	Norm total time = 3.53e-04 seconds, average time = 2.35e-05 seconds
	Arrange total time = 3.47e-04 seconds, average time = 3.47e-04 seconds
```

# More information and how to cite

For more information on the algorithms used in Genten with Kokkos, or to cite
Genten, please see
* Eric T. Phipps and Tamara G. Kolda, *Software for Sparse Tensor Decomposition
  on Emerging Computing Architectures*, SIAM Journal on Scientific Computing
  2019 41:3, C269-C290
  (available [here](https://epubs.siam.org/doi/ref/10.1137/18M1210691)).

# Updating Kokkos

Genten uses `git subtree` to manage the bundled Kokkos source.  Below is a summary of the steps required to update Genten's clone to the latest Kokkos sources.

First add a remote referencing the Kokkos github repo:
```
git remote add -f kokkos git@github.com:kokkos/kokkos.git
```
Then the Kokkos clone can be udpated via
```
git fetch kokkos master
git subtree pull --prefix tpls/kokkos kokkos master --squash
```
The new Kokkos sources are then pushed to the Genten repo as usual:
```
git push
```
