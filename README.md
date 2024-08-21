# Genten: Software for Generalized Tensor Decompositions by Sandia National Laboratories

Sandia National Laboratories is a multimission laboratory managed and operated
by National Technology and Engineering Solutions of Sandia, LLC, a wholly owned
subsidiary of Honeywell International, Inc., for the U.S. Department
of Energy's National Nuclear Security Administration under contract
DE-NA0003525.

Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

# About GenTen

GenTen is a tool for computing Canonical Polyadic (CP, also called CANDECOMP/PARAFAC) decompositions of tensor data.  It is geared towards analysis of extreme-scale data and implements several CP decomposition algorithms that are parallel and scalable, including:
* CP-ALS:  The workhorse algorithm for Gaussian sparse or dense tensor data.
* [CP-OPT](https://doi.org/10.1002/cem.1335):  CP decomposition of (sparse or dense) Gaussian data using a quasi-Newton optimization algorithm incorporating possible upper and lower bound constraints.
* [GCP](https://epubs.siam.org/doi/abs/10.1137/18M1203626):  Generalized CP supporting arbitrary loss functions (Gaussian, Poisson, Bernoulli, ...), solved using [quasi-Newton](https://epubs.siam.org/doi/abs/10.1137/18M1203626) (dense tensors) or [stochastic gradient descent](https://doi.org/10.1137/19M1266265) (sparse or dense tensors) optimization methods.
* [Streaming GCP](https://doi.org/10.1145/3592979.3593405): A GCP algorithm that incrementally updates a GCP decomposition as new data is observed, suitable for in situ analysis of streaming data.
* Federated GCP:  A federated learning algorithm for GCP supporting asynchronous parallel communication.

GenTen does not provide CP-APR for Poisson data (see [SparTen](https://github.com/sandialabs/sparten) instead) nor other tensor decompositions methods such as Tucker (see [TuckerMPI](https://gitlab.com/tensors/TuckerMPI) instead) or Tensor Train. 

GenTen builds on [Kokkos](https://github.com/kokkos/kokkos) and [Kokkos Kernels](https://github.com/kokkos/kokkos-kernels) to support shared memory parallel programming models on a variety of contemporary architectures, including:
* OpenMP for CPUs.
* CUDA for NVIDIA GPUs.
* HIP for AMD GPUs.
* SYCL for Intel GPUs.

GenTen also supports distributed memory parallelism using MPI.

GenTen is implemented in C++ for high-performance and supports several modes of operation.  It provides a standalone command line tool for reading tensor data from files, and also provides Matlab and Python bindings that integrate with the [Matlab Tensor Toolbox](https://www.tensortoolbox.org/) and [Python Tensor Toolbox (pyttb)](https://github.com/sandialabs/pyttb) to provide high-performance, parallel tensor decomposition capabilities in those environments.

For more information on the algorithms used in Genten with Kokkos, or to cite Genten, please see
* Eric T. Phipps and Tamara G. Kolda, *Software for Sparse Tensor Decomposition on Emerging Computing Architectures*, SIAM Journal on Scientific Computing 2019 41:3, C269-C290, [DOI: 10.1137/18M1210691](https://epubs.siam.org/doi/ref/10.1137/18M1210691).

Additional papers describing recent GenTen research and development:
* E. Phipps, N. Johnson and T. Kolda, *Streaming Generalized Canonical Polyadic Tensor Decompositions,* in Proceedings of the Platform for Advanced Scientific Computing Conference (PASC ’23), 2023 [DOI: 10.1145/3592979.3593405](https://doi.org/10.1145/3592979.3593405).
* C. Lewis and E. Phipps, *Low-Communication Asynchronous Distributed Generalized Canonical Polyadic Tensor Decomposition,* 2021 IEEE High Performance Extreme Computing Conference (HPEC), 2021 [DOI: 0.1109/HPEC49654.2021.9622844](https://doi.org/10.1109/HPEC49654.2021.9622844).

# Installing GenTen

## Required dependencies

GenTen bundles most of what it needs to build and run, including Kokkos/Kokkos-Kernels but requires the following components:
* A C++ compiler [supported by Kokkos](https://kokkos.org/kokkos-core-wiki/requirements.html#compiler-versions) for the intended architecture/programming model.
* The corresponding toolkit for GPU builds (CUDA, ROCm, or oneAPI for NVIDIA, AMD, or Intel GPUs, respectively).
* A version of CMake [supported by Kokkos](https://kokkos.org/kokkos-core-wiki/requirements.html#build-system).
* BLAS and LAPACK libraries when intending to run GenTen algorithms on a CPU architecture (GenTen uses BLAS/LAPACK functionality provided by the NVIDIA, AMD, and Intel toolkits for GPU builds)

## Optional dependencies

Genten can optionally make use the following components:
* MPI for distributed memory parallelism.
  * For distributed parallelism with GPUs, GenTen assumes the MPI library is "GPU-aware" in that it can read and write data directly from GPU memory.  On NVIDIA GPUs, one can also enable UVM in the Kokkos build if the MPI library is not GPU-aware.  
* Python for Python bindings.
  * pytest is required for testing GenTen Python bindings.
  * Integration with pyttb requires several other Python packages required by pyttb, including numpy, numpy_groupies, and scipy.
* MATLAB for integration with the Matlab Tensor Toolbox.
* Boost for reading compressed sparse tensors.
* GenTen can use several packages from Trilinos, including:
  * ROL for large-scale, parallel, derivative-based optimization methods.
  * Tpetra for certain distributed parallelism approaches.
  * Teuchos for timing-related utilities.
  * SEACAS for reading tensor data from Exodus files.
  * Trilinos builds can also be used to provide Kokkos/Kokkos-Kernels.

## Building Genten

Genten bundles Kokkos and Kokkos Kernels (along with several other support libraries) using `git subtree` that are compiled along with GenTen to make building GenTen easier (referred to as inline builds below), which are generally kept up-to-date with the most recent release of these libraries.  GenTen can also link against externally compiled versions of these libraries, however compatibility is only guaranteed for the same version bundled by GenTen.  We first describe the basics of building GenTen with inline builds of Kokkos/Kokkos-Kernels followed by the modifications needed for linking to externally compiled Kokkos/Kokkos-Kernels libraries.

### Inline build of Kokkos and Kokkos Kernels

The simplest approach for building GenTen in standalone situations is to compile Kokkos and Kokkos Kernels along with GenTen, which means the correct configuration options for configuring Kokkos/Kokkos-Kernels for the intended architecture(s) must be provided when configuring GenTen.  A full description of how to configure these packages is beyond the scope of this document, and we refer the reader to [here](https://kokkos.org/kokkos-core-wiki/building.html) and [here](https://github.com/kokkos/kokkos-kernels/blob/develop/BUILD.md) for more information.

The instructions below assume a directory structure similar to the following.  For concreteness, assume we will build an optimized version of the code using GNU compilers and OpenMP parallelism.

```
top-level
| -- genten
     | -- genten
          | -- tpls
               | -- kokkos
     | -- build
          | -- opt_gnu_openmp
```

Of course that structure isn't required, but modifying it will require adjusting the paths in the scripts below.

GenTen is built using [CMake](cmake.org), an open-souce build system that supports multiple operating systems. You must download and install CMake to build Genten.

Using our example above, the GenTen source goes in top-level/genten/genten.  To build the code with CMake, we create a simple bash script (in top-level/genten/genten/build/opt_gnu_openmp), such as the following:

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

The script uses Kokkos options to specify the type of parallelism (OpenMP) and the host architecture (SKX for Intel Skylake CPU).

Execute this script to configure GenTen using CMake.  This will use Kokkos for setting the necessary CXX flags for your architecture. Then run `make`.  To run the tests, you can run `ctest`.

### Build options

#### BLAS/LAPACK

Most computations in Genten are implemented directly with Kokkos/Kokkos-Kernels, however BLAS and LAPACK routines are used when possible.  Therefore these libraries are required for CPU builds.  CMake will attempt to locate these libraries automatically, but in many situations they must be specified through the LAPACK_LIBS and LAPACK_ADD_LIBS CMake variables, e.g., for Intel MKL:

```
 -D LAPACK_LIBS=${MKLROOT}/lib/intel64/libmkl_rt.so \
 -D LAPACK_ADD_LIBS="-liomp5;-lpthread;-lm;-ldl" \
```

#### MPI

GenTen supports distributed memory parallelism using MPI.  This is enabled by replacing the compilers with the MPI compilers and telling GenTen to enable MPI support, e.g.,:

```
#!/bin/bash

rm -f CMakeCache.txt;
rm -rf CMakeFiles

EXTRA_ARGS=$@

cmake \
 -D CMAKE_CXX_COMPILER=mpicxx \
 -D CMAKE_C_COMPILER=mpicc \
 -D Kokkos_ENABLE_OPENMP=ON \
 -D Kokkos_ARCH_SKX=ON \
 -D ENABLE_MPI=ON \
 ${EXTRA_ARGS} \
 ../../genten
```

If you are also going to run GenTen's tests, it is recommended to specify a maximum number of MPI processors to use (otherwise it will use the maximum number available on the machine) and tell MPI to bind processes to cores to improve performance.  When using OpenMPI, this can be done by adding the following configuration options:
```
 -D MPIEXEC_MAX_NUMPROCS=4 \
 -D MPIEXEC_PREFLAGS="--bind-to;core" \
```

#### GPU architectures

GenTen supports building for NVIDIA, AMD, and Intel GPU architectures supported by Kokkos.  Enabling these generally just requires enabling the corresponding Kokkos backend and target architecture, e.g., for NVIDIA V100 architecture:
```
 -D Kokkos_ENABLE_CUDA=ON \
 -D Kokkos_ARCH_VOLTA70=ON \
```
and set the CMake CXX compiler to be the appropriate compiler provided by the vendor toolkit.  For CUDA however, one must use the `nvcc_wrapper` script provided by Kokkos as the compiler, e.g.,:
```
 -D CMAKE_CXX_COMPILER=${PWD}/../../genten/tpls/kokkos/bin/nvcc_wrapper \
```
Note that `nvcc_wrapper` uses g++ as the host compiler by default.  If this is not correct, the compiler can be changed by setting the `NVCC_WRAPPER_DEFAULT_COMPILER` environment variable, e.g.,
```
export NVCC_WRAPPER_DEFAULT_COMPILER=/path/to/my/gnu/compiler/g++
```
Also, when enabling MPI and CUDA, the supported approach is to still use the MPI compilers as the CMake CXX compiler as shown in the MPI section above, but override the compiler it calls to be `nvcc_wrapper`, e.g. for OpenMPI,
```
export OMPI_CXX=${PWD}/../genten/tpls/kokkos/bin/nvcc_wrapper
```

#### Boost

GenTen can use [Boost](www.boost.org) for reading compressed tensors.  This is enabled by adding the following to your genten configure script:

```
 -D ENABLE_BOOST=ON \
 -D Boost_ROOT=PATH-TO-BOOST \
```

where PATH-TO-BOOST is the path to the top-level of your boost installation.

#### MATLAB

GenTen includes a limited MATLAB interface designed to be integrated with the [Tensor Toolbox](https://www.tensortoolbox.org/).  To enable it, simply add the configure options:
```
 -D ENABLE_MATLAB=ON \
 -D Matlab_ROOT=/path/to/MATLAB/R2018b \
```
where `/path/to/MATLAB/R2018b` is the path to your toplevel MATLAB installation (R2018b in this case).  On recent Mac OS X architectures, you may also need to add
```
 -D INDEX_TYPE="unsigned long long" \
```
to your configure options.

#### Python

GenTen provides Python bindings through a Python module called pygenten generated by [pybind11](https://github.com/pybind/pybind11).  This can be enabled by adding the configuration option
```
 -D ENABLE_PYTHON=ON \
```
which will compile GenTen to link against the version of python found in your environment.  If the tests are going to be run, one must also have `pytest` installed.  Furthermore, GenTen provides support for interoperability with pyttb.  Several examples of using pygenten with and without pyttb can be found in [python/example](python/example).

Alternatively, pygenten can be directly installed using pip.  See [python/README.md](python/README.md) for more details.

### Building on MacOS

Building GenTen on MacOS has a few challenges.  First, MacOS does not provide a Fortran compiler, used by Kokkos-Kernels to discover Fortran name mangling for calling BLAS/LAPACK functions from C++.  There are two ways to address this:
* Tell Kokkos-Kernels what the name mangling scheme is, e.g., `-D F77_BLAS_MANGLE='(name,NAME) name ## _'` seems to be the correct scheme on recent MacOS versions.
* Install a Fortran compiler using your favorite package manager (e.g., `brew install gcc`).  If the compiler has a standard name (e.g., `gfortran`) and is in your path, CMake will automatically find it.  Otherwise you can set the Fortran compiler with `-D CMAKE_Fortran_COMPILER=gfortran`).

Second, the MacOS toolchain does not include the OpenMP libraries.  If you want to enable OpenMP, you can install the libraries using your package manager, e.g., `brew install libomp`, and then tell CMake where they were installed, e.g., `-D OpenMP_ROOT=$(brew --prefix)/opt/libomp`.

Note, that CMake will automatically discover the BLAS/LAPACK libraries included natively in MacOS, so you don't need to specify them (unless you want to use something else).

## External builds of Kokkos/Kokkos-Kernels

For external builds of Kokkos and/or Kokkos Kernels, you must configure, build and install them first using their CMake files.  Once that is completed, you just configure GenTen to point to their corresponding installations via:
```
-D Kokkos_ROOT=/path/to/kokkos/install
-D KokkosKernels_ROOT=/path/to/kokkos-kernels/install
```
using the paths to their respective top-level installations.  You then do not need to supply any Kokkos-related configuration options when configuring GenTen.

## Building with Trilinos

To enable Trilinos support, you must use an external build of Trilinos and then specify the path to this installation via
```
-D Trilinos_DIR=path/to/trilinos/install/lib64/cmake/Trilinos \
```
CMake will determine which packages in Trilinos were enabled and enable the corresponding support in GenTen.  CMake can also deduce the compilers, BLAS/LAPACK libraries, and MPI support from the Trilinos build saving the need to specify these in the GenTen build as well.

# Testing GenTen

Once GenTen has been compiled, it can be tested by executing `ctest`.

# Using GenTen

The primary executable for GenTen is `bin/genten` in your build tree, which is a driver for reading in a tensor from a file and performing a CP decomposition of it.  The driver accepts numerous command line options controlling various aspects of the computation.  Run `genten --help` for a full listing.  For example
```
./bin/genten --input data/aminoacid_data.txt --rank 16 --output aa.ktns
```
will perform a rank 16 CP decomposition of the amino-acid tensor data set included with Genten in the data directory, and save the resulting factors in `aa.ktns`.  One should see output similar to:
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

For larger tensor datasets, consider those available from the [FROSTT](https://frost.io) collection.  Note that GenTen *does not* require a header at the top of the sparse tensor file indicating the number of modes, their dimensions, and the number of nonzeros.  Any textfile consisting of a list of nonzeros in coordinate format (i.e., nonzero indices and value) can be read.  If configured with Boost support, compressed tensors can be read directly without first decompressing them.  GenTen can also read sparse and dense tensors in a binary format that can be generated using the provided `bin/convert_tensor` utility (once a tensor has been converted to binary format, reading it into GenTen for decomposition is much faster and can be executed in parallel).

## Using the MATLAB interface

To use GenTen within MATLAB, you must first add the Tensor Toolbox and the path to the `matlab` directory in your GenTen build tree to your MATLAB path. GenTen provides a MATLAB class `sptensor_gt` that works similarly to the Tensor Toolbox sparse tensor class `sptensor` that can be passed to various MATLAB functions provided by Genten for manipulating the tensor.  A given tensor `X` in `sptensor` format can be converted to `sptensor_gt` format by calling the constructor:
```
>> X = sptenrand([10 20 30],100);
>> X_gt = sptensor_gt(X);
```
Genten then provides overloads of several functions in the Tensor Toolbox that call the corresponding implementation in Genten when passed a tensor in `sptensor_gt` format, e.g.,
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
Note that in addition to the normal options accepted by `cp_als`, all options accepted by the `genten` command-line driver (without the leading '--') are also accepted, e.g.,
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

## Using the Python interface

Examples of using the Python interface can be found in [python/example](python/example).  This interface is most useful when used in conjunction with pyttb, as pygenten provides limited support for manipulating tensors and their CP decompositions on its own.  If MPI support is enabled in the GenTen build, pygenten can also use distributed parallelism by launching python in parallel using `mpirun`/`mpiexec`.  GPU parallelism is also supported whereby Python tensor data will be copied to the GPU by GenTen within the Python implementation of the decomposition routine (e.g., `cp_als`) and the resulting CP model will be copied from the GPU when it is returned to Python.

# Updating bundled libraries (for developers)

GenTen uses `git subtree` to manage the bundled sources for the several bundled libraries it depends on.  Below is a summary of the steps required to update GenTen's clone to the latest sources using Kokkos as an example.

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
