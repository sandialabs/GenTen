# pygenten:  Python bindings for the GenTen package

The python package pygenten provides python bindings for the [GenTen](https://github.com/sandialabs/GenTen) package.  GenTen is a tool for computing Canonical Polyadic (CP, also called CANDECOMP/PARAFAC) decompositions of tensor data.  It is geared towards analysis of extreme-scale data and implements several CP decomposition algorithms that are parallel and scalable, including:
* CP-ALS:  The workhorse algorithm for Gaussian sparse or dense tensor data.
* [CP-OPT](https://doi.org/10.1002/cem.1335):  CP decomposition of (sparse or dense) Gaussian data using a quasi-Newton optimization algorithm incorporating possible upper and lower bound constraints.
* [GCP](https://epubs.siam.org/doi/abs/10.1137/18M1203626):  Generalized CP supporting arbitrary loss functions (Gaussian, Poisson, Bernoulli, ...), solved using [quasi-Newton](https://epubs.siam.org/doi/abs/10.1137/18M1203626) (dense tensors) or [stochastic gradient descent](https://doi.org/10.1137/19M1266265) (sparse or dense tensors) optimization methods.
* [Streaming GCP](https://doi.org/10.1145/3592979.3593405): A GCP algorithm that incrementally updates a GCP decomposition as new data is observed, suitable for in situ analysis of streaming data.
* Federated GCP:  A federated learning algorithm for GCP supporting asynchronous parallel communication.

GenTen builds on [Kokkos](https://github.com/kokkos/kokkos) and [Kokkos Kernels](https://github.com/kokkos/kokkos-kernels) to support shared memory parallel programming models on a variety of contemporary architectures, including:
* OpenMP for CPUs.
* CUDA for NVIDIA GPUs.
* HIP for AMD GPUs.
* SYCL for Intel GPUs.

GenTen also supports distributed memory parallelism using MPI.

# Installing pygenten

There are two general approaches for building pygenten:
* Enable python in the generic CMake build process described [here](https://github.com/sandialabs/GenTen#installing-genten).
* Install using pip which automates the CMake build to some degree.

## Installing with pip

pygenten has experimental support for installation using pip from the source distribution on [pypi](https://pypi.org/project/pygenten/).  Furthermore, binary wheels are provided in limited circumstances (currently just linux with OpenMP support only, but more may be provided in the future), enabling immediate installation.  The pip installation leverages [scikit-build-core](https://github.com/scikit-build/scikit-build-core) to provide a CMake build backend for pip, which allows the user to provide CMake defines that control the pygenten build process and determine which architectures/parallel programming models are enabled.  We thus recommend becoming familiar with the CMake build process for GenTen in general as described [here](https://github.com/sandialabs/GenTen#installing-genten) before continuing.  In particular, the user must have BLAS and LAPACK libraries available in their build environment that can either be automatically discovered by CMake or manually specified through `LAPACK_LIBS`.

### Basic installation

A basic installation of pygenten can be done simply by:
```
pip install pygenten
```
This will install the binary wheel if it is available, and if it isn't, build GenTen and pygenten for a CPU architecture using OpenMP parallelism using a default compiler from the user's path.  During the build of pygenten, CMake will attempt to locate valid BLAS and LAPACK libraries in the user environment.  If these cannot be found, the user can customize the build by specifying `LAPACK_LIBS` as described below.

**Note that when installing pygenten from a binary wheel, the `repairwheel` step that makes the wheel usable on a wide variety of architectures seems to make the included `genten` and related executables unusable.  If you want to use GenTen outside of python, you should install it from source as described below.**

### Customized installation

To customize the GenTen/pygenten build, you must first instruct pip to compile from source by adding the `--no-binary pygenten` command-line argument.  The can then be customized by passing CMake defines to specify compilers, BLAS/LAPACK libraries, host/device architectures, and enabled programming models.  This is done by adding command-line arguments to pip of the form
```
--config-settings=cmake.define.SOME_DEFINE=value
```
Any CMake define accepted by GenTen/Kokkos/KokkosKernels can be passed this way.  Since this is fairly verbose and GenTen can require several defines, several meta-options are provided to enable supported parallel programming models:
| CMake Define    | What it enables |
| --------------- | --------------- |
| PYGENTEN_MPI    | Enable distributed parallelism with MPI.  Sets the execution space to Serial by default. |
| PYGENTEN_OPENMP | Enable shared memory host parallelism using OpenMP |
| PYGENTEN_SERIAL | No host shared memory parallelism.  Useful for builds targeting GPU architectures or distributed memory parallelism |
| PYGENTEN_CUDA   | Enable CUDA parallelism for NVIDIA GPU architectures |
| PYGENTEN_HIP    | Enable HIP parallelism for AMD GPU architectures |
| PYGENTEN_SYCL   | Enable SYCL parallelism for Intel GPU architectures |

When enabling GPU architectures, one also needs to specify the corresponding architecture via `Kokkos_ARCH_*` defines described [here](https://kokkos.org/kokkos-core-wiki/keywords.html#architectures). 

For example, an MPI+CUDA build for a Volta V100 GPU architecture can be obtained with
```
pip install -v --no-binary pygenten --config-settings=cmake.define.PYGENTEN_CUDA=ON --config-settings=cmake.define.Kokkos_ARCH_VOLTA70=ON --config-settings=cmake.define.PYGENTEN_MPI=ON pygenten
```
For MPI builds, pygenten assumes the MPI compiler wrappers `mpicxx` and `mpicc` are available in the user's path.  If this is not correct, the user can specify the appropriate compiler by setting the appropriate CMake define, e.g., `CMAKE_CXX_COMPILER`.  Furthermore, for CUDA builds, pygenten will build with the `nvcc_wrapper` script as the compiler as required by Kokkos, which calls `g++` as the host compiler by default.  This can be changed by setting the `NVCC_WRAPPER_DEFAULT_COMPILER` environment variable.  Moreover, for MPI+CUDA, pygenten will set environment variables to override the compiler wrapped by `mpicxx` to use `nvcc_wrapper`, which currently works only with OpenMPI and MPICH.  Finally, for MPI+HIP or MPI+SYCL builds, pygenten assumes the compiler wrappers call the appropriate device-enabled compiler, e.g., `hipcc` for AMD and `icpx` for Intel.

# Installing numpy

pygenten relies on numpy, both of which are compiled extension libraries leveraging OpenMP and BLAS/LAPACK.  Therefore, module import errors can occur if pygenten and numpy are compiled in very different environments, due to, e.g., symbol conflicts in `libstdc++`.  This typically happens when pygenten is compiled with a much newer compiler than what was used to compile the numpy wheel.  Futhermore, we have observed slower performance in pygenten in some cases when numpy is imported before pygenten, which we believe is due to inconsistent OpenMP and/or BLAS/LAPACK libraries between the two packages.  Thus, the most robust way to use pygenten is to also install numpy from source, using the same build environment as pygenten.  This can be done in a similar manner as pygenten by providing configure options to numpy through pip, e.g.,
```
pip install --no-binary numpy -Csetup-args=-Dblas=my_blas -Csetup-args=-Dlapack=my_lapack numpy
```
to specify the appropriate BLAS and LAPACK libraries (called `my_blas` and `my_lapack` in this case).  Compilers can be specified through the `CC`, `CXX`, and `FC` environment variables.  More details can be found [here](https://numpy.org/doc/stable/building/compilers_and_options.html).  However, we only recommend doing this if you see errors when importing pygenten or you observe slower performance than what would be observed with the `genten` command-line tool.
