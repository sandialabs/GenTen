# Porting GenTen to SYCL - Developer diary

## Compilation with CUDA Thrust
1. I didn't find any example of Nvidia Thrust and SYCL combination
2. Trying to use Thrust and SYCL combo in GenTen ends up with a compilation error:
```bash
[build] [30/55  12% :: 19.976] Building CXX object CMakeFiles/gentenlib.dir/src/Genten_GCP_SS_Grad_SA_Poisson.cpp.o
[build] FAILED: CMakeFiles/gentenlib.dir/src/Genten_GCP_SS_Grad_SA_Poisson.cpp.o
[build] ccache /opt/sycl/bin/clang++ -DENABLE_SYCL_WITH_CUDA -DKOKKOS_DEPENDENCE -DLAPACK_FOUND -Dgentenlib_EXPORTS -I../../src -I../../src/mathlib -I../../src/rol -I../../src/lbfgsb -ICMakeInclude -I../../driver -I../../tpls/lbfgsb -Itpls/kokkos -Itpls/kokkos/core/src -I../../tpls/kokkos/core/src -Itpls/kokkos/containers/src -I../../tpls/kokkos/containers/src -Itpls/kokkos/algorithms/src -I../../tpls/kokkos/algorithms/src -Wno-unknown-cuda-version -Wno-gnu-zero-variadic-macro-arguments -Wno-deprecated-declarations -Wno-linker-warnings --cuda-path=/usr/local/cuda-11.6/ -g -fPIC -fsycl -fno-sycl-id-queries-fit-in-int -fsycl-unnamed-lambda -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -std=c++17 -MD -MT CMakeFiles/gentenlib.dir/src/Genten_GCP_SS_Grad_SA_Poisson.cpp.o -MF CMakeFiles/gentenlib.dir/src/Genten_GCP_SS_Grad_SA_Poisson.cpp.o.d -o CMakeFiles/gentenlib.dir/src/Genten_GCP_SS_Grad_SA_Poisson.cpp.o -c ../../src/Genten_GCP_SS_Grad_SA_Poisson.cpp
[build] clang-15: warning: argument 'nvptx64-nvidia-cuda-sycldevice' is deprecated, use 'nvptx64-nvidia-cuda' instead [-Wdeprecated]
[build] In file included from ../../src/Genten_GCP_SS_Grad_SA_Poisson.cpp:41:
[build] In file included from ../../src/Genten_GCP_SS_Grad_SA_Def.hpp:47:
[build] In file included from ../../src/Genten_KokkosAlgs.hpp:60:
[build] In file included from /usr/local/cuda-11.6//include/thrust/sort.h:1358:
[build] In file included from /usr/local/cuda-11.6//include/thrust/detail/sort.inl:26:
[build] In file included from /usr/local/cuda-11.6//include/thrust/system/detail/generic/sort.h:152:
[build] /usr/local/cuda-11.6//include/thrust/system/detail/generic/sort.inl:190:3: error: static_assert failed due to requirement 'thrust::detail::depend_on_instantiation<thrust::device_ptr<unsigned long>, false>::value' "unimplemented for this system"
[build]   THRUST_STATIC_ASSERT_MSG(
[build]   ^
[build] /usr/local/cuda-11.6//include/thrust/detail/static_assert.h:50:44: note: expanded from macro 'THRUST_STATIC_ASSERT_MSG'
[build] #  define THRUST_STATIC_ASSERT_MSG(B, msg) static_assert(B, msg)
[build]                                            ^             ~
[build] /usr/local/cuda-11.6//include/thrust/detail/sort.inl:82:10: note: in instantiation of function template specialization 'thrust::system::detail::generic::stable_sort<thrust::cuda_cub::tag, thrust::device_ptr<unsigned long>, (lambda at ../../src/Genten_KokkosAlgs.hpp:130:22)>' requested here
[build]   return stable_sort(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, comp);
[build]          ^
[build] /usr/local/cuda-11.6//include/thrust/detail/sort.inl:261:18: note: in instantiation of function template specialization 'thrust::stable_sort<thrust::cuda_cub::tag, thrust::device_ptr<unsigned long>, (lambda at ../../src/Genten_KokkosAlgs.hpp:130:22)>' requested here
[build]   return thrust::stable_sort(select_system(system), first, last, comp);
[build]                  ^
[build] ../../src/Genten_KokkosAlgs.hpp:103:13: note: in instantiation of function template specialization 'thrust::stable_sort<thrust::device_ptr<unsigned long>, (lambda at ../../src/Genten_KokkosAlgs.hpp:130:22)>' requested here
[build]     thrust::stable_sort(thrust::device_ptr<perm_val_type>(perm.data()),
[build]             ^
[build] ../../src/Genten_KokkosAlgs.hpp:130:3: note: in instantiation of function template specialization 'Genten::perm_sort_op<Kokkos::View<unsigned long *, Kokkos::Experimental::SYCL>, (lambda at ../../src/Genten_KokkosAlgs.hpp:130:22)>' requested here
[build]   perm_sort_op(perm, KOKKOS_LAMBDA(const size_type& a, const size_type& b)
[build]   ^
[build] ../../src/Genten_GCP_SS_Grad_SA_Def.hpp:338:17: note: in instantiation of function template specialization 'Genten::perm_sort<Kokkos::View<unsigned long *, Kokkos::Experimental::SYCL>, Kokkos::View<unsigned long *, Kokkos::LayoutLeft, Kokkos::Experimental::SYCL>>' requested here
[build]         Genten::perm_sort(perm, Gind_n);
[build]                 ^
```
3. As an alternative to the Thrust, we could use SYCL Parallel STL (https://github.com/KhronosGroup/SyclParallelSTL), but at the moment, it doesn't implement `stable_sort`. How to use it - https://www.khronos.org/assets/uploads/developers/library/2016-supercomputing/SYCL_ParallelSTL-Ruyman-Reyes_Nov16.pdf
4. Also, it's possible to use `std::execution::parallel_policy` for STL algorithms.
5. For now, I'm going to use Kokkos::sort.

## Building GenTen + SYCL on AWS
1. EC2 instance type - G4dn.xlarge, Ubuntu 20.04
2. requirements: GCC, CMake, CUDA - https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation, https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local
3. `nvidia-cuda-toolkit` shouldn't be installed - it comes with CUDA 10, which doesn't support newer GCC versions.
4. add CUDA to PATH `export PATH=$PATH:/usr/local/cuda-11.6/bin`
5. At the moment on AWS Kokkos architecture should be `Kokkos_ARCH_TURING75=ON`
6. Configuring and building Kokkos for CUDA:
```bash
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=../install_kokkos -D Kokkos_ENABLE_EXAMPLES=OFF -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_CUDA_LAMBDA=ON -D Kokkos_ENABLE_TESTS=ON -D Kokkos_ARCH_TURING75=ON ../kokkos
make -j"$(nproc)"
```
7. Both Kokkos and GenTen compiled on AWS with CUDA work just fine.
8. For SYCL compiler installation on AWS cloud, `python` and `ninja-build` are required.
9. Installation can be done with `docs/sycl/install-dpc++.sh <dpc++-version>` script. Available versions can be found here -> https://github.com/intel/llvm/releases. For example, to install version `DPC++ daily 2022-05-10`: `./install-dpc++.sh 20220510`.
10. After installation do `export LD_LIBRARY_PATH=/opt/sycl/lib/`
11. Configuring Kokkos + SYCL:
```bash
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_C_COMPILER=/opt/sycl/bin/clang -D CMAKE_CXX_COMPILER=/opt/sycl/bin/clang++ -D CMAKE_CXX_FLAGS="-Wno-unknown-cuda-version -Wno-gnu-zero-variadic-macro-arguments -Wno-deprecated-declarations -Wno-linker-warnings" -D CMAKE_CXX_STANDARD=17 -D CMAKE_INSTALL_PREFIX=../install_kokkos_sycl -D Kokkos_ARCH_TURING75=ON -D Kokkos_ENABLE_COMPILER_WARNINGS=ON -D Kokkos_ENABLE_DEPRECATED_CODE_3=ON -D Kokkos_ENABLE_DEPRECATION_WARNINGS=OFF -D Kokkos_ENABLE_EXAMPLES=OFF -D Kokkos_ENABLE_SYCL=ON -D Kokkos_ENABLE_TESTS=ON -D Kokkos_ENABLE_UNSUPPORTED_ARCHS=ON ../kokkos
```
12. Kokkos compiled on AWS with SYCL for CUDA works just fine.
13. Configuring GenTen + SYCL:
```bash
cmake -D BUILD_SHARED_LIBS=ON -D CMAKE_BUILD_TYPE=Release -D CMAKE_C_COMPILER=/opt/sycl/bin/clang -D CMAKE_CXX_COMPILER=/opt/sycl/bin/clang++ -D CMAKE_CXX_FLAGS="-Wno-unknown-cuda-version -Wno-gnu-zero-variadic-macro-arguments -Wno-deprecated-declarations -Wno-linker-warnings" -D GENTEN_ENABLE_SYCL_WITH_CUDA=ON -D Kokkos_ARCH_PASCAL61=ON -D Kokkos_ENABLE_DEPRECATED_CODE_3=ON -D Kokkos_ENABLE_DEPRECATION_WARNINGS=OFF -D Kokkos_ENABLE_SYCL=ON -D Kokkos_ENABLE_UNSUPPORTED_ARCHS=ON -D LIBCUBLAS_PATH=/usr/local/cuda-11.6/lib64 -D LIBCUSOLVER_PATH=/usr/local/cuda-11.6/lib64 ../genten
```

## Failing GenTen's unit tests
1. GenTen unit tests fail just at the beginning
```bash
❯ ./unit_tests 1
----------------------------------------------------
Tests on Genten::TTM (sycl)
----------------------------------------------------
INFO: preparing to calculate ttm:  5x2 * 3x4x2x2 along mode 0 (File: ../../test/Genten_Test_TTM.cpp, Line: 162)
INFO: Testing default DGEMM ttm along mode: 0 (File: ../../test/Genten_Test_TTM.cpp, Line: 89)
genten_ttm_batched_cublas
PASS: DGEMM (File: ../../test/Genten_Test_TTM.cpp, Line: 93)
INFO: Testing Parfor_DGEMM ttm along mode: 0 (File: ../../test/Genten_Test_TTM.cpp, Line: 94)
PASS: Parfor_DGEMM (File: ../../test/Genten_Test_TTM.cpp, Line: 99)
INFO: Testing parfor dgemm along mode: 0 (File: ../../test/Genten_Test_TTM.cpp, Line: 102)
PASS: parfor_dgemm (File: ../../test/Genten_Test_TTM.cpp, Line: 111)
INFO: Testing serial dgemm along mode: 0 (File: ../../test/Genten_Test_TTM.cpp, Line: 113)
PASS: parfor_dgemm (File: ../../test/Genten_Test_TTM.cpp, Line: 118)
INFO: preparing to calculate ttm:  5x4 * 3x4x2x2 along mode 1 (File: ../../test/Genten_Test_TTM.cpp, Line: 212)
INFO: Testing default DGEMM ttm along mode: 1 (File: ../../test/Genten_Test_TTM.cpp, Line: 89)
genten_ttm_batched_cublas
Z[0]: 0   unit_test: 210
Z[0]: 0   unit_test: 210
```

After some digging and print debugging, it turns out that printing tensor at the end of the `genten_ttm_batched_cublas()` makes a test passing. It seems like a problem with compiler and machine code generation.

Next problematic test - `Tests on Genten::FacMatrix (sycl)` - ends with segfault:
```bash
----------------------------------------------------
Tests on Genten::FacMatrix (sycl)
----------------------------------------------------
INFO: Testing empty constructor (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 62)
PASS: Empty constructor works as expected (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 64)
INFO: Testing size constructor (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 67)
PASS: Size constructor works as expected (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 69)
INFO: Testing data constructor (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 72)
PASS: Data constructor works as expected (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 89)
INFO: Copy constructor not tested explicitly (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 92)
INFO: Destructor is not tested explicitly (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 95)
PASS: Entry for const works as expected (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 101)
INFO: Entry for non-const not tested explicitly (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 105)
PASS: Resize works as expected (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 109)
PASS: Resize works as expected (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 110)
PASS: Operator= for scalar works as expected (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 114)
PASS: Operator= for scalar works as expected (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 115)
PASS: Operator= for scalar works as expected (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 116)
PASS: Operator= for scalar works as expected (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 117)
PASS: Reset works as expected (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 122)
PASS: Reset works as expected (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 123)
PASS: Reset works as expected (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 124)
PASS: Gramian works (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 137)
PASS: Gramian works (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 138)
FAIL: Gramian yields expected answer (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 157)
INFO: Checking error on wrong sized matrices (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 160)
PASS: Expected exception is caught (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 172)
INFO: Now checking actual correctness (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 174)
PASS: Hadamard works (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 181)
PASS: Hadamard works (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 182)
PASS: Hadamard works as expected (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 202)
PASS: Transpose # columns ok (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 207)
PASS: Transpose # rows ok (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 208)
PASS: Transpose works as expected (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 226)
PASS: Oprod # rows ok (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 238)
PASS: Oprod # cols ok (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 239)
PASS: Oprod works as expected (File: ../../test/Genten_Test_FacMatrix.cpp, Line: 258)
Segmentation fault (core dumped)
```

Segfault comes from a second call to `dsysv()` from `Genten::sysv()`.
