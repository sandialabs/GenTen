//@HEADER
// ************************************************************************
//     Genten: Software for Generalized Tensor Decompositions
//     by Sandia National Laboratories
//
// Sandia National Laboratories is a multimission laboratory managed
// and operated by National Technology and Engineering Solutions of Sandia,
// LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
// U.S. Department of Energy's National Nuclear Security Administration under
// contract DE-NA0003525.
//
// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// ************************************************************************
//@HEADER


#pragma once

// Include this file whenever Kokkos needs to be included, to make sure any
// Kokkos-related definitions are consistent.

// Force Kokkos to always pad views when padding is enabled for that view,
// regardless of view dimensions
#ifdef KOKKOS_MEMORY_ALIGNMENT_THRESHOLD
#undef KOKKOS_MEMORY_ALIGNMENT_THRESHOLD
#endif
#define KOKKOS_MEMORY_ALIGNMENT_THRESHOLD 0

#include "Kokkos_Core.hpp"

namespace Genten {

  // A helper trait to determine whether an execution space is Serial or not,
  // and his always defined, regardless if Serial is enabled.
  template <typename ExecSpace>
  struct is_serial_space {
    static const bool value = false;
  };

  // A helper trait to determine whether an execution space is OpenMP or not,
  // and his always defined, regardless if OpenMP is enabled.
  template <typename ExecSpace>
  struct is_openmp_space {
    static constexpr bool value = false;
  };

  // A helper trait to determine whether an execution space is Threads or not,
  // and his always defined, regardless if Threads is enabled.
  template <typename ExecSpace>
  struct is_threads_space {
    static constexpr bool value = false;
  };

  // A helper trait to determine whether an execution space is Cuda or not,
  // and his always defined, regardless if Cuda is enabled.
  template <typename ExecSpace>
  struct is_cuda_space {
    static constexpr bool value = false;
  };

  // A helper trait to determine the cuda architecture,
  // and his always defined, regardless if Cuda is enabled.
  template <typename ExecSpace>
  struct cuda_device_arch {
    static typename ExecSpace::size_type eval() { return 0; }
  };

  // A helper trait to determine whether an execution space is HIP or not,
  // and his always defined, regardless if HIP is enabled.
  template <typename ExecSpace>
  struct is_hip_space {
    static constexpr bool value = false;
  };

  // A helper trait to determine whether an execution space is SYCL or not,
  // and his always defined, regardless if SYCL is enabled.
  template <typename ExecSpace>
  struct is_sycl_space {
    static constexpr bool value = false;
  };

  // A helper trait to determine whether an execution space is GPU or not,
  // and his always defined, regardless if any GPU is enabled.
  template <typename ExecSpace>
  struct is_gpu_space {
    static constexpr bool value =
      is_cuda_space<ExecSpace>::value ||
      is_hip_space<ExecSpace>::value ||
      is_sycl_space<ExecSpace>::value;
  };

  template <typename ExecSpace>
  struct is_host_space {
    static constexpr bool value =
      is_serial_space<ExecSpace>::value ||
      is_threads_space<ExecSpace>::value ||
      is_openmp_space<ExecSpace>::value;
  };

#if defined(KOKKOS_ENABLE_SERIAL)
  template <>
  struct is_serial_space<Kokkos::Serial> {
    static constexpr bool value = true;
  };
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
  template <>
  struct is_openmp_space<Kokkos::OpenMP> {
    static constexpr bool value = true;
  };
#endif

#if defined(KOKKOS_ENABLE_THREADS)
  template <>
  struct is_threads_space<Kokkos::Threads> {
    static constexpr bool value = true;
  };
#endif

#if defined(KOKKOS_ENABLE_CUDA)
  template <>
  struct is_cuda_space<Kokkos::Cuda> {
    static constexpr bool value = true;
  };

  template <>
  struct cuda_device_arch<Kokkos::Cuda> {
    static typename Kokkos::Cuda::size_type eval() {
      return Kokkos::Cuda::device_arch();
    }
  };

  using Kokkos_GPU_Space = Kokkos::Cuda;
#endif

#if defined(KOKKOS_ENABLE_HIP)
  template <>
  struct is_hip_space<Kokkos::Experimental::HIP> {
    static constexpr bool value = true;
  };

  using Kokkos_GPU_Space = Kokkos::Experimental::HIP;
#endif

#if defined(KOKKOS_ENABLE_SYCL)
  template <>
  struct is_sycl_space<Kokkos::Experimental::SYCL> {
    static constexpr bool value = true;
  };

  using Kokkos_GPU_Space = Kokkos::Experimental::SYCL;
#endif

  // Set of traits and functions for inquiring about the execution space
  template <typename ExecSpace>
  struct SpaceProperties {
    using exec_space = ExecSpace;
    using size_type  = typename exec_space::size_type;

    static constexpr bool is_serial = is_serial_space<exec_space>::value;
    static constexpr bool is_openmp = is_openmp_space<exec_space>::value;
    static constexpr bool is_threads = is_threads_space<exec_space>::value;
    static constexpr bool is_cuda = is_cuda_space<exec_space>::value;
    static constexpr bool is_hip = is_hip_space<exec_space>::value;
    static constexpr bool is_sycl = is_sycl_space<exec_space>::value;
    static constexpr bool is_gpu = is_cuda || is_hip || is_sycl;

    static std::string name() {
      if (is_serial)
        return "serial";
      else if (is_openmp)
        return "openmp";
      else if (is_threads)
        return "threads";
      else if (is_cuda)
        return "cuda";
      else if (is_hip)
        return "hip";
      else if (is_sycl)
        return "sycl";
      return "";
    }

    static std::string verbose_name() {
      std::string str = name();
#ifdef KOKKOS_ENABLE_CUDA
      if (is_cuda) {
        Kokkos::Cuda cuda;
        auto prop = cuda.cuda_device_prop();
        str += " (device " + std::to_string(cuda.cuda_device()) + ", " +
          prop.name + ")";
      }
#endif

#ifdef KOKKOS_ENABLE_HIP
      if (is_hip) {
        auto prop = Kokkos::Experimental::HIP::hip_device_prop();
        str +=
          " (device " +
          std::to_string(Kokkos::Experimental::HIP{}.hip_device()) +
          ", " + prop.name + ")";
      }
#endif

#ifdef KOKKOS_ENABLE_SYCL
      if (is_sycl) {
        Kokkos::Experimental::SYCL sycl;
        auto sycl_device =
          sycl.impl_internal_space_instance()->m_queue->get_device();
        str +=
          " (device " + sycl_device.get_info<sycl::info::device::name>() + ")";
      }
#endif

      if (is_openmp || is_threads)
        str += " (" + std::to_string(concurrency()) + " threads)";

      // We don't add any details for serial
      return str;
    }

    // Level of concurrency (i.e., threads) supported by the architecture
    static size_type concurrency() {
      using Kokkos::Experimental::UniqueToken;
      using Kokkos::Experimental::UniqueTokenScope;
      UniqueToken<exec_space, UniqueTokenScope::Global> token;
      return token.size();
    }

    // The Cuda architecture type (e.g., 35, 61, 70, ...)
    static size_type cuda_arch() {
      return cuda_device_arch<exec_space>::eval();
    }
  };

  // Our own versions of atomic_oper_fetch() suitable for using a lambda
  // for the operator

  // 32-bit version
  template < class Oper, typename T >
  KOKKOS_INLINE_FUNCTION
  T atomic_oper_fetch(
    const Oper& op, volatile T * const dest ,
    typename std::enable_if< sizeof(T) == sizeof(int), const T >::type& val )
  {
    union { int i ; T t ; } oldval , assume , newval ;

    oldval.t = *dest ;

    do {
      assume.i = oldval.i ;
      newval.t = op(assume.t, val) ;
      oldval.i =
        Kokkos::atomic_compare_exchange( (int*)dest , assume.i , newval.i );
    } while ( assume.i != oldval.i );

    return newval.t ;
  }

  // 64-bit version
  template < class Oper, typename T >
  KOKKOS_INLINE_FUNCTION
  T atomic_oper_fetch(
    const Oper& op, volatile T * const dest ,
    typename std::enable_if<
      sizeof(T) != sizeof(int) && sizeof(T) == sizeof(unsigned long long int) ,
      const T >::type& val )
  {
    union { unsigned long long int i ; T t ; } oldval , assume , newval ;

    oldval.t = *dest ;

    do {
      assume.i = oldval.i ;
      newval.t = op(assume.t, val) ;
      oldval.i = Kokkos::atomic_compare_exchange( (unsigned long long int*)dest , assume.i , newval.i );
    } while ( assume.i != oldval.i );

    return newval.t ;
  }

}
