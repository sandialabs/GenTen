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
#define KOKKOS_MEMORY_ALIGNMENT_THRESHOLD 0

#include "Kokkos_Core.hpp"

namespace Genten {

  // A helper trait to determine whether an execution space is Cuda or not,
  // and his always defined, regardless if Cuda is enabled.
  template <typename ExecSpace>
  struct is_cuda_space {
    static const bool value = false;
  };

#if defined(KOKKOS_HAVE_CUDA)
  template <>
  struct is_cuda_space<Kokkos::Cuda> {
    static const bool value = true;
  };
#endif

  // A helper trait to determine whether an execution space is Serial or not,
  // and his always defined, regardless if Serial is enabled.
  template <typename ExecSpace>
  struct is_serial_space {
    static const bool value = false;
  };

#if defined(KOKKOS_HAVE_SERIAL)
  template <>
  struct is_serial_space<Kokkos::Serial> {
    static const bool value = true;
  };
#endif

}
