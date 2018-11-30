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

/* ----- Standard Include Statements ----- */
#include <iostream>
#include <string>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <limits>

#include "CMakeInclude.h"
#include "Genten_Kokkos.hpp"

namespace Genten {
  typedef Kokkos::DefaultExecutionSpace DefaultExecutionSpace;
  typedef Kokkos::DefaultHostExecutionSpace DefaultHostExecutionSpace;
}
#ifdef KOKKOS_HAVE_CUDA
#define GENTEN_INST_CUDA(INSTMACRO) \
  INSTMACRO(Kokkos::Cuda)
#else
#define GENTEN_INST_CUDA(INSTMACRO) /* */
#endif
#ifdef KOKKOS_HAVE_OPENMP
#define GENTEN_INST_OPENMP(INSTMACRO) \
  INSTMACRO(Kokkos::OpenMP)
#else
#define GENTEN_INST_OPENMP(INSTMACRO) /* */
#endif
#ifdef KOKKOS_HAVE_THREADS
#define GENTEN_INST_THREADS(INSTMACRO) \
  INSTMACRO(Kokkos::Threads)
#else
#define GENTEN_INST_THREADS(INSTMACRO) /* */
#endif
#ifdef KOKKOS_HAVE_SERIAL
#define GENTEN_INST_SERIAL(INSTMACRO) \
  INSTMACRO(Kokkos::Serial)
#else
#define GENTEN_INST_SERIAL(INSTMACRO) /* */
#endif

#define GENTEN_INST(INSTMACRO)                  \
namespace Genten {                              \
  GENTEN_INST_CUDA(INSTMACRO)                   \
  GENTEN_INST_OPENMP(INSTMACRO)                 \
  GENTEN_INST_THREADS(INSTMACRO)                \
  GENTEN_INST_SERIAL(INSTMACRO)                 \
}

// What we align memory to (in bytes)
#define GENTEN_MEMORY_ALIGNMENT 64

/* ----- Typedefs ----- */
/* We use typedefs to make the code portable, especially for
   varying sizes of integers, etc.
   Note that ttb_indx should always be an unsigned integer type.
   Note that ttb_real, ttb_indx are now set at configure time in CMakeInclude.h.
*/
typedef bool ttb_bool;

constexpr ttb_real MACHINE_EPSILON = std::numeric_limits<ttb_real>::epsilon();
constexpr ttb_real DOUBLE_MAX = std::numeric_limits<ttb_real>::max();
constexpr ttb_real DOUBLE_MIN = std::numeric_limits<ttb_real>::min();

/* ----- Enums ----- */
namespace Genten {

  enum NormType { NormOne, NormTwo, NormInf };

  // MTTKRP algorithm
  struct MTTKRP_Method {
    enum type {
      Default,     // Use default method based on architecture
      OrigKokkos,  // Use original Kokkos implementation
      Atomic,      // Use atomics factor matrix update
      Duplicated,  // Duplicate factor matrix then inter-thread reduce
      Single,      // Single-thread algorithm (no atomics or duplication)
      Perm         // Permutation-based algorithm
    };
    static constexpr unsigned num_types = 6;
    static constexpr type types[] = {
      Default,
      OrigKokkos,
      Atomic,
      Duplicated,
      Single,
      Perm
    };
    static constexpr const char* names[] = {
      "default", "orig-kokkos", "atomic", "duplicated", "single", "perm"
    };
    static constexpr type default_type = Default;

    template <typename ExecSpace>
    static type computeDefault() {
      typedef SpaceProperties<ExecSpace> space_prop;

      type method = Atomic;

      // Always use Single if there is only a single thread
      if (space_prop::concurrency() == 1)
        method = Single;

      // Use Atomic on Cuda if it supports fast atomics for ttb_real.
      // This is true with float on all arch's or float/double on Pascal (6.0)
      // or later
      else if (space_prop::is_cuda && (space_prop::cuda_arch() >= 600 ||
                                       sizeof(ttb_real) == 4))
        method = Atomic;

      // Otherwise use Perm
      else
        method = Perm;

      return method;
    }
  };

  // Loss functions supported by GCP
  struct GCP_LossFunction {
    enum type {
      Gaussian,
      Rayleigh,
      Gamma,
      Bernoulli,
      Poisson
    };
    static constexpr unsigned num_types = 5;
    static constexpr type types[] = {
      Gaussian, Rayleigh, Gamma, Bernoulli, Poisson
    };
    static constexpr const char* names[] = {
      "gaussian", "rayleigh", "gamma", "bernoulli", "poisson"
    };
    static constexpr type default_type = Gaussian;
  };
}

/* ----- Utility Functions ----- */
namespace Genten {

  // Throw an error.
  void error(std::string s);

  //! Return true if two real numbers are equal to a specified tolerance.
  /*!
   *  Numbers are equal if
   *
   *  <pre>
   *         fabs(d1 - d2)
   *    --------------------------  < dTol .
   *    max(1, fabs(d1), fabs(d2))
   *  </pre>
   *
   *  @param d1    1st real number
   *  @param d2    2nd real number
   *  @param dTol  Relative tolerance, must be >= 0
   *  @return      true if d1 and d2 are within dTol
   */
  bool  isEqualToTol(ttb_real  d1,
                     ttb_real  d2,
                     ttb_real  dTol);

  // Return the offcial version for this release of the Tensor Toolbox.
  char *  getGentenVersion(void);

  // Connect executable to vtune for profiling
  void connect_vtune(const int p_rank = 0);

  // Struct for passing various algorithmic parameters
  struct AlgParams {
    // MTTKRP algorithm
    MTTKRP_Method::type mttkrp_method;

    // Factor matrix tile size for MTTKRP_Duplicated algorithm
    unsigned mttkrp_duplicated_factor_matrix_tile_size;

    AlgParams() :
      mttkrp_method(MTTKRP_Method::default_type),
      mttkrp_duplicated_factor_matrix_tile_size(0)  // Use default choice
      {}
  };

  template <typename T>
  typename T::type parse_enum(const std::string& name) {
    for (unsigned i=0; i<T::num_types; ++i) {
      if (name == T::names[i])
        return T::types[i];
    }

    // if we got here, name wasn't found
    std::ostringstream error_string;
    error_string << "Invalid enum choice " << name
                 << ",  must be one of the values: ";
    for (unsigned i=0; i<T::num_types; ++i) {
      error_string << T::names[i];
      if (i != T::num_types-1)
        error_string << ", ";
    }
    error_string << "." << std::endl;
    Genten::error(error_string.str());
    return T::default_type;
  }

}
