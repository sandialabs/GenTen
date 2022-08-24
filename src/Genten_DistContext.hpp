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

namespace Genten {
bool InitializeGenten(int *argc, char ***argv);
bool FinalizeGenten();
}

#include "CMakeInclude.h"
#if defined(HAVE_DIST)

#include <cassert>
#include <memory>
#include <sstream>
#include <string>
#include <typeinfo>

#include <mpi.h>

#ifdef HAVE_BOOST
#include <boost/core/demangle.hpp>
#endif

#include "Genten_Util.hpp"
#include "Genten_Ptree.hpp"
#include "Genten_SmallVector.hpp"

namespace Genten {
std::stringstream debugInput();

namespace detail {
int bcastStr(std::string &s, int root);
} // namespace detail

struct DistContext {
  DistContext(DistContext const &) = delete;
  DistContext(DistContext &&) = delete;
  DistContext &operator=(DistContext const &) = delete;
  DistContext &operator=(DistContext &&) = delete;
  // Clean up MPI and Kokkos in the deleter
  ~DistContext();

  static int rank() {
    assert(instance_ != nullptr);
    return instance_->rank_;
  }

  static int nranks() {
    assert(instance_ != nullptr);
    return instance_->nranks_;
  }

  static bool initialized() { return instance_ != nullptr; }

  static MPI_Comm commWorld() {
    assert(instance_ != nullptr);
    return instance_->commWorld_;
  }

  static ptree const &input() {
    assert(instance_ != nullptr);
    return instance_->input_;
  }

  static bool isDebug() {
    assert(instance_ != nullptr);
    return instance_->input_.get<bool>("debug", false);
  }

  static void Barrier() {
    assert(instance_ != nullptr);
    MPI_Barrier(instance_->commWorld());
  }

  // This would be better with some if constexprs but I'll leave them out for
  // now
  template <typename T> static MPI_Datatype toMpiType() {
    if (std::is_same<T, double>::value) {
      return MPI_DOUBLE;
    }

    if (std::is_same<T, float>::value) {
      return MPI_FLOAT;
    }

    if (std::is_same<T, int>::value) {
      return MPI_INT;
    }

    if (std::is_same<T, long>::value) {
      return MPI_LONG;
    }

    if (std::is_same<T, long long>::value) {
      return MPI_LONG_LONG;
    }

    if (std::is_same<T, unsigned int>::value) {
      return MPI_UNSIGNED;
    }

    if (std::is_same<T, unsigned long>::value) {
      return MPI_UNSIGNED_LONG_LONG;
    }

    if (std::is_same<T, unsigned long long>::value) {
      return MPI_UNSIGNED_LONG_LONG;
    }

    if (std::is_same<T, char>::value) {
      return MPI_CHAR;
    }

    std::stringstream ss;
    ss << "Not able to convert type "
#ifdef HAVE_BOOST
       << boost::core::demangle(typeid(T).name())
#else
       << typeid(T).name()
#endif
       << " to an MPI_Datatype.";
    throw std::logic_error(ss.str());
  }

  // Bcasts that I don't want to figure out how to do the other way right now
  static int Bcast(small_vector<int> &t, int root);
  static int Bcast(std::size_t &t, int root);
  static int Bcast(ptree &t, int root);

private:
  int rank_;
  int nranks_;
  MPI_Comm commWorld_;
  ptree input_;
  friend bool InitializeGenten(int *argc, char ***argv);
  friend bool FinalizeGenten();
  DistContext() = default;
  static std::unique_ptr<DistContext> instance_;
};

} // namespace Genten

#else

namespace Genten {

struct DistContext {
  DistContext(DistContext const &) = delete;
  DistContext(DistContext &&) = delete;
  DistContext &operator=(DistContext const &) = delete;
  DistContext &operator=(DistContext &&) = delete;
  ~DistContext() = default;

  static int rank() { return 0; }

  static int nranks() { return 1; }

  static bool initialized() { return true; }

  static bool isDebug() { return false; }

  static void Barrier() {}

};

} // namespace Genten

#endif