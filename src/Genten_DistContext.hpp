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
#include "Genten_Boost.hpp"
#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <assert.h>
#include <memory.h>
#include <sstream>
#include <string>

namespace Genten {
bool InitializeGenten(int *argc, char ***argv);
bool FinalizeGenten();

namespace detail {
template <typename T> std::string serializeToStr(T const &t);
template <typename T> T deserializeFromStr(std::string const &s);

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

  static MPI_Comm commWorld() {
    assert(instance_ != nullptr);
    return instance_->commWorld_;
  }

  static ptree const &input() {
    assert(instance_ != nullptr);
    return instance_->input_;
  }

  // TODO we might want to specialize this function to be more efficent for
  // specific types, like say a Kokkos View ...
  template <typename T> static int Bcast(T &t, int root) {
    assert(instance_ != nullptr);
    if (DistContext::nranks() == 1 && root == 0) {
      return MPI_SUCCESS;
    }

    int bcast_result = MPI_SUCCESS;
    if (DistContext::rank() == root) {
      auto data = detail::serializeToStr(t);
      bcast_result = detail::bcastStr(data, root);
    } else {
      std::string data;
      bcast_result = detail::bcastStr(data, root);
      t = detail::deserializeFromStr<T>(data);
    }
    return bcast_result;
  }

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

namespace detail {
template <typename T> std::string serializeToStr(T const &t) {
  std::ostringstream oss;
  boost::archive::binary_oarchive ar(oss);
  ar << t;
  return oss.str();
}

template <typename T> T deserializeFromStr(std::string const &s) {
  std::stringstream ss;
  ss.write(s.data(), s.size());
  T t{};
  boost::archive::binary_iarchive oi(ss);
  oi >> t;
  return t;
}

} // namespace detail

} // namespace Genten
