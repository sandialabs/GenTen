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
//
#include "Genten_DistContext.hpp"

#if defined(HAVE_DIST)

#include <algorithm>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

namespace Genten {
namespace detail {
int bcastStr(std::string &str, int root) {
  int size = (DistContext::rank() == root) ? str.size() : 0;
  MPI_Bcast(&size, 1, MPI_INT, root, DistContext::commWorld());

  if (DistContext::rank() != root) {
    str.resize(size);
  }

  return MPI_Bcast(&str[0], size, MPI_CHAR, root, DistContext::commWorld());
}
} // namespace detail

namespace {
bool contains(std::string const &s, std::string const &target) {
  return s.find(target) != std::string::npos;
}

ptree readInput(std::string const &json_file) {
  ptree tree;
  if (DistContext::rank() == 0) {
    read_json(json_file, tree);
  }

  // Have rank 0 serialize the ptree and bcast it to all other ranks
  if (DistContext::Bcast(tree, 0) != MPI_SUCCESS) {
    printf("Error with Bcasting the input tree.\n");
    std::abort();
  }

  return tree;
}
} // namespace

int DistContext::Bcast(ptree &t, int root) {
  assert(instance_ != nullptr);
  if (DistContext::nranks() == 1 && root == 0) {
    return MPI_SUCCESS;
  }

  std::string s = t.dump();
  int ret = detail::bcastStr(s, root);
  if (DistContext::rank() != root)
    t.parse(s);

  return ret;
}

std::stringstream debugInput() {
  ptree in = DistContext::input();

  in.add<int>("mpi_ranks", DistContext::nranks());
  if (Kokkos::hwloc::available()) {
    in.add<int>("cores_per_numa",
                Kokkos::hwloc::get_available_cores_per_numa());
    in.add<int>("numa_count", Kokkos::hwloc::get_available_numa_count());
  }

  std::stringstream ss;
  write_json(ss, in);
  return ss;
}

bool InitializeGenten(int *argc, char ***argv) {
  static bool initialized = [&] {
    int provided = 0;
    MPI_Init_thread(argc, argv, MPI_THREAD_SINGLE, &provided);
    Kokkos::initialize(*argc, *argv);
    DistContext::instance_ = std::unique_ptr<DistContext>(new DistContext());
    auto &dc = *DistContext::instance_;

    MPI_Comm_dup(MPI_COMM_WORLD, &(dc.commWorld_));
    MPI_Comm_rank(dc.commWorld_, &(dc.rank_));
    MPI_Comm_size(dc.commWorld_, &(dc.nranks_));

    auto real_argv = *argv;
    for (auto i = 0; i < *argc; ++i) {
      if (contains(real_argv[i], ".json")) {
        dc.input_ = readInput(real_argv[i]);
      }
    }

    // Check for dump on command line
    for (auto i = 0; i < *argc; ++i) {
      if (std::string(real_argv[i]) == "--dump") {
        if(dc.input_.count("dump") == 0){
          dc.input_.add("dump", true);
        }
      }
    }

    // Check for debug on command line
    for (auto i = 0; i < *argc; ++i) {
      if (std::string(real_argv[i]) == "--debug") {
        if (dc.input_.count("debug") == 0) {
          dc.input_.add("debug", true);
        }
      }
    }

    return true;
  }();

  return initialized;
}

bool InitializeGenten() {
  if (DistContext::instance_ == nullptr) {
    int provided = 0;
    int argc = 0;
    char **argv = nullptr;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

    if (!Kokkos::is_initialized()) {
      Kokkos::InitializationSettings args;
      Kokkos::initialize(args);
    }

    DistContext::instance_ = std::unique_ptr<DistContext>(new DistContext());
    auto &dc = *DistContext::instance_;
    MPI_Comm_dup(MPI_COMM_WORLD, &(dc.commWorld_));
    MPI_Comm_rank(dc.commWorld_, &(dc.rank_));
    MPI_Comm_size(dc.commWorld_, &(dc.nranks_));
    return true; // we initialized
  }
  return false; // we did not initialize
}

bool FinalizeGenten() {
  MPI_Barrier(DistContext::commWorld());
  DistContext::instance_ = nullptr;
  return true;
}

DistContext::~DistContext() {
  MPI_Comm_free(&commWorld_);
  Kokkos::finalize();
  MPI_Finalize();
}

std::unique_ptr<DistContext> DistContext::instance_ = nullptr;
} // namespace Genten

#else

#include <Kokkos_Core.hpp>

namespace Genten {

bool InitializeGenten(int *argc, char ***argv) {
  Kokkos::initialize(*argc, *argv);
  return true;
}

bool InitializeGenten() {
  if (!Kokkos::is_initialized()) {
    Kokkos::InitializationSettings args;
    Kokkos::initialize(args);
  }
  return true;
}

bool FinalizeGenten() {
  Kokkos::finalize();
  return true;
}

}

#endif
