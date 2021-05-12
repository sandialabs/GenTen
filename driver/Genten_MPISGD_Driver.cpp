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

#include "Genten_DistContext.hpp"
#include "Genten_DistSpSystem.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Pmap.hpp"
#include "Genten_TensorBlockSystem.hpp"

#include <Kokkos_Core.hpp>
#include <iostream>

namespace GT = Genten;

GT::Sptensor getSparseTensor();

void real_main(int argc, char **argv);

int main(int argc, char **argv) {
  GT::InitializeGenten(&argc, &argv);
  { real_main(argc, argv); }
  GT::FinalizeGenten();
  return 0;
}

void check_input() {
  auto &in = GT::DistContext::input();
  if (GT::DistContext::rank() == 0) {
    if (in.get_optional<std::string>("tensor.file") == boost::none) {
      throw std::logic_error{"Input must contain tensor.file"};
    }
  }
}

void real_main(int argc, char **argv) {
  try {
    check_input();
    const auto size = GT::DistContext::nranks();
    const auto rank = GT::DistContext::rank();

    if (rank == 0) {
      std::cout << "Running Geten-MPI-SGD with: " << size << " mpi-ranks\n";
      std::cout << "\tdecomposing file: "
                << GT::DistContext::input().get<std::string>("tensor.file")
                << "\n";
      std::cout << "\tusing method: "
                << GT::DistContext::input().get<std::string>("tensor.method")
                << std::endl;

      if(GT::DistContext::input().get<bool>("debug", false)){
        std::cout << "Input file: " << argv[1] << ":\n";
        std::ifstream json_input(argv[1]);
        if(json_input.is_open()){
          std::cout << json_input.rdbuf();
        }
        std::cout << std::flush;
      }
    }
    GT::DistContext::Barrier();

    GT::TensorBlockSystem<double, Kokkos::OpenMP> tbs(GT::DistContext::input());
    tbs.SGD();
  } catch (std::exception &e) {
    auto rank = GT::DistContext::rank();
    std::cerr << "Rank: " << rank << " " << e.what() << "\n";
    MPI_Abort(GT::DistContext::commWorld(), 0);
  }
}
