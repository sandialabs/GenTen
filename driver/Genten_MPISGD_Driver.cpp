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
#include "Genten_DistTensorContext.hpp"
#include "Genten_DistGCP.hpp"
#include "Genten_DistCpAls.hpp"

namespace GT = Genten;

GT::Sptensor getSparseTensor();

void real_main(int argc, char **argv);

int main(int argc, char **argv) {
  GT::InitializeGenten(&argc, &argv);

  std::size_t output_signal = 0;
  if (GT::DistContext::rank() == 0) {
    std::string help_output =
        "Input to the mpi driver is a single json file (optionally --dump or "
        "--debug "
        "maybe passed as a command line flags instead of changing the json "
        "file). At the top "
        "level there are three arguments:\n\tdebug: a boolean that "
        "turns on extra printing if true.\n\ttensor: a json object "
        "which controls the tensor decomposition.\n\tdump: a boolean that asks "
        "the requested method to dump possible input values instead of running "
        "the calculation.\n";

    if (argc < 2) {
      std::cout << help_output;
      output_signal = 1; // Need to return non-zero
    } else if (std::string(argv[1]) == "-h" ||
               std::string(argv[1]) == "--help") {
      std::cout << help_output;
      output_signal = 2; // Need to return early, but not error
    }
    GT::DistContext::Bcast(output_signal, 0);
  } else {
    GT::DistContext::Bcast(output_signal, 0);
  }

  switch (output_signal) {
  case 0:
    break;
  case 2:
    GT::FinalizeGenten();
    return 0;
  default:
    GT::FinalizeGenten();
    return -1; // If we get unknown what can we do
  }

  { real_main(argc, argv); }
  GT::FinalizeGenten();
  return 0;
}

void check_input() {
  auto const &in = GT::DistContext::input();
  if (GT::DistContext::rank() == 0) {
    if (!in.contains("tensor.file")) {
      throw std::logic_error{"Input must contain tensor.file"};
    }
  }
}

void real_main(int argc, char **argv) {
  try {
    check_input();
    const auto size = GT::DistContext::nranks();
    const auto rank = GT::DistContext::rank();

    std::string solver_method = GT::DistContext::input().get<std::string>("solver-method","gcp");

    if (rank == 0) {
      std::cout << "Running Geten-MPI-SGD with: " << size << " mpi-ranks\n";
      std::cout << "\tdecomposing file: "
                << GT::DistContext::input().get<std::string>("tensor.file")
                << "\n";
      std::cout << "\tusing solution method: "
                << solver_method
                << std::endl;

      if (GT::DistContext::isDebug()) {
        std::cout << "Input file: " << argv[1] << ":\n";
        auto ss = GT::debugInput();
        std::cout << ss.str() << std::endl;
      }
    }
    GT::DistContext::Barrier();

    using Space = Kokkos::DefaultHostExecutionSpace;
    GT::DistTensorContext dtc;
    GT::SptensorT<Space> X =
      dtc.distributeTensor<Space>(GT::DistContext::input());
    GT::KtensorT<Space> u =
      dtc.computeInitialGuess<Space>(X, GT::DistContext::input());

    if (solver_method == "cp-als") {
      GT::DistCpAls<Space> cpals(dtc, X, u, GT::DistContext::input());
      cpals.compute();
    }
    else if (solver_method == "gcp") {
      GT::DistGCP<Space> gcp(dtc, X, u, GT::DistContext::input());
      gcp.compute();
    }
    else
      Genten::error("Unknown solver-method: " + solver_method);

    std::string output =
      GT::DistContext::input().get<std::string>("k-tensor.output", "");
    if (output != "")
      dtc.exportToFile(u, output);

  } catch (std::exception &e) {
    auto rank = GT::DistContext::rank();
    std::cerr << "Rank: " << rank << " " << e.what() << "\n";
    MPI_Abort(GT::DistContext::commWorld(), 0);
  }
}
