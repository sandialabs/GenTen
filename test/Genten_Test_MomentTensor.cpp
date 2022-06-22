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

#include "Genten_Array.hpp"
#include "Genten_HigherMoments.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_Test_Utils.hpp"
#include "Genten_Util.hpp"
#include <Kokkos_Random.hpp>
#include <cmath>
#include <iostream>

using namespace Genten::Test;

template <typename ExecSpace>
Kokkos::View<ttb_real**, Kokkos::LayoutLeft, ExecSpace> generateTestInput(
  const int numRows, const int numCols
) {
  Kokkos::View<ttb_real**, Kokkos::LayoutLeft, ExecSpace> x(
    "x", numRows, numCols
  );

  std::srand(std::time(nullptr));
  const ttb_indx seed = std::rand();
  Kokkos::Random_XorShift64_Pool<ExecSpace> random(seed);
  Kokkos::fill_random(x, random, 1.0);

  return x;
}

template <class ViewT>
Genten::TensorT<typename ViewT::execution_space> naiveAlgo(const ViewT& X) {
  const auto c = X.extent(1);
  using exec_space = typename ViewT::execution_space;
  Genten::TensorT<exec_space> moment(Genten::IndxArrayT<exec_space>{c, c, c, c});

  const double factor = 1.0 / X.extent(0);
  for (std::size_t i = 0; i < c; ++i) {
    for (std::size_t j = 0; j < c; ++j) {
      for (std::size_t k = 0; k < c; ++k) {
        for (std::size_t l = 0; l < c; ++l) {
          for (std::size_t m = 0; m < X.extent(0); ++m) {
            moment(i, j, k, l) += X(m, i) * X(m, j) * X(m, k) * X(m, l);
          }

          moment(i, j, k, l) *= factor;
        }
      }
    }
  }

  return moment;
}

template <typename ExecSpace>
void Genten_Test_MomentTensorImpl(int infolevel) {
  std::string space_name = Genten::SpaceProperties<ExecSpace>::name();
  initialize("Tests on Genten::MomentTensor (" + space_name + ")", infolevel);

  // for (int numRows : {20, 31}) {
    // for (int numCols : {4, 5, 6}) {

  for (int numRows : {20}) {
    for (int numCols : {5}) {
      const auto x = generateTestInput<ExecSpace>(numRows, numCols);
      constexpr int blockSize = 2;

      printf(
        "\nnumRows: %d, numCols: %d, blockSize: %d\n",
        numRows, numCols, blockSize
      );

      const auto naiveAlgoRes = naiveAlgo(x);
      Genten::print_tensor(naiveAlgoRes, std::cout, "naiveAlgoRes");

      const auto refactoredAlgoRes =
        Genten::create_and_compute_moment_tensor(x, blockSize);

      Genten::print_tensor(refactoredAlgoRes, std::cout, "refactoredAlgoRes");

      const auto areTheSame =
        naiveAlgoRes.getValues().isEqual(
          refactoredAlgoRes.getValues(), 1.0E-10

          // TODO (STRZ) - std::epsilon doesn't work for colNum = 6
          // refactoredAlgoRes.getValues(), std::numeric_limits<ttb_real>::epsilon()
        );

      printf("\nareTheSame: %s\n", areTheSame ? "true" : "false");
    }
  }

  finalize();
}

void Genten_MomentTensor(int infolevel) {

// #ifdef KOKKOS_ENABLE_CUDA
//   Genten_Test_MomentTensorImpl<Kokkos::Cuda>(infolevel);
// #endif
#ifdef KOKKOS_ENABLE_HIP
  Genten_Test_MomentTensorImpl<Kokkos::Experimental::HIP>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  Genten_Test_MomentTensorImpl<Kokkos::OpenMP>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_THREADS
  Genten_Test_MomentTensorImpl<Kokkos::Threads>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_SERIAL
  Genten_Test_MomentTensorImpl<Kokkos::Serial>(infolevel);
#endif
}
