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
#include <sstream>

#include <gtest/gtest.h>
// using namespace Genten::Test;

namespace Genten {
namespace UnitTests {
template <typename ExecSpace> struct TestJMMT : public ::testing::Test {
  using exec_space = ExecSpace;
};

TYPED_TEST_SUITE(TestMomentTensor, genten_test_types);

// Unit Test for the computation of the Joint Matricized Moment Tensor (JMMT)


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
Genten::TensorT<Kokkos::DefaultHostExecutionSpace> naiveAlgo(ViewT x) {
  const auto x_h = Kokkos::create_mirror_view_and_copy(
    Kokkos::DefaultHostExecutionSpace(), x
  );
  const auto c = x_h.extent(1);
  Genten::TensorT<Kokkos::DefaultHostExecutionSpace> moment(
    Genten::IndxArrayT<Kokkos::DefaultHostExecutionSpace>{c, c, c, c}
  );

  const double factor = 1.0 / x.extent(0);
  for (std::size_t i = 0; i < c; ++i) {
    for (std::size_t j = 0; j < c; ++j) {
      for (std::size_t k = 0; k < c; ++k) {
        for (std::size_t l = 0; l < c; ++l) {
          for (std::size_t m = 0; m < x.extent(0); ++m) {
            moment(i, j, k, l) += x_h(m, i) * x_h(m, j) * x_h(m, k) * x_h(m, l);
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

  int n_configs_tested=0;
  // The original combinations of sizes to test:
  // for (const int numRows : {20, 35, 40, 91}) {
  //   for (const int numCols : { 4, 5, 6, 7, 8, 9, 10, 11, 12}) {
  //     for (const int blockSize : {2, 3, 4, 5, 6}) {
  //       if (numCols >= (blockSize)) {
  //         for (const int teamSize : {1, 2, 4}) {
  // That takes over 6 minutes to run. Pared down to something more reasonable:
  for (const int numRows : {20, 35}) {
    for (const int numCols : { 4, 5, 6}) {
      for (const int blockSize : {2, 5}) {
        if (numCols >= (blockSize)) {
          for (const int teamSize : {1, 4}) {
            std::cout << "Computing 4th joint moment for matrix " << numRows
                      << "x" << numCols << ", blockSize: " << blockSize
                      << ", teamSize: " << teamSize << std::endl;

            const auto x = generateTestInput<ExecSpace>(numRows, numCols);

            const auto naiveAlgoRes = naiveAlgo(x);
            // Genten::print_tensor(naiveAlgoRes, std::cout, "naiveAlgoRes");

            const auto refactoredAlgoRes =
              Genten::create_and_compute_moment_tensor(x, blockSize, teamSize);
            // Genten::print_tensor(refactoredAlgoRes, std::cout, "refactoredAlgoRes");

            constexpr auto epsilon = 1.0E-10;
            const auto tensorsAreEqual =
              naiveAlgoRes.getValues().isEqual(
                refactoredAlgoRes.getValues(), epsilon
              );

            if (not tensorsAreEqual) {
              for (int i = 0; i < numCols; ++i) {
                for (int j = 0; j < numCols; ++j) {
                  for (int k = 0; k < numCols; ++k) {
                    for (int l = 0; l < numCols; ++l) {
                      const auto nar = naiveAlgoRes(i, j, k, l);
                      const auto rar = refactoredAlgoRes(i, j, k, l);
                      if (not Genten::isEqualToTol(nar, rar, epsilon)) {
                        std::stringstream resDiff;
                        std::cout << "(" << i << "," << j << "," << k << "," << l
                                << "): naiveAlgoRes = " << nar
                                << ", refactoredAlgoRes = " << rar<<std::endl;
                      }
                    }
                  }
                }
              }
            }
            std::cout<<"tensorsAreEqual: "<<tensorsAreEqual<<std::endl;
            std::cout << "FINISHED for matrix " << numRows
                      << "x" << numCols << ", blockSize: " << blockSize
                      << ", teamSize: " << teamSize <<std::endl;
            std::cout<<"wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww"<<std::endl;
            ASSERT_EQ(tensorsAreEqual, 1);
            n_configs_tested++;
          }
        }
      }
    }
  }
  std::cout<<"n_configs_tested: "<<n_configs_tested<<std::endl;

}
TEST(TestMoment, ComputeMoment) {

  int infolevel = 1;
  #ifdef KOKKOS_ENABLE_CUDA
    Genten_Test_MomentTensorImpl<Kokkos::Cuda>(infolevel);
  #endif
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
} //end TestMoment
} //end namespace UnitTests
} //end namespace Genten
