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

#include <Genten_CpAls.hpp>
#include <Genten_DistTensorContext.hpp>
#include <Genten_FacTestSetGenerator.hpp>
#include <Genten_Kokkos.hpp>
#include <Genten_Ktensor.hpp>
#include <Genten_Sptensor.hpp>
#include <Genten_Tensor.hpp>
#include <Genten_Util.hpp>

#include "Genten_Test_Utils.hpp"

#include <gtest/gtest.h>

namespace Genten {
namespace UnitTests {

template <typename ExecSpace> struct TestCpAlsT : public ::testing::Test {
  using exec_space = ExecSpace;
};

TYPED_TEST_SUITE(TestCpAlsT, genten_test_types);

static void evaluateResult(const ttb_indx itersCompleted,
                           const ttb_real stopTol, const Ktensor &result) {
  std::stringstream sMsg;
  sMsg << "CpAls finished after " << itersCompleted << " iterations";
  INFO_MSG(sMsg.str().c_str());

  constexpr ttb_real tol = 2.5e-3;

  // Check the final weights, which can be in any order.
  const ttb_real wght0 = result.weights(0);
  const ttb_real wght1 = result.weights(1);
  if (wght0 >= wght1) {
    const ttb_real diffA = fabs(wght0 - 2.828427);
    const ttb_real diffB = fabs(wght1 - 2.0);
    ASSERT_LE(diffA, tol);
    ASSERT_LE(diffB, tol);
    INFO_MSG("Result ktensor weights match");

    GENTEN_LE(fabs(result[0].entry(0, 0) - 0.7071), tol,
              "Result ktensor[0](0,0) matches");
    GENTEN_LE(fabs(result[0].entry(1, 0) - 0.7071), tol,
              "Result ktensor[0](1,0) matches");
    GENTEN_LE(fabs(result[0].entry(0, 1) - 1.0), tol,
              "Result ktensor[0](0,1) matches");
    GENTEN_LE(fabs(result[0].entry(1, 1) - 0.0), tol,
              "Result ktensor[0](1,1) matches");

    GENTEN_LE(fabs(result[1].entry(0, 0) - 0.7071), tol,
              "Result ktensor[1](0,0) matches");
    GENTEN_LE(fabs(result[1].entry(1, 0) - 0.7071), tol,
              "Result ktensor[1](1,0) matches");
    GENTEN_LE(fabs(result[1].entry(2, 0) - 0.0), tol,
              "Result ktensor[1](2,0) matches");
    GENTEN_LE(fabs(result[1].entry(0, 1) - 0.7071), tol,
              "Result ktensor[1](0,1) matches");
    GENTEN_LE(fabs(result[1].entry(1, 1) - 0.0), tol,
              "Result ktensor[1](1,1) matches");
    GENTEN_LE(fabs(result[1].entry(2, 1) - 0.7071), tol,
              "Result ktensor[1](2,1) matches");

    GENTEN_LE(fabs(result[2].entry(0, 0) - 0.7071), tol,
              "Result ktensor[2](0,0) matches");
    GENTEN_LE(fabs(result[2].entry(1, 0) - 0.0), tol,
              "Result ktensor[2](1,0) matches");
    GENTEN_LE(fabs(result[2].entry(2, 0) - 0.0), tol,
              "Result ktensor[2](2,0) matches");
    GENTEN_LE(fabs(result[2].entry(3, 0) - 0.7071), tol,
              "Result ktensor[2](3,0) matches");
    GENTEN_LE(fabs(result[2].entry(0, 1) - 0.7071), tol,
              "Result ktensor[2](0,1) matches");
    GENTEN_LE(fabs(result[2].entry(1, 1) - 0.7071), tol,
              "Result ktensor[2](1,1) matches");
    GENTEN_LE(fabs(result[2].entry(2, 1) - 0.0), tol,
              "Result ktensor[2](2,1) matches");
    GENTEN_LE(fabs(result[2].entry(3, 1) - 0.0), tol,
              "Result ktensor[2](2,1) matches");
  } else {
    const ttb_real diffA = fabs(wght0 - 2.0);
    const ttb_real diffB = fabs(wght1 - 2.8284);
    ASSERT_LE(diffA, tol);
    ASSERT_LE(diffB, tol);
    INFO_MSG("Result ktensor weights match");

    GENTEN_LE(fabs(result[0].entry(0, 0) - 1.0), tol,
              "Result ktensor[0](0,0) matches");
    GENTEN_LE(fabs(result[0].entry(1, 0) - 0.0), tol,
              "Result ktensor[0](1,0) matches");
    GENTEN_LE(fabs(result[0].entry(0, 1) - 0.7071), tol,
              "Result ktensor[0](0,1) matches");
    GENTEN_LE(fabs(result[0].entry(1, 1) - 0.7071), tol,
              "Result ktensor[0](1,1) matches");

    GENTEN_LE(fabs(result[1].entry(0, 0) - 0.7071), tol,
              "Result ktensor[1](0,0) matches");
    GENTEN_LE(fabs(result[1].entry(1, 0) - 0.0), tol,
              "Result ktensor[1](1,0) matches");
    GENTEN_LE(fabs(result[1].entry(2, 0) - 0.7071), tol,
              "Result ktensor[1](2,0) matches");
    GENTEN_LE(fabs(result[1].entry(0, 1) - 0.7071), tol,
              "Result ktensor[1](0,1) matches");
    GENTEN_LE(fabs(result[1].entry(1, 1) - 0.7071), tol,
              "Result ktensor[1](1,1) matches");
    GENTEN_LE(fabs(result[1].entry(2, 1) - 0.0), tol,
              "Result ktensor[1](2,1) matches");

    GENTEN_LE(fabs(result[2].entry(0, 0) - 0.7071), tol,
              "Result ktensor[2](0,0) matches");
    GENTEN_LE(fabs(result[2].entry(1, 0) - 0.7071), tol,
              "Result ktensor[2](1,0) matches");
    GENTEN_LE(fabs(result[2].entry(2, 0) - 0.0), tol,
              "Result ktensor[2](2,0) matches");
    GENTEN_LE(fabs(result[2].entry(3, 0) - 0.0), tol,
              "Result ktensor[2](2,0) matches");
    GENTEN_LE(fabs(result[2].entry(0, 1) - 0.7071), tol,
              "Result ktensor[2](0,1) matches");
    GENTEN_LE(fabs(result[2].entry(1, 1) - 0.0), tol,
              "Result ktensor[2](1,1) matches");
    GENTEN_LE(fabs(result[2].entry(2, 1) - 0.0), tol,
              "Result ktensor[2](2,1) matches");
    GENTEN_LE(fabs(result[2].entry(3, 1) - 0.7071), tol,
              "Result ktensor[2](3,1) matches");
  }
}

/*!
 *  The test factors a simple 2x3x4 sparse tensor into known components.
 *  Matlab formulation:
 *    subs = [1 1 1 ; 2 1 1 ; 1 2 1 ; 2 2 1 ; 1 3 1 ; 1 1 2 ; 1 3 2 ; 1 1 4 ;
 *            2 1 4 ; 1 2 4 ; 2 2 4]
 *    vals = [2 1 1 1 1 1 1 1 1 1 1]
 *    X = sptensor (subs, vals', [2 3 4])
 *    X0 = { rand(2,2), rand(3,2), rand(4,2) }, or values below (it matters!)
 *    F = cp_als (X,2, 'init',X0)
 *  Exact solution (because this is how X was constructed):
 *    lambda = [1 1]
 *    A = [1 1 ; 0 1]
 *    B = [1 1 ; 0 1 ; 1 0]
 *    C = [1 1 ; 1 0 ; 0 0 ; 0 1]
 *  Exact solution as a normalized ktensor:
 *    lambda = [2.0 2.8284]
 *    A = [1.0    0.7071 ; 0.0    0.7071]
 *    B = [0.7071 0.7071 ; 0.0    0.7071 ; 0.7071 0.0   ]
 *    C = [0.7071 0.7071 ; 0.7071 0.0    ; 0.0    0.0    ; 0.0    0.7071]
 *  Random start point can converge to a different (worse) solution, so the
 *  test uses a particular start point:
 *    A0 = [0.8 0.5 ; 0.2 0.5]
 *    B0 = [0.5 0.5 ; 0.1 0.5 ; 0.5 0.1]
 *    C0 = [0.7 0.7 ; 0.7 0.1 ; 0.1 0.1 ; 0.1 0.7]
 *
 * Note ETP 03/20/2018:  Originally this comment as the weights as [2.8284 2.0],
 * but based on my arithmetic, it should be [2.8284 2.0] based on the order of
 * the factor matrix columns shown above.  This is consistent with what the code
 * produces.
 */

template <typename exec_space>
void RunCpAlsTestSparse(MTTKRP_Method::type mttkrp_method,
                        const std::string &label) {
  using host_exec_space = DefaultHostExecutionSpace;

  IndxArray dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 4;

  SptensorT<host_exec_space> X(dims, 11);

  X.subscript(0, 0) = 0;
  X.subscript(0, 1) = 0;
  X.subscript(0, 2) = 0;
  X.value(0) = 2.0;
  X.subscript(1, 0) = 1;
  X.subscript(1, 1) = 0;
  X.subscript(1, 2) = 0;
  X.value(1) = 1.0;
  X.subscript(2, 0) = 0;
  X.subscript(2, 1) = 1;
  X.subscript(2, 2) = 0;
  X.value(2) = 1.0;
  X.subscript(3, 0) = 1;
  X.subscript(3, 1) = 1;
  X.subscript(3, 2) = 0;
  X.value(3) = 1.0;
  X.subscript(4, 0) = 0;
  X.subscript(4, 1) = 2;
  X.subscript(4, 2) = 0;
  X.value(4) = 1.0;
  X.subscript(5, 0) = 0;
  X.subscript(5, 1) = 0;
  X.subscript(5, 2) = 1;
  X.value(5) = 1.0;
  X.subscript(6, 0) = 0;
  X.subscript(6, 1) = 2;
  X.subscript(6, 2) = 1;
  X.value(6) = 1.0;
  X.subscript(7, 0) = 0;
  X.subscript(7, 1) = 0;
  X.subscript(7, 2) = 3;
  X.value(7) = 1.0;
  X.subscript(8, 0) = 1;
  X.subscript(8, 1) = 0;
  X.subscript(8, 2) = 3;
  X.value(8) = 1.0;
  X.subscript(9, 0) = 0;
  X.subscript(9, 1) = 1;
  X.subscript(9, 2) = 3;
  X.value(9) = 1.0;
  X.subscript(10, 0) = 1;
  X.subscript(10, 1) = 1;
  X.subscript(10, 2) = 3;
  X.value(10) = 1.0;

  GENTEN_EQ(X.nnz(), 11, "Data tensor has 11 nonzeroes");

  Genten::DistTensorContext<exec_space> dtc;
  SptensorT<exec_space> X_dev = dtc.distributeTensor(X);
  const ProcessorMap *pmap = dtc.pmap_ptr().get();
  X_dev.setProcessorMap(pmap);
  if (mttkrp_method == MTTKRP_Method::Perm) {
    X_dev.createPermutation();
  }

  INFO_MSG("Creating a ktensor with initial guess of lin indep basis vectors");
  ttb_indx nNumComponents = 2;
  Ktensor initialBasis(nNumComponents, dims.size(), dims);
  initialBasis.setWeights(1.0);
  initialBasis.setMatrices(0.0);
  initialBasis[0].entry(0, 0) = 0.8;
  initialBasis[0].entry(1, 0) = 0.2;
  initialBasis[0].entry(0, 1) = 0.5;
  initialBasis[0].entry(1, 1) = 0.5;
  initialBasis[1].entry(0, 0) = 0.5;
  initialBasis[1].entry(1, 0) = 0.1;
  initialBasis[1].entry(2, 0) = 0.5;
  initialBasis[1].entry(0, 1) = 0.5;
  initialBasis[1].entry(1, 1) = 0.5;
  initialBasis[1].entry(2, 1) = 0.1;
  initialBasis[2].entry(0, 0) = 0.7;
  initialBasis[2].entry(1, 0) = 0.7;
  initialBasis[2].entry(2, 0) = 0.1;
  initialBasis[2].entry(3, 0) = 0.1;
  initialBasis[2].entry(0, 1) = 0.7;
  initialBasis[2].entry(1, 1) = 0.1;
  initialBasis[2].entry(2, 1) = 0.1;
  initialBasis[2].entry(3, 1) = 0.7;
  initialBasis.weights(0) = 2.0; // Test with weights different from one.

  KtensorT<exec_space> initialBasis_dev = dtc.exportFromRoot(initialBasis);

  // Factorize.
  AlgParams algParams;
  algParams.rank = nNumComponents;
  algParams.tol = 1.0e-6;
  algParams.maxiters = 100;
  algParams.maxsecs = -1.0;
  algParams.printitn = 1;
  algParams.mttkrp_method = mttkrp_method;
  ttb_indx itersCompleted;
  ttb_real resNorm;

  std::ostream& out = pmap->gridRank() == 0 ? std::cout : Genten::bhcout;

  {
    KtensorT<exec_space> result_dev =
        create_mirror_view(exec_space(), initialBasis_dev);

    EXPECT_NO_THROW({
      // Request performance information on every 3rd iteration.
      // Allocation adds two more for start and stop states of the algorithm.
      PerfHistory perfInfo;

      deep_copy(result_dev, initialBasis_dev);
      result_dev.setProcessorMap(pmap);

      cpals_core(X_dev, result_dev, algParams, itersCompleted, resNorm, 3,
                 perfInfo, out);
      // Check performance information.
      for (ttb_indx i = 0; i < perfInfo.size(); i++) {
        if ((perfInfo[i].iteration > 0)) {
          ASSERT_GE(perfInfo[i].fit, 0.99);
          ASSERT_LE(perfInfo[i].fit, 1.00);
          ASSERT_LE(perfInfo[i].residual, 0.03);
          ASSERT_GE(perfInfo[i].cum_time, 0.0);
        }
      }

      INFO_MSG("Performance info from cpals_core is reasonable.");
    });

    Ktensor result = dtc.template importToRoot<host_exec_space>(result_dev);
    if (dtc.gridRank() == 0) {
      evaluateResult(itersCompleted, algParams.tol, result);
    }
  }

  {
    // Test factorization from a bad initial guess.
    INFO_MSG("Creating a ktensor with initial guess all zero");

    Ktensor initialZero(nNumComponents, dims.size(), dims);
    initialZero.setWeights(0.0);
    initialZero.setMatrices(0.0);

    KtensorT<exec_space> initialZero_dev =
      dtc.exportFromRoot(initialZero);
    KtensorT<exec_space> result_dev =
        create_mirror_view(exec_space(), initialZero_dev);
    deep_copy(result_dev, initialZero_dev);
    result_dev.setProcessorMap(pmap);

    INFO_MSG("Checking if linear solver detects singular guess");

    PerfHistory history;
    EXPECT_ANY_THROW(cpals_core(X_dev, result_dev, algParams, itersCompleted,
                                resNorm, 0, history));

    if (DistContext::nranks() == 1) {
      // Repeat the tests using the same data, but in a dense Tensor.

      INFO_MSG("Creating a dense tensor with data to model");

      TensorT<exec_space> Xd_dev(X_dev);

      // Factorize.
      EXPECT_NO_THROW({
        deep_copy(result_dev, initialBasis_dev);
        PerfHistory history;
        algParams.mttkrp_method = MTTKRP_Method::RowBased;
        cpals_core(Xd_dev, result_dev, algParams, itersCompleted, resNorm, 0,
                   history, out);
      });

      Ktensor result = dtc.template importToRoot<host_exec_space>(result_dev);
      if (dtc.gridRank() == 0) {
        evaluateResult(itersCompleted, algParams.tol, result);
      }
    }
  }
}

TYPED_TEST(TestCpAlsT, CpAlsSparse) {
  using exec_space = typename TestFixture::exec_space;

  struct TestCase {
    TestCase(const MTTKRP_Method::type mttkrp_method, const char *label)
        : mttkrp_method{mttkrp_method}, label{label} {}

    const MTTKRP_Method::type mttkrp_method;
    const char *label;

    const bool run{not SpaceProperties<exec_space>::is_gpu ||
                   mttkrp_method != MTTKRP_Method::Duplicated};
  };

  TestCase test_cases[]{TestCase{MTTKRP_Method::type::Atomic, "Atomic"},
                        TestCase{MTTKRP_Method::type::Duplicated, "Duplicated"},
                        TestCase{MTTKRP_Method::type::Perm, "Perm"}};

  for (const auto &tc : test_cases) {
    if (tc.run) {
      RunCpAlsTestSparse<exec_space>(tc.mttkrp_method, tc.label);
    }
  }
}

template <typename exec_space>
void RunCpAlsTestDense(const TensorLayout& layout, const std::string &label) {
  using host_exec_space = DefaultHostExecutionSpace;

  IndxArray dims = {2, 3, 4};
  TensorT<host_exec_space> X(dims);
  X(0, 0, 0) = 2.0;
  X(1, 0, 0) = 1.0;
  X(0, 1, 0) = 1.0;
  X(1, 1, 0) = 1.0;
  X(0, 2, 0) = 1.0;
  X(0, 0, 1) = 1.0;
  X(0, 2, 1) = 1.0;
  X(0, 0, 3) = 1.0;
  X(1, 0, 3) = 1.0;
  X(0, 1, 3) = 1.0;
  X(1, 1, 3) = 1.0;

  Genten::DistTensorContext<exec_space> dtc;
  SptensorT<exec_space> X_dev = dtc.distributeTensor(X);
  const ProcessorMap *pmap = dtc.pmap_ptr().get();
  X_dev.setProcessorMap(pmap);

  INFO_MSG("Creating a ktensor with initial guess of lin indep basis vectors");
  ttb_indx nNumComponents = 2;
  Ktensor initialBasis(nNumComponents, dims.size(), dims);
  initialBasis.setWeights(1.0);
  initialBasis.setMatrices(0.0);
  initialBasis[0].entry(0, 0) = 0.8;
  initialBasis[0].entry(1, 0) = 0.2;
  initialBasis[0].entry(0, 1) = 0.5;
  initialBasis[0].entry(1, 1) = 0.5;
  initialBasis[1].entry(0, 0) = 0.5;
  initialBasis[1].entry(1, 0) = 0.1;
  initialBasis[1].entry(2, 0) = 0.5;
  initialBasis[1].entry(0, 1) = 0.5;
  initialBasis[1].entry(1, 1) = 0.5;
  initialBasis[1].entry(2, 1) = 0.1;
  initialBasis[2].entry(0, 0) = 0.7;
  initialBasis[2].entry(1, 0) = 0.7;
  initialBasis[2].entry(2, 0) = 0.1;
  initialBasis[2].entry(3, 0) = 0.1;
  initialBasis[2].entry(0, 1) = 0.7;
  initialBasis[2].entry(1, 1) = 0.1;
  initialBasis[2].entry(2, 1) = 0.1;
  initialBasis[2].entry(3, 1) = 0.7;
  initialBasis.weights(0) = 2.0; // Test with weights different from one.

  KtensorT<exec_space> initialBasis_dev = dtc.exportFromRoot(initialBasis);

  // Factorize.
  AlgParams algParams;
  algParams.rank = nNumComponents;
  algParams.tol = 1.0e-6;
  algParams.maxiters = 100;
  algParams.maxsecs = -1.0;
  algParams.printitn = 1;
  algParams.mttkrp_method = MTTKRP_Method::type::Atomic;
  ttb_indx itersCompleted;
  ttb_real resNorm;

  std::ostream& out = pmap->gridRank() == 0 ? std::cout : Genten::bhcout;

  {
    KtensorT<exec_space> result_dev =
        create_mirror_view(exec_space(), initialBasis_dev);

    EXPECT_NO_THROW({
      // Request performance information on every 3rd iteration.
      // Allocation adds two more for start and stop states of the algorithm.
      PerfHistory perfInfo;

      deep_copy(result_dev, initialBasis_dev);
      result_dev.setProcessorMap(pmap);

      cpals_core(X_dev, result_dev, algParams, itersCompleted, resNorm, 3,
                 perfInfo, out);
      // Check performance information.
      for (ttb_indx i = 0; i < perfInfo.size(); i++) {
        if ((perfInfo[i].iteration > 0)) {
          ASSERT_GE(perfInfo[i].fit, 0.99);
          ASSERT_LE(perfInfo[i].fit, 1.00);
          ASSERT_LE(perfInfo[i].residual, 0.03);
          ASSERT_GE(perfInfo[i].cum_time, 0.0);
        }
      }

      INFO_MSG("Performance info from cpals_core is reasonable.");
    });

    Ktensor result = dtc.template importToRoot<host_exec_space>(result_dev);
    if (dtc.gridRank() == 0) {
      evaluateResult(itersCompleted, algParams.tol, result);
    }
  }

  {
    // Test factorization from a bad initial guess.
    INFO_MSG("Creating a ktensor with initial guess all zero");

    Ktensor initialZero(nNumComponents, dims.size(), dims);
    initialZero.setWeights(0.0);
    initialZero.setMatrices(0.0);

    KtensorT<exec_space> initialZero_dev =
      dtc.exportFromRoot(initialZero);
    KtensorT<exec_space> result_dev =
        create_mirror_view(exec_space(), initialZero_dev);
    deep_copy(result_dev, initialZero_dev);
    result_dev.setProcessorMap(pmap);

    INFO_MSG("Checking if linear solver detects singular guess");

    PerfHistory history;
    EXPECT_ANY_THROW(cpals_core(X_dev, result_dev, algParams, itersCompleted,
                                resNorm, 0, history));
  }
}

TYPED_TEST(TestCpAlsT, CpAlsDense) {
  using exec_space = typename TestFixture::exec_space;
  RunCpAlsTestDense<exec_space>(TensorLayout::Left, "LayoutLeft");
  RunCpAlsTestDense<exec_space>(TensorLayout::Right, "Layoutright");
}

} // namespace UnitTests
} // namespace Genten
