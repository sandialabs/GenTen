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

#ifdef HAVE_ROL

#include <Genten_GCP_Opt.hpp>
#include <Genten_IndxArray.hpp>
#include <Genten_Ktensor.hpp>
#include <Genten_Sptensor.hpp>

#include "Genten_Test_Utils.hpp"

#include <gtest/gtest.h>

namespace Genten {
namespace UnitTests {

template <typename ExecSpace> struct TestGcpOptT : public ::testing::Test {
  using exec_space = ExecSpace;
};

TYPED_TEST_SUITE(TestGcpOptT, genten_test_types);

/*!
 *  The test factors a simple 2x3x4 sparse tensor into known components.
 *  Matlab formulation:
 *    subs = [1 1 1 ; 2 1 1 ; 1 2 1 ; 2 2 1 ; 1 3 1 ; 1 1 2 ; 1 3 2 ; 1 1 4 ;
 *            2 1 4 ; 1 2 4 ; 2 2 4]
 *    vals = [2 1 1 1 1 1 1 1 1 1 1]
 *    X = sptensor (subs, vals', [2 3 4])
 *    X0 = { rand(2,2), rand(3,2), rand(4,2) }, or values below (it matters!)
 *    F = cp_als (X,2, 'init',X0)
 *  There are many possible factors, so instead of comparing to an exact
 *  solution, just check that when you multiply the factors together, you
 *  get the original tensor.
 */
template <typename exec_space>
void RunGCPOptTest(const std::string &label, MTTKRP_Method::type mttkrp_method,
                   const GCP_LossFunction::type loss_type) {
  using host_exec_space = DefaultHostExecutionSpace;

  INFO_MSG("Creating a sparse tensor with data to model");

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

  SptensorT<exec_space> X_dev = create_mirror_view(exec_space(), X);
  deep_copy(X_dev, X);
  if (mttkrp_method == MTTKRP_Method::Perm) {
    X_dev.createPermutation();
  }

  INFO_MSG("Creating a ktensor with initial guess of lin indep basis vectors");
  ttb_indx nNumComponents = 2;
  Ktensor initialBasis(nNumComponents, dims.size(), dims);
  ttb_indx seed = 12345;
  RandomMT cRMT(seed);
  initialBasis.setMatricesScatter(false, false, cRMT);
  initialBasis.setWeights(1.0);

  KtensorT<exec_space> initialBasis_dev =
      create_mirror_view(exec_space(), initialBasis);
  deep_copy(initialBasis_dev, initialBasis);

  // Factorize.
  AlgParams algParams;
  algParams.tol = 1.0e-6;
  algParams.maxiters = 100;
  algParams.mttkrp_method = mttkrp_method;
  KtensorT<exec_space> result_dev;
  std::ostream *stream = nullptr;
  EXPECT_NO_THROW({
    result_dev = initialBasis_dev;
    gcp_opt<SptensorT<exec_space>>(X_dev, result_dev, algParams, stream);
  });

  Ktensor result = initialBasis;
  deep_copy(result, result_dev);

  // Multiply Ktensor entries and compare to tensor
  const ttb_real tol = 1.0e-3;
  const ttb_indx nnz = X.nnz();
  const IndxArray subs(3);
  for (ttb_indx i = 0; i < nnz; ++i) {
    X.getSubscripts(i, subs);
    const ttb_real x_val = X.value(i);
    const ttb_real val = result.entry(subs);
    GENTEN_LE(fabs(x_val - val), tol, "Result matches");
  }
}

TYPED_TEST(TestGcpOptT, GCPOpt) {
  using exec_space = typename TestFixture::exec_space;

  struct TestCase {
    TestCase(const char *label, const MTTKRP_Method::type mttkrp_method,
             const GCP_LossFunction::type loss_type)
        : label{label}, mttkrp_method{mttkrp_method}, loss_type{loss_type} {}

    const char *label;
    const MTTKRP_Method::type mttkrp_method;
    const GCP_LossFunction::type loss_type;
  };

  TestCase test_cases[]{TestCase{"Atomic, Gaussian", MTTKRP_Method::Atomic,
                                 GCP_LossFunction::Gaussian},
                        TestCase{"Duplicated, Gaussian",
                                 MTTKRP_Method::Duplicated,
                                 GCP_LossFunction::Gaussian},
                        TestCase{"Perm, Gaussian", MTTKRP_Method::Perm,
                                 GCP_LossFunction::Gaussian}};

  for (const auto &tc : test_cases) {
    RunGCPOptTest<exec_space>(tc.label, tc.mttkrp_method, tc.loss_type);
  }
}

} // namespace UnitTests
} // namespace Genten

#endif
