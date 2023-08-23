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

#include <Genten_DistTensorContext.hpp>
#include <Genten_GCP_SGD.hpp>
#include <Genten_GCP_SGD_SA.hpp>
#include <Genten_Sptensor.hpp>

#include "Genten_Test_Utils.hpp"

#include <gtest/gtest.h>

namespace Genten {
namespace UnitTests {

template <typename ExecSpace> struct TestGcpSgdT : public ::testing::Test {
  using exec_space = ExecSpace;
};

TYPED_TEST_SUITE(TestGcpSgdT, genten_test_types);

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
void RunGcpSgdTest(const std::string &label, GCP_Sampling::type sampling_type,
                   MTTKRP_All_Method::type mttkrp_all_method,
                   MTTKRP_Method::type mttkrp_method, const bool fuse,
                   const bool fuse_sa, const std::string& loss_type) {
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

  Genten::DistTensorContext<exec_space> dtc;
  SptensorT<exec_space> X_dev = dtc.distributeTensor(X);
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

  // Set parallel maps
  const ProcessorMap *pmap = dtc.pmap_ptr().get();
  X_dev.setProcessorMap(pmap);

  std::ostream& out = pmap->gridRank() == 0 ? std::cout : Genten::bhcout;

  // Factorize.
  AlgParams algParams;
  algParams.rate = 7.0e-2;
  algParams.epoch_iters = 50;
  algParams.gcp_tol = 1.0e-6;
  algParams.maxiters = 100;
  algParams.printitn = 1;
  algParams.sampling_type = sampling_type;
  algParams.mttkrp_method = mttkrp_method;
  algParams.mttkrp_all_method = mttkrp_all_method;
  algParams.fuse = fuse;
  algParams.loss_function_type = loss_type;
  algParams.oversample_factor = 5;
  algParams.gcp_seed = 12345;

  ttb_indx numIters;
  ttb_real resNorm;
  Ktensor result(nNumComponents, dims.size(), dims);
  deep_copy(result, initialBasis);
  KtensorT<exec_space> result_dev = dtc.exportFromRoot(result);
  result_dev.setProcessorMap(pmap);
  PerfHistory history;
  EXPECT_NO_THROW({
    if (!fuse_sa) {
      gcp_sgd(X_dev, result_dev, algParams, numIters, resNorm, history,
              out);
    } else {
      gcp_sgd_sa(X_dev, result_dev, algParams, numIters, resNorm, history,
                 out);
    }
  });

  result = dtc.template importToRoot<typename Ktensor::exec_space>(result_dev);
  if (dtc.gridRank() == 0) {
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
}

TYPED_TEST(TestGcpSgdT, GCP_SGD) {
  using exec_space = typename TestFixture::exec_space;

  struct TestCase {
    TestCase(const char *label, const GCP_Sampling::type sampling_type,
             const MTTKRP_All_Method::type mttkrp_all_method,
             const MTTKRP_Method::type mttkrp_method, const bool fuse,
             const bool fuse_sa, const std::string& loss_type)
        : label{label}, sampling_type{sampling_type},
          mttkrp_all_method{mttkrp_all_method}, mttkrp_method{mttkrp_method},
          fuse{fuse}, fuse_sa{fuse_sa}, loss_type{loss_type} {}

    const char *label;
    const GCP_Sampling::type sampling_type;
    const MTTKRP_All_Method::type mttkrp_all_method;
    const MTTKRP_Method::type mttkrp_method;
    const bool fuse;
    const bool fuse_sa;
    const std::string loss_type;

    const bool run{not SpaceProperties<exec_space>::is_gpu ||
                   mttkrp_method != MTTKRP_Method::type::Duplicated};
  };

  TestCase test_cases[]{
      TestCase{"Stratified, Atomic (iterated), Gaussian",
               GCP_Sampling::Stratified, MTTKRP_All_Method::Iterated,
               MTTKRP_Method::Atomic, false, false, "gaussian"},
      TestCase{"Stratified, Atomic (all), Gaussian", GCP_Sampling::Stratified,
               MTTKRP_All_Method::Atomic, MTTKRP_Method::Atomic, false, false,
              "gaussian"},
      TestCase{"Stratified, Duplicated (all), Gaussian",
               GCP_Sampling::Stratified, MTTKRP_All_Method::Duplicated,
               MTTKRP_Method::Duplicated, false, false,
               "gaussian"}};

  for (const auto &tc : test_cases) {
    if (tc.run) {
      RunGcpSgdTest<exec_space>(tc.label, tc.sampling_type,
                                tc.mttkrp_all_method, tc.mttkrp_method, tc.fuse,
                                tc.fuse_sa, tc.loss_type);
    }
  }
}

} // namespace UnitTests
} // namespace Genten
