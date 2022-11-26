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

#include <Genten_Ktensor.hpp>

#include "Genten_Test_Utils.hpp"

#include <gtest/gtest.h>

namespace Genten {
namespace UnitTests {

template <typename ExecSpace> struct TestKtensorT : public ::testing::Test {
  using exec_space = ExecSpace;
};

TYPED_TEST_SUITE(TestKtensorT, genten_test_types);

TEST(TestKtensor, EmptyConstructor) {
  Ktensor kt;
  ASSERT_EQ(kt.ncomponents(), 0);
  ASSERT_EQ(kt.ndims(), 0);
}

TEST(TestKtensor, ConstructorWithArguments) {
  const ttb_indx nc = 2;
  Ktensor kt_a(nc, 3);
  ASSERT_EQ(kt_a.ncomponents(), nc);
  ASSERT_EQ(kt_a.ndims(), 3);

  IndxArray dims{1, 2, 3};
  Ktensor kt_b(nc, 3, dims);
  ASSERT_TRUE(kt_b.isConsistent());
  ASSERT_EQ(kt_b[0].nRows(), 1);
  ASSERT_EQ(kt_b[0].nCols(), nc);
  ASSERT_EQ(kt_b[1].nRows(), 2);
  ASSERT_EQ(kt_b[1].nCols(), nc);
  ASSERT_EQ(kt_b[2].nRows(), 3);
  ASSERT_EQ(kt_b[2].nCols(), nc);
}

TEST(TestKtensor, FactorsAndWeights) {
  const ttb_indx nc = 2;
  IndxArray dims{1, 2, 3};
  Ktensor kt(nc, 3, dims);

  kt.weights(0) = 1.0;
  kt.weights(1) = 2.0;
  kt[0].entry(0, 0) = 1.0;
  kt[0].entry(0, 1) = 2.0;
  kt[2].entry(2, 1) = 3.0;
  kt[1].entry(1, 1) = 4.0;
  ASSERT_EQ(kt[2].entry(2, 1), 3.0);
  ASSERT_FLOAT_EQ(kt.normFsq(), 48 * 48);
}

TEST(TestKtensor, CopyConstructor) {
  const ttb_indx nc = 2;
  IndxArray dims{1, 2, 3};
  Ktensor kt(nc, 3, dims);
  kt[0].entry(0, 0) = 1.0;
  kt[0].entry(0, 1) = 2.0;
  kt[2].entry(2, 1) = 3.0;
  kt[1].entry(1, 1) = 4.0;

  Ktensor kt_copy(kt);

  ASSERT_EQ(kt_copy.ncomponents(), kt.ncomponents());
  ASSERT_EQ(kt_copy.ndims(), kt.ndims());
  ASSERT_EQ(kt_copy.weights(0), 1.0);
  ASSERT_FLOAT_EQ(kt_copy[0].entry(0, 0), 1.0);
}

TEST(TestKtensor, AssignmentOperator) {
  const ttb_indx nc = 2;
  IndxArray dims{1, 2, 3};
  Ktensor kt(nc, 3, dims);
  kt[0].entry(0, 0) = 1.0;
  kt[0].entry(0, 1) = 2.0;
  kt[2].entry(2, 1) = 3.0;
  kt[1].entry(1, 1) = 4.0;

  Ktensor kt_copy;
  kt_copy = kt;

  ASSERT_EQ(kt_copy.ncomponents(), kt.ncomponents());
  ASSERT_EQ(kt_copy.ndims(), kt.ndims());
  ASSERT_EQ(kt_copy.weights(0), 1.0);
  ASSERT_FLOAT_EQ(kt_copy[0].entry(0, 0), 1.0);
}

TEST(TestKtensor, Arrange) {
  IndxArray dims{1, 2, 3};
  Ktensor kt(3, 3, dims);

  kt.weights(0) = 1.0;
  kt.weights(1) = 2.0;
  kt.weights(2) = 3.0;

  kt[0].entry(0, 0) = 1.0;
  kt[0].entry(0, 1) = 2.0;
  kt[0].entry(0, 2) = 3.0;
  kt[1].entry(0, 0) = 4.0;
  kt[1].entry(1, 0) = 5.0;
  kt[1].entry(0, 1) = 6.0;
  kt[1].entry(1, 1) = 7.0;
  kt[1].entry(0, 2) = 8.0;
  kt[1].entry(1, 2) = 9.0;
  kt[2].entry(0, 0) = 1.0;
  kt[2].entry(1, 0) = 2.0;
  kt[2].entry(2, 0) = 3.0;
  kt[2].entry(0, 1) = 4.0;
  kt[2].entry(1, 1) = 5.0;
  kt[2].entry(2, 1) = 6.0;
  kt[2].entry(0, 2) = 7.0;
  kt[2].entry(1, 2) = 8.0;
  kt[2].entry(2, 2) = 9.0;

  ASSERT_FLOAT_EQ(kt.normFsq(), 3443252.0);

  kt.arrange();

  ASSERT_FLOAT_EQ(kt.weights(0), 3.0);
  ASSERT_FLOAT_EQ(kt.weights(1), 2.0);
  ASSERT_FLOAT_EQ(kt.weights(2), 1.0);
  ASSERT_FLOAT_EQ(kt[0].entry(0, 0), 3.0);
  ASSERT_FLOAT_EQ(kt[0].entry(0, 1), 2.0);
  ASSERT_FLOAT_EQ(kt[0].entry(0, 2), 1.0);
  ASSERT_FLOAT_EQ(kt.normFsq(), 3443252.0);
}

} // namespace UnitTests
} // namespace Genten
