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

#include <Genten_FacMatrix.hpp>
#include <Genten_IOtext.hpp>
#include <Genten_Ktensor.hpp>
#include <Genten_Sptensor.hpp>

#include "Genten_Test_Utils.hpp"

#include <gtest/gtest.h>

namespace Genten {
namespace UnitTests {

template <typename ExecSpace> struct TestIOtextT : public ::testing::Test {
  using exec_space = ExecSpace;
};

TYPED_TEST_SUITE(TestIOtextT, genten_test_types);

TEST(TestIOtext, Sptensor) {
  Sptensor oSpt;
  const std::string input_path = "data/C_sptensor.txt";
  import_sptensor(input_path, oSpt);

  ASSERT_EQ(oSpt.ndims(), 3);
  ASSERT_EQ(oSpt.nnz(), 5);
  ASSERT_EQ(oSpt.numel(), 24);
  ASSERT_EQ(oSpt.subscript(4, 0), 3);
  ASSERT_EQ(oSpt.subscript(4, 1), 0);
  ASSERT_EQ(oSpt.subscript(4, 2), 1);
  ASSERT_FLOAT_EQ(oSpt.value(4), 5.0);

  const std::string output_path = "tmp_Test_IOtext.txt";
  export_sptensor(output_path, oSpt);

  Sptensor oSpt2;
  import_sptensor(output_path, oSpt2);
  ASSERT_TRUE(oSpt.isEqual(oSpt2, MACHINE_EPSILON));
  ASSERT_EQ(remove(output_path.c_str()), 0);
}

TEST(TestIOtext, FacMatrix) {
  FacMatrix oFM;
  const std::string input_path = "data/B_matrix.txt";
  import_matrix(input_path, oFM);
  ASSERT_EQ(oFM.nRows(), 3);
  ASSERT_EQ(oFM.nCols(), 2);

  double dExpectedValue = 0.1;
  for (ttb_indx j = 0; j < oFM.nCols(); j++) {
    for (ttb_indx i = 0; i < oFM.nRows(); i++) {
      ASSERT_FLOAT_EQ(oFM.entry(i, j), dExpectedValue);
      dExpectedValue += 0.1;
    }
  }

  const std::string output_path = "tmp_Test_IOtext.txt";
  export_matrix(output_path, oFM);
  FacMatrix oFM2;
  import_matrix(output_path, oFM2);
  ASSERT_TRUE(oFM.isEqual(oFM2, MACHINE_EPSILON));
  ASSERT_EQ(remove(output_path.c_str()), 0);
}

TEST(TestIOtext, Ktensor) {
  Ktensor oK;
  const std::string input_path = "data/E_ktensor.txt";
  import_ktensor(input_path, oK);
  ASSERT_EQ(oK.ndims(), 3);
  ASSERT_EQ(oK.ncomponents(), 2);
  ASSERT_FLOAT_EQ(oK.weights(0), 1.0);
  ASSERT_FLOAT_EQ(oK.weights(1), 2.0);

  ASSERT_FLOAT_EQ(oK[0].entry(1, 1), 0.9);
  ASSERT_FLOAT_EQ(oK[1].entry(0, 1), 2.0);
  ASSERT_FLOAT_EQ(oK[2].entry(2, 0), 0.03);

  const std::string output_path = "tmp_Test_IOtext.txt";
  export_ktensor(output_path, oK);
  Genten::Ktensor oK2;
  import_ktensor(output_path, oK2);
  ASSERT_TRUE(oK.isEqual(oK2, MACHINE_EPSILON));
  ASSERT_EQ(remove(output_path.c_str()), 0);
}

} // namespace UnitTests
} // namespace Genten
