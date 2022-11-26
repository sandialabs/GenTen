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

#include <Genten_IndxArray.hpp>

#include <gtest/gtest.h>

namespace Genten {
namespace UnitTests {

TEST(TestIndxArray, EmptyConstruct) {
  IndxArray arr;
  ASSERT_TRUE(arr.empty());
}

TEST(TestIndxArray, ConstructorOfLengthZero) {
  IndxArray arr(0);
  ASSERT_TRUE(arr.empty());
  ASSERT_EQ(arr.size(), 0);
}

TEST(TestIndxArray, ConstructorOfLengthN) {
  IndxArray arr(3);
  ASSERT_FALSE(arr.empty());
  ASSERT_EQ(arr.size(), 3);
}

TEST(TestIndxArray, CopyConstructorFromArray) {
  ttb_indx arr_data[]{1, 2, 3};
  IndxArray arr(3, arr_data);
  ASSERT_EQ(arr.size(), 3);
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(arr[i], i + 1);
  }
}

TEST(TestIndxArray, CopyConstructorFromDoubleArray) {
  ttb_indx arr_data[]{0, 1, 2};
  IndxArray arr(3, arr_data);
  ASSERT_EQ(arr.size(), 3);
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(arr[i], i);
  }
}

TEST(TestIndxArray, CopyConstruct) {
  ttb_indx arr_data[]{1, 2, 3};
  IndxArray arr_a(3, arr_data);

  IndxArray arr_b(arr_a.size());
  deep_copy(arr_b, arr_a);
  ASSERT_EQ(arr_a, arr_b);
}

TEST(TestIndxArray, LeaveOneOutCopyConstructor) {
  ttb_indx arr_data[]{1, 2, 3};
  IndxArray arr_a(3, arr_data);

  IndxArray arr_b(arr_a, 1);
  ASSERT_EQ(arr_b.size(), 2);
  ASSERT_EQ(arr_b.size(), arr_a.size() - 1);
  ASSERT_EQ(arr_b[0], 1);
  ASSERT_EQ(arr_b[1], 3);
}

TEST(TestIndxArray, OperatorIsEqual) {
  IndxArray arr_a;
  ASSERT_TRUE(arr_a.empty());
  ASSERT_EQ(arr_a.size(), 0);

  ttb_indx arr_data[]{1, 2, 3};

  IndxArray arr_b(3, arr_data);
  ASSERT_FALSE(arr_b.empty());
  ASSERT_EQ(arr_b.size(), 3);

  IndxArray arr_c(3, arr_data);
  ASSERT_FALSE(arr_c.empty());
  ASSERT_EQ(arr_c.size(), 3);

  ASSERT_NE(arr_a, arr_b);
  ASSERT_NE(arr_a, arr_c);
  ASSERT_EQ(arr_b, arr_c);
}

TEST(TestIndxArray, Resize) {
  IndxArray arr_a(0);
  ASSERT_TRUE(arr_a.empty());
  ASSERT_EQ(arr_a.size(), 0);

  arr_a = IndxArray(5);
  ASSERT_FALSE(arr_a.empty());
  ASSERT_EQ(arr_a.size(), 5);

  ttb_indx arr_data[]{1, 2, 3};

  IndxArray arr_b(3, arr_data);
  ASSERT_FALSE(arr_b.empty());
  ASSERT_EQ(arr_b.size(), 3);

  arr_b = IndxArray(0);
  ASSERT_TRUE(arr_b.empty());
  ASSERT_EQ(arr_b.size(), 0);
}

TEST(TestIndxArray, OperatorSubscriptConst) {
  ttb_indx arr_data[]{0, 1, 2};
  IndxArray arr_a(3, arr_data);
  const IndxArray arr_b(arr_a);
  for (ttb_indx i = 0; i < arr_b.size(); i++) {
    ASSERT_EQ(arr_b[i], i);
  }
}

TEST(TestIndxArray, OperatorSubscriptNonCost) {
  ttb_indx arr_data[]{0, 1, 2};
  IndxArray arr_a(3, arr_data);
  IndxArray arr_b(arr_a);

  for (ttb_indx i = 0; i < arr_b.size(); i++) {
    arr_b[i] = i + 5;
  }

  for (ttb_indx i = 0; i < arr_b.size(); i++) {
    ASSERT_EQ(arr_b[i], i + 5);
  }
}

TEST(TestIndxArray, Prod) {
  IndxArray arr_a;

  ttb_indx arr_b_data[]{5, 6, 7, 8, 9};
  IndxArray arr_b(5, arr_b_data);

  ttb_indx arr_c_data[]{1, 2, 3};
  IndxArray arr_c(3, arr_c_data);

  ttb_indx arr_d_data[]{0, 1, 2};
  const IndxArray arr_d(3, arr_d_data);

  ASSERT_EQ(arr_a.prod(), 0);
  ASSERT_EQ(arr_b.prod(), 15120);
  ASSERT_EQ(arr_c.prod(), 6);
  ASSERT_EQ(arr_d.prod(), 0);
}

TEST(TestIndxArray, ProdWithDefault) {
  IndxArray arr_a;

  ttb_indx arr_b_data[]{5, 6, 7, 8, 9};
  IndxArray arr_b(5, arr_b_data);

  ttb_indx arr_c_data[]{1, 2, 3};
  IndxArray arr_c(3, arr_c_data);

  ttb_indx arr_d_data[]{0, 1, 2};
  const IndxArray arr_d(3, arr_d_data);

  ASSERT_EQ(arr_a.prod(1), 1);
  ASSERT_EQ(arr_b.prod(1), 15120);
  ASSERT_EQ(arr_c.prod(1), 6);
  ASSERT_EQ(arr_d.prod(1), 0);
}

TEST(TestIndxArray, ProdWithStartEnd) {
  ttb_indx arr_data[]{5, 6, 7, 8, 9};
  IndxArray arr(5, arr_data);

  ASSERT_EQ(arr.prod(1, 5), 3024);
  ASSERT_EQ(arr.prod(2, 2), 0);
}

TEST(TestIndxArray, ProdWithStartEndAndDefault) {
  ttb_indx arr_data[]{5, 6, 7, 8, 9};
  IndxArray arr(5, arr_data);
  ASSERT_EQ(arr.prod(2, 2, 1), 1);
}

TEST(TestIndxArray, Cumprod) {
  ttb_indx arr_a_data[]{5, 6, 7, 8, 9};
  IndxArray arr_a(5, arr_a_data);
  IndxArray arr_b(arr_a.size() + 1);
  arr_b.cumprod(arr_a);
  ASSERT_EQ(arr_b.size(), arr_a.size() + 1);

  ttb_real tmp = 1;
  for (ttb_indx i = 0; i < arr_b.size(); i++) {
    ASSERT_EQ(arr_b[i], tmp);

    if (i < arr_a.size()) {
      tmp = tmp * arr_a[i];
    }
  }
}

TEST(TestIndxArray, Zero) {
  ttb_indx arr_data[]{0, 1, 2};
  IndxArray arr(3, arr_data);
  ASSERT_FALSE(arr.empty());
  ASSERT_EQ(arr.size(), 3);

  arr.zero();
  for (ttb_indx i = 0; i < arr.size(); i++) {
    ASSERT_EQ(arr[i], 0);
  }
}

TEST(TestIndxArray, IsPermutation) {
  IndxArray arr_a(5);
  for (int i = 0; i < 5; i++) {
    arr_a[i] = 4 - i;
  }

  ASSERT_TRUE(arr_a.isPermutation());

  IndxArray arr_b(2);
  arr_b[0] = 1;
  arr_b[1] = 3;

  ASSERT_FALSE(arr_b.isPermutation());
}

TEST(TestIndxArray, Increment) {
  IndxArray dims(4), check(4), ii(4);

  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 2;
  dims[3] = 4;

  ii[0] = 0;
  ii[1] = 0;
  ii[2] = 0;
  ii[3] = 0;
  ii.increment(dims);

  check[0] = 0;
  check[1] = 0;
  check[2] = 0;
  check[3] = 1;
  ASSERT_EQ(ii, check);

  ii[0] = 0;
  ii[1] = 2;
  ii[2] = 1;
  ii[3] = 3;
  ii.increment(dims);

  check[0] = 1;
  check[1] = 0;
  check[2] = 0;
  check[3] = 0;
  ASSERT_EQ(ii, check);

  ii[0] = 1;
  ii[1] = 2;
  ii[2] = 1;
  ii[3] = 3;
  ii.increment(dims);

  check[0] = 2;
  check[1] = 0;
  check[2] = 0;
  check[3] = 0;
  ASSERT_EQ(ii, check);
}

} // namespace UnitTests
} // namespace Genten
