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

#include <Genten_Array.hpp>
#include <Genten_Util.hpp>

#include "Genten_Test_Utils.hpp"

#include <gtest/gtest.h>

namespace Genten {
namespace UnitTests {

template <typename ExecSpace> struct TestArrayT : public ::testing::Test {
  using exec_space = ExecSpace;
};

TYPED_TEST_SUITE(TestArrayT, genten_test_types);

TEST(TestArray, EmptyConstructor) {
  Array arr;
  ASSERT_TRUE(arr.empty());
}

TEST(TestArray, ArrayWithLengthOfZero) {
  Array arr(0);
  ASSERT_TRUE(arr.empty());
}

TEST(TestArray, ArrayWithSpecifiedLength) {
  Array arr(5);
  ASSERT_EQ(arr.size(), 5);
}

TEST(TestArray, ArrayWithShadowing) {
  // arr = [ 0 1 2 3 4 ] SHADOW
  ttb_real arr_data[]{0, 1, 2, 3, 4};
  Array arr(5, arr_data);
  ASSERT_EQ(arr.size(), 5);
  for (int i = 0; i < 5; ++i) {
    ASSERT_EQ(arr[i], i);
  }

  // arr = [ -5 1 2 3 4 ] SHADOW
  arr_data[0] = -5;
  ASSERT_EQ(arr[0], -5);
}

TEST(TestArray, ArrayWithoutShadowing) {
  ttb_real *arr_data = (ttb_real *)malloc(5 * sizeof(ttb_real));
  for (int i = 0; i < 5; i++) {
    arr_data[i] = i;
  }

  const bool with_shadowing = false;
  Array arr(5, arr_data, with_shadowing);
  free(arr_data);

  ASSERT_EQ(arr.size(), 5);
  for (int i = 0; i < 5; ++i) {
    ASSERT_EQ(arr[i], i);
  }
}

TEST(TestArray, CopyConstructor) {
  ttb_real arr_data[]{0, 1, 2, 3, 4};
  Array arr(5, arr_data);
  Array new_arr(arr.size());
  deep_copy(new_arr, arr);
  ASSERT_EQ(new_arr, arr);

  new_arr[0] = -1;
  ASSERT_NE(new_arr[0], arr[0]);
}

TEST(TestArray, Destructor) {
  ttb_real *arr_data = (ttb_real *)malloc(5 * sizeof(ttb_real));
  for (int i = 0; i < 5; i++) {
    arr_data[i] = i;
  }

  Array *arr = new Array(5, arr_data);
  delete arr;
  ASSERT_NE(arr_data, nullptr);

  free(arr_data);
}

TEST(TestArray, ResizingShadowdArray) {
  ttb_real arr_data[]{0, 1, 2, 3, 4};
  Array arr(5, arr_data);

  arr = Array(3);
  ASSERT_EQ(arr.size(), 3);
}

TEST(TestArray, AssignmentOperatorArray) {
  ttb_real arr_data_lhs[]{0, 1, 2, 3, 4};
  Array arr_lhs(5, arr_data_lhs);
  ASSERT_EQ(arr_lhs.size(), 5);

  Array arr_rhs;
  ASSERT_EQ(arr_rhs.size(), 0);

  arr_rhs = arr_lhs;
  ASSERT_EQ(arr_lhs, arr_rhs);
}

TEST(TestArray, AssignmentOperatorScalar) {
  Array arr(5, 0.0);
  ASSERT_EQ(arr.size(), 5);
  for (ttb_indx i = 0; i < arr.size(); ++i) {
    ASSERT_FLOAT_EQ(arr[i], 0.0);
  }

  arr = 0.5;
  ASSERT_EQ(arr.size(), 5);
  for (ttb_indx i = 0; i < arr.size(); ++i) {
    ASSERT_FLOAT_EQ(arr[i], 0.5);
  }
}

TEST(TestArray, Reset) {
  ttb_real arr_data[]{0, 1, 2, 3, 4};
  Array arr(5, arr_data);
  ASSERT_EQ(arr.size(), 5);

  arr = Array(3, 0.5);
  ASSERT_EQ(arr.size(), 3);
  for (ttb_indx i = 0; i < arr.size(); ++i) {
    ASSERT_FLOAT_EQ(arr[i], 0.5);
  }

  arr = Array(0, 0.0);
  ASSERT_TRUE(arr.empty());
  ASSERT_EQ(arr.size(), 0);
}

TEST(TestArray, OperatorSubscriptConst) {
  ttb_real arr_data[]{0, 1, 2, 3, 4};
  Array arr(5, arr_data);

  const Array new_arr(arr);
  ASSERT_EQ(new_arr, arr);
  for (int i = 0; i < 5; ++i) {
    ASSERT_EQ(new_arr[i], i);
  }
}

TEST(TestArray, OperatorSubscriptNonConst) {
  Array arr(5);
  for (int i = 0; i < 5; ++i) {
    arr[i] = static_cast<ttb_real>(i);
  }

  for (int i = 0; i < 5; ++i) {
    ASSERT_FLOAT_EQ(arr[i], static_cast<ttb_real>(i));
  }
}

TYPED_TEST(TestArrayT, NormTwo) {
  Array arr(5);
  ttb_real ans = 0.0;
  for (int i = 0; i < 5; i++) {
    arr[i] = i / 11.0;
    ans += (i / 11.0) * (i / 11.0);
  }
  ans = sqrt(ans);

  using exec_space = typename TestFixture::exec_space;
  ArrayT<exec_space> arr_dev = create_mirror_view(exec_space(), arr);
  deep_copy(arr_dev, arr);
  ASSERT_FLOAT_EQ(arr_dev.norm(NormTwo), ans);
}

TYPED_TEST(TestArrayT, NormOne) {
  Array arr(5);
  ttb_real ans = 0.0;
  for (int i = 0; i < 5; i++) {
    arr[i] = pow(-1.0, i) * i / 11.0;
    ans += i / 11.0;
  }

  using exec_space = typename TestFixture::exec_space;
  ArrayT<exec_space> arr_dev = create_mirror_view(exec_space(), arr);
  deep_copy(arr_dev, arr);
  ASSERT_FLOAT_EQ(arr_dev.norm(NormOne), ans);
}

TEST(TestArray, NormInf) {
  Array arr(5);
  ttb_real ans = 0.0;
  for (int i = 0; i < 5; i++) {
    arr[i] = pow(-1.0, i) * i / 11.0;
    ans += i / 11.0;
  }

  ans = 4.0 / 11.0;

  ASSERT_FLOAT_EQ(arr.norm(NormInf), ans);
}

TEST(TestArray, NNZ) {
  Array arr(5);
  for (int i = 0; i < 5; i++) {
    arr[i] = pow(-1.0, i) * i / 11.0;
  }

  ASSERT_EQ(arr.nnz(), 4);
}

TEST(TestArray, Dot) {
  Array arr_a(5);
  for (int i = 0; i < 5; i++) {
    arr_a[i] = pow(-1.0, i) * i / 11.0;
  }

  Array arr_b(5, 2.0);

  ttb_real ans = 0.0;
  for (int i = 0; i < 5; i++) {
    ans += 2.0 * arr_a[i];
  }

  ASSERT_FLOAT_EQ(arr_a.dot(arr_b), ans);
}

TEST(TestArray, Equal) {
  Array arr_a(5);
  for (int i = 0; i < 5; i++) {
    arr_a[i] = pow(-1.0, i) * i / 11.0;
  }

  Array arr_b(5, 2.0);
  ASSERT_FALSE(arr_b.isEqual(arr_a, MACHINE_EPSILON));

  Array arr_c = arr_a;
  ASSERT_TRUE(arr_c.isEqual(arr_a, MACHINE_EPSILON));
}

TEST(TestArray, Times) {
  Array arr_a(5, 2.5);
  arr_a.times(3.0);
  Array arr_answ(arr_a.size(), 7.5);
  ASSERT_TRUE(arr_a.isEqual(arr_answ, MACHINE_EPSILON));
}

TEST(TestArray, Invert) {
  Array arr_a(5, 7.5);
  arr_a.invert(9.375);
  Array arr_answ(arr_a.size(), 1.25);
  ASSERT_TRUE(arr_a.isEqual(arr_answ, MACHINE_EPSILON));
}

TEST(TestArray, Shift) {
  Array arr_a(5, 1.25);
  arr_a.shift(3.0);
  Array arr_answ(5, 4.25);
  ASSERT_TRUE(arr_a.isEqual(arr_answ, MACHINE_EPSILON));
}

TEST(TestArray, Power) {
  Array arr_a(5, 4.25);
  arr_a.power(2.0);
  Array arr_answ(5, 18.0625);
  ASSERT_TRUE(arr_a.isEqual(arr_answ, MACHINE_EPSILON));
}

TEST(TestArray, TimesArray) {
  Array arr_a(5, 18.0625);
  Array arr_b(5, 2.5);
  arr_a.times(3.0, arr_b);
  Array arr_answ(5, 7.5);
  ASSERT_TRUE(arr_a.isEqual(arr_answ, MACHINE_EPSILON));
}

TEST(TestArray, InvertArray) {
  Array arr_a(5, 7.5);
  Array arr_b = arr_a;
  arr_a.invert(9.375, arr_b);
  Array arr_answ(5, 1.25);
  ASSERT_TRUE(arr_a.isEqual(arr_answ, MACHINE_EPSILON));
}

TEST(TestArray, ShiftArray) {
  Array arr_a(5, 1.25);
  Array arr_b = arr_a;
  arr_a.shift(3.0, arr_b);
  Array arr_answ(5, 4.25);
  ASSERT_TRUE(arr_a.isEqual(arr_answ, MACHINE_EPSILON));
}

TEST(TestArray, PowerArray) {
  Array arr_a(5, 4.25);
  Array arr_b = arr_a;
  arr_a.power(2.0, arr_b);
  Array arr_answ(5, 18.0625);
  ASSERT_TRUE(arr_a.isEqual(arr_answ, MACHINE_EPSILON));
}

TEST(TestArray, PlusArray) {
  Array arr_a(5, 2.3);
  Array arr_b(5, 2.5);
  arr_a.plus(arr_b);
  Array arr_answ(5, 4.8);
  ASSERT_TRUE(arr_a.isEqual(arr_answ, MACHINE_EPSILON));
}

TEST(TestArray, MinusArray) {
  Array arr_a(5, 2.3);
  Array arr_b(5, 2.5);
  arr_a.minus(arr_b);
  Array arr_answ(5, -0.2);
  ASSERT_TRUE(arr_a.isEqual(arr_answ, MACHINE_EPSILON));
}

TYPED_TEST(TestArrayT, TimesArray) {
  using exec_space = typename TestFixture::exec_space;

  Array arr_a(5, 2.3);
  ArrayT<exec_space> arr_a_dev = create_mirror_view(exec_space(), arr_a);
  deep_copy(arr_a_dev, arr_a);

  Array arr_b(5, 2.5);
  ArrayT<exec_space> arr_b_dev = create_mirror_view(exec_space(), arr_b);
  deep_copy(arr_b_dev, arr_b);

  arr_a_dev.times(arr_b_dev);
  deep_copy(arr_a, arr_a_dev);

  Array arr_answ(5, 5.75);
  ASSERT_TRUE(arr_a.isEqual(arr_answ, MACHINE_EPSILON));
}

TEST(TestArray, Divide) {
  Array arr_a(5, 2.3);
  Array arr_b(5, 2.5);
  arr_a.divide(arr_b);
  Array arr_answ(5, 0.92);
  ASSERT_TRUE(arr_a.isEqual(arr_answ, MACHINE_EPSILON));
}

TEST(TestArray, ArrayEqualsArrayPlusArray) {
  Array arr_a(5, 2.3);
  Array arr_b(5, 2.5);
  Array arr_c(5);

  arr_c.plus(arr_a, arr_b);
  Array arr_answ(5, 4.8);
  ASSERT_TRUE(arr_c.isEqual(arr_answ, MACHINE_EPSILON));
}

TEST(TestArray, ArrayEqualsArrayMinusArray) {
  Array arr_a(5, 2.3);
  Array arr_b(5, 2.5);
  Array arr_c(5);

  arr_c.minus(arr_a, arr_b);
  Array arr_answ(5, -0.2);
  ASSERT_TRUE(arr_c.isEqual(arr_answ, MACHINE_EPSILON));
}

TEST(TestArray, ArrayEqualsArrayTimesArray) {
  Array arr_a(5, 2.3);
  Array arr_b(5, 2.5);
  Array arr_c(5);

  arr_c.times(arr_a, arr_b);
  Array arr_answ(5, 5.75);
  ASSERT_TRUE(arr_c.isEqual(arr_answ, MACHINE_EPSILON));
}

TEST(TestArray, ArrayEqualsArrayDivideArray) {
  Array arr_a(5, 2.3);
  Array arr_b(5, 2.5);
  Array arr_c(5);

  arr_c.divide(arr_a, arr_b);
  Array arr_answ(5, 0.92);
  ASSERT_TRUE(arr_c.isEqual(arr_answ, MACHINE_EPSILON));
}

} // namespace UnitTests
} // namespace Genten
