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

#include <Genten_Tensor.hpp>
#include <Genten_Sptensor.hpp>
#include <Genten_Util.hpp>

#include "Genten_Test_Utils.hpp"

#include <gtest/gtest.h>

namespace Genten {
namespace UnitTests {

template <typename ExecSpace> struct TestTensorT : public ::testing::Test {
  using exec_space = ExecSpace;
};

TYPED_TEST_SUITE(TestTensorT, genten_test_types);

TEST(TestTensor, EmptyConstructor) {
  Tensor t;
  ASSERT_EQ(t.nnz(), 0);
  ASSERT_EQ(t.ndims(), 0);
}

TEST(TestTensor, ZerothOrderTensor) {
  Tensor t_a(0);
  ASSERT_EQ(t_a.nnz(), 0);
  ASSERT_EQ(t_a.ndims(), 0);

  IndxArray dims;
  Tensor t_b(dims);
  ASSERT_EQ(t_b.nnz(), 0);
  ASSERT_EQ(t_b.ndims(), 0);
}

TEST(TestTensor, AccessingElementsLeft) {
  IndxArray dims(3);
  dims[0] = 4;
  dims[1] = 2;
  dims[2] = 3;
  Tensor t(dims, 0.0, TensorLayout::Left);
  t[0] = 1.0;
  t[23] = 2.0;

  ASSERT_EQ(t.nnz(), 24);

  IndxArray oSub(3);
  oSub[0] = 0;
  oSub[1] = 0;
  oSub[2] = 0;

  ASSERT_EQ(t[oSub], 1.0);

  oSub[0] = 3;
  oSub[1] = 1;
  oSub[2] = 2;

  ASSERT_EQ(t[oSub], 2.0);

  t[oSub] = 3.0;

  ASSERT_EQ(t[23], 3.0);

  t[5] = 4.0;
  oSub[0] = 1;
  oSub[1] = 1;
  oSub[2] = 0;

  ASSERT_EQ(t[oSub], 4.0);
}

TEST(TestTensor, AccessingElementsRight) {
  IndxArray dims(3);
  dims[0] = 4;
  dims[1] = 2;
  dims[2] = 3;
  Tensor t(dims, 0.0, TensorLayout::Right);
  t[0] = 1.0;
  t[23] = 2.0;

  ASSERT_EQ(t.nnz(), 24);

  IndxArray oSub(3);
  oSub[0] = 0;
  oSub[1] = 0;
  oSub[2] = 0;

  ASSERT_EQ(t[oSub], 1.0);

  oSub[0] = 3;
  oSub[1] = 1;
  oSub[2] = 2;

  ASSERT_EQ(t[oSub], 2.0);

  t[oSub] = 3.0;

  ASSERT_EQ(t[23], 3.0);

  t[5] = 4.0;
  oSub[0] = 0;
  oSub[1] = 1;
  oSub[2] = 2;

  ASSERT_EQ(t[oSub], 4.0);
}

#define ASSERT_EQ_INDXARRAY(a, r, ae, re)       \
  ASSERT_EQ(a[0], ae[0]);                       \
  ASSERT_EQ(a[1], ae[1]);                       \
  ASSERT_EQ(a[2], ae[2]);                       \
  ASSERT_EQ(r, re)                              \

TEST(TestTensor, IncrementSubsLeft) {
  IndxArray dims = { 2, 2, 2 };
  Tensor t(dims, 0.0, TensorLayout::Left);

  IndxArray sub;
  bool rem;

  sub = { 0, 0, 0 };
  rem = t.increment_sub(sub, 0);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({0,1,0}), true);
  rem = t.increment_sub(sub, 0);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({0,0,1}), true);
  rem = t.increment_sub(sub, 0);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({0,1,1}), true);
  rem = t.increment_sub(sub, 0);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({0,0,2}), false);

  sub = { 0, 0, 0 };
  rem = t.increment_sub(sub, 1);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({1,0,0}), true);
  rem = t.increment_sub(sub, 1);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({0,0,1}), true);
  rem = t.increment_sub(sub, 1);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({1,0,1}), true);
  rem = t.increment_sub(sub, 1);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({0,0,2}), false);

  sub = { 0, 0, 0 };
  rem = t.increment_sub(sub, 2);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({1,0,0}), true);
  rem = t.increment_sub(sub, 2);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({0,1,0}), true);
  rem = t.increment_sub(sub, 2);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({1,1,0}), true);
  rem = t.increment_sub(sub, 2);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({0,2,0}), false);
}

TEST(TestTensor, IncrementSubsRight) {
  IndxArray dims = { 2, 2, 2 };
  Tensor t(dims, 0.0, TensorLayout::Right);

  IndxArray sub;
  bool rem;

  sub = { 0, 0, 0 };
  rem = t.increment_sub(sub, 0);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({0,0,1}), true);
  rem = t.increment_sub(sub, 0);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({0,1,0}), true);
  rem = t.increment_sub(sub, 0);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({0,1,1}), true);
  rem = t.increment_sub(sub, 0);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({0,2,0}), false);

  sub = { 0, 0, 0 };
  rem = t.increment_sub(sub, 1);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({0,0,1}), true);
  rem = t.increment_sub(sub, 1);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({1,0,0}), true);
  rem = t.increment_sub(sub, 1);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({1,0,1}), true);
  rem = t.increment_sub(sub, 1);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({2,0,0}), false);

  sub = { 0, 0, 0 };
  rem = t.increment_sub(sub, 2);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({0,1,0}), true);
  rem = t.increment_sub(sub, 2);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({1,0,0}), true);
  rem = t.increment_sub(sub, 2);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({1,1,0}), true);
  rem = t.increment_sub(sub, 2);
  ASSERT_EQ_INDXARRAY(sub, rem, IndxArray({2,0,0}), false);
}

TEST(TestTensor, SwitchLayout) {
  IndxArray dims = {2, 3, 4};
  Tensor t_left(dims, 0.0, TensorLayout::Left);
  const ttb_indx ne = t_left.numel();
  for (ttb_indx i=0; i<ne; ++i)
    t_left[i] = i+1;

  Tensor t_right = t_left.switch_layout(TensorLayout::Right);

  const ttb_indx nd = t_left.ndims();
  IndxArray sub(nd);
  for (ttb_indx i=0; i<ne; ++i) {
    t_left.ind2sub(sub, i);
    const ttb_real val = t_left[i];
    const ttb_real val_left = t_left[sub];
    const ttb_real val_right = t_right[sub];
    ASSERT_EQ(val_left, val);
    ASSERT_EQ(val_right, val);
  }
}

TEST(TestTensor, Norm) {
  IndxArray dims(2);
  dims[0] = 1;
  dims[1] = 2;

  Tensor t(dims);
  t[0] = 1.0;
  t[1] = 3.0;

  ASSERT_FLOAT_EQ(t.norm(), std::sqrt(10.0));
}

TYPED_TEST(TestTensorT, DenseTensorFromSparseTensor) {
  using exec_space = typename TestFixture::exec_space;
  using host_exec_space = DefaultHostExecutionSpace;

  IndxArray dims(3);
  dims[0] = 3;
  dims[1] = 4;
  dims[2] = 5;

  SptensorT<host_exec_space> st(dims, 5);
  for (ttb_indx i = 0; i < st.nnz(); i++) {
    st.subscript(i, 0) = i % 3;
    st.subscript(i, 1) = (i + 1) % 4;
    st.subscript(i, 2) = (i + 2) % 5;
    st.value(i) = i * 1.5 + 1;
  }

  SptensorT<exec_space> st_dev = create_mirror_view(exec_space(), st);
  deep_copy(st_dev, st);
  TensorT<exec_space> t_dev(st_dev);
  ASSERT_FLOAT_EQ(t_dev.norm(), st_dev.norm());
}

TYPED_TEST(TestTensorT, DenseTensorFromKruskalTensor) {
  using exec_space = typename TestFixture::exec_space;
  using host_exec_space = DefaultHostExecutionSpace;

  IndxArray dims(2);
  dims[0] = 1;
  dims[1] = 2;

  KtensorT<host_exec_space> kt(3, 2, dims);
  kt.setWeightsRand();
  kt.setMatricesRand();

  KtensorT<exec_space> kt_dev = create_mirror_view(exec_space(), kt);
  deep_copy(kt_dev, kt);

  TensorT<exec_space> t_dev(kt_dev);
  TensorT<host_exec_space> t = create_mirror_view(host_exec_space(), t_dev);
  deep_copy(t, t_dev);

  IndxArray sub(2);
  sub[0] = 0;
  sub[1] = 0;

  ASSERT_FLOAT_EQ(t[sub], compute_Ktensor_value(kt, sub));

  sub[0] = 0;
  sub[1] = 1;

  ASSERT_FLOAT_EQ(t[sub], compute_Ktensor_value(kt, sub));
}

} // namespace UnitTests
} // namespace Genten
