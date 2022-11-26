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
#include <sstream>

#include "Genten_Test_Utils.hpp"

#include <gtest/gtest.h>

namespace Genten {
namespace UnitTests {

template <typename ExecSpace> struct TestFacMatrixT : public ::testing::Test {
  using exec_space = ExecSpace;
};

TYPED_TEST_SUITE(TestFacMatrixT, genten_test_types);

TEST(TestFacMatrix, EmptyConstructor) {
  FacMatrix fm;
  ASSERT_EQ(fm.nRows(), 0);
  ASSERT_EQ(fm.nCols(), 0);
}

TEST(TestFacMatrix, SizeConstructor) {
  FacMatrix fm(3, 2);
  ASSERT_EQ(fm.nRows(), 3);
  ASSERT_EQ(fm.nCols(), 2);
}

TEST(TestFacMatrix, DataConstructor) {
  ttb_real fm_data[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
  FacMatrix fm(3, 3, fm_data);
  double val = 1;
  for (ttb_indx j = 0; j < fm.nRows(); ++j) {
    for (ttb_indx i = 0; i < fm.nCols(); ++i) {
      ASSERT_FLOAT_EQ(fm.entry(i, j), val);
      ++val;
    }
  }
}

TEST(TestFacMatrix, EntryConst) {
  ttb_real fm_data[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
  FacMatrix fm(3, 3, fm_data);

  FacMatrix fm_const(fm);
  ASSERT_TRUE(fm_const.isEqual(fm, MACHINE_EPSILON));
}

TEST(TestFacMatrix, Resize) {
  FacMatrix fm;
  fm = FacMatrix(2, 2);
  ASSERT_EQ(fm.nRows(), 2);
  ASSERT_EQ(fm.nCols(), 2);
}

TEST(TestFacMatrix, AssignmentOperator) {
  FacMatrix fm(2, 2);
  fm = 5;
  ASSERT_FLOAT_EQ(fm.entry(0, 0), 5);
  ASSERT_FLOAT_EQ(fm.entry(0, 1), 5);
  ASSERT_FLOAT_EQ(fm.entry(1, 0), 5);
  ASSERT_FLOAT_EQ(fm.entry(1, 1), 5);
}

TEST(TestFacMatrix, Reset) {
  FacMatrix fm(2, 2);
  fm = 5;

  fm = FacMatrix(1, 1);
  fm = 3;
  ASSERT_EQ(fm.nRows(), 1);
  ASSERT_EQ(fm.nCols(), 1);
  ASSERT_FLOAT_EQ(fm.entry(0, 0), 3);
}

TYPED_TEST(TestFacMatrixT, Gramian) {
  FacMatrix fm_b;
  FacMatrix fm_d;
  import_matrix("data/B_matrix.txt", fm_b);
  import_matrix("data/D_matrix.txt", fm_d);

  FacMatrix fm_e(fm_b.nCols(), fm_b.nCols());

  using exec_space = typename TestFixture::exec_space;
  FacMatrixT<exec_space> fm_b_dev = create_mirror_view(exec_space(), fm_b);
  FacMatrixT<exec_space> fm_e_dev = create_mirror_view(exec_space(), fm_e);
  deep_copy(fm_b_dev, fm_b);
  fm_e_dev.gramian(fm_b_dev, true);
  deep_copy(fm_e, fm_e_dev);

  ASSERT_EQ(fm_e.nCols(), fm_d.nCols());
  ASSERT_EQ(fm_e.nRows(), fm_d.nRows());

  for (ttb_indx j = 0; j < fm_e.nCols(); j++) {
    for (ttb_indx i = 0; i < fm_e.nRows(); i++) {
      ASSERT_FLOAT_EQ(fm_e.entry(i, j), fm_d.entry(i, j));
    }
  }
}

TYPED_TEST(TestFacMatrixT, Gemm) {
  // gemm (matrix-matrix multiply)
  constexpr ttb_real alpha = 1.5;
  constexpr ttb_real beta = 2.1;
  constexpr ttb_real c0 = 0.6;

  using exec_space = typename TestFixture::exec_space;

  auto test_gemm = [=](const FacMatrix &fm_a, const FacMatrix &fm_b,
                       const FacMatrix &fm_c, const bool trans_a,
                       const bool trans_b, const std::string &label) {
    FacMatrixT<exec_space> fm_a_dev = create_mirror_view(exec_space(), fm_a);
    FacMatrixT<exec_space> fm_b_dev = create_mirror_view(exec_space(), fm_b);
    FacMatrixT<exec_space> fm_c_dev = create_mirror_view(exec_space(), fm_c);
    deep_copy(fm_a_dev, fm_a);
    deep_copy(fm_b_dev, fm_b);
    deep_copy(fm_c_dev, fm_c);
    fm_c_dev.gemm(trans_a, trans_b, alpha, fm_a_dev, fm_b_dev, beta);
    deep_copy(fm_c, fm_c_dev);
    const ttb_indx p = trans_a ? fm_a.nRows() : fm_a.nCols();
    for (ttb_indx i = 0; i < fm_c.nRows(); ++i) {
      for (ttb_indx j = 0; j < fm_c.nCols(); ++j) {
        ttb_real tmp = beta * c0;
        for (ttb_indx k = 0; k < p; ++k) {
          if (trans_a && trans_b) {
            tmp += alpha * fm_a(k, i) * fm_b(j, k);
          } else if (trans_a) {
            tmp += alpha * fm_a(k, i) * fm_b(k, j);
          } else if (trans_b) {
            tmp += alpha * fm_a(i, k) * fm_b(j, k);
          } else {
            tmp += alpha * fm_a(i, k) * fm_b(k, j);
          }
        }

        std::ostringstream oss;
        oss << "fm_c(" << i << "," << j << ") = " << fm_c(i, j)
            << " tmp = " << tmp;
        GENTEN_FLOAT_EQ(fm_c.entry(i, j), tmp, oss.str().c_str());
      }
    }

    const std::string msg = label + std::string(" yields expected answer");
    INFO_MSG(msg.c_str());
  };

  // no-trans, no-trans
  {
    FacMatrix a(3, 5);
    FacMatrix b(5, 7);
    FacMatrix c(3, 7);
    a.rand();
    b.rand();
    c = c0;
    test_gemm(a, b, c, false, false, "gemm(false,false)");
  }

  // no-trans, trans
  {
    FacMatrix a(3, 5);
    FacMatrix b(7, 5);
    FacMatrix c(3, 7);
    a.rand();
    b.rand();
    c = c0;
    test_gemm(a, b, c, false, true, "gemm(false,true)");
  }

  // trans, no-trans
  {
    FacMatrix a(5, 3);
    FacMatrix b(5, 7);
    FacMatrix c(3, 7);
    a.rand();
    b.rand();
    c = c0;
    test_gemm(a, b, c, true, false, "gemm(true,false)");
  }

  // trans, trans
  {
    FacMatrix a(5, 3);
    FacMatrix b(7, 5);
    FacMatrix c(3, 7);
    a.rand();
    b.rand();
    c = c0;
    test_gemm(a, b, c, true, true, "gemm(true,true)");
  }
}

TEST(TestFacMatrix, HadamardProductThrow) {
  FacMatrix fm_b;
  import_matrix("data/B_matrix.txt", fm_b);
  FacMatrix fm_d;
  import_matrix("data/D_matrix.txt", fm_d);

  ASSERT_ANY_THROW(fm_b.times(fm_d));
}

TYPED_TEST(TestFacMatrixT, HadamardProduct) {
  using exec_space = typename TestFixture::exec_space;

  FacMatrix fm_b;
  import_matrix("data/B_matrix.txt", fm_b);

  FacMatrixT<exec_space> fm_b_dev = create_mirror_view(exec_space(), fm_b);
  deep_copy(fm_b_dev, fm_b);

  FacMatrix fm_f(3, 2);
  fm_f = 2;

  FacMatrixT<exec_space> fm_f_dev = create_mirror_view(exec_space(), fm_f);
  deep_copy(fm_f_dev, fm_f);
  fm_f_dev.times(fm_b_dev);
  deep_copy(fm_f, fm_f_dev);

  ASSERT_EQ(fm_f.nRows(), 3);
  ASSERT_EQ(fm_f.nCols(), 2);

  ttb_real val = 0.1;
  for (ttb_indx j = 0; j < fm_f.nCols(); j++) {
    for (ttb_indx i = 0; i < fm_f.nRows(); i++) {
      ASSERT_FLOAT_EQ(fm_f.entry(i, j), 2 * val);
      val += 0.1;
    }
  }
}

TEST(TestFacMatrix, Transpose) {
  FacMatrix fm_b;
  import_matrix("data/B_matrix.txt", fm_b);

  FacMatrix fm(fm_b.nCols(), fm_b.nRows());
  fm.transpose(fm_b);
  ASSERT_EQ(fm.nCols(), fm_b.nRows());
  ASSERT_EQ(fm.nRows(), fm_b.nCols());

  for (ttb_indx j = 0; j < fm.nCols(); j++) {
    for (ttb_indx i = 0; i < fm.nRows(); i++) {
      ASSERT_FLOAT_EQ(fm.entry(i, j), fm_b.entry(j, i));
    }
  }
}

TYPED_TEST(TestFacMatrixT, Oprod) {
  using exec_space = typename TestFixture::exec_space;

  ttb_real arr_data[]{0.1, 0.2, 0.3};
  const Array arr(3, arr_data);
  FacMatrix fm(arr.size(), arr.size());
  FacMatrixT<exec_space> fm_dev = create_mirror_view(exec_space(), fm);
  ArrayT<exec_space> arr_dev = create_mirror_view(exec_space(), arr);
  deep_copy(fm_dev, fm);
  deep_copy(arr_dev, arr);
  fm_dev.oprod(arr_dev);
  deep_copy(fm, fm_dev);
  ASSERT_EQ(fm.nRows(), 3);
  ASSERT_EQ(fm.nCols(), 3);
  for (ttb_indx j = 0; j < fm.nCols(); j++) {
    for (ttb_indx i = 0; i < fm.nRows(); i++) {
      const ttb_real val = arr[i] * arr[j];
      ASSERT_FLOAT_EQ(fm.entry(i, j), val);
    }
  }
}

TYPED_TEST(TestFacMatrixT, LinearSolverDiagonalMatrix) {
  using exec_space = typename TestFixture::exec_space;

  FacMatrix fm_a(2, 2);
  fm_a.entry(0, 0) = 1.0;
  fm_a.entry(1, 0) = 0.0;
  fm_a.entry(0, 1) = 0.0;
  fm_a.entry(1, 1) = 2.0;

  FacMatrix fm_b(1, 2);
  fm_b.entry(0, 0) = 3.0;
  fm_b.entry(0, 1) = 4.0;

  FacMatrixT<exec_space> fm_a_dev = create_mirror_view(exec_space(), fm_a);
  FacMatrixT<exec_space> fm_b_dev = create_mirror_view(exec_space(), fm_b);
  deep_copy(fm_a_dev, fm_a);
  deep_copy(fm_b_dev, fm_b);

  fm_b_dev.solveTransposeRHS(fm_a_dev);
  deep_copy(fm_b, fm_b_dev);

  FacMatrix fm_c(1, 2);
  fm_c.entry(0, 0) = 3.0;
  fm_c.entry(0, 1) = 2.0;

  // Very slightly loosening of tolerance for GPU
  ASSERT_TRUE(fm_c.isEqual(fm_b, MACHINE_EPSILON * 10.0));
}

TYPED_TEST(TestFacMatrixT, LinearSolverIndefiniteMatrix) {
  using exec_space = typename TestFixture::exec_space;
  using host_exec_space = DefaultHostExecutionSpace;

  FacMatrix fm_a(2, 2);
  fm_a.entry(0, 0) = 1.0;
  fm_a.entry(1, 0) = 2.0;
  fm_a.entry(0, 1) = 2.0;
  fm_a.entry(1, 1) = 1.0;

  FacMatrix fm_b(3, 2);
  fm_b.entry(0, 0) = 1.0;
  fm_b.entry(0, 1) = 0.0;
  fm_b.entry(1, 0) = 0.0;
  fm_b.entry(1, 1) = 1.0;
  fm_b.entry(2, 0) = -1.0;
  fm_b.entry(2, 1) = 2.0;

  FacMatrix fm_c(3, 2);
  fm_c.entry(0, 0) = -1.0 / 3.0;
  fm_c.entry(0, 1) = 2.0 / 3.0;
  fm_c.entry(1, 0) = 2.0 / 3.0;
  fm_c.entry(1, 1) = -1.0 / 3.0;
  fm_c.entry(2, 0) = 5.0 / 3.0;
  fm_c.entry(2, 1) = -4.0 / 3.0;

  FacMatrixT<exec_space> fm_a_dev = create_mirror_view(exec_space(), fm_a);
  deep_copy(fm_a_dev, fm_a);

  FacMatrixT<exec_space> fm_b1_dev(fm_b.nRows(), fm_b.nCols());
  deep_copy(fm_b1_dev, fm_b);
  fm_b1_dev.solveTransposeRHS(fm_a_dev, true);
  auto fm_b1 = create_mirror_view(host_exec_space(), fm_b1_dev);
  deep_copy(fm_b1, fm_b1_dev);

  ASSERT_TRUE(fm_c.isEqual(fm_b1, 10.0 * MACHINE_EPSILON));
}

TYPED_TEST(TestFacMatrixT, LinearSolverSymetricIndefiniteMatrix) {
  // Symmetric, indefinite solver currently doesn't work on GPU (solver not
  // fully implemented in cuSOLVER)
  using exec_space = typename TestFixture::exec_space;
  using host_exec_space = DefaultHostExecutionSpace;

  if (is_gpu_space<exec_space>::value) {
    return;
  }

  FacMatrix fm_a(2, 2);
  fm_a.entry(0, 0) = 1.0;
  fm_a.entry(1, 0) = 2.0;
  fm_a.entry(0, 1) = 2.0;
  fm_a.entry(1, 1) = 1.0;

  FacMatrix fm_b(3, 2);
  fm_b.entry(0, 0) = 1.0;
  fm_b.entry(0, 1) = 0.0;
  fm_b.entry(1, 0) = 0.0;
  fm_b.entry(1, 1) = 1.0;
  fm_b.entry(2, 0) = -1.0;
  fm_b.entry(2, 1) = 2.0;

  FacMatrix fm_c(3, 2);
  fm_c.entry(0, 0) = -1.0 / 3.0;
  fm_c.entry(0, 1) = 2.0 / 3.0;
  fm_c.entry(1, 0) = 2.0 / 3.0;
  fm_c.entry(1, 1) = -1.0 / 3.0;
  fm_c.entry(2, 0) = 5.0 / 3.0;
  fm_c.entry(2, 1) = -4.0 / 3.0;

  FacMatrixT<exec_space> fm_a_dev = create_mirror_view(exec_space(), fm_a);
  deep_copy(fm_a_dev, fm_a);

  FacMatrixT<exec_space> fm_b1_dev(fm_b.nRows(), fm_b.nCols());
  deep_copy(fm_b1_dev, fm_b);
  fm_b1_dev.solveTransposeRHS(fm_a_dev, false, Upper, true);
  auto fm_b1 = create_mirror_view(host_exec_space(), fm_b1_dev);
  deep_copy(fm_b1, fm_b1_dev);
  ASSERT_TRUE(fm_c.isEqual(fm_b1, MACHINE_EPSILON));

  FacMatrixT<exec_space> fm_b2_dev(fm_b.nRows(), fm_b.nCols());
  deep_copy(fm_b2_dev, fm_b);
  fm_b2_dev.solveTransposeRHS(fm_a_dev, false, Upper, false);
  auto fm_b2 = create_mirror_view(host_exec_space(), fm_b2_dev);
  deep_copy(fm_b2, fm_b2_dev);
  ASSERT_TRUE(fm_c.isEqual(fm_b2, MACHINE_EPSILON));
}

TYPED_TEST(TestFacMatrixT, ColNorms) {
  using exec_space = typename TestFixture::exec_space;

  Array nrms(3), nrms_chk(3);

  // set fm_a = [3 0 0; 4 1 0; 0 0 0]
  FacMatrix fm_a(3, 3);
  fm_a.entry(0, 0) = 3;
  fm_a.entry(1, 0) = 4;
  fm_a.entry(1, 1) = 1;

  FacMatrixT<exec_space> fm_a_dev = create_mirror_view(exec_space(), fm_a);
  deep_copy(fm_a_dev, fm_a);

  ArrayT<exec_space> nrms_dev = create_mirror_view(exec_space(), nrms);

  fm_a_dev.colNorms(NormInf, nrms_dev, 0.0);
  deep_copy(nrms, nrms_dev);
  nrms_chk[0] = 4;
  nrms_chk[1] = 1;
  nrms_chk[2] = 0;
  ASSERT_TRUE(nrms.isEqual(nrms_chk, MACHINE_EPSILON));

  fm_a_dev.colNorms(NormOne, nrms_dev, 0.0);
  deep_copy(nrms, nrms_dev);
  nrms_chk[0] = 7;
  ASSERT_TRUE(nrms.isEqual(nrms_chk, MACHINE_EPSILON));

  fm_a_dev.colNorms(NormTwo, nrms_dev, 0.0);
  deep_copy(nrms, nrms_dev);
  nrms_chk[0] = 5;
  ASSERT_TRUE(nrms.isEqual(nrms_chk, MACHINE_EPSILON));
}

TYPED_TEST(TestFacMatrixT, ColScale) {
  using exec_space = typename TestFixture::exec_space;

  Array weights(3);
  weights[0] = 3;
  weights[1] = 2;
  weights[2] = 1;

  FacMatrix fm_a(3, 3);
  fm_a.entry(0, 0) = 3;
  fm_a.entry(1, 0) = 4;
  fm_a.entry(1, 1) = 1;

  FacMatrixT<exec_space> fm_a_dev = create_mirror_view(exec_space(), fm_a);
  deep_copy(fm_a_dev, fm_a);

  FacMatrix fm_b(fm_a.nRows(), fm_a.nCols());
  deep_copy(fm_b, fm_a);

  ArrayT<exec_space> weights_dev = create_mirror_view(exec_space(), weights);
  deep_copy(weights_dev, weights);
  fm_a_dev.colScale(weights_dev, false);
  deep_copy(fm_a, fm_a_dev);

  for (ttb_indx i = 0; i < 3; i++) {
    for (ttb_indx j = 0; j < 3; j++) {
      ASSERT_FLOAT_EQ(fm_a.entry(i, j), fm_b.entry(i, j) * weights[j]);
    }
  }

  fm_a_dev.colScale(weights_dev, true);
  deep_copy(fm_a, fm_a_dev);
  ASSERT_TRUE(fm_a.isEqual(fm_b, MACHINE_EPSILON));
}

TYPED_TEST(TestFacMatrixT, RowScale) {
  using exec_space = typename TestFixture::exec_space;

  FacMatrix fm_a(3, 3);
  fm_a.entry(0, 0) = 3;
  fm_a.entry(1, 0) = 4;
  fm_a.entry(1, 1) = 1;

  FacMatrixT<exec_space> fm_a_dev = create_mirror_view(exec_space(), fm_a);
  deep_copy(fm_a_dev, fm_a);

  FacMatrix fm_b(fm_a.nRows(), fm_a.nCols());
  deep_copy(fm_b, fm_a);

  Array weights(3);
  weights[0] = 3;
  weights[1] = 2;
  weights[2] = 1;

  ArrayT<exec_space> weights_dev = create_mirror_view(exec_space(), weights);
  deep_copy(weights_dev, weights);

  fm_a_dev.rowScale(weights_dev, false);
  deep_copy(fm_a, fm_a_dev);

  for (ttb_indx i = 0; i < 3; i++) {
    for (ttb_indx j = 0; j < 3; j++) {
      ASSERT_FLOAT_EQ(fm_a.entry(i, j), fm_b.entry(i, j) * weights[i]);
    }
  }

  fm_a_dev.rowScale(weights_dev, true);
  deep_copy(fm_a, fm_a_dev);
  ASSERT_TRUE(fm_a.isEqual(fm_b, MACHINE_EPSILON));
}

TYPED_TEST(TestFacMatrixT, Permute) {
  using exec_space = typename TestFixture::exec_space;

  ttb_real pdata[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
  ttb_real pdata_new[]{7, 8, 9, 4, 5, 6, 1, 2, 3};
  ttb_indx idata[]{2, 1, 0};

  IndxArray ind(3, idata);
  FacMatrix p(3, 3, pdata);
  FacMatrix p_new(3, 3, pdata_new);

  FacMatrixT<exec_space> p_dev = create_mirror_view(exec_space(), p);
  deep_copy(p_dev, p);
  p_dev.permute(ind);
  deep_copy(p, p_dev);

  for (ttb_indx i = 0; i < 3; i++) {
    for (ttb_indx j = 0; j < 3; j++) {
      ASSERT_FLOAT_EQ(p.entry(i, j), p_new.entry(i, j));
    }
  }
}

TYPED_TEST(TestFacMatrixT, Innerprod) {
  using exec_space = typename TestFixture::exec_space;

  const ttb_indx m = 50;
  const ttb_indx n = 20;

  FacMatrix fm_a(m, n);
  FacMatrix fm_b(m, n);
  Array w(n);

  ttb_real ip_true = 0.0;
  for (ttb_indx i = 0; i < m; ++i) {
    for (ttb_indx j = 0; j < n; ++j) {
      fm_a.entry(i, j) = i + j;
      fm_b.entry(i, j) = 10 * (i + j);
      w[j] = j + 1;
      ip_true += w[j] * fm_a.entry(i, j) * fm_b.entry(i, j);
    }
  }

  FacMatrixT<exec_space> fm_a_dev = create_mirror_view(exec_space(), fm_a);
  FacMatrixT<exec_space> fm_b_dev = create_mirror_view(exec_space(), fm_b);
  ArrayT<exec_space> w_dev = create_mirror_view(exec_space(), w);

  deep_copy(fm_a_dev, fm_a);
  deep_copy(fm_b_dev, fm_b);
  deep_copy(w_dev, w);
  ttb_real ip = fm_a_dev.innerprod(fm_b_dev, w_dev);

  ASSERT_FLOAT_EQ(ip, ip_true);
}

} // namespace UnitTests
} // namespace Genten
