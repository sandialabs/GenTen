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
#include <Genten_Sptensor.hpp>

#include "Genten_Test_Utils.hpp"

#include <gtest/gtest.h>

namespace Genten {
namespace UnitTests {

TEST(TestSptensor, EmptyConstructor) {
  Sptensor st;
  ASSERT_EQ(st.nnz(), 0);
}

TEST(TestSptensor, LengthConstructor) {
  IndxArray dims;
  Sptensor st_a(dims, 0);
  ASSERT_EQ(st_a.nnz(), 0);

  dims = IndxArray(3);
  dims[0] = 3;
  dims[1] = 4;
  dims[2] = 5;
  Sptensor st_b(dims, 5);
  ASSERT_EQ(st_b.nnz(), 5);

  Genten::DistTensorContext<DefaultHostExecutionSpace> dtc;
  Sptensor st_distributed = dtc.distributeTensor(st_b);

  const ProcessorMap *pmap = dtc.pmap_ptr().get();
  st_distributed.setProcessorMap(pmap);

  ASSERT_EQ(st_distributed.global_nnz(), 5);
  ASSERT_EQ(st_distributed.global_numel(), 3 * 4 * 5);
  ASSERT_FLOAT_EQ(st_distributed.global_numel_float(), 3 * 4 * 5);
  if (DistContext::nranks() > 1) {
    ASSERT_NE(st_distributed.global_numel(), st_distributed.numel());
  }
}

TEST(TestSptensor, DataConstructorFromMatlab) {
  // nnz = 5, size = [5 2 4]
  // vls = [ 0 1 2 3 4 ]
  // sbs = [[1 1 1], [2 2 2], [3 1 3], [4 2 4], [5 1 1]]

  ttb_indx nd = 3, nz = 5;
  ttb_real sz[3]{5, 2, 4};
  ttb_real *vls = (ttb_real *)malloc(5 * sizeof(ttb_real));
  ttb_real *sbs = (ttb_real *)malloc(15 * sizeof(ttb_real));
  for (int i = 0; i < 5; i++) {
    vls[i] = i;
    sbs[i] = i % 5 + 1;
    sbs[5 + i] = i % 2 + 1;
    sbs[10 + i] = i % 4 + 1;
  }

  Sptensor st(nd, sz, nz, vls, sbs);

  ASSERT_EQ(st.nnz(), 5);
  ASSERT_EQ(st.ndims(), 3);
  for (ttb_indx i = 0; i < st.nnz(); i++) {
    ASSERT_EQ(st.value(i), i);

    for (ttb_indx j = 0; j < st.ndims(); j++) {
      ASSERT_EQ(st.subscript(i, j), sbs[i + j * nz] - 1);
    }
  }

  free(vls);
  free(sbs);
}

TEST(TestSptensor, SearchUnsorted) {
  IndxArray dims = IndxArray(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 4;

  Sptensor st(dims, 10);

  st.subscript(0, 0) = 1;
  st.subscript(0, 1) = 0;
  st.subscript(0, 2) = 0;
  st.value(0) = 1.0;

  st.subscript(1, 0) = 0;
  st.subscript(1, 1) = 1;
  st.subscript(1, 2) = 0;
  st.value(1) = 1.0;

  st.subscript(2, 0) = 1;
  st.subscript(2, 1) = 1;
  st.subscript(2, 2) = 0;
  st.value(2) = 1.0;

  st.subscript(3, 0) = 0;
  st.subscript(3, 1) = 2;
  st.subscript(3, 2) = 0;
  st.value(3) = 1.0;

  st.subscript(4, 0) = 0;
  st.subscript(4, 1) = 0;
  st.subscript(4, 2) = 1;
  st.value(4) = 1.0;

  st.subscript(5, 0) = 0;
  st.subscript(5, 1) = 2;
  st.subscript(5, 2) = 1;
  st.value(5) = 1.0;

  st.subscript(6, 0) = 0;
  st.subscript(6, 1) = 0;
  st.subscript(6, 2) = 3;
  st.value(6) = 1.0;

  st.subscript(7, 0) = 1;
  st.subscript(7, 1) = 0;
  st.subscript(7, 2) = 3;
  st.value(7) = 1.0;

  st.subscript(8, 0) = 0;
  st.subscript(8, 1) = 1;
  st.subscript(8, 2) = 3;
  st.value(8) = 1.0;

  st.subscript(9, 0) = 1;
  st.subscript(9, 1) = 1;
  st.subscript(9, 2) = 3;
  st.value(9) = 1.0;

  ASSERT_EQ(st.nnz(), 10);

  ASSERT_EQ(st.index(1, 0, 0), 0);
  ASSERT_EQ(st.index(1, 0, 0), 0);
  ASSERT_EQ(st.index(0, 1, 0), 1);
  ASSERT_EQ(st.index(1, 1, 0), 2);
  ASSERT_EQ(st.index(0, 2, 0), 3);
  ASSERT_EQ(st.index(0, 0, 1), 4);
  ASSERT_EQ(st.index(0, 2, 1), 5);
  ASSERT_EQ(st.index(0, 0, 3), 6);
  ASSERT_EQ(st.index(1, 0, 3), 7);
  ASSERT_EQ(st.index(0, 1, 3), 8);
  ASSERT_EQ(st.index(1, 1, 3), 9);
  ASSERT_EQ(st.index(0, 0, 0), 10);
  ASSERT_EQ(st.index(0, 1, 2), 10);
  ASSERT_EQ(st.index(1, 2, 3), 10);
  ASSERT_EQ(st.index(3, 0, 0), 10);
}

TEST(TestSptensor, SearchSorted) {
  IndxArray dims = IndxArray(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 4;

  Sptensor st(dims, 10);

  st.subscript(0, 0) = 1;
  st.subscript(0, 1) = 0;
  st.subscript(0, 2) = 0;
  st.value(0) = 1.0;

  st.subscript(1, 0) = 0;
  st.subscript(1, 1) = 1;
  st.subscript(1, 2) = 0;
  st.value(1) = 1.0;

  st.subscript(2, 0) = 1;
  st.subscript(2, 1) = 1;
  st.subscript(2, 2) = 0;
  st.value(2) = 1.0;

  st.subscript(3, 0) = 0;
  st.subscript(3, 1) = 2;
  st.subscript(3, 2) = 0;
  st.value(3) = 1.0;

  st.subscript(4, 0) = 0;
  st.subscript(4, 1) = 0;
  st.subscript(4, 2) = 1;
  st.value(4) = 1.0;

  st.subscript(5, 0) = 0;
  st.subscript(5, 1) = 2;
  st.subscript(5, 2) = 1;
  st.value(5) = 1.0;

  st.subscript(6, 0) = 0;
  st.subscript(6, 1) = 0;
  st.subscript(6, 2) = 3;
  st.value(6) = 1.0;

  st.subscript(7, 0) = 1;
  st.subscript(7, 1) = 0;
  st.subscript(7, 2) = 3;
  st.value(7) = 1.0;

  st.subscript(8, 0) = 0;
  st.subscript(8, 1) = 1;
  st.subscript(8, 2) = 3;
  st.value(8) = 1.0;

  st.subscript(9, 0) = 1;
  st.subscript(9, 1) = 1;
  st.subscript(9, 2) = 3;
  st.value(9) = 1.0;

  ASSERT_EQ(st.nnz(), 10);

  st.sort();

  ASSERT_EQ(st.index(0, 0, 1), 0);
  ASSERT_EQ(st.index(0, 0, 3), 1);
  ASSERT_EQ(st.index(0, 1, 0), 2);
  ASSERT_EQ(st.index(0, 1, 3), 3);
  ASSERT_EQ(st.index(0, 2, 0), 4);
  ASSERT_EQ(st.index(0, 2, 1), 5);
  ASSERT_EQ(st.index(1, 0, 0), 6);
  ASSERT_EQ(st.index(1, 0, 3), 7);
  ASSERT_EQ(st.index(1, 1, 0), 8);
  ASSERT_EQ(st.index(1, 1, 3), 9);
  ASSERT_EQ(st.index(0, 0, 0), 10);
  ASSERT_EQ(st.index(0, 1, 2), 10);
  ASSERT_EQ(st.index(1, 2, 3), 10);
  ASSERT_EQ(st.index(3, 0, 0), 10);
}

template <typename ExecSpace> struct TestSpTensorT : public ::testing::Test {
  using exec_space = ExecSpace;
};

TYPED_TEST_SUITE(TestSpTensorT, genten_test_types);

TYPED_TEST(TestSpTensorT, SpTensorFromKruskalTensor) {
  using exec_space = typename TestFixture::exec_space;
  using host_exec_space = DefaultHostExecutionSpace;

  // Create ktensor whose reconstruction is sparse
  IndxArray dims(3);
  dims[0] = 2;
  dims[1] = 2;
  dims[2] = 2;
  KtensorT<host_exec_space> kt(2, 3, dims);
  FacMatrixT<host_exec_space> A(2,2);
  A.entry(0,0) = 1.0;
  A.entry(0,1) = 0.0;
  A.entry(1,0) = 0.0;
  A.entry(1,1) = 1.0;
  deep_copy(kt[0],A);
  deep_copy(kt[1],A);
  deep_copy(kt[2],A);
  kt.weights() = 1.0;
  KtensorT<exec_space> kt_dev = create_mirror_view(exec_space(), kt);
  deep_copy(kt_dev, kt);

  // Create sparse reconstruction and check values (ordering of nonzeros is not deterministic)
  SptensorT<exec_space> x_dev(kt_dev);
  SptensorT<host_exec_space> x = create_mirror_view(host_exec_space(), x_dev);
  deep_copy(x,x_dev);
  ASSERT_EQ(x.nnz(),2);
  ASSERT_TRUE((x.subscript(0,0) == 0 && x.subscript(0,1) == 0 &&  x.subscript(0,2) == 0 &&
               x.subscript(1,0) == 1 && x.subscript(1,1) == 1 &&  x.subscript(1,2) == 1) ||
              (x.subscript(0,0) == 1 && x.subscript(0,1) == 1 &&  x.subscript(0,2) == 1 &&
               x.subscript(1,0) == 0 && x.subscript(1,1) == 0 &&  x.subscript(1,2) == 0));
  ASSERT_FLOAT_EQ(x.value(0),1.0);
  ASSERT_FLOAT_EQ(x.value(1),1.0);

  // Create second sparse reconstruction using sparsity pattern from x and check values
  SptensorT<exec_space> x_dev2(x_dev, kt_dev);
  SptensorT<host_exec_space> x2 = create_mirror_view(host_exec_space(), x_dev2);
  deep_copy(x2,x_dev2);
  ASSERT_EQ(x2.nnz(),2);
  ASSERT_TRUE((x2.subscript(0,0) == 0 && x2.subscript(0,1) == 0 &&  x2.subscript(0,2) == 0 &&
               x2.subscript(1,0) == 1 && x2.subscript(1,1) == 1 &&  x2.subscript(1,2) == 1) ||
              (x2.subscript(0,0) == 1 && x2.subscript(0,1) == 1 &&  x2.subscript(0,2) == 1 &&
               x2.subscript(1,0) == 0 && x2.subscript(1,1) == 0 &&  x2.subscript(1,2) == 0));
  ASSERT_FLOAT_EQ(x2.value(0),1.0);
  ASSERT_FLOAT_EQ(x2.value(1),1.0);
}

TYPED_TEST(TestSpTensorT, SpTensorImportToRoot) {
  using exec_space = typename TestFixture::exec_space;
  using host_exec_space = DefaultHostExecutionSpace;

  // Create a sparse tensor
  const ttb_indx nz = 10;
  const ttb_indx nd = 3;
  IndxArray dims = { 3, 4, 5 };
  Sptensor X(dims, nz);
  ASSERT_EQ(X.nnz(),nz);
  ASSERT_EQ(X.ndims(),nd); 
  for (ttb_indx i=0; i<nz; ++i) {
    X.value(i) = i;
    for (ttb_indx j=0; j<nd; ++j) {
      X.subscript(i,j) = (i+j) % dims[j]; // mod is necessary to get subscript in valid range
    }
  }

  // Iterate over all distribution approaches (most don't change the tensor distribution approach, but some do)
  for (unsigned t=0; t<Dist_Update_Method::num_types; ++t) {
    AlgParams algParams;
    algParams.dist_update_method = Dist_Update_Method::types[t];

    // Skip tpetra if tpetra is not enabled
#ifndef HAVE_TPETRA
    if (algParams.dist_update_method == Dist_Update_Method::Tpetra)
      continue;
#endif
    std::string message = std::string("Testing dist_update_method = ") + Dist_Update_Method::names[t];
    INFO_MSG(message.c_str()); 

    // Distribute the sparse tensor
    DistTensorContext<exec_space> dtc;
    SptensorT<exec_space> Xd = dtc.distributeTensor(X, algParams);

    // Import the tensor back to root
    Sptensor Xi = dtc.template importToRoot<host_exec_space>(Xd);

    // Check values and indices are correct, but ordering is not deterministic, so have to search
    if (dtc.gridRank() == 0) {
      ASSERT_EQ(Xi.nnz(),nz);
      ASSERT_EQ(Xi.ndims(),nd);
      for (ttb_indx i=0; i<nz; ++i) {
        const ttb_indx ind = Xi.index(X.getSubscripts(i));
        ASSERT_TRUE(ind < nz); // ind >= nz means indices were not found
        ASSERT_FLOAT_EQ(Xi.value(ind),X.value(i));
      }
    }
  }
}


} // namespace UnitTests
} // namespace Genten
