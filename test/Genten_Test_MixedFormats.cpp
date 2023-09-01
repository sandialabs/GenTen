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

#include <Genten_DistContext.hpp>
#include <Genten_DistTensorContext.hpp>
#include <Genten_IndxArray.hpp>
#include <Genten_Ktensor.hpp>
#include <Genten_MixedFormatOps.hpp>
#include <Genten_Sptensor.hpp>
#include <Genten_Util.hpp>
#include <Genten_DistKtensorUpdate.hpp>
#include <Genten_IOtext.hpp>

#include "Genten_Test_Utils.hpp"

#include <gtest/gtest.h>

namespace Genten {
namespace UnitTests {

template <typename ExecSpace>
struct TestMixedFormatsT : public ::testing::Test {
  using exec_space = ExecSpace;
};

TYPED_TEST_SUITE(TestMixedFormatsT, genten_test_types);

TYPED_TEST(TestMixedFormatsT, SptensorTensorKtensorInnerprod) {
  DistContext::Barrier();

  using exec_space = typename TestFixture::exec_space;
  using host_exec_space = DefaultHostExecutionSpace;

  Genten::AlgParams algParams;

  INFO_MSG("Creating an Sptensor for innerprod test");
  IndxArray dims(3);
  dims[0] = 4;
  dims[1] = 2;
  dims[2] = 3;
  SptensorT<host_exec_space> a(dims, 4);
  a.subscript(0, 0) = 2;
  a.subscript(0, 1) = 0;
  a.subscript(0, 2) = 0;
  a.value(0) = 1.0;
  a.subscript(1, 0) = 1;
  a.subscript(1, 1) = 1;
  a.subscript(1, 2) = 1;
  a.value(1) = 2.0;
  a.subscript(2, 0) = 3;
  a.subscript(2, 1) = 0;
  a.subscript(2, 2) = 2;
  a.value(2) = 3.0;
  a.subscript(3, 0) = 0;
  a.subscript(3, 1) = 1;
  a.subscript(3, 2) = 2;
  a.value(3) = 4.0;

  Genten::DistTensorContext<exec_space> dtc;
  SptensorT<exec_space> a_dev = dtc.distributeTensor(a);
  const ProcessorMap *pmap = dtc.pmap_ptr().get();
  a_dev.setProcessorMap(pmap);

  INFO_MSG("Creating a (dense) Tensor for innerprod test");
  TensorT<exec_space> t_dev(a_dev);

  INFO_MSG("Creating a Ktensor of matching shape");
  dims = IndxArray(3);
  dims[0] = 4;
  dims[1] = 2;
  dims[2] = 3;
  ttb_indx nc = 2;
  KtensorT<host_exec_space> oKtens(nc, 3, dims);

  // The sparse tensor has 4 nonzeroes.  Populate the Ktensor so that two
  // of its nonzeroes match, testing different weights.
  // Answers were obtained using Matlab tensor toolbox.

  oKtens.weights(0) = 1.0;
  oKtens.weights(1) = 2.0;

  oKtens[0].entry(2, 0) = 1.0;
  oKtens[1].entry(0, 0) = 1.0;
  oKtens[2].entry(0, 0) = 1.0;

  oKtens[0].entry(1, 0) = -1.0;

  oKtens[0].entry(3, 1) = 0.3;
  oKtens[1].entry(0, 1) = 0.3;
  oKtens[2].entry(2, 1) = 0.3;

  KtensorT<exec_space> oKtens_dev = dtc.exportFromRoot(oKtens);
  oKtens_dev.setProcessorMap(pmap);

  DistKtensorUpdate<exec_space> *dku =
    createKtensorUpdate(a_dev, oKtens_dev, algParams);
  KtensorT<exec_space> oKtens_dev_overlap =
    dku->createOverlapKtensor(oKtens_dev);
  dku->doImport(oKtens_dev_overlap, oKtens_dev);

  ttb_real d;
  d = innerprod(a_dev, oKtens_dev_overlap);
  GENTEN_FLOAT_EQ(d, 1.162, "Inner product between sptensor and ktensor");
  d = innerprod(t_dev, oKtens_dev);
  GENTEN_FLOAT_EQ(d, 1.162, "Inner product between tensor and ktensor");

  Array altLambda(2);
  altLambda[0] = 3.0;
  altLambda[1] = 1.0;
  ArrayT<exec_space> altLambda_dev =
      create_mirror_view(exec_space(), altLambda);
  deep_copy(altLambda_dev, altLambda);
  d = innerprod(a_dev, oKtens_dev_overlap, altLambda_dev);
  GENTEN_FLOAT_EQ(d, 3.081, "Inner product with alternate lambda is correct");
  d = innerprod(t_dev, oKtens_dev, altLambda_dev);
  GENTEN_FLOAT_EQ(d, 3.081, "Inner product with alternate lambda is correct");

  delete dku;
}

TYPED_TEST(TestMixedFormatsT, SptensorKtensorTimesDivide) {
  DistContext::Barrier();

  using host_exec_space = DefaultHostExecutionSpace;

  INFO_MSG("Resizing Sptensor for times/divide test");
  IndxArray dims(3);
  dims[0] = 3;
  dims[1] = 4;
  dims[2] = 2;
  SptensorT<host_exec_space> a(dims, 3);
  a.subscript(0, 0) = 1;
  a.subscript(0, 1) = 0;
  a.subscript(0, 2) = 0;
  a.value(0) = 1.0;
  a.subscript(1, 0) = 1;
  a.subscript(1, 1) = 0;
  a.subscript(1, 2) = 1;
  a.value(1) = 1.0;
  a.subscript(2, 0) = 1;
  a.subscript(2, 1) = 1;
  a.subscript(2, 2) = 0;
  a.value(2) = 1.0;
  GENTEN_EQ(a.nnz(), 3, "Sptensor resized to correct nnz");

  INFO_MSG("Resizing Ktensor for times test");
  dims = IndxArray(3);
  dims[0] = 3;
  dims[1] = 4;
  dims[2] = 2;
  ttb_indx nc = 2;
  KtensorT<host_exec_space> oKtens(nc, 3, dims);
  oKtens.setWeights(1.0);
  GENTEN_TRUE(oKtens.isConsistent(), "Ktensor resized consistently");

  // Set elements in the factors to unique values, counting up by 1.
  // The Ktensor has a nonzero value for every entry in the reconstructed
  // tensor; element-wise multiplication will result in modifications to
  // just the nonzeroes of the Sptensor.
  ttb_real d = 1.0;
  for (ttb_indx i = 0; i < oKtens.ndims(); i++) {
    for (ttb_indx j = 0; j < oKtens[i].nRows(); j++) {
      for (ttb_indx r = 0; r < oKtens.ncomponents(); r++) {
        oKtens[i].entry(j, r) = d;
        d = d + 1.0;
      }
    }
  }

  // Uncomment to manually check what the answer should be.
  // Genten::print_sptensor(a, std::cout, "Sparse tensor for times/divide
  // test"); Genten::print_ktensor(oKtens, std::cout, "Ktensor for times/divide
  // test");

  // Test times().
  SptensorT<host_exec_space> oTest(a.size(), a.nnz());
  oTest.times(oKtens, a);
  GENTEN_FLOAT_EQ(oTest.value(0), (3 * 7 * 15 + 4 * 8 * 16),
                  "times() element 0 OK");
  GENTEN_FLOAT_EQ(oTest.value(1), (3 * 7 * 17 + 4 * 8 * 18),
                  "times() element 1 OK");
  GENTEN_FLOAT_EQ(oTest.value(2), (3 * 9 * 15 + 4 * 10 * 16),
                  "times() element 2 OK");

  // Test that divide() undoes the multiplication.
  oTest.divide(oKtens, oTest, 1.0e-10);
  GENTEN_FLOAT_EQ(oTest.value(0), 1.0, "divide() element 0 OK");
  GENTEN_FLOAT_EQ(oTest.value(1), 1.0, "divide() element 1 OK");
  GENTEN_FLOAT_EQ(oTest.value(2), 1.0, "divide() element 2 OK");
}

template <typename exec_space>
void RunTensorKtensorMTTKRP(const TensorLayout layout, const char *label)
{
  DistContext::Barrier();

  using host_exec_space = DefaultHostExecutionSpace;

  //----------------------------------------------------------------------
  // Test mttkrp() between Tensor and Ktensor.
  //
  // Corresponding Matlab code for the tests:
  //   Adense = tenzeros([2 3 4]);
  //   Adense(1,1,1) = 1.0
  //   Afac = [10 ; 11];  Bfac = [12 ; 13 ; 14];  Cfac = [15 ; 16 ; 17 ; 18];
  //   K = ktensor({Afac,Bfac,Cfac})
  //   mttkrp(Adense,K.U,1)     % Matricizes on 1st index
  //   mttkrp(Adense,K.U,2)     % Matricizes on 2nd index
  //   mttkrp(Adense,K.U,3)     % Matricizes on 3rd index
  //   Adense(2,3,4) = 1.0
  //   mttkrp(Adense,K.U,1)     % Matricizes on 1st index
  //   mttkrp(Adense,K.U,2)     % Matricizes on 2nd index
  //   mttkrp(Adense,K.U,3)     % Matricizes on 3rd index
  //----------------------------------------------------------------------

  INFO_MSG("Resizing Tensor for mttkrp test");
  IndxArray dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 4;
  TensorT<host_exec_space> t2(dims, 0.0, layout);
  t2(0, 0, 0) = 1.0;
  TensorT<exec_space> t2_dev = create_mirror_view(exec_space(), t2);
  deep_copy(t2_dev, t2);

  INFO_MSG("Resizing Ktensor of matching shape");
  ttb_indx nc = 5;
  KtensorT<host_exec_space> oKtens(nc, 3, dims);
  oKtens.setWeights(1.0);
  GENTEN_TRUE(oKtens.isConsistent(), "Ktensor resized consistently");

  // Set elements in the factors to unique values.
  for (ttb_indx j=0; j<nc; ++j) {
    oKtens[0](0, j) = 10.0*(j+1);
    oKtens[0](1, j) = 11.0*(j+1);
    oKtens[1](0, j) = 12.0*(j+1);
    oKtens[1](1, j) = 13.0*(j+1);
    oKtens[1](2, j) = 14.0*(j+1);
    oKtens[2](0, j) = 15.0*(j+1);
    oKtens[2](1, j) = 16.0*(j+1);
    oKtens[2](2, j) = 17.0*(j+1);
    oKtens[2](3, j) = 18.0*(j+1);
  }
  KtensorT<exec_space> oKtens_dev = create_mirror_view(exec_space(), oKtens);
  deep_copy(oKtens_dev, oKtens);

  Genten::AlgParams algParams;
  //algParams.mttkrp_method = Genten::MTTKRP_Method::Phan;

  // Matricizing on index 0 has result 12*15 = 180.
  FacMatrix oFM(t2.size(0), oKtens.ncomponents());
  FacMatrixT<exec_space> oFM_dev = create_mirror_view(exec_space(), oFM);
  deep_copy(oFM_dev, oFM);
  mttkrp(t2_dev, oKtens_dev, 0, oFM_dev, algParams);
  deep_copy(oFM, oFM_dev);

  ASSERT_EQ(oFM.nRows(), 2);
  ASSERT_EQ(oFM.nCols(), nc);
  INFO_MSG("mttkrp result shape correct for index [0]");

  for (ttb_indx j=0; j<nc; ++j) {
    ASSERT_FLOAT_EQ(oFM.entry(0, j), 180.0*(j+1)*(j+1));
    ASSERT_FLOAT_EQ(oFM.entry(1, j), 0.0);
  }
  INFO_MSG("mttkrp result values correct for index [0]");

  // Uncomment to manually check what the answer should be.
  // Genten::print_tensor(t2, std::cout, "Tensor for mttkrp test");
  // Genten::print_ktensor(oKtens, std::cout, "Ktensor for mttkrp test");
  // Genten::print_matrix(oFM, std::cout, "Matrix result from mttkrp");

  // Matricizing on index 1 has result 10*15 = 150.
  oFM = FacMatrix(t2.size(1), oKtens.ncomponents());
  oFM_dev = create_mirror_view(exec_space(), oFM);
  deep_copy(oFM_dev, oFM);
  mttkrp(t2_dev, oKtens_dev, 1, oFM_dev, algParams);
  deep_copy(oFM, oFM_dev);

  ASSERT_EQ(oFM.nRows(), 3);
  ASSERT_EQ(oFM.nCols(), nc);
  INFO_MSG("mttkrp result shape correct for index [1]");

  for (ttb_indx j=0; j<nc; ++j) {
    ASSERT_FLOAT_EQ(oFM.entry(0, j), 150.0*(j+1)*(j+1));
    ASSERT_FLOAT_EQ(oFM.entry(1, j), 0.0);
  }
  INFO_MSG("mttkrp result values correct for index [1]");

  // Matricizing on index 2 has result 10*12 = 120.
  oFM = FacMatrix(t2.size(2), oKtens.ncomponents());
  oFM_dev = create_mirror_view(exec_space(), oFM);
  deep_copy(oFM_dev, oFM);
  mttkrp(t2_dev, oKtens_dev, 2, oFM_dev, algParams);
  deep_copy(oFM, oFM_dev);

  ASSERT_EQ(oFM.nRows(), 4);
  ASSERT_EQ(oFM.nCols(), nc);
  INFO_MSG("mttkrp result shape correct for index [2]");

  for (ttb_indx j=0; j<nc; ++j) {
    ASSERT_FLOAT_EQ(oFM.entry(0, j), 120.0*(j+1)*(j+1));
    ASSERT_FLOAT_EQ(oFM.entry(1, j), 0.0);
  }
  INFO_MSG("mttkrp result values correct for index [2]");

  // Add another nonzero and repeat the three tests.
  t2(1, 2, 3) = 1.0;
  deep_copy(t2_dev, t2);
  oFM = FacMatrix(t2.size(0), oKtens.ncomponents());
  oFM_dev = create_mirror_view(exec_space(), oFM);
  deep_copy(oFM_dev, oFM);
  mttkrp(t2_dev, oKtens_dev, 0, oFM_dev, algParams);
  deep_copy(oFM, oFM_dev);

  for (ttb_indx j=0; j<nc; ++j) {
    ASSERT_FLOAT_EQ(oFM.entry(0, j), 180.0*(j+1)*(j+1));
    ASSERT_FLOAT_EQ(oFM.entry(1, j), 252.0*(j+1)*(j+1));
  }
  INFO_MSG("mttkrp result values correct for index [0], 2 sparse nnz");

  oFM = FacMatrix(t2.size(1), oKtens.ncomponents());
  oFM_dev = create_mirror_view(exec_space(), oFM);
  deep_copy(oFM_dev, oFM);
  mttkrp(t2_dev, oKtens_dev, 1, oFM_dev, algParams);
  deep_copy(oFM, oFM_dev);

  for (ttb_indx j=0; j<nc; ++j) {
    ASSERT_FLOAT_EQ(oFM.entry(0, j), 150.0*(j+1)*(j+1));
    ASSERT_FLOAT_EQ(oFM.entry(2, j), 198.0*(j+1)*(j+1));
  }
  INFO_MSG("mttkrp result values correct for index [1], 2 sparse nnz");

  oFM = FacMatrix(t2.size(2), oKtens.ncomponents());
  oFM_dev = create_mirror_view(exec_space(), oFM);
  deep_copy(oFM_dev, oFM);
  mttkrp(t2_dev, oKtens_dev, 2, oFM_dev, algParams);
  deep_copy(oFM, oFM_dev);

  for (ttb_indx j=0; j<nc; ++j) {
    ASSERT_FLOAT_EQ(oFM.entry(0, j), 120.0*(j+1)*(j+1));
    ASSERT_FLOAT_EQ(oFM.entry(3, j), 154.0*(j+1)*(j+1));
  }
  INFO_MSG("mttkrp result values correct for index [3], 2 sparse nnz");
}

TYPED_TEST(TestMixedFormatsT, TensorKtensorMTTKRP) {
   using exec_space = typename TestFixture::exec_space;

   RunTensorKtensorMTTKRP<exec_space>(TensorLayout::Left, "TensorLayout::Left");
   RunTensorKtensorMTTKRP<exec_space>(TensorLayout::Right, "TensorLayout::Right");
}

TYPED_TEST(TestMixedFormatsT, TensorPhanAminoAcid) {
   using exec_space = typename TestFixture::exec_space;
   DistContext::Barrier();

  // Read layout-left amino acid tensor
  Tensor Xlh;
  import_tensor("./data/aminoacid_data_dense.txt", Xlh);
  TensorT<exec_space> Xl = create_mirror_view(exec_space(), Xlh);
  deep_copy(Xl, Xlh);

  // Create matching ktensor
  //const ttb_indx nc = 5;
  const ttb_indx nc = 3;
  const ttb_indx nd = Xl.ndims();
  KtensorT<exec_space> u(nc, nd, Xl.size());
  auto uh = create_mirror_view(u);
  RandomMT rng(12345);
  uh.setMatricesScatter(true, false, rng);
  uh.setWeights(1.0);
  deep_copy(u, uh);

  // Test Phan MTTKRP algorithm for layout-left matches Row-based algorithm
  KtensorT<exec_space> vr(nc, nd, Xl.size()), vp(nc, nd, Xl.size());
  auto vrh = create_mirror_view(vr);
  auto vph = create_mirror_view(vp);
  AlgParams algParams;
  INFO_MSG("mttkrp result correct for Phan algorihtm with LayoutLeft tensor");
  for (ttb_indx n=0; n<nd; ++n) {
    algParams.mttkrp_method = MTTKRP_Method::RowBased;
    mttkrp(Xl, u, n, vr[n], algParams);
    algParams.mttkrp_method = MTTKRP_Method::Phan;
    mttkrp(Xl, u, n, vp[n], algParams);
    deep_copy(vrh[n], vr[n]);
    deep_copy(vph[n], vp[n]);

    for (ttb_indx i=0; i<Xl.size(n); ++i)
      for (ttb_indx j=0; j<nc; ++j)
        ASSERT_FLOAT_EQ(vrh[n](i,j), vph[n](i,j));
  }

  // Test Phan MTTKRP algorithm for layout-right matches Row-based algorithm
  TensorT<exec_space> Xr = Xl.switch_layout(TensorLayout::Right);
  INFO_MSG("mttkrp result correct for Phan algorihtm with LayoutRight tensor");
  vr.setMatrices(0.0);
  vp.setMatrices(0.0);
  for (ttb_indx n=0; n<nd; ++n) {
    algParams.mttkrp_method = MTTKRP_Method::RowBased;
    mttkrp(Xr, u, n, vr[n], algParams);
    algParams.mttkrp_method = MTTKRP_Method::Phan;
    mttkrp(Xr, u, n, vp[n], algParams);
    deep_copy(vrh[n], vr[n]);
    deep_copy(vph[n], vp[n]);

    for (ttb_indx i=0; i<Xl.size(n); ++i)
      for (ttb_indx j=0; j<nc; ++j)
        ASSERT_FLOAT_EQ(vrh[n](i,j), vph[n](i,j));
  }
}

TYPED_TEST(TestMixedFormatsT, LayoutRightTensorAminoAcid) {
   using exec_space = typename TestFixture::exec_space;
   DistContext::Barrier();

  // Read layout-left amino acid tensor
  Tensor Xlh;
  import_tensor("./data/aminoacid_data_dense.txt", Xlh);
  TensorT<exec_space> Xl = create_mirror_view(exec_space(), Xlh);
  deep_copy(Xl, Xlh);

  // Create layout-right version
  TensorT<exec_space> Xr = Xl.switch_layout(TensorLayout::Right);

  // Create matching ktensor
  const ttb_indx nc = 5;
  const ttb_indx nd = Xl.ndims();
  KtensorT<exec_space> u(nc, nd, Xl.size());
  auto uh = create_mirror_view(u);
  uh.setMatricesRand();
  uh.setWeights(1.0);
  deep_copy(u, uh);

  // Test MTTKRP for layout-right matches layout-left
  KtensorT<exec_space> vl(nc, nd, Xl.size()), vr(nc, nd, Xl.size());
  auto vlh = create_mirror_view(vl);
  auto vrh = create_mirror_view(vr);
  INFO_MSG("mttkrp result correct for LayoutRight tensor");
  for (ttb_indx n=0; n<nd; ++n) {
    mttkrp(Xl, u, n, vl[n]);
    mttkrp(Xr, u, n, vr[n]);
    deep_copy(vlh[n], vl[n]);
    deep_copy(vrh[n], vr[n]);

    for (ttb_indx i=0; i<Xl.size(n); ++i)
      for (ttb_indx j=0; j<nc; ++j)
        ASSERT_FLOAT_EQ(vlh[n](i,j), vrh[n](i,j));
  }

  // Test inner product for layout-right matches layout-left
  const ttb_real ipl = innerprod(Xl, u);
  const ttb_real ipr = innerprod(Xr, u);
  INFO_MSG("innerprod result correct for LayoutRight tensor");
  ASSERT_FLOAT_EQ(ipl, ipr);

  // Test norm
  const ttb_real nrml = Xl.norm();
  const ttb_real nrmr = Xr.norm();
  INFO_MSG("norm result correct for LayoutRight tensor");
  ASSERT_FLOAT_EQ(nrml, nrmr);
}

template <typename exec_space>
void RunMTTKRPTypeTest(MTTKRP_Method::type mttkrp_method,
                       const std::string &label) {
  DistContext::Barrier();

  using host_exec_space = DefaultHostExecutionSpace;

  //----------------------------------------------------------------------
  // Test mttkrp() between Sptensor and Ktensor.
  //
  // Corresponding Matlab code for the tests:
  //   Asparse = sptensor([], [], [2 3 4]);
  //   Asparse(1,1,1) = 1.0
  //   Afac = [10 ; 11];  Bfac = [12 ; 13 ; 14];  Cfac = [15 ; 16 ; 17 ; 18];
  //   K = ktensor({Afac,Bfac,Cfac})
  //   mttkrp(Asparse,K.U,1)     % Matricizes on 1st index
  //   mttkrp(Asparse,K.U,2)     % Matricizes on 2nd index
  //   mttkrp(Asparse,K.U,3)     % Matricizes on 3rd index
  //   Asparse(2,3,4) = 1.0
  //   mttkrp(Asparse,K.U,1)     % Matricizes on 1st index
  //   mttkrp(Asparse,K.U,2)     % Matricizes on 2nd index
  //   mttkrp(Asparse,K.U,3)     % Matricizes on 3rd index
  //----------------------------------------------------------------------

  INFO_MSG("Resizing Sptensor for mttkrp test");
  const ttb_indx nd = 3;
  IndxArray dims(nd);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 4;
  SptensorT<host_exec_space> a(dims, 2);
  a.subscript(0, 0) = 0;
  a.subscript(0, 1) = 0;
  a.subscript(0, 2) = 0;
  a.value(0) = 1.0;
  a.subscript(1, 0) = 1;
  a.subscript(1, 1) = 2;
  a.subscript(1, 2) = 3;
  a.value(1) = 0.0;

  INFO_MSG("Resizing Ktensor of matching shape");
  ttb_indx nc = 1;
  Ktensor oKtens(nc, 3, dims);
  oKtens.setWeights(1.0);
  GENTEN_TRUE(oKtens.isConsistent(), "Ktensor resized consistently");

  // Set elements in the factors to unique values.
  // The sparse tensor has only one nonzero value; hence, mttkrp will involve
  // just one element from two of the factor matrices.
  oKtens[0].entry(0, 0) = 10.0;
  oKtens[0].entry(1, 0) = 11.0;
  oKtens[1].entry(0, 0) = 12.0;
  oKtens[1].entry(1, 0) = 13.0;
  oKtens[1].entry(2, 0) = 14.0;
  oKtens[2].entry(0, 0) = 15.0;
  oKtens[2].entry(1, 0) = 16.0;
  oKtens[2].entry(2, 0) = 17.0;
  oKtens[2].entry(3, 0) = 18.0;

  // Copy a and oKtens to device
  Genten::DistTensorContext<exec_space> dtc;
  SptensorT<exec_space> a_dev_dist = dtc.distributeTensor(a);
  const ProcessorMap *pmap = dtc.pmap_ptr().get();
  a_dev_dist.setProcessorMap(pmap);
  if (mttkrp_method == MTTKRP_Method::Perm) {
    a_dev_dist.createPermutation();
  }

  KtensorT<exec_space> oKtens_dev_dist = dtc.exportFromRoot(oKtens);
  oKtens_dev_dist.setProcessorMap(pmap);

  FacMatrix oFM;
  KtensorT<exec_space> res_dev_dist = Genten::clone(oKtens_dev_dist);

  AlgParams algParams;
  algParams.mttkrp_method = mttkrp_method;
  algParams.mttkrp_duplicated_threshold = 1.0e6; // Ensure duplicated is used

  DistKtensorUpdate<exec_space> *dku =
    createKtensorUpdate(a_dev_dist, oKtens_dev_dist, algParams);
  KtensorT<exec_space> oKtens_dev_overlap =
    dku->createOverlapKtensor(oKtens_dev_dist);
   KtensorT<exec_space> res_dev_overlap =
    dku->createOverlapKtensor(res_dev_dist);
  dku->doImport(oKtens_dev_overlap, oKtens_dev_dist);

  // Matricizing on index 0 has result 12*15 = 180.
  mttkrp(a_dev_dist, oKtens_dev_overlap, 0, res_dev_overlap[0], algParams);
  dku->doExport(res_dev_dist, res_dev_overlap, 0);
  oFM = dtc.template importToRoot<DefaultHostExecutionSpace>(0, res_dev_dist[0]);

  ASSERT_EQ(oFM.nRows(), 2);
  ASSERT_EQ(oFM.nCols(), 1);
  INFO_MSG("mttkrp result shape correct for index [0]");

  if (dtc.gridRank() == 0) {
    ASSERT_FLOAT_EQ(oFM.entry(0, 0), 180.0);
    ASSERT_FLOAT_EQ(oFM.entry(1, 0), 0.0);
    INFO_MSG("mttkrp result values correct for index [0]");
  }

  // Uncomment to manually check what the answer should be.
  // Genten::print_sptensor(a, std::cout, "Sparse tensor for mttkrp test");
  // Genten::print_ktensor(oKtens, std::cout, "Ktensor for mttkrp test");
  // Genten::print_matrix(oFM, std::cout, "Matrix result from mttkrp");

  // Matricizing on index 1 has result 10*15 = 150.
  mttkrp(a_dev_dist, oKtens_dev_overlap, 1, res_dev_overlap[1], algParams);
  dku->doExport(res_dev_dist, res_dev_overlap, 1);
  oFM = dtc.template importToRoot<DefaultHostExecutionSpace>(1, res_dev_dist[1]);

  ASSERT_EQ(oFM.nRows(), 3);
  ASSERT_EQ(oFM.nCols(), 1);
  INFO_MSG("mttkrp result shape correct for index [1]");

  if (dtc.gridRank() == 0) {
    ASSERT_FLOAT_EQ(oFM.entry(0, 0), 150.0);
    ASSERT_FLOAT_EQ(oFM.entry(1, 0), 0.0);
    INFO_MSG("mttkrp result values correct for index [1]");
  }

  // Matricizing on index 2 has result 10*12 = 120.
  mttkrp(a_dev_dist, oKtens_dev_overlap, 2, res_dev_overlap[2], algParams);
  dku->doExport(res_dev_dist, res_dev_overlap, 2);
  oFM = dtc.template importToRoot<DefaultHostExecutionSpace>(2, res_dev_dist[2]);

  ASSERT_EQ(oFM.nRows(), 4);
  ASSERT_EQ(oFM.nCols(), 1);
  INFO_MSG("mttkrp result shape correct for index [2]");

  if (dtc.gridRank() == 0) {
    ASSERT_FLOAT_EQ(oFM.entry(0, 0), 120.0);
    ASSERT_FLOAT_EQ(oFM.entry(1, 0), 0.0);
    INFO_MSG("mttkrp result values correct for index [2]");
  }

  // Add another nonzero and repeat the three tests.
  a.subscript(1, 0) = 1;
  a.subscript(1, 1) = 2;
  a.subscript(1, 2) = 3;
  a.value(1) = 1.0;

  a_dev_dist = dtc.distributeTensor(a);
  pmap = dtc.pmap_ptr().get();
  a_dev_dist.setProcessorMap(pmap);
  if (mttkrp_method == MTTKRP_Method::Perm) {
    a_dev_dist.createPermutation();
  }

  oKtens_dev_dist = dtc.exportFromRoot(oKtens);
  oKtens_dev_dist.setProcessorMap(pmap);

  dku->updateTensor(a_dev_dist);
  oKtens_dev_overlap = dku->createOverlapKtensor(oKtens_dev_dist);
  res_dev_overlap = dku->createOverlapKtensor(res_dev_dist);
  dku->doImport(oKtens_dev_overlap, oKtens_dev_dist);

  mttkrp(a_dev_dist, oKtens_dev_overlap, 0, res_dev_overlap[0], algParams);
  dku->doExport(res_dev_dist, res_dev_overlap, 0);
  oFM = dtc.template importToRoot<DefaultHostExecutionSpace>(0, res_dev_dist[0]);

  if (dtc.gridRank() == 0) {
    ASSERT_FLOAT_EQ(oFM.entry(0, 0), 180.0);
    ASSERT_FLOAT_EQ(oFM.entry(1, 0), 252.0);
    INFO_MSG("mttkrp result values correct for index [0], 2 sparse nnz");
  }

  mttkrp(a_dev_dist, oKtens_dev_overlap, 1, res_dev_overlap[1], algParams);
  dku->doExport(res_dev_dist, res_dev_overlap, 1);
  oFM = dtc.template importToRoot<DefaultHostExecutionSpace>(1, res_dev_dist[1]);

  if (dtc.gridRank() == 0) {
    ASSERT_FLOAT_EQ(oFM.entry(0, 0), 150.0);
    ASSERT_FLOAT_EQ(oFM.entry(2, 0), 198.0);
    INFO_MSG("mttkrp result values correct for index [0], 2 sparse nnz");
  }

  mttkrp(a_dev_dist, oKtens_dev_overlap, 2, res_dev_overlap[2], algParams);
  dku->doExport(res_dev_dist, res_dev_overlap, 2);
  oFM = dtc.template importToRoot<DefaultHostExecutionSpace>(2, res_dev_dist[2]);

  if (dtc.gridRank() == 0) {
    ASSERT_FLOAT_EQ(oFM.entry(0, 0), 120.0);
    ASSERT_FLOAT_EQ(oFM.entry(3, 0), 154.0);
    INFO_MSG("mttkrp result values correct for index [0], 2 sparse nnz");
  }

  delete dku;
}

TYPED_TEST(TestMixedFormatsT, MTTKRP_Type) {
  using exec_space = typename TestFixture::exec_space;

  struct TestCase {
    TestCase(const MTTKRP_Method::type mttkrp_method, const char *label)
        : mttkrp_method{mttkrp_method}, label{label} {}

    const MTTKRP_Method::type mttkrp_method;
    const char *label;

    const bool run{not SpaceProperties<exec_space>::is_gpu ||
                   mttkrp_method != MTTKRP_Method::type::Duplicated};
  };

  TestCase test_cases[]{
      TestCase{MTTKRP_Method::type::Atomic, "Atomic"},
      TestCase{MTTKRP_Method::type::Perm, "Perm"},
      TestCase{MTTKRP_Method::type::Duplicated, "Duplicated"}};

  for (const auto &tc : test_cases) {
    if (tc.run) {
      RunMTTKRPTypeTest<exec_space>(tc.mttkrp_method, tc.label);
    }
  }
}

template <typename exec_space>
void RunMTTKRAllTypeTest(MTTKRP_All_Method::type mttkrp_method,
                         const std::string &label) {
  DistContext::Barrier();

  using host_exec_space = DefaultHostExecutionSpace;

  //----------------------------------------------------------------------
  // Test mttkrp() between Sptensor and Ktensor.
  //
  // Corresponding Matlab code for the tests:
  //   Asparse = sptensor([], [], [2 3 4]);
  //   Asparse(1,1,1) = 1.0
  //   Afac = [10 ; 11];  Bfac = [12 ; 13 ; 14];  Cfac = [15 ; 16 ; 17 ; 18];
  //   K = ktensor({Afac,Bfac,Cfac})
  //   mttkrp(Asparse,K.U,1)     % Matricizes on 1st index
  //   mttkrp(Asparse,K.U,2)     % Matricizes on 2nd index
  //   mttkrp(Asparse,K.U,3)     % Matricizes on 3rd index
  //   Asparse(2,3,4) = 1.0
  //   mttkrp(Asparse,K.U,1)     % Matricizes on 1st index
  //   mttkrp(Asparse,K.U,2)     % Matricizes on 2nd index
  //   mttkrp(Asparse,K.U,3)     % Matricizes on 3rd index
  //----------------------------------------------------------------------

  INFO_MSG("Resizing Sptensor for mttkrp test");
  const ttb_indx nd = 3;
  IndxArray dims(nd);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 4;
  SptensorT<host_exec_space> a(dims, 2);
  a.subscript(0, 0) = 0;
  a.subscript(0, 1) = 0;
  a.subscript(0, 2) = 0;
  a.value(0) = 1.0;
  a.subscript(1, 0) = 1;
  a.subscript(1, 1) = 2;
  a.subscript(1, 2) = 3;
  a.value(1) = 0.0;

  INFO_MSG("Resizing Ktensor of matching shape");
  const ttb_indx nc = 1;
  Ktensor oKtens(nc, nd, dims);
  oKtens.setWeights(1.0);
  GENTEN_TRUE(oKtens.isConsistent(), "Ktensor resized consistently");

  // Set elements in the factors to unique values.
  // The sparse tensor has only one nonzero value; hence, mttkrp will involve
  // just one element from two of the factor matrices.
  oKtens[0].entry(0, 0) = 10.0;
  oKtens[0].entry(1, 0) = 11.0;
  oKtens[1].entry(0, 0) = 12.0;
  oKtens[1].entry(1, 0) = 13.0;
  oKtens[1].entry(2, 0) = 14.0;
  oKtens[2].entry(0, 0) = 15.0;
  oKtens[2].entry(1, 0) = 16.0;
  oKtens[2].entry(2, 0) = 17.0;
  oKtens[2].entry(3, 0) = 18.0;

  // Copy a and oKtens to device
  Genten::DistTensorContext<exec_space> dtc;
  SptensorT<exec_space> a_dev_dist = dtc.distributeTensor(a);
  const ProcessorMap *pmap = dtc.pmap_ptr().get();
  a_dev_dist.setProcessorMap(pmap);

  KtensorT<exec_space> oKtens_dev_dist = dtc.exportFromRoot(oKtens);
  oKtens_dev_dist.setProcessorMap(pmap);

  Ktensor v(nc, nd, dims);
  v.setWeights(1.0);
  KtensorT<exec_space> v_dev_dist = dtc.exportFromRoot(v);
  v_dev_dist.setProcessorMap(pmap);

  AlgParams algParams;
  algParams.mttkrp_all_method = mttkrp_method;
  algParams.mttkrp_method = Genten::MTTKRP_Method::Atomic;

  DistKtensorUpdate<exec_space> *dku =
    createKtensorUpdate(a_dev_dist, oKtens_dev_dist, algParams);
  KtensorT<exec_space> oKtens_dev_overlap =
    dku->createOverlapKtensor(oKtens_dev_dist);
   KtensorT<exec_space> v_dev_overlap =
    dku->createOverlapKtensor(v_dev_dist);
  dku->doImport(oKtens_dev_overlap, oKtens_dev_dist);

  // Matricizing on index 0 has result 12*15 = 180.
  // Matricizing on index 1 has result 10*15 = 150.
  // Matricizing on index 2 has result 10*12 = 120.
  mttkrp_all(a_dev_dist, oKtens_dev_overlap, v_dev_overlap, algParams);
  dku->doExport(v_dev_dist, v_dev_overlap);
  v = dtc.template importToRoot<DefaultHostExecutionSpace>(v_dev_dist);

  if (DistContext::rank() == 0) {
    ASSERT_EQ(v[0].nRows(), 2);
    ASSERT_EQ(v[0].nCols(), 1);
    INFO_MSG("mttkrp result shape correct for index [0]");

    ASSERT_FLOAT_EQ(v[0].entry(0, 0), 180.0);
    ASSERT_FLOAT_EQ(v[0].entry(1, 0), 0.0);
    INFO_MSG("mttkrp result values correct for index [0]");

    ASSERT_EQ(v[1].nRows(), 3);
    ASSERT_EQ(v[1].nCols(), 1);
    INFO_MSG("mttkrp result shape correct for index [1]");

    ASSERT_FLOAT_EQ(v[1].entry(0, 0), 150.0);
    ASSERT_FLOAT_EQ(v[1].entry(1, 0), 0.0);
    INFO_MSG("mttkrp result values correct for index [1]");

    ASSERT_EQ(v[2].nRows(), 4);
    ASSERT_EQ(v[2].nCols(), 1);
    INFO_MSG("mttkrp result shape correct for index [2]");

    ASSERT_FLOAT_EQ(v[2].entry(0, 0), 120.0);
    ASSERT_FLOAT_EQ(v[2].entry(1, 0), 0.0);
    INFO_MSG("mttkrp result values correct for index [2]");
  }

  // Uncomment to manually check what the answer should be.
  // Genten::print_sptensor(a, std::cout, "Sparse tensor for mttkrp test");
  // Genten::print_ktensor(oKtens, std::cout, "Ktensor for mttkrp test");
  // Genten::print_ktensor(v, std::cout, "Ktensor result from mttkrp_all");

  // Add another nonzero and repeat the three tests.
  a.subscript(1, 0) = 1;
  a.subscript(1, 1) = 2;
  a.subscript(1, 2) = 3;
  a.value(1) = 1.0;
  a_dev_dist = dtc.distributeTensor(a);
  a_dev_dist.setProcessorMap(pmap);

  dku->updateTensor(a_dev_dist);
  oKtens_dev_overlap = dku->createOverlapKtensor(oKtens_dev_dist);
  v_dev_overlap = dku->createOverlapKtensor(v_dev_dist);
  dku->doImport(oKtens_dev_overlap, oKtens_dev_dist);

  mttkrp_all(a_dev_dist, oKtens_dev_overlap, v_dev_overlap, algParams);
  dku->doExport(v_dev_dist, v_dev_overlap);
  v = dtc.template importToRoot<DefaultHostExecutionSpace>(v_dev_dist);

  if (DistContext::rank() == 0) {
    ASSERT_FLOAT_EQ(v[0].entry(0, 0), 180.0);
    ASSERT_FLOAT_EQ(v[0].entry(1, 0), 252.0);
    INFO_MSG("mttkrp result values correct for index [0], 2 sparse nnz");

    ASSERT_FLOAT_EQ(v[1].entry(0, 0), 150.0);
    ASSERT_FLOAT_EQ(v[1].entry(2, 0), 198.0);
    INFO_MSG("mttkrp result values correct for index [0], 2 sparse nnz");

    ASSERT_FLOAT_EQ(v[2].entry(0, 0), 120.0);
    ASSERT_FLOAT_EQ(v[2].entry(3, 0), 154.0);
    INFO_MSG("mttkrp result values correct for index [0], 2 sparse nnz");
  }

  delete dku;
}

TYPED_TEST(TestMixedFormatsT, MTTKRP_All_Type) {
  using exec_space = typename TestFixture::exec_space;

  struct TestCase {
    TestCase(const MTTKRP_All_Method::type mttkrp_method, const char *label)
        : mttkrp_method{mttkrp_method}, label{label} {}

    const MTTKRP_All_Method::type mttkrp_method;
    const char *label;

    const bool run{not SpaceProperties<exec_space>::is_gpu ||
                   mttkrp_method != MTTKRP_All_Method::type::Duplicated};
  };

  TestCase test_cases[]{TestCase{MTTKRP_All_Method::Iterated, "Iterated"},
                        TestCase{MTTKRP_All_Method::Atomic, "Atomic"},
                        TestCase{MTTKRP_All_Method::Duplicated, "Duplicated"}};

  for (const auto &tc : test_cases) {
    if (tc.run) {
      RunMTTKRAllTypeTest<exec_space>(tc.mttkrp_method, tc.label);
    }
  }
}

} // namespace UnitTests
} // namespace Genten
