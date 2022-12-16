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

#include <Genten_CP_Model.hpp>
#include <Genten_CpAls.hpp>
#include <Genten_DistContext.hpp>
#include <Genten_DistTensorContext.hpp>
#include <Genten_FacTestSetGenerator.hpp>
#include <Genten_Ktensor.hpp>
#include <Genten_Sptensor.hpp>
#include <Genten_Tensor.hpp>
#include <sstream>

#include "Genten_Test_Utils.hpp"

#include <gtest/gtest.h>

namespace Genten {
namespace UnitTests {

template <typename ExecSpace> struct TestHessVecT : public ::testing::Test {
  using exec_space = ExecSpace;
};

TYPED_TEST_SUITE(TestHessVecT, genten_test_types);

template <typename exec_space>
void RunHessVecDenseTest(Genten::Hess_Vec_Tensor_Method::type hess_vec_method,
                         const std::string &label) {
  DistContext::Barrier();

  const ttb_indx nd = 4;
  const ttb_indx nc = 3;
  const ttb_indx nnz = 20;
  IndxArray dims = {3, 4, 5, 6};
  RandomMT rng(12345);
  Ktensor sol_host;
  Sptensor x_host;
  FacTestSetGenerator testGen;
  ASSERT_TRUE(
      testGen.genSpFromRndKtensor(dims, nc, nnz, rng, x_host, sol_host));

  SptensorT<exec_space> x = create_mirror_view(exec_space(), x_host);
  deep_copy(x, x_host);

  // Create random Ktensors for multiply
  KtensorT<exec_space> a(nc, nd, x.size()), v(nc, nd, x.size());
  auto a_host = create_mirror_view(a);
  auto v_host = create_mirror_view(v);
  a_host.setMatricesScatter(true, false, rng);
  v_host.setMatricesScatter(true, false, rng);
  a_host.setWeights(1.0);
  v_host.setWeights(1.0);
  deep_copy(a, a_host);
  deep_copy(v, v_host);

  AlgParams algParams;
  algParams.hess_vec_tensor_method = hess_vec_method;
  algParams.mttkrp_method = MTTKRP_Method::Atomic;
  algParams.mttkrp_all_method = Genten::MTTKRP_All_Method::Atomic;

  // Compute full hess-vec
  algParams.hess_vec_method = Hess_Vec_Method::Full;
  CP_Model<SptensorT<exec_space>> cp_model_full(x, a, algParams);
  KtensorT<exec_space> u(nc, nd, x.size());
  cp_model_full.update(a);
  cp_model_full.hess_vec(u, a, v);

  // Compute finite-difference approximation to hess-vec
  algParams.hess_vec_method = Hess_Vec_Method::FiniteDifference;
  CP_Model<SptensorT<exec_space>> cp_model_fd(x, a, algParams);
  KtensorT<exec_space> u_fd(nc, nd, x.size());
  cp_model_fd.update(a);
  cp_model_fd.hess_vec(u_fd, a, v);

  // Compute dense hess_vec
  algParams.hess_vec_method = Hess_Vec_Method::Full;
  TensorT<exec_space> x_dense(x);
  CP_Model<TensorT<exec_space>> cp_model_dense(x_dense, a, algParams);
  KtensorT<exec_space> u_dense(nc, nd, x.size());
  cp_model_dense.update(a);
  cp_model_dense.hess_vec(u_dense, a, v);

  // Check hess-vec values
  const ttb_real tol2 = 100.0 * MACHINE_EPSILON;
  auto u_host = create_mirror_view(u);
  deep_copy(u_host, u);
  auto u_dense_host = create_mirror_view(u_dense);
  deep_copy(u_dense_host, u_dense);
  for (ttb_indx n = 0; n < nd; ++n) {
    for (ttb_indx i = 0; i < x.size(n); ++i) {
      for (ttb_indx j = 0; j < nc; ++j) {
        std::stringstream ss;
        ss << "dense hess-vec values correct for dim " << n << ", entry (" << i
           << "," << j << ")";
        GENTEN_TRUE(
            FLOAT_EQ(u_host[n].entry(i, j), u_dense_host[n].entry(i, j), tol2),
            ss.str().c_str());
      }
    }
  }
}

TYPED_TEST(TestHessVecT, HessVecDense) {
  using exec_space = typename TestFixture::exec_space;

  struct TestCase {
    TestCase(const Hess_Vec_Tensor_Method::type hess_vec_method,
             const char *label)
        : hess_vec_method{hess_vec_method}, label{label} {}

    const Hess_Vec_Tensor_Method::type hess_vec_method;
    const char *label;

    const bool run{not SpaceProperties<exec_space>::is_gpu ||
                   hess_vec_method != Hess_Vec_Tensor_Method::type::Duplicated};
  };

  TestCase test_cases[]{
      TestCase{Hess_Vec_Tensor_Method::type::Atomic, "Atomic"},
      TestCase{Hess_Vec_Tensor_Method::type::Duplicated, "Duplicated"}};

  for (const auto &tc : test_cases) {
    if (tc.run) {
      RunHessVecDenseTest<exec_space>(tc.hess_vec_method, tc.label);
    }
  }
}

template <typename exec_space>
void RunHessVecTest(Genten::Hess_Vec_Tensor_Method::type hess_vec_method,
                    const std::string &label) {
  DistContext::Barrier();
  Genten::DistTensorContext<exec_space> dtc;

  const ttb_indx nd = 4;
  const ttb_indx nc = 3;
  const ttb_indx nnz = 20;
  IndxArray dims = {3, 4, 5, 6};
  RandomMT rng(12345);
  Ktensor sol_host;
  Sptensor x_host;
  FacTestSetGenerator testGen;
  ASSERT_TRUE(
      testGen.genSpFromRndKtensor(dims, nc, nnz, rng, x_host, sol_host));

  SptensorT<exec_space> x_dist = dtc.distributeTensor(x_host);
  const ProcessorMap *pmap = dtc.pmap_ptr().get();
  x_dist.setProcessorMap(pmap);

  // Create random Ktensors for multiply
  KtensorT<exec_space> a_dev(nc, nd, x_dist.size()),
      v_dev(nc, nd, x_dist.size());
  auto a_host = create_mirror_view(a_dev);
  auto v_host = create_mirror_view(v_dev);
  a_host.setMatricesScatter(true, false, rng);
  v_host.setMatricesScatter(true, false, rng);
  a_host.setWeights(1.0);
  v_host.setWeights(1.0);
  deep_copy(a_dev, a_host);
  deep_copy(v_dev, v_host);

  KtensorT<exec_space> a_dev_dist = dtc.exportFromRoot(a_dev);
  a_dev_dist.setProcessorMap(pmap);
  KtensorT<exec_space> v_dev_dist = dtc.exportFromRoot(v_dev);
  v_dev_dist.setProcessorMap(pmap);

  AlgParams algParams;
  algParams.hess_vec_tensor_method = hess_vec_method;
  algParams.mttkrp_method = MTTKRP_Method::Atomic;
  algParams.mttkrp_all_method = Genten::MTTKRP_All_Method::Atomic;

  // Compute full hess-vec
  algParams.hess_vec_method = Hess_Vec_Method::Full;
  CP_Model<SptensorT<exec_space>> cp_model_full(x_dist, a_dev_dist, algParams);
  KtensorT<exec_space> u(nc, nd, x_dist.size());
  KtensorT<exec_space> u_dist = dtc.exportFromRoot(u);
  u_dist.setProcessorMap(pmap);
  cp_model_full.update(a_dev_dist);
  cp_model_full.hess_vec(u_dist, a_dev_dist, v_dev_dist);

  // Compute finite-difference approximation to hess-vec
  algParams.hess_vec_method = Hess_Vec_Method::FiniteDifference;
  CP_Model<SptensorT<exec_space>> cp_model_fd(x_dist, a_dev_dist, algParams);
  KtensorT<exec_space> u_fd(nc, nd, x_dist.size());
  KtensorT<exec_space> u_fd_dist = dtc.exportFromRoot(u_fd);
  u_fd_dist.setProcessorMap(pmap);
  cp_model_fd.update(a_dev_dist);
  cp_model_fd.hess_vec(u_fd_dist, a_dev_dist, v_dev_dist);

  // Check values
  u = dtc.template importToRoot<exec_space>(u_dist);
  u_fd = dtc.template importToRoot<exec_space>(u_fd_dist);

  if (DistContext::rank() == 0) {
    const ttb_real tol = 1e-6;

    auto u_host = create_mirror_view(u);
    deep_copy(u_host, u);
    auto u_fd_host = create_mirror_view(u_fd);
    deep_copy(u_fd_host, u_fd);

    for (ttb_indx n = 0; n < nd; ++n) {
      for (ttb_indx i = 0; i < x_dist.size(n); ++i) {
        for (ttb_indx j = 0; j < nc; ++j) {
          const auto uh_ij = u_host[n].entry(i, j);
          const auto ufdh_ij = u_fd_host[n].entry(i, j);

          std::stringstream ss;
          ss << "hess-vec values correct for dim " << n << ", entry (" << i
             << "," << j << "), u_host[n].entry(i, j): " << uh_ij
             << ", u_fd_host[n].entry(i, j): " << ufdh_ij;

          GENTEN_TRUE(FLOAT_EQ(uh_ij, ufdh_ij, tol), ss.str().c_str());
        }
      }
    }
  }
}

TYPED_TEST(TestHessVecT, HessVec) {
  using exec_space = typename TestFixture::exec_space;

  struct TestCase {
    TestCase(const Hess_Vec_Tensor_Method::type hess_vec_method,
             const char *label)
        : hess_vec_method{hess_vec_method}, label{label} {}

    const Hess_Vec_Tensor_Method::type hess_vec_method;
    const char *label;

    const bool run{not SpaceProperties<exec_space>::is_gpu ||
                   hess_vec_method != Hess_Vec_Tensor_Method::type::Duplicated};
  };

  TestCase test_cases[]{
      TestCase{Hess_Vec_Tensor_Method::type::Atomic, "Atomic"},
      TestCase{Hess_Vec_Tensor_Method::type::Duplicated, "Duplicated"}};

  for (const auto &tc : test_cases) {
    if (tc.run) {
      RunHessVecTest<exec_space>(tc.hess_vec_method, tc.label);
    }
  }
}

TYPED_TEST(TestHessVecT, HessVecGaussNewton) {
  DistContext::Barrier();

  using exec_space = typename TestFixture::exec_space;

  const ttb_indx nd = 3;
  const ttb_indx nc = 3;
  IndxArray dims{3, 4, 5};
  Ktensor sol_host(nc, nd, dims);
  RandomMT rng(12345);
  sol_host.setMatricesScatter(true, false, rng);
  sol_host.setWeights(1.0);
  KtensorT<exec_space> sol = create_mirror_view(exec_space(), sol_host);
  deep_copy(sol, sol_host);

  // Create dense tensor from ktensor.  The Gauss-Newton and full Hessian
  // should be equal when the residual is zero, i.e., the ktensor exactly
  // matches the tensor
  TensorT<exec_space> x(sol);

  // Create random Ktensors for multiply
  KtensorT<exec_space> v(nc, nd, x.size());
  auto v_host = create_mirror_view(v);
  v_host.setMatricesScatter(true, false, rng);
  v_host.setWeights(1.0);
  deep_copy(v, v_host);

  AlgParams algParams;

  // Compute full hess-vec
  algParams.hess_vec_method = Hess_Vec_Method::Full;
  CP_Model<TensorT<exec_space>> cp_model_full(x, sol, algParams);
  KtensorT<exec_space> u_full(nc, nd, x.size());

  cp_model_full.update(sol);
  cp_model_full.hess_vec(u_full, sol, v);

  // Compute Gauss-Newton approximation to hess-vec
  algParams.hess_vec_method = Hess_Vec_Method::GaussNewton;
  CP_Model<TensorT<exec_space>> cp_model_gn(x, sol, algParams);
  KtensorT<exec_space> u_gn(nc, nd, x.size());
  cp_model_gn.update(sol);
  cp_model_gn.hess_vec(u_gn, sol, v);

  // Check residuals are in fact 0
  const ttb_real tol = 100.0 * MACHINE_EPSILON;
  cp_model_full.update(sol);
  ttb_real residual = cp_model_full.value(sol);
  GENTEN_TRUE(FLOAT_EQ(residual, ttb_real(0.0), tol), "residual value correct");

  // Check hess-vec values
  auto u_full_host = create_mirror_view(u_full);
  auto u_gn_host = create_mirror_view(u_gn);
  deep_copy(u_full_host, u_full);
  deep_copy(u_gn_host, u_gn);
  for (ttb_indx n = 0; n < nd; ++n) {
    for (ttb_indx i = 0; i < x.size(n); ++i) {
      for (ttb_indx j = 0; j < nc; ++j) {
        std::stringstream ss;
        ss << "hess-vec values correct for dim " << n << ", entry (" << i << ","
           << j << ")";

        GENTEN_TRUE(
            FLOAT_EQ(u_full_host[n].entry(i, j), u_gn_host[n].entry(i, j), tol),
            ss.str().c_str());
      }
    }
  }
}

} // namespace UnitTests
} // namespace Genten
