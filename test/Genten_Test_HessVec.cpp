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


#include "Genten_Array.hpp"
#include "Genten_FacMatrix.hpp"
#include "Genten_IOtext.hpp"             // In case debug lines are uncommented
#include "Genten_Ktensor.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_CP_Model.hpp"
#include "Genten_Test_Utils.hpp"
#include "Genten_Util.hpp"
#include "Genten_FacTestSetGenerator.hpp"

using namespace Genten::Test;


/* This file contains unit tests for Hessian-vector product operations.
 */

template <typename ExecSpace>
void Genten_Test_HessVec_Type(Genten::MTTKRP_All_Method::type mttkrp_method,
                                int infolevel, const std::string& label)
{
  std::string space_name = Genten::SpaceProperties<ExecSpace>::name();
  initialize("HessVec tests ("+label+", "+space_name+")", infolevel);

  // Create random sparse tensor
  const ttb_indx nd = 3;
  const ttb_indx nc = 3;
  const ttb_indx nnz = 10;
  Genten::IndxArray dims = { 3, 4, 5 };
  Genten::RandomMT rng (12345);
  Genten::Ktensor sol_host;
  Genten::Sptensor x_host;
  Genten::FacTestSetGenerator testGen;
  bool r = testGen.genSpFromRndKtensor(dims, nc, nnz, rng, x_host, sol_host);
  if (!r)
    Genten::error("*** Call to genSpFromRndKtensor failed.\n");
  Genten::SptensorT<ExecSpace> x = create_mirror_view( ExecSpace(), x_host );
  deep_copy( x, x_host );

  // Create random Ktensors for multiply
  Genten::KtensorT<ExecSpace> a(nc, nd, x.size()), v(nc, nd, x.size());
  auto a_host = create_mirror_view(a);
  auto v_host = create_mirror_view(v);
  a_host.setMatricesScatter(true, false, rng);
  v_host.setMatricesScatter(true, false, rng);
  a_host.setWeights(1.0);
  v_host.setWeights(1.0);
  deep_copy(a, a_host);
  deep_copy(v, v_host);

  Genten::AlgParams algParams;
  algParams.mttkrp_all_method = mttkrp_method;
  algParams.mttkrp_method = Genten::MTTKRP_Method::Atomic;

  // Compute full hess-vec
  algParams.hess_vec_method = Genten::Hess_Vec_Method::Full;
  Genten::CP_Model<Genten::SptensorT<ExecSpace> >
    cp_model_full(x, a, algParams);
  Genten::KtensorT<ExecSpace> u(nc, nd, x.size());
  cp_model_full.update(a);
  cp_model_full.hess_vec(u, a, v);

  // Compute finite-difference approximation to hess-vec
  algParams.hess_vec_method = Genten::Hess_Vec_Method::FiniteDifference;
  Genten::CP_Model<Genten::SptensorT<ExecSpace> >
    cp_model_fd(x, a, algParams);
  Genten::KtensorT<ExecSpace> u_fd(nc, nd, x.size());
  cp_model_fd.update(a);
  cp_model_fd.hess_vec(u_fd, a, v);

  // Check values
  const ttb_real tol = 1e-6;
  auto u_host = create_mirror_view(u);
  auto u_fd_host = create_mirror_view(u_fd);
  deep_copy(u_host, u);
  deep_copy(u_fd_host, u);
  //print_ktensor(u_host, std::cout, "u");
  //print_ktensor(u_fd_host, std::cout, "u_fd");
  for (ttb_indx n=0; n<nd; ++n) {
    for (ttb_indx i=0; i<x.size(n); ++i) {
      for (ttb_indx j=0; j<nc; ++j) {
        std::string msg = "hess-vec values correct for dim " +
          std::to_string(n) + ", entry (" +
          std::to_string(i) + "," +
          std::to_string(j) + ")";
        ASSERT(FLOAT_EQ(u_host[n].entry(i,j), u_fd_host[n].entry(i,j), tol),
               msg);
      }
    }
  }

  finalize();
  return;
}

template <typename ExecSpace>
void Genten_Test_HessVec_Space(int infolevel)
{
  typedef Genten::SpaceProperties<ExecSpace> space_prop;

  Genten_Test_HessVec_Type<ExecSpace>(
    Genten::MTTKRP_All_Method::Atomic, infolevel, "Atomic");
  if (!space_prop::is_gpu)
    Genten_Test_HessVec_Type<ExecSpace>(
      Genten::MTTKRP_All_Method::Duplicated, infolevel, "Duplicated");
}

void Genten_Test_HessVec(int infolevel) {
#ifdef KOKKOS_ENABLE_CUDA
  Genten_Test_HessVec_Space<Kokkos::Cuda>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_HIP
  Genten_Test_HessVec_Space<Kokkos::Experimental::HIP>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  Genten_Test_HessVec_Space<Kokkos::OpenMP>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_THREADS
  Genten_Test_HessVec_Space<Kokkos::Threads>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_SERIAL
  Genten_Test_HessVec_Space<Kokkos::Serial>(infolevel);
#endif
}
