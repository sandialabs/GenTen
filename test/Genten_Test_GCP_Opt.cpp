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

#include <sstream>

#include "Genten_GCP_Opt.hpp"
#include "Genten_IndxArray.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Test_Utils.hpp"

using namespace Genten::Test;

/*!
 *  The test factors a simple 2x3x4 sparse tensor into known components.
 *  Matlab formulation:
 *    subs = [1 1 1 ; 2 1 1 ; 1 2 1 ; 2 2 1 ; 1 3 1 ; 1 1 2 ; 1 3 2 ; 1 1 4 ;
 *            2 1 4 ; 1 2 4 ; 2 2 4]
 *    vals = [2 1 1 1 1 1 1 1 1 1 1]
 *    X = sptensor (subs, vals', [2 3 4])
 *    X0 = { rand(2,2), rand(3,2), rand(4,2) }, or values below (it matters!)
 *    F = cp_als (X,2, 'init',X0)
 *  There are many possible factors, so instead of comparing to an exact
 *  solution, just check that when you multiply the factors together, you
 *  get the original tensor.
 */
template <typename ExecSpace>
void Genten_Test_GCP_Opt_Type (int infolevel, const std::string& label,
                               Genten::MTTKRP_Method::type mttkrp_method,
                               const Genten::GCP_LossFunction::type loss_type)
{
  typedef ExecSpace exec_space;
  typedef Genten::DefaultHostExecutionSpace host_exec_space;
  typedef Genten::SptensorT<exec_space> Sptensor_type;
  typedef Genten::SptensorT<host_exec_space> Sptensor_host_type;

  std::string space_name = Genten::SpaceProperties<exec_space>::name();
  initialize("Test of Genten::GCP_Opt ("+label+", "+space_name+")", infolevel);

  MESSAGE("Creating a sparse tensor with data to model");
  Genten::IndxArray  dims(3);
  dims[0] = 2;  dims[1] = 3;  dims[2] = 4;
  Sptensor_host_type  X(dims,11);
  X.subscript(0,0) = 0;  X.subscript(0,1) = 0;  X.subscript(0,2) = 0;
  X.value(0) = 2.0;
  X.subscript(1,0) = 1;  X.subscript(1,1) = 0;  X.subscript(1,2) = 0;
  X.value(1) = 1.0;
  X.subscript(2,0) = 0;  X.subscript(2,1) = 1;  X.subscript(2,2) = 0;
  X.value(2) = 1.0;
  X.subscript(3,0) = 1;  X.subscript(3,1) = 1;  X.subscript(3,2) = 0;
  X.value(3) = 1.0;
  X.subscript(4,0) = 0;  X.subscript(4,1) = 2;  X.subscript(4,2) = 0;
  X.value(4) = 1.0;
  X.subscript(5,0) = 0;  X.subscript(5,1) = 0;  X.subscript(5,2) = 1;
  X.value(5) = 1.0;
  X.subscript(6,0) = 0;  X.subscript(6,1) = 2;  X.subscript(6,2) = 1;
  X.value(6) = 1.0;
  X.subscript(7,0) = 0;  X.subscript(7,1) = 0;  X.subscript(7,2) = 3;
  X.value(7) = 1.0;
  X.subscript(8,0) = 1;  X.subscript(8,1) = 0;  X.subscript(8,2) = 3;
  X.value(8) = 1.0;
  X.subscript(9,0) = 0;  X.subscript(9,1) = 1;  X.subscript(9,2) = 3;
  X.value(9) = 1.0;
  X.subscript(10,0) = 1;  X.subscript(10,1) = 1;  X.subscript(10,2) = 3;
  X.value(10) = 1.0;
  ASSERT(X.nnz() == 11, "Data tensor has 11 nonzeroes");

  // Copy X to device
  Sptensor_type X_dev = create_mirror_view( exec_space(), X );
  deep_copy( X_dev, X );
  if (mttkrp_method == Genten::MTTKRP_Method::Perm)
    X_dev.createPermutation();

  // Load a known initial guess.
  MESSAGE("Creating a ktensor with initial guess of lin indep basis vectors");
  ttb_indx  nNumComponents = 2;
  Genten::Ktensor  initialBasis (nNumComponents, dims.size(), dims);
  ttb_indx seed = 12345;
  Genten::RandomMT cRMT(seed);
  initialBasis.setMatricesScatter(false, false, cRMT);
  initialBasis.setWeights(1.0);

  if (infolevel == 1)
    print_ktensor(initialBasis,std::cout,"Initial guess for GCP-Opt");

  // Copy initialBasis to the device
  Genten::KtensorT<exec_space> initialBasis_dev =
    create_mirror_view( exec_space(), initialBasis );
  deep_copy( initialBasis_dev, initialBasis );

  // Factorize.
  Genten::AlgParams algParams;
  algParams.tol = 1.0e-6;
  algParams.maxiters = 100;
  algParams.mttkrp_method = mttkrp_method;
  Genten::KtensorT<exec_space> result_dev;
  std::ostream* stream = (infolevel == 1) ? &std::cout : nullptr;
  try
  {
    result_dev = initialBasis_dev;
    Genten::gcp_opt <Sptensor_type> (X_dev, result_dev, algParams, stream);
  }
  catch(std::string sExc)
  {
    // Should not happen.
    MESSAGE(sExc);
    ASSERT( true, "Call to gcp_opt threw an exception." );
    return;
  }

  // Copy result to host
  Genten::Ktensor result = initialBasis;
  deep_copy( result, result_dev );

  if (infolevel == 1)
    print_ktensor(result, std::cout, "Factorization result in ktensor form");

  // Multiply Ktensor entries and compare to tensor
  if (infolevel == 1)
    std::cout << "Checking factorization matches original tensor:" << std::endl;
  const ttb_real tol = 1.0e-3;
  const ttb_indx nnz = X.nnz();
  const Genten::IndxArray subs(3);
  for (ttb_indx i=0; i<nnz; ++i) {
    X.getSubscripts(i, subs);
    const ttb_real x_val = X.value(i);
    const ttb_real val = result.entry(subs);
    if (infolevel == 1) {
      std::cout << "X(" << subs[0] << "," << subs[1] << "," << subs[2] << ") = "
                << x_val << ", Ktensor = " << val << std::endl;
    }
    ASSERT( fabs(x_val-val) <= tol, "Result matches" );
  }

  finalize();
  return;
}

template <typename ExecSpace>
void Genten_Test_GCP_Opt_Space (int infolevel)
{
  Genten_Test_GCP_Opt_Type<ExecSpace>(infolevel,"Atomic, Gaussian",
                                      Genten::MTTKRP_Method::Atomic,
                                      Genten::GCP_LossFunction::Gaussian);
  Genten_Test_GCP_Opt_Type<ExecSpace>(infolevel,"Duplicated, Gaussian",
                                      Genten::MTTKRP_Method::Duplicated,
                                      Genten::GCP_LossFunction::Gaussian);
  Genten_Test_GCP_Opt_Type<ExecSpace>(infolevel,"Perm, Gaussian",
                                      Genten::MTTKRP_Method::Perm,
                                      Genten::GCP_LossFunction::Gaussian);
}

void Genten_Test_GCP_Opt(int infolevel) {
#ifdef KOKKOS_ENABLE_CUDA
  Genten_Test_GCP_Opt_Space<Kokkos::Cuda>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_HIP
  Genten_Test_GCP_Opt_Space<Kokkos::Experimental::HIP>(infolevel);
#endif
#ifdef ENABLE_SYCL_FOR_CUDA
  Genten_Test_GCP_Opt_Space<Kokkos::Experimental::SYCL>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  Genten_Test_GCP_Opt_Space<Kokkos::OpenMP>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_THREADS
  Genten_Test_GCP_Opt_Space<Kokkos::Threads>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_SERIAL
  Genten_Test_GCP_Opt_Space<Kokkos::Serial>(infolevel);
#endif
}
