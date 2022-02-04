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

#include "Genten_CpAls.hpp"
#include "Genten_IndxArray.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_Test_Utils.hpp"

using namespace Genten::Test;


static void  evaluateResult (const int             infolevel,
                             const ttb_indx        itersCompleted,
                             const ttb_real        stopTol,
                             const Genten::Ktensor &  result)
{
  std::stringstream  sMsg;
  sMsg << "CpAls finished after " << itersCompleted << " iterations";
  MESSAGE(sMsg.str());
  ttb_real tol = 2.5e-3;

  if (infolevel == 1)
    print_ktensor(result, std::cout,"Factorization result in ktensor form");

  // Check the final weights, which can be in any order.
  ttb_real  wght0 = result.weights(0);
  ttb_real  wght1 = result.weights(1);
  if (wght0 >= wght1)
  {
    ttb_real  diffA = fabs(wght0 - 2.828427);
    ttb_real  diffB = fabs(wght1 - 2.0);
    ASSERT( (diffA <= tol) && (diffB <= tol),
            "Result ktensor weights match" );

    ASSERT( fabs(result[0].entry(0,0)-0.7071) <= tol,
            "Result ktensor[0](0,0) matches");
    ASSERT( fabs(result[0].entry(1,0)-0.7071) <= tol,
            "Result ktensor[0](1,0) matches");
    ASSERT( fabs(result[0].entry(0,1)-1.0) <= tol,
            "Result ktensor[0](0,1) matches");
    ASSERT( fabs(result[0].entry(1,1)-0.0) <= tol,
            "Result ktensor[0](1,1) matches");

    ASSERT( fabs(result[1].entry(0,0)-0.7071) <= tol,
            "Result ktensor[1](0,0) matches");
    ASSERT( fabs(result[1].entry(1,0)-0.7071) <= tol,
            "Result ktensor[1](1,0) matches");
    ASSERT( fabs(result[1].entry(2,0)-0.0) <= tol,
            "Result ktensor[1](2,0) matches");
    ASSERT( fabs(result[1].entry(0,1)-0.7071) <= tol,
            "Result ktensor[1](0,1) matches");
    ASSERT( fabs(result[1].entry(1,1)-0.0) <= tol,
            "Result ktensor[1](1,1) matches");
    ASSERT( fabs(result[1].entry(2,1)-0.7071) <= tol,
            "Result ktensor[1](2,1) matches");

    ASSERT( fabs(result[2].entry(0,0)-0.7071) <= tol,
            "Result ktensor[2](0,0) matches");
    ASSERT( fabs(result[2].entry(1,0)-0.0) <= tol,
            "Result ktensor[2](1,0) matches");
    ASSERT( fabs(result[2].entry(2,0)-0.0) <= tol,
            "Result ktensor[2](2,0) matches");
    ASSERT( fabs(result[2].entry(3,0)-0.7071) <= tol,
            "Result ktensor[2](3,0) matches");
    ASSERT( fabs(result[2].entry(0,1)-0.7071) <= tol,
            "Result ktensor[2](0,1) matches");
    ASSERT( fabs(result[2].entry(1,1)-0.7071) <= tol,
            "Result ktensor[2](1,1) matches");
    ASSERT( fabs(result[2].entry(2,1)-0.0) <= tol,
            "Result ktensor[2](2,1) matches");
    ASSERT( fabs(result[2].entry(3,1)-0.0) <= tol,
            "Result ktensor[2](2,1) matches");
  }
  else
  {
    ttb_real  diffA = fabs(wght0 - 2.0);
    ttb_real  diffB = fabs(wght1 - 2.8284);
    ASSERT( (diffA <= tol) && (diffB <= tol),
            "Result ktensor weights match" );

    ASSERT( fabs(result[0].entry(0,1)-0.7071) <= tol,
            "Result ktensor[0](0,1) matches");
    ASSERT( fabs(result[0].entry(1,1)-0.7071) <= tol,
            "Result ktensor[0](1,1) matches");
    ASSERT( fabs(result[0].entry(0,0)-1.0) <= tol,
            "Result ktensor[0](0,0) matches");
    ASSERT( fabs(result[0].entry(1,0)-0.0) <= tol,
            "Result ktensor[0](1,0) matches");

    ASSERT( fabs(result[1].entry(0,1)-0.7071) <= tol,
            "Result ktensor[1](0,1) matches");
    ASSERT( fabs(result[1].entry(1,1)-0.7071) <= tol,
            "Result ktensor[1](1,1) matches");
    ASSERT( fabs(result[1].entry(2,1)-0.0) <= tol,
            "Result ktensor[1](2,1) matches");
    ASSERT( fabs(result[1].entry(0,0)-0.7071) <= tol,
            "Result ktensor[1](0,0) matches");
    ASSERT( fabs(result[1].entry(1,0)-0.0) <= tol,
            "Result ktensor[1](1,0) matches");
    ASSERT( fabs(result[1].entry(2,0)-0.7071) <= tol,
            "Result ktensor[1](2,0) matches");

    ASSERT( fabs(result[2].entry(0,1)-0.7071) <= tol,
            "Result ktensor[2](0,1) matches");
    ASSERT( fabs(result[2].entry(1,1)-0.0) <= tol,
            "Result ktensor[2](1,1) matches");
    ASSERT( fabs(result[2].entry(2,1)-0.0) <= tol,
            "Result ktensor[2](2,1) matches");
    ASSERT( fabs(result[2].entry(3,1)-0.7071) <= tol,
            "Result ktensor[2](3,1) matches");
    ASSERT( fabs(result[2].entry(0,0)-0.7071) <= tol,
            "Result ktensor[2](0,0) matches");
    ASSERT( fabs(result[2].entry(1,0)-0.7071) <= tol,
            "Result ktensor[2](1,0) matches");
    ASSERT( fabs(result[2].entry(2,0)-0.0) <= tol,
            "Result ktensor[2](2,0) matches");
    ASSERT( fabs(result[2].entry(3,0)-0.0) <= tol,
            "Result ktensor[2](2,0) matches");
  }

  return;
}


/*!
 *  The test factors a simple 2x3x4 sparse tensor into known components.
 *  Matlab formulation:
 *    subs = [1 1 1 ; 2 1 1 ; 1 2 1 ; 2 2 1 ; 1 3 1 ; 1 1 2 ; 1 3 2 ; 1 1 4 ;
 *            2 1 4 ; 1 2 4 ; 2 2 4]
 *    vals = [2 1 1 1 1 1 1 1 1 1 1]
 *    X = sptensor (subs, vals', [2 3 4])
 *    X0 = { rand(2,2), rand(3,2), rand(4,2) }, or values below (it matters!)
 *    F = cp_als (X,2, 'init',X0)
 *  Exact solution (because this is how X was constructed):
 *    lambda = [1 1]
 *    A = [1 1 ; 0 1]
 *    B = [1 1 ; 0 1 ; 1 0]
 *    C = [1 1 ; 1 0 ; 0 0 ; 0 1]
 *  Exact solution as a normalized ktensor:
 *    lambda = [2.0 2.8284]
 *    A = [1.0    0.7071 ; 0.0    0.7071]
 *    B = [0.7071 0.7071 ; 0.0    0.7071 ; 0.7071 0.0   ]
 *    C = [0.7071 0.7071 ; 0.7071 0.0    ; 0.0    0.0    ; 0.0    0.7071]
 *  Random start point can converge to a different (worse) solution, so the
 *  test uses a particular start point:
 *    A0 = [0.8 0.5 ; 0.2 0.5]
 *    B0 = [0.5 0.5 ; 0.1 0.5 ; 0.5 0.1]
 *    C0 = [0.7 0.7 ; 0.7 0.1 ; 0.1 0.1 ; 0.1 0.7]
 *
 * Note ETP 03/20/2018:  Originally this comment as the weights as [2.8284 2.0],
 * but based on my arithmetic, it should be [2.8284 2.0] based on the order of
 * the factor matrix columns shown above.  This is consistent with what the code
 * produces.
 */
template <typename ExecSpace>
void Genten_Test_CpAls_Type (Genten::MTTKRP_Method::type mttkrp_method,
                             int infolevel, const std::string& label)
{
  typedef ExecSpace exec_space;
  typedef Genten::DefaultHostExecutionSpace host_exec_space;
  typedef Genten::SptensorT<exec_space> Sptensor_type;
  typedef Genten::SptensorT<host_exec_space> Sptensor_host_type;
  typedef Genten::TensorT<exec_space> Tensor_type;
  typedef Genten::TensorT<host_exec_space> Tensor_host_type;

  SETUP_DISABLE_CERR;

  std::string space_name = Genten::SpaceProperties<exec_space>::name();
  initialize("Test of Genten::CpAls ("+label+", "+space_name+")", infolevel);

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
  initialBasis.setWeights(1.0);
  initialBasis.setMatrices(0.0);
  initialBasis[0].entry(0,0) = 0.8;
  initialBasis[0].entry(1,0) = 0.2;
  initialBasis[0].entry(0,1) = 0.5;
  initialBasis[0].entry(1,1) = 0.5;
  initialBasis[1].entry(0,0) = 0.5;
  initialBasis[1].entry(1,0) = 0.1;
  initialBasis[1].entry(2,0) = 0.5;
  initialBasis[1].entry(0,1) = 0.5;
  initialBasis[1].entry(1,1) = 0.5;
  initialBasis[1].entry(2,1) = 0.1;
  initialBasis[2].entry(0,0) = 0.7;
  initialBasis[2].entry(1,0) = 0.7;
  initialBasis[2].entry(2,0) = 0.1;
  initialBasis[2].entry(3,0) = 0.1;
  initialBasis[2].entry(0,1) = 0.7;
  initialBasis[2].entry(1,1) = 0.1;
  initialBasis[2].entry(2,1) = 0.1;
  initialBasis[2].entry(3,1) = 0.7;
  initialBasis.weights(0) = 2.0; // Test with weights different from one.
  if (infolevel == 1)
    print_ktensor(initialBasis,std::cout,"Initial guess for CpAls");

  // Copy initialBasis to the device
  Genten::KtensorT<exec_space> initialBasis_dev =
    create_mirror_view( exec_space(), initialBasis );
  deep_copy( initialBasis_dev, initialBasis );

  // Factorize.
  Genten::AlgParams algParams;
  algParams.rank = nNumComponents;
  algParams.tol = 1.0e-6;
  algParams.maxiters = 100;
  algParams.maxsecs = -1.0;
  algParams.printitn = infolevel;
  algParams.mttkrp_method = mttkrp_method;
  Genten::Ktensor result(nNumComponents, dims.size(), dims);
  Genten::KtensorT<exec_space> result_dev =
    create_mirror_view( exec_space(), result );
  ttb_indx  itersCompleted;
  ttb_real  resNorm;
  try
  {
    // Request performance information on every 3rd iteration.
    // Allocation adds two more for start and stop states of the algorithm.
    ttb_indx  nMaxPerfSize = 2 + (algParams.maxiters / 3);
    Genten::CpAlsPerfInfo *  perfInfo = new Genten::CpAlsPerfInfo[nMaxPerfSize];
    deep_copy(result_dev, initialBasis_dev);
    Genten::cpals_core(X_dev, result_dev, algParams, itersCompleted, resNorm,
                       3, perfInfo);
    // Check performance information.
    bool  bIsOK = true;
    for (ttb_indx  i = 0; i < nMaxPerfSize; i++)
    {
      if ((perfInfo[i].nIter != -1) && (perfInfo[i].nIter > 0))
      {
        if ((perfInfo[i].dFit < 0.99) || (perfInfo[i].dFit > 1.00))
          bIsOK = false;
        if (perfInfo[i].dResNorm > 0.03)
          bIsOK = false;
        if (perfInfo[i].dCumTime < 0.0)
          bIsOK = false;
      }
    }
    ASSERT( bIsOK, "Performance info from cpals_core is reasonable." );
    delete[] perfInfo;
  }
  catch(std::string sExc)
  {
    // Should not happen.
    MESSAGE(sExc);
    ASSERT( true, "Call to cpals_core threw an exception." );
    return;
  }

  // Copy result to host
  deep_copy(result, result_dev);

  evaluateResult(infolevel, itersCompleted, algParams.tol, result);

  // Test factorization from a bad initial guess.
  MESSAGE("Creating a ktensor with initial guess all zero");
  Genten::Ktensor  initialZero (nNumComponents, dims.size(), dims);
  initialZero.setWeights(0.0);
  initialZero.setMatrices(0.0);
  Genten::KtensorT<exec_space> initialZero_dev =
    create_mirror_view( exec_space(), initialZero );
  deep_copy( initialZero_dev, initialZero );
  MESSAGE("Checking if linear solver detects singular guess");
  DISABLE_CERR;
  try
  {
    deep_copy(result_dev, initialZero_dev);
    algParams.printitn = 0;
    Genten::cpals_core(X_dev, result_dev, algParams, itersCompleted, resNorm,
                       0, NULL);
  }
  catch(std::string sExc)
  {
    // The test expects this to happen.
    std::stringstream  sMsg;
    sMsg << "Exception caught: " << sExc;
    ASSERT( true, sMsg.str() );
  }
  REENABLE_CERR;

  // Repeat the tests using the same data, but in a dense Tensor.

  MESSAGE("Creating a dense tensor with data to model");
  Tensor_type Xd_dev(X_dev);

  // Factorize.
  try
  {
    algParams.printitn = infolevel;
    deep_copy(result_dev, initialBasis_dev);
    Genten::cpals_core(Xd_dev, result_dev, algParams, itersCompleted, resNorm,
                       0, nullptr);
  }
  catch(std::string sExc)
  {
    // Should not happen.
    MESSAGE(sExc);
    ASSERT( true, "Call to cpals_core threw an exception." );
    return;
  }

  deep_copy(result, result_dev);
  evaluateResult(infolevel, itersCompleted, algParams.tol, result);

  finalize();
  return;
}

template <typename ExecSpace>
void Genten_Test_CpAls_Space (int infolevel)
{
  typedef Genten::SpaceProperties<ExecSpace> space_prop;

  Genten_Test_CpAls_Type<ExecSpace>(Genten::MTTKRP_Method::Atomic,infolevel,
                                    "Atomic");
  if (!space_prop::is_cuda)
    Genten_Test_CpAls_Type<ExecSpace>(
      Genten::MTTKRP_Method::Duplicated,infolevel, "Duplicated");
  Genten_Test_CpAls_Type<ExecSpace>(Genten::MTTKRP_Method::Perm,infolevel,
                                    "Perm");
}

void Genten_Test_CpAls(int infolevel) {
#ifdef KOKKOS_ENABLE_CUDA
  Genten_Test_CpAls_Space<Kokkos::Cuda>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_HIP
  Genten_Test_CpAls_Space<Kokkos::Experimental::HIP>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  Genten_Test_CpAls_Space<Kokkos::OpenMP>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_THREADS
  Genten_Test_CpAls_Space<Kokkos::Threads>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_SERIAL
  Genten_Test_CpAls_Space<Kokkos::Serial>(infolevel);
#endif
}
