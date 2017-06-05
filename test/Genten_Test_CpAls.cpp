//@HEADER
// ************************************************************************
//     Genten Tensor Toolbox
//     Software package for tensor math by Sandia National Laboratories
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
// ************************************************************************
//@HEADER


#include <sstream>

#include "Genten_CpAls.h"
#include "Genten_IndxArray.h"
#include "Genten_IOtext.h"
#include "Genten_Ktensor.h"
#include "Genten_MixedFormatOps.h"
#include "Genten_Sptensor.h"
#include "Genten_Test_Utils.h"

using namespace Genten::Test;


static void  evaluateResult (const int             infolevel,
                             const ttb_indx        itersCompleted,
                             const ttb_real        stopTol,
                             const Genten::Ktensor &  result)
{
  std::stringstream  sMsg;
  sMsg << "CpAls finished after " << itersCompleted << " iterations";
  MESSAGE(sMsg.str());

  if (infolevel == 1)
    print_ktensor(result, std::cout,"Factorization result in ktensor form");

  // Check the final weights, which can be in any order.
  ttb_real  wght0 = result.weights(0);
  ttb_real  wght1 = result.weights(1);
  if (wght0 >= wght1)
  {
    ttb_real  diffA = fabs(wght0 - 2.828427);
    ttb_real  diffB = fabs(wght1 - 2.0);
    ASSERT( (diffA <= 1.0e-4) && (diffB <= 1.e-04),
            "Result ktensor weights match" );
  }
  else
  {
    ttb_real  diffA = fabs(wght0 - 2.0);
    ttb_real  diffB = fabs(wght1 - 2.8284);
    ASSERT( (diffA <= 1.0e-4) && (diffB <= 1.0e-4),
            "Result ktensor weights match" );
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
 *    lambda = [2.8284 2.0]
 *    A = [1.0    0.7071 ; 0.0    0.7071]
 *    B = [0.7071 0.7071 ; 0.0    0.7071 ; 0.7071 0.0   ]
 *    C = [0.7071 0.7071 ; 0.7071 0.0    ; 0.0    0.0    ; 0.0    0.7071]
 *  Random start point can converge to a different (worse) solution, so the
 *  test uses a particular start point:
 *    A0 = [0.8 0.5 ; 0.2 0.5]
 *    B0 = [0.5 0.5 ; 0.1 0.5 ; 0.5 0.1]
 *    C0 = [0.7 0.7 ; 0.7 0.1 ; 0.1 0.1 ; 0.1 0.7]
 */
void Genten_Test_CpAls (int infolevel)
{
  SETUP_DISABLE_CERR;

  initialize("Test of Genten::CpAls", infolevel);

  MESSAGE("Creating a sparse tensor with data to model");
  Genten::IndxArray  dims(3);
  dims[0] = 2;  dims[1] = 3;  dims[2] = 4;
  Genten::Sptensor  X(dims,11);
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
  if (infolevel == 1)
    print_ktensor(initialBasis,std::cout,"Initial guess for CpAls");

  // Factorize.
  ttb_real  stopTol = 1.0e-6;
  ttb_indx  maxIters = 100;
  Genten::Ktensor  result;
  ttb_indx  itersCompleted;
  ttb_real  resNorm;
  try
  {
    // Request performance information on every 3rd iteration.
    // Allocation adds two more for start and stop states of the algorithm.
    ttb_indx  nMaxPerfSize = 2 + (maxIters / 3);
    Genten::CpAlsPerfInfo *  perfInfo = new Genten::CpAlsPerfInfo[nMaxPerfSize];
    result = initialBasis;
    // Test with weights different from one.
    result.weights(0) = 2.0;
    Genten::cpals_core <Genten::Sptensor> (X, result,
                                           stopTol, maxIters, -1.0, infolevel,
                                           itersCompleted, resNorm,
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

  evaluateResult(infolevel, itersCompleted, stopTol, result);

  // Test factorization from a bad initial guess.
  MESSAGE("Creating a ktensor with initial guess all zero");
  Genten::Ktensor  initialZero (nNumComponents, dims.size(), dims);
  initialZero.setWeights(0.0);
  initialZero.setMatrices(0.0);
  MESSAGE("Checking if linear solver detects singular guess");
  DISABLE_CERR;
  try
  {
    result = initialZero;
    Genten::cpals_core <Genten::Sptensor> (X, result,
                                           stopTol, maxIters, -1.0, 0,
                                           itersCompleted, resNorm,
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

  finalize();
  return;
}
