//@HEADER
// ************************************************************************
//     Genten Tensor Toolbox
//     Software package for tensor math by Sandia National Laboratories
//
// Sandia National Laboratories is a multimission laboratory managed
// and operated by National Technology and Engineering Solutions of Sandia,
// LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
// U.S. Department of Energy’s National Nuclear Security Administration under
// contract DE-NA0003525.
//
// Copyright 20XX, Sandia Corporation.
// ************************************************************************
//@HEADER


/*!
  @file Genten_CpAlsAminoAcid.cpp
  @brief Main program that analyzes 3-way data using CP-ALS algorithm.
*/

#include <iostream>
#include <stdio.h>

#include "Genten_CpAls.h"
#include "Genten_IOtext.h"
#include "Genten_Ktensor.h"
#include "Genten_Sptensor.h"

#include "Kokkos_Core.hpp"

using namespace std;

static void  makeInitialGuess (const ttb_indx  nNumComponents,
                               const Genten::IndxArray &  dataSize,
                               Genten::Ktensor &  initialGuess)
{
  // Construct an initialize guess that is simple to produce, but does not
  // lead to linearly dependent factor matrices.  This example is a sort of
  // circulant matrix for each factor.  Matlab code is the following:
  //   a0 = [ 1:5   ; 2:5 , 1       ; 3:5 , 1:2 ]';
  //   b0 = [ 1:201 ; 50:201 , 1:49 ; 90:201 , 1:89 ]';
  //   c0 = [ 1:61  ; 15:61 , 1:14  ; 30:61 , 1:29 ]';
  //   X0 = {a0, b0, c0};

  // Set weights to one to match Matlab cp_als internal behavior.
  initialGuess.setWeights (1.0);

  initialGuess.setMatrices (0.0);
  for (ttb_indx  i = 0; i < dataSize[0]; i++)
  {
    initialGuess[0].entry(i,0) = 1.0 + (ttb_real) (i);
    initialGuess[0].entry(i,1) = 1.0 + (ttb_real) ((i + 1) % dataSize[0]);
    initialGuess[0].entry(i,2) = 1.0 + (ttb_real) ((i + 2) % dataSize[0]);
  }
  for (ttb_indx  i = 0; i < dataSize[1]; i++)
  {
    initialGuess[1].entry(i,0) = 1.0 + (ttb_real) (i);
    initialGuess[1].entry(i,1) = 1.0 + (ttb_real) ((i + 49) % dataSize[1]);
    initialGuess[1].entry(i,2) = 1.0 + (ttb_real) ((i + 89) % dataSize[1]);
  }
  for (ttb_indx  i = 0; i < dataSize[2]; i++)
  {
    initialGuess[2].entry(i,0) = 1.0 + (ttb_real) (i);
    initialGuess[2].entry(i,1) = 1.0 + (ttb_real) ((i + 14) % dataSize[2]);
    initialGuess[2].entry(i,2) = 1.0 + (ttb_real) ((i + 29) % dataSize[2]);
  }

  return;
}

template <typename KtensorT>
static void  evaluateResult (const ttb_indx  itersCompleted,
                             const KtensorT &  result)
{
  // Do a simple check on the weights to verify the solution.
  bool  bIsAnswerOK = true;
  if (fabs(result.weights(0) - 3.3588e+4) > 1.0)
    bIsAnswerOK = false;
  if (fabs(result.weights(1) - 2.3382e+4) > 1.0)
    bIsAnswerOK = false;
  if (fabs(result.weights(2) - 2.1188e+4) > 1.0)
    bIsAnswerOK = false;
  cout << "Quick check: does result match known solution? ";
  if (bIsAnswerOK)
    cout << "true\n";
  else
    cout << "false\n";

  // print_matrix(result[0],cout,"Mode[0] components of the solution");
  return;
}


//! Main routine for the executable.
/*!
 *  The test constructs a sparse data tensor from a file, calls CP-ALS
 *  to compute a factorization, and shows how to access various results.
 *  Data is from a well-known amino acid chemometrics problem in the
 *  tensor literature.
 *
 *  The test is then repeated using a dense data tensor.  The factorization
 *  is identical, but takes longer to execute.
 *
 *  The same problem can be solved in Matlab for comparison,
 *  see matlab_CpAlsAminoAcid.m.
 */
int main(int argc, char* argv[])
{

  //Genten::connect_vtune();

  Kokkos::initialize(argc, argv);
  {

    // Load a tensor with chemical flourescence data from a file.
    //   http://www.models.kvl.dk/Amino_Acit_fluo
    //   R. Bro, "Multi-way analysis in the food industry", 1998,
    //   PhD dissertation, University of Amsterdam & Royal Veterinary
    //   and Agricultural University.
    //
    // The data could be treated as dense because every tensor element is
    // defined, but this first test loads it as a sparse tensor.
    //
    // A Matlab version of the data file will increase the three indices in each
    // record by one (Matlab indexes from 1, Genten tensors index from 0).
    string  filename = "aminoacid_data.txt";
    Genten::Sptensor  data;
    try
    {
      Genten::import_sptensor ("./test_data/" + filename, data);
    }
    catch (string)
    {
      cout << "*** Exiting, failed to open ./test_data\n";
      return( -1 );
    }

    // Specify the number of components to factorize ("rank").
    // In the literature rank=3 captures all the essential features.
    ttb_indx  nNumComponents = 3;

    // Compute an initial guess.
    Genten::Ktensor  initialGuess (nNumComponents, data.ndims(), data.size());
    makeInitialGuess (nNumComponents, data.size(), initialGuess);

    ttb_real  stopTol = 1.0e-5;
    ttb_indx  maxIters = 100;
    Genten::Ktensor  result;
    ttb_indx  itersCompleted;
    ttb_real  resNorm;

    // Compute factors for the sparse data tensor.
    cout << "Factorize SPARSE amino acid tensor using CpAls\n";
    cout << "  Data tensor has size [";
    for (ttb_indx  i = 0; i < data.ndims() - 1; i++)
      cout << data.size(i) << ", ";
    cout << data.size(data.ndims() - 1) << "]\n";
    cout << "  " << data.nnz() << " structural nonzero elements\n";

    cout << "Calling CpAls to compute " << nNumComponents << " components\n";
    cout << "  Max iters = " << maxIters << "\n";
    cout << "  Stop tol  = " << stopTol << "\n";

    try
    {
      result = initialGuess;
      Genten::cpals_core (data, result,
                          stopTol, maxIters, -1.0, 10,
                          itersCompleted, resNorm,
                          0, NULL);
    }
    catch(std::string sExc)
    {
      // Should not happen.
      cout << "Call to cpals_core threw an exception:\n";
      cout << "  " << sExc << "\n";
      return( -1 );
    }
    evaluateResult (itersCompleted, result);

    //Genten::export_ktensor("result.txt", result);


    cout << endl;
  }

  Kokkos::finalize();

  return( 0 );
}
