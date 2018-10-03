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


/*!
  @file Genten_CpAlsRandomKtensor.cpp
  @brief Main program that factorizes synthetic data using the CP-ALS algorithm.
*/

#include <iostream>
#include <stdio.h>

#include "Genten_CpAls.hpp"
#include "Genten_FacTestSetGenerator.hpp"
#include "Genten_IndxArray.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_Driver_Utils.hpp"

#include "Genten_MixedFormatOps.hpp"

using namespace std;

enum SPTENSOR_TYPE {
  SPTENSOR,
  SPTENSOR_PERM
};
const unsigned num_sptensor_types = 2;
SPTENSOR_TYPE sptensor_types[] =
{ SPTENSOR, SPTENSOR_PERM };
std::string sptensor_names[] =
{ "kokkos", "perm" };

template <template<class> class Sptensor_template, typename Space>
int run_cpals(const Genten::IndxArray& cFacDims_host,
              ttb_indx  nNumComponents,
              ttb_indx  nMaxNonzeroes,
              unsigned long  nRNGseed,
              ttb_indx  nMaxIters,
              ttb_real  dStopTol,
              SPTENSOR_TYPE tensor_type)
{
  typedef Sptensor_template<Space> Sptensor_type;
  typedef Sptensor_template<Genten::DefaultHostExecutionSpace> Sptensor_host_type;
  typedef Genten::KtensorT<Space> Ktensor_type;
  typedef Genten::KtensorT<Genten::DefaultHostExecutionSpace> Ktensor_host_type;

  Genten::IndxArrayT<Space> cFacDims =
    create_mirror_view( Space(), cFacDims_host );
  deep_copy( cFacDims, cFacDims_host );

  cout << "Will construct a random Ktensor/Sptensor_";
  cout << sptensor_names[tensor_type] << " pair:\n";
  cout << "  Ndims = " << cFacDims_host.size() << ",  Size = [ ";
  for (ttb_indx  n = 0; n < cFacDims_host.size(); n++)
    cout << cFacDims_host[n] << ' ';
  cout << "]\n";
  cout << "  Ncomps = " << nNumComponents << "\n";
  cout << "  Maximum nnz = " << nMaxNonzeroes << "\n";

  // Construct a random number generator that matches Matlab.
  Genten::RandomMT  cRNG (nRNGseed);

  // Generate a random Ktensor, and from it a representative sparse
  // data tensor.
  Sptensor_host_type  cData_host;
  Ktensor_host_type   cSol_host;
  Genten::FacTestSetGenerator  cTestGen;

  Genten::SystemTimer  timer(2);
  timer.start(0);
  if (cTestGen.genSpFromRndKtensor (cFacDims_host, nNumComponents,
                                    nMaxNonzeroes,
                                    cRNG, cData_host, cSol_host) == false)
  {
    cout << "*** Call to genSpFromRndKtensor failed.\n";
    return( -1 );
  }
  timer.stop(0);
  printf ("  (data generation took %6.3f seconds)\n", timer.getTotalTime(0));
  cout << "  Actual nnz  = " << cData_host.nnz() << "\n";

  Sptensor_type cData = create_mirror_view( Space(), cData_host );
  Ktensor_type cSol = create_mirror_view( Space(), cSol_host );
  deep_copy( cData, cData_host );
  deep_copy( cSol, cSol_host );

  // Set a random initial guess, matching the Matlab code.
  Ktensor_host_type  cInitialGuess_host (nNumComponents, cFacDims.size(),
                                         cFacDims_host);
  cInitialGuess_host.setWeights (1.0);
  cInitialGuess_host.setMatrices (0.0);
  for (ttb_indx  n = 0; n < cFacDims_host.size(); n++)
  {
    for (ttb_indx  c = 0; c < nNumComponents; c++)
    {
      for (ttb_indx  i = 0; i < cFacDims_host[n]; i++)
      {
        cInitialGuess_host[n].entry(i,c) = cRNG.genMatlabMT();
      }
    }
  }
  Ktensor_type cInitialGuess = create_mirror_view( Space(), cInitialGuess_host );
  deep_copy( cInitialGuess, cInitialGuess_host );

  // Do a pass through the mttkrp to warm up and make sure the tensor
  // is copied to the device before generating any timings.  Use
  // Sptensor mttkrp and do this before fillComplete() so that
  // fillComplete() timings are not polluted by UVM transfers
  Ktensor_type  tmp (nNumComponents, cFacDims.size(), cFacDims);
  Genten::SptensorT<Genten::DefaultExecutionSpace>& cData_tmp = cData;
  for (ttb_indx  n = 0; n < cFacDims.size(); n++)
    Genten::mttkrp(cData_tmp, cInitialGuess, n, tmp[n]);

  // Perform any post-processing (e.g., permutation and row ptr generation)
  timer.start(1);
  cData.fillComplete();
  timer.stop(1);
  printf ("  (fillComplete() took %6.3f seconds)\n", timer.getTotalTime(1));

  // Call CpAls to factorize, timing the performance.
  Ktensor_type  cResult;
  ttb_indx  nItersCompleted;
  ttb_real  dResNorm;

  cout << "Calling CpAls with random initial guess and parameters:\n";
  cout << "  Max iters = " << nMaxIters << "\n";
  cout << "  Stop tol  = " << dStopTol << "\n";

  // Request performance information on every iteration.
  // Allocation adds two more for start and stop states of the algorithm.
  ttb_indx  nMaxPerfSize = 2 + nMaxIters;
  Genten::CpAlsPerfInfo *  perfInfo = new Genten::CpAlsPerfInfo[nMaxPerfSize];
  cResult = cInitialGuess;
  Genten::cpals_core (cData, cResult,
                      dStopTol, nMaxIters, -1.0, 1,
                      nItersCompleted, dResNorm,
                      1, perfInfo);
  printf ("Performance information per iteration:\n");
  for (ttb_indx  i = 0; i <= nItersCompleted; i++)
  {
    printf (" %2d: fit = %.6e, resnorm = %.2e, time = %.3f secs\n",
            perfInfo[i].nIter, perfInfo[i].dFit,
            perfInfo[i].dResNorm, perfInfo[i].dCumTime);
  }
  delete[] perfInfo;

  printf ("  Final residual norm = %10.3e\n", dResNorm);
  printf ("  Weights (lambda):\n");
  auto weights_host = create_mirror_view(cResult.weights());
  deep_copy(weights_host, cResult.weights());
  for (ttb_indx  c = 0; c < nNumComponents; c++)
    printf("    [%d] = %f\n", (int)c, weights_host[c]);

  //    print_ktensor(cResult, std::cout, "TBD computed solution");

  // There is no attempt to verify the answer.

  return 0;
}

void usage(char **argv)
{
  std::cout << "Usage: "<< argv[0]<<" [options]" << std::endl;
  std::cout << "options: " << std::endl;
  std::cout << "  --dims <[n1,n2,...]> tensor dimensions" << std::endl;
  std::cout << "  --nc <int>           number of factor components" << std::endl;
  std::cout << "  --nnz <int>          maximum number of tensor nonzeros" << std::endl;
  std::cout << "  --maxiters <int>     maximum iterations to perform" << std::endl;
  std::cout << "  --tol <float>        stopping tolerance" << std::endl;
  std::cout << "  --seed <int>         seed for random number generator used in initial guess" << std::endl;
  std::cout << "  --tensor <type>      Sptensor format: ";
  for (unsigned i=0; i<num_sptensor_types; ++i) {
    std::cout << sptensor_names[i];
    if (i != num_sptensor_types-1)
      std::cout << ", ";
  }
  std::cout << std::endl;
  std::cout << "  --vtune              connect to vtune for Intel-based profiling (assumes vtune profiling tool, amplxe-cl, is in your path)" << std::endl;
}

//! Main routine for the executable.
/*!
 *  The test constructs a random Ktensor, derives a sparse data tensor,
 *  and calls CP-ALS to compute a factorization.  Parameters allow different
 *  data sizes with the intent of understanding CP-ALS performance issues.
 *  The same problem can be solved in Matlab for comparison.
 *
 *  On matador (2013 quad core workstation) C++ code is 2-3 times faster than
 *  Matlab on problems that take 15-30secs (in C++):
 *    3000 x 4000 x 5000, R=2, 1M nnz, tol 1.0e-9
 *    300 x 400 x 500, R=50, 1M nnz, maxit 15
 *    500 x 25000 x 500 x 10, R=10, 5M nnz, tol 1.0e-7
 *  Experiments on matador with very large problems:
 *    1k x  10k x 1k x 1k, R=1000, 10M nnz:  C++ 15m for iter #1, Matlab 60m
 *   10k x 100k x 1k x 1k, R=1000, 10M nnz:  C++ killed after 60m on iter #1
 *   10k x 1M x 10k, R=1000, 10M nnz:  both killed, swapping badly on iter #1
 */
int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  int ret = 0;

  try {

    ttb_bool help = parse_ttb_bool(argc, argv, "--help", false);
    if (help) {
      usage(argv);
      Kokkos::finalize();
      return 0;
    }

    ttb_bool vtune = parse_ttb_bool(argc, argv, "--vtune", false);
    if (vtune)
      Genten::connect_vtune();

    // Choose parameters: ndims, dim sizes, ncomps.
    // Values below match those in matlab_CpAlsRandomKTensor.m,
    // solves in just a few seconds.
    Genten::IndxArray  cFacDims = { 3000, 4000, 5000 };
    cFacDims =
      parse_ttb_indx_array(argc, argv, "--dims", cFacDims, 1, INT_MAX);
    ttb_indx  nNumComponents =
      parse_ttb_indx(argc, argv, "--nc", 32, 1, INT_MAX);
    ttb_indx  nMaxNonzeroes =
      parse_ttb_indx(argc, argv, "--nnz", 1 * 1000 * 1000, 1, INT_MAX);
    unsigned long  nRNGseed =
      parse_ttb_indx(argc, argv, "--seed", 1, 0, INT_MAX);
    ttb_indx  nMaxIters =
      parse_ttb_indx(argc, argv, "--maxiters", 100, 1, INT_MAX);
    ttb_real  dStopTol =
      parse_ttb_real(argc, argv, "--tol", 1.0e-7, 0.0, 1.0);
    SPTENSOR_TYPE tensor_type =
      parse_ttb_enum(argc, argv, "--tensor", SPTENSOR,
                     num_sptensor_types, sptensor_types, sptensor_names);

    if (tensor_type == SPTENSOR)
      ret = run_cpals< Genten::SptensorT, Genten::DefaultExecutionSpace >(
        cFacDims, nNumComponents, nMaxNonzeroes, nRNGseed, nMaxIters, dStopTol,
        tensor_type);
    else if (tensor_type == SPTENSOR_PERM)
      ret = run_cpals< Genten::SptensorT_perm, Genten::DefaultExecutionSpace >(
        cFacDims, nNumComponents, nMaxNonzeroes, nRNGseed, nMaxIters, dStopTol,
        tensor_type);

  }
  catch(std::string sExc)
  {
    cout << "*** Call to cpals_core threw an exception:\n";
    cout << "  " << sExc << "\n";
    ret = 0;
  }

  Kokkos::finalize();
  return ret;
}
