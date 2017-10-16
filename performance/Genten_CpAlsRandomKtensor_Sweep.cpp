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
  SPTENSOR_PERM,
  SPTENSOR_ROW
};
const unsigned num_sptensor_types = 3;
SPTENSOR_TYPE sptensor_types[] =
{ SPTENSOR, SPTENSOR_PERM, SPTENSOR_ROW };
std::string sptensor_names[] =
{ "kokkos", "perm", "row" };

template <typename Sptensor_type>
int run_cpals(const Genten::IndxArray& cFacDims,
              ttb_indx  nNumComponentsMin,
              ttb_indx  nNumComponentsMax,
              ttb_indx  nNumComponentsStep,
              ttb_indx  nMaxNonzeroes,
              unsigned long  nRNGseed,
              ttb_indx  nMaxIters,
              ttb_real  dStopTol,
              SPTENSOR_TYPE tensor_type)
{
  cout << "Will construct a random Ktensor/Sptensor_";
  cout << sptensor_names[tensor_type] << " pair:\n";
  cout << "  Ndims = " << cFacDims.size() << ",  Size = [ ";
  for (ttb_indx  n = 0; n < cFacDims.size(); n++)
    cout << cFacDims[n] << ' ';
  cout << "]\n";
  cout << "  Ncomps = [" << nNumComponentsMin << ":" << nNumComponentsStep << ":" << nNumComponentsMax << "]\n";
  cout << "  Maximum nnz = " << nMaxNonzeroes << "\n";

  // Construct a random number generator that matches Matlab.
  Genten::RandomMT  cRNG (nRNGseed);

  // Generate a random Ktensor, and from it a representative sparse
  // data tensor.
  Sptensor_type  cData;
  Genten::Ktensor   cSol;
  Genten::FacTestSetGenerator  cTestGen;

  Genten::SystemTimer  timer(2);
  timer.start(0);
  if (cTestGen.genSpFromRndKtensor (cFacDims, nNumComponentsMax, nMaxNonzeroes,
                                    cRNG, cData, cSol) == false)
  {
    cout << "*** Call to genSpFromRndKtensor failed.\n";
    return( -1 );
  }
  timer.stop(0);
  printf ("  (data generation took %6.3f seconds)\n", timer.getTotalTime(0));
  cout << "  Actual nnz  = " << cData.nnz() << "\n";

  // Set a random initial guess, matching the Matlab code.
  Genten::Ktensor  cInitialGuess (nNumComponentsMax, cFacDims.size(), cFacDims);
  cInitialGuess.setWeights (1.0);
  cInitialGuess.setMatrices (0.0);
  for (ttb_indx  n = 0; n < cFacDims.size(); n++)
  {
    for (ttb_indx  c = 0; c < nNumComponentsMax; c++)
    {
      for (ttb_indx  i = 0; i < cFacDims[n]; i++)
      {
        cInitialGuess[n].entry(i,c) = cRNG.genMatlabMT();
      }
    }
  }

  // Do a pass through the mttkrp to warm up and make sure the tensor
  // is copied to the device before generating any timings.  Use
  // Sptensor mttkrp and do this before fillComplete() so that
  // fillComplete() timings are not polluted by UVM transfers
  Genten::Ktensor  tmp (nNumComponentsMax, cFacDims.size(), cFacDims);
  Genten::Sptensor& cData_tmp = cData;
  for (ttb_indx  n = 0; n < cFacDims.size(); n++)
    Genten::mttkrp(cData_tmp, cInitialGuess, n, tmp[n]);

  // Perform any post-processing (e.g., permutation and row ptr generation)
  timer.start(1);
  cData.fillComplete();
  timer.stop(1);
  printf ("  (fillComplete() took %6.3f seconds)\n", timer.getTotalTime(1));

  // Call CpAls to factorize, timing the performance.
  cout << "Calling CpAls with random initial guess and parameters:\n";
  cout << "  Max iters = " << nMaxIters << "\n";
  cout << "  Stop tol  = " << dStopTol << "\n";

  printf("\t R \tMTTKRP GFLOP/s\n");
  printf("\t===\t==============\n");
  for (ttb_indx R=nNumComponentsMin; R<=nNumComponentsMax; R+=nNumComponentsStep)
  {
    Genten::Ktensor cResult(R, cFacDims.size(), cFacDims);
    for (ttb_indx  n = 0; n < cFacDims.size(); n++)
    {
      for (ttb_indx  c = 0; c < R; c++)
      {
        for (ttb_indx  i = 0; i < cFacDims[n]; i++)
        {
          cResult[n].entry(i,c) = cInitialGuess[n].entry(i,c);
        }
      }
    }
    ttb_indx  nItersCompleted;
    ttb_real  dResNorm;
    ttb_indx  nMaxPerfSize = 2 + nMaxIters;
    Genten::CpAlsPerfInfo * perfInfo = new Genten::CpAlsPerfInfo[nMaxPerfSize];
    Genten::cpals_core (cData, cResult,
                        dStopTol, nMaxIters, -1.0, 0,
                        nItersCompleted, dResNorm,
                        1, perfInfo);
    ttb_real mttkrp_gflops = perfInfo[nMaxPerfSize-1].dmttkrp_gflops;
    printf("\t%3d\t    %.3f\n", R, mttkrp_gflops);
    delete[] perfInfo;
  }

  return 0;
}

void usage(char **argv)
{
  std::cout << "Usage: "<< argv[0]<<" [options]" << std::endl;
  std::cout << "options: " << std::endl;
  std::cout << "  --dims <[n1,n2,...]> tensor dimensions" << std::endl;
  std::cout << "  --nc_min <int>           minumum number of factor components" << std::endl;
  std::cout << "  --nc_max <int>           maximum number of factor components" << std::endl;
  std::cout << "  --nc_step <int>          step size in number of factor components" << std::endl;
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
    ttb_indx  nNumComponentsMin =
      parse_ttb_indx(argc, argv, "--nc_min", 32, 1, INT_MAX);
    ttb_indx  nNumComponentsMax =
      parse_ttb_indx(argc, argv, "--nc_max", 64, 1, INT_MAX);
    ttb_indx  nNumComponentsStep =
      parse_ttb_indx(argc, argv, "--nc_step", 8, 1, INT_MAX);
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
      ret = run_cpals<Genten::Sptensor>(
        cFacDims, nNumComponentsMin, nNumComponentsMax, nNumComponentsStep,
        nMaxNonzeroes, nRNGseed, nMaxIters, dStopTol,
        tensor_type);
    else if (tensor_type == SPTENSOR_PERM)
      ret = run_cpals<Genten::Sptensor_perm>(
        cFacDims, nNumComponentsMin, nNumComponentsMax, nNumComponentsStep,
        nMaxNonzeroes, nRNGseed, nMaxIters, dStopTol,
        tensor_type);
    else if (tensor_type == SPTENSOR_ROW)
      ret = run_cpals<Genten::Sptensor_row>(
        cFacDims, nNumComponentsMin, nNumComponentsMax, nNumComponentsStep,
        nMaxNonzeroes, nRNGseed, nMaxIters, dStopTol,
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
