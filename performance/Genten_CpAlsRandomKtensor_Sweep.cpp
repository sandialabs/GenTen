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
#include "Genten_AlgParams.hpp"

#include "Genten_MixedFormatOps.hpp"

using namespace std;

template <typename Space>
int run_cpals(const Genten::IndxArray& cFacDims_host,
              ttb_indx  nNumComponentsMin,
              ttb_indx  nNumComponentsMax,
              ttb_indx  nNumComponentsStep,
              ttb_indx  nMaxNonzeroes,
              Genten::AlgParams& algParams)
{
  typedef Genten::SptensorT<Space> Sptensor_type;
  typedef Genten::SptensorT<Genten::DefaultHostExecutionSpace> Sptensor_host_type;
  typedef Genten::KtensorT<Space> Ktensor_type;
  typedef Genten::KtensorT<Genten::DefaultHostExecutionSpace> Ktensor_host_type;

  Genten::IndxArrayT<Space> cFacDims =
    create_mirror_view( Space(), cFacDims_host );
  deep_copy( cFacDims, cFacDims_host );

  cout << "Will construct a random Ktensor/Sptensor pair:\n";
  cout << "  Ndims = " << cFacDims_host.size() << ",  Size = [ ";
  for (ttb_indx  n = 0; n < cFacDims_host.size(); n++)
    cout << cFacDims_host[n] << ' ';
  cout << "]\n";
  cout << "  Ncomps = [" << nNumComponentsMin << ":" << nNumComponentsStep << ":" << nNumComponentsMax << "]\n";
  cout << "  Maximum nnz = " << nMaxNonzeroes << "\n";

  // Construct a random number generator that matches Matlab.
  Genten::RandomMT  cRNG (algParams.seed);

  // Generate a random Ktensor, and from it a representative sparse
  // data tensor.
  Sptensor_host_type  cData_host;
  Ktensor_host_type   cSol_host;
  Genten::FacTestSetGenerator  cTestGen;

  Genten::SystemTimer  timer(2);
  timer.start(0);
  if (cTestGen.genSpFromRndKtensor (cFacDims_host, nNumComponentsMax,
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
  Ktensor_host_type  cInitialGuess_host (nNumComponentsMax, cFacDims.size(),
                                         cFacDims_host);
  cInitialGuess_host.setWeights (1.0);
  cInitialGuess_host.setMatrices (0.0);
  for (ttb_indx  n = 0; n < cFacDims.size(); n++)
  {
    for (ttb_indx  c = 0; c < nNumComponentsMax; c++)
    {
      for (ttb_indx  i = 0; i < cFacDims_host[n]; i++)
      {
        cInitialGuess_host[n].entry(i,c) = cRNG.genMatlabMT();
      }
    }
  }
  Ktensor_type cInitialGuess = create_mirror_view( Space(), cInitialGuess_host );
  deep_copy( cInitialGuess, cInitialGuess_host );

  // Compute default MTTKRP method if that is what was chosen
  if (algParams.mttkrp_method == Genten::MTTKRP_Method::Default)
    algParams.mttkrp_method = Genten::MTTKRP_Method::computeDefault<Space>();

  // Do a pass through the mttkrp to warm up and make sure the tensor
  // is copied to the device before generating any timings.  Use
  // Sptensor mttkrp and do this before createPermutation() so that
  // createPermutation() timings are not polluted by UVM transfers
  Ktensor_type  tmp (nNumComponentsMax, cFacDims.size(), cFacDims);
  Genten::SptensorT<Genten::DefaultExecutionSpace>& cData_tmp = cData;
  Genten::AlgParams ap = algParams;
    ap.mttkrp_method = Genten::MTTKRP_Method::Atomic;
  for (ttb_indx  n = 0; n < cFacDims.size(); n++)
    Genten::mttkrp(cData_tmp, cInitialGuess, n, tmp[n], ap);

  // Perform any post-processing (e.g., permutation and row ptr generation)
  timer.start(1);
  if (algParams.mttkrp_method == Genten::MTTKRP_Method::Perm)
    cData.createPermutation();
  timer.stop(1);
  printf ("  (createPermutation() took %6.3f seconds)\n", timer.getTotalTime(1));

  // Call CpAls to factorize, timing the performance.
  cout << "Calling CpAls with random initial guess and parameters:\n";
  cout << "  Max iters = " << algParams.maxiters << "\n";
  cout << "  Stop tol  = " << algParams.tol << "\n";

  printf("\t R \tMTTKRP GFLOP/s\n");
  printf("\t===\t==============\n");
  for (ttb_indx R=nNumComponentsMin; R<=nNumComponentsMax; R+=nNumComponentsStep)
  {
    Ktensor_host_type  cResult_host (R, cFacDims.size(), cFacDims_host);
    for (ttb_indx  n = 0; n < cFacDims.size(); n++)
    {
      for (ttb_indx  c = 0; c < R; c++)
      {
        for (ttb_indx  i = 0; i < cFacDims_host[n]; i++)
        {
          cResult_host[n].entry(i,c) = cInitialGuess_host[n].entry(i,c);
        }
      }
    }
    Ktensor_type cResult = create_mirror_view( Space(), cResult_host );
    deep_copy( cResult, cResult_host );

    ttb_indx  nItersCompleted;
    ttb_real  dResNorm;
    ttb_indx  nMaxPerfSize = 2 + algParams.maxiters;
    Genten::CpAlsPerfInfo * perfInfo = new Genten::CpAlsPerfInfo[nMaxPerfSize];
    algParams.rank = R;
    Genten::cpals_core (cData, cResult, algParams, nItersCompleted, dResNorm,
                        1, perfInfo);
    ttb_indx last_perf =
      nItersCompleted > algParams.maxiters ? algParams.maxiters+1 : nItersCompleted+1;
    ttb_real mttkrp_gflops = perfInfo[last_perf].dmttkrp_gflops;
    printf("\t%3d\t    %.3f\n", int(R), mttkrp_gflops);
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
  std::cout << "  --mttkrp_method <method> MTTKRP algorithm: ";
  for (unsigned i=0; i<Genten::MTTKRP_Method::num_types; ++i) {
    std::cout << Genten::MTTKRP_Method::names[i];
    if (i != Genten::MTTKRP_Method::num_types-1)
      std::cout << ", ";
  }
  std::cout << std::endl;
  std::cout << "  --mttkrp_tile_size <int> tile size for mttkrp algorithm" << std::endl;
  std::cout << "  --vtune              connect to vtune for Intel-based profiling (assumes vtune profiling tool, amplxe-cl, is in your path)" << std::endl;
}

//! Main routine for the executable.
int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  int ret = 0;

  try {

    ttb_bool help = Genten::parse_ttb_bool(argc, argv, "--help", "--no-help", false);
    if (help) {
      usage(argv);
      Kokkos::finalize();
      return 0;
    }

    ttb_bool vtune = Genten::parse_ttb_bool(argc, argv, "--vtune", "--no-vtune", false);
    if (vtune)
      Genten::connect_vtune();

    // Choose parameters: ndims, dim sizes, ncomps.
    // Values below match those in matlab_CpAlsRandomKTensor.m,
    // solves in just a few seconds.
    Genten::IndxArray  cFacDims = { 3000, 4000, 5000 };
    cFacDims =
      Genten::parse_ttb_indx_array(argc, argv, "--dims", cFacDims, 1, INT_MAX);
    ttb_indx  nNumComponentsMin =
      Genten::parse_ttb_indx(argc, argv, "--nc_min", 32, 1, INT_MAX);
    ttb_indx  nNumComponentsMax =
      Genten::parse_ttb_indx(argc, argv, "--nc_max", 64, 1, INT_MAX);
    ttb_indx  nNumComponentsStep =
      Genten::parse_ttb_indx(argc, argv, "--nc_step", 8, 1, INT_MAX);
    ttb_indx  nMaxNonzeroes =
      Genten::parse_ttb_indx(argc, argv, "--nnz", 1 * 1000 * 1000, 1, INT_MAX);
    unsigned long  nRNGseed =
      Genten::parse_ttb_indx(argc, argv, "--seed", 1, 0, INT_MAX);
    ttb_indx  nMaxIters =
      Genten::parse_ttb_indx(argc, argv, "--maxiters", 100, 1, INT_MAX);
    ttb_real  dStopTol =
      Genten::parse_ttb_real(argc, argv, "--tol", 1.0e-7, 0.0, 1.0);
    Genten::MTTKRP_Method::type mttkrp_method =
      Genten::parse_ttb_enum(argc, argv, "--mttkrp_method",
                     Genten::MTTKRP_Method::default_type,
                     Genten::MTTKRP_Method::num_types,
                     Genten::MTTKRP_Method::types,
                     Genten::MTTKRP_Method::names);
    ttb_indx mttkrp_tile_size =
      Genten::parse_ttb_indx(argc, argv, "--mttkrp_tile_size", 0, 0, INT_MAX);

    Genten::AlgParams algParams;
    algParams.seed = nRNGseed;
    algParams.maxiters = nMaxIters;
    algParams.maxsecs = -1.0;
    algParams.printitn = 0;
    algParams.tol = dStopTol;
    algParams.mttkrp_method = mttkrp_method;
    algParams.mttkrp_duplicated_factor_matrix_tile_size = mttkrp_tile_size;

    ret = run_cpals< Genten::DefaultExecutionSpace >(
      cFacDims, nNumComponentsMin, nNumComponentsMax, nNumComponentsStep,
      nMaxNonzeroes, algParams);

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
