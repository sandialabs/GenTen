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

#include "Genten_FacTestSetGenerator.hpp"
#include "Genten_IndxArray.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_AlgParams.hpp"
#include "Genten_MixedFormatOps.hpp"

template <typename Space>
void run_mttkrp(const std::string& inputfilename,
                const ttb_indx index_base,
                const bool gz,
                const Genten::IndxArray& cFacDims_rnd_host,
                const ttb_indx  nNumComponentsMin,
                const ttb_indx  nNumComponentsMax,
                const ttb_indx  nNumComponentsStep,
                const ttb_indx  nMaxNonzeroes,
                const unsigned long  nRNGseed,
                const ttb_indx  nIters,
                Genten::AlgParams& algParams)
{
  typedef Genten::SptensorT<Space> Sptensor_type;
  typedef Genten::SptensorT<Genten::DefaultHostExecutionSpace> Sptensor_host_type;
  typedef Genten::KtensorT<Space> Ktensor_type;
  typedef Genten::KtensorT<Genten::DefaultHostExecutionSpace> Ktensor_host_type;

  // Construct a random number generator that matches Matlab.
  Genten::RandomMT cRNG(nRNGseed);

  std::cout << "Genten sparse MTTKRP running on "
            << Genten::SpaceProperties<Space>::verbose_name()
            << std::endl;

  Sptensor_host_type cData_host;
  Sptensor_type cData;
  Genten::IndxArray cFacDims_host;
  Genten::IndxArrayT<Space> cFacDims;
  ttb_indx nDims = 0;
  if (inputfilename != "") {
    // Read tensor from file
    std::string fname(inputfilename);
    Genten::SystemTimer read_timer(1);
    read_timer.start(0);
    Genten::import_sptensor(fname, cData_host, index_base, gz, true);
    cData = create_mirror_view( Space(), cData_host );
    deep_copy( cData, cData_host );
    read_timer.stop(0);
    printf("Data import took %6.3f seconds\n", read_timer.getTotalTime(0));
    cFacDims_host = cData_host.size();
    cFacDims = cData.size();
    nDims = cData_host.ndims();
  }
  else {
    // Generate random tensor
    cFacDims_host = cFacDims_rnd_host;
    cFacDims = create_mirror_view( Space(), cFacDims_host );
    deep_copy( cFacDims, cFacDims_host );
    nDims = cFacDims_host.size();

    std::cout << "Will construct a random Ktensor/Sptensor pair:\n";
    std::cout << "  Ndims = " << nDims << ",  Size = [ ";
    for (ttb_indx n=0; n<nDims; ++n)
      std::cout << cFacDims_host[n] << ' ';
    std::cout << "]\n";
    std::cout << "  Ncomps = [" << nNumComponentsMin << ":" << nNumComponentsStep << ":" << nNumComponentsMax << "]\n";
    std::cout << "  Maximum nnz = " << nMaxNonzeroes << "\n";

    // Generate a random Ktensor, and from it a representative sparse
    // data tensor.
    Ktensor_host_type cSol_host;
    Genten::FacTestSetGenerator cTestGen;

    // Fixup algorithmic choices
    algParams.fixup<Space>(std::cout);

    Genten::SystemTimer gen_timer(1);
    gen_timer.start(0);
    if (cTestGen.genSpFromRndKtensor(cFacDims_host, nNumComponentsMax,
                                     nMaxNonzeroes,
                                     cRNG, cData_host, cSol_host) == false)
    {
      throw "*** Call to genSpFromRndKtensor failed.\n";
    }
    cData = create_mirror_view( Space(), cData_host );
    deep_copy( cData, cData_host );
    gen_timer.stop(0);
    std::printf("  (data generation took %6.3f seconds)\n",
                gen_timer.getTotalTime(0));
    std::cout << "  Actual nnz  = " << cData_host.nnz() << "\n";
  }
  if (algParams.mttkrp_method == Genten::MTTKRP_Method::Perm)
    cData.createPermutation();

  // Set a random input Ktensor, matching the Matlab code.
  Ktensor_host_type  cInput_host(nNumComponentsMax, nDims, cFacDims_host);
  cInput_host.setWeights(1.0);
  cInput_host.setMatrices(0.0);
  for (ttb_indx n=0; n<nDims; ++n)
  {
    for (ttb_indx c=0; c<nNumComponentsMax; ++c)
    {
      for (ttb_indx i=0; i<cFacDims_host[n]; ++i)
      {
        cInput_host[n].entry(i,c) = cRNG.genMatlabMT();
      }
    }
  }

  // Perform nIters iterations of MTTKRP on each mode, timing performance
  // We do each mode sequentially as this is more representative of CpALS
  // (as opposed to running all nIters iterations on each mode before moving
  // to the next one).
  std::cout << "Performing " << nIters << " iterations of MTTKRP" << std::endl;
  std::cout << "\t R \tMTTKRP GFLOP/s" << std::endl;
  std::cout << "\t===\t==============" << std::endl;
  for (ttb_indx R=nNumComponentsMin; R<=nNumComponentsMax; R+=nNumComponentsStep)
  {
    Ktensor_host_type  cInput_host2(R, nDims, cFacDims_host);
    for (ttb_indx  n = 0; n < nDims; n++)
    {
      for (ttb_indx  c = 0; c < R; c++)
      {
        for (ttb_indx  i = 0; i < cFacDims_host[n]; i++)
        {
          cInput_host2[n].entry(i,c) = cInput_host[n].entry(i,c);
        }
      }
    }
    Ktensor_type cInput2 = create_mirror_view( Space(), cInput_host2 );
    deep_copy( cInput2, cInput_host2 );
    Ktensor_type cResult(R, nDims, cFacDims);

    Genten::SystemTimer timer(1);
    timer.start(0);
    for (ttb_indx iter=0; iter<nIters; ++iter) {
      for (ttb_indx n=0; n<nDims; ++n) {
        Genten::mttkrp(cData, cInput2, n, cResult[n], algParams);
      }
    }
    timer.stop(0);
    const double atomic = 1.0; // cost of atomic measured in flops
    const double mttkrp_flops =
      cData.nnz()*R*(nDims+atomic)*nIters*nDims;
    const double mttkrp_total_time = timer.getTotalTime(0);
    const double mttkrp_total_throughput =
      ( mttkrp_flops / mttkrp_total_time ) / (1024.0 * 1024.0 * 1024.0);
    std::printf("\t%3d\t    %.3f\n", int(R), mttkrp_total_throughput);
  }
}

void usage(char **argv)
{
  std::cout << "Usage: "<< argv[0]<<" [options]" << std::endl;
  std::cout << "options: " << std::endl;
  std::cout << "  --exec-space <space> execution space to run on: ";
  for (unsigned i=0; i<Genten::Execution_Space::num_types; ++i) {
    std::cout << Genten::Execution_Space::names[i];
    if (i != Genten::Execution_Space::num_types-1)
      std::cout << ", ";
  }
  std::cout << std::endl;
  std::cout << "  --input <string>     path to input sptensor data" << std::endl;
  std::cout << "  --index-base <int>   starting index for tensor nonzeros" << std::endl;
  std::cout << "  --gz                 read tensor in gzip compressed format" << std::endl;
  std::cout << "  --dims <[n1,n2,...]> random tensor dimensions" << std::endl;
  std::cout << "  --nnz <int>          maximum number of random tensor nonzeros" << std::endl;
  std::cout << "  --nc-min <int>       minumum number of factor components" << std::endl;
  std::cout << "  --nc-max <int>       maximum number of factor components" << std::endl;
  std::cout << "  --nc-step <int>      step size in number of factor components" << std::endl;
  std::cout << "  --iters <int>        number of iterations to perform" << std::endl;
  std::cout << "  --seed <int>         seed for random number generator used in initial guess" << std::endl;
  std::cout << "  --check <0/1>        check the result for correctness" << std::endl;
  std::cout << "  --warmup <0/1>       do an MTTKRP to warm up first" << std::endl;
  std::cout << "  --mttkrp-method <method> MTTKRP algorithm: ";
  for (unsigned i=0; i<Genten::MTTKRP_Method::num_types; ++i) {
    std::cout << Genten::MTTKRP_Method::names[i];
    if (i != Genten::MTTKRP_Method::num_types-1)
      std::cout << ", ";
  }
  std::cout << std::endl;
  std::cout << "  --mttkrp-tile-size <int> tile size for mttkrp algorithm" << std::endl;
  std::cout << "  --vtune              connect to vtune for Intel-based profiling (assumes vtune profiling tool, amplxe-cl, is in your path)" << std::endl;
}

//! Main routine for the executable.
/*!
 *  The test constructs a random Ktensor, derives a sparse data tensor,
 *  and calls MTTKRP.  Parameters allow different
 *  data sizes with the intent of understanding MTTKRP performance issues.
 */
int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  int ret = 0;

  try {

    // Convert argc,argv to list of arguments
    auto args = Genten::build_arg_list(argc,argv);

    ttb_bool help = Genten::parse_ttb_bool(args, "--help", "--no-help", false);
    if (help) {
      usage(argv);
      Kokkos::finalize();
      return 0;
    }

    ttb_bool vtune =
      Genten::parse_ttb_bool(args, "--vtune", "--no-vtune", false);
    if (vtune)
      Genten::connect_vtune();

    // Choose parameters: ndims, dim sizes, ncomps.
     Genten::Execution_Space::type exec_space =
      parse_ttb_enum(args, "--exec-space",
                     Genten::Execution_Space::default_type,
                     Genten::Execution_Space::num_types,
                     Genten::Execution_Space::types,
                     Genten::Execution_Space::names);
    std::string inputfilename =
      Genten::parse_string(args,"--input","");
    ttb_indx index_base =
      Genten::parse_ttb_indx(args, "--index_base", 0, 0, INT_MAX);
    ttb_bool gz =
      Genten::parse_ttb_bool(args, "--gz", "--no-gz", false);
    Genten::IndxArray  cFacDims = { 3000, 4000, 5000 };
    cFacDims =
      Genten::parse_ttb_indx_array(args, "--dims", cFacDims, 1, INT_MAX);
    ttb_indx  nNumComponentsMin =
      Genten::parse_ttb_indx(args, "--nc-min", 32, 1, INT_MAX);
    ttb_indx  nNumComponentsMax =
      Genten::parse_ttb_indx(args, "--nc-max", 64, 1, INT_MAX);
    ttb_indx  nNumComponentsStep =
      Genten::parse_ttb_indx(args, "--nc-step", 8, 1, INT_MAX);
    ttb_indx  nMaxNonzeroes =
      Genten::parse_ttb_indx(args, "--nnz", 1 * 1000 * 1000, 1, INT_MAX);
    unsigned long  nRNGseed =
      Genten::parse_ttb_indx(args, "--seed", 1, 0, INT_MAX);
    ttb_indx  nIters =
      Genten::parse_ttb_indx(args, "--iters", 10, 1, INT_MAX);
    Genten::MTTKRP_Method::type mttkrp_method =
      Genten::parse_ttb_enum(args, "--mttkrp-method",
                     Genten::MTTKRP_Method::default_type,
                     Genten::MTTKRP_Method::num_types,
                     Genten::MTTKRP_Method::types,
                     Genten::MTTKRP_Method::names);
    ttb_indx mttkrp_tile_size =
      Genten::parse_ttb_indx(args, "--mttkrp-tile-size", 0, 0, INT_MAX);

    // Check for unrecognized arguments
    if (Genten::check_and_print_unused_args(args, std::cout)) {
      usage(argv);
      // Use throw instead of exit for proper Kokkos shutdown
      throw std::string("Invalid command line arguments.");
    }

    Genten::AlgParams algParams;
    algParams.mttkrp_method = mttkrp_method;
    algParams.mttkrp_duplicated_factor_matrix_tile_size = mttkrp_tile_size;

    if (exec_space == Genten::Execution_Space::Default)
      run_mttkrp< Genten::DefaultExecutionSpace >(
        inputfilename, index_base, gz,
        cFacDims, nNumComponentsMin, nNumComponentsMax, nNumComponentsStep,
        nMaxNonzeroes, nRNGseed, nIters, algParams);
#ifdef HAVE_CUDA
    else if (exec_space == Genten::Execution_Space::Cuda)
      run_mttkrp< Kokkos::Cuda >(
        inputfilename, index_base, gz,
        cFacDims, nNumComponentsMin, nNumComponentsMax, nNumComponentsStep,
        nMaxNonzeroes, nRNGseed, nIters, algParams);
#endif
#ifdef HAVE_HIP
    else if (exec_space == Genten::Execution_Space::HIP)
      run_mttkrp< Kokkos::Experimental::HIP >(
        inputfilename, index_base, gz,
        cFacDims, nNumComponentsMin, nNumComponentsMax, nNumComponentsStep,
        nMaxNonzeroes, nRNGseed, nIters, algParams);
#endif
#ifdef HAVE_SYCL
    else if (exec_space == Genten::Execution_Space::SYCL)
      run_mttkrp< Kokkos::Experimental::SYCL >(
        inputfilename, index_base, gz,
        cFacDims, nNumComponentsMin, nNumComponentsMax, nNumComponentsStep,
        nMaxNonzeroes, nRNGseed, nIters, algParams);
#endif
#ifdef HAVE_OPENMP
    else if (exec_space == Genten::Execution_Space::OpenMP)
      run_mttkrp< Kokkos::OpenMP >(
        inputfilename, index_base, gz,
        cFacDims, nNumComponentsMin, nNumComponentsMax, nNumComponentsStep,
        nMaxNonzeroes, nRNGseed, nIters, algParams);

#endif
#ifdef HAVE_THREADS
    else if (exec_space == Genten::Execution_Space::Threads)
      run_mttkrp< Kokkos::Threads >(
        inputfilename, index_base, gz,
        cFacDims, nNumComponentsMin, nNumComponentsMax, nNumComponentsStep,
        nMaxNonzeroes, nRNGseed, nIters, algParams);

#endif
#ifdef HAVE_SERIAL
    else if (exec_space == Genten::Execution_Space::Serial)
      run_mttkrp< Kokkos::Serial >(
        inputfilename, index_base, gz,
        cFacDims, nNumComponentsMin, nNumComponentsMax, nNumComponentsStep,
        nMaxNonzeroes, nRNGseed, nIters, algParams);
#endif
  }
  catch(std::string sExc)
  {
    std::cout << "*** Call to mttkrp threw an exception:\n";
    std::cout << "  " << sExc << "\n";
    ret = -1;
  }
  catch(...)
  {
    std::cout << "*** Call to mttkrp threw an exception:\n";
    ret = -1;
  }

  Kokkos::finalize();
  return ret;
}
