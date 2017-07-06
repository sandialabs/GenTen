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

#include "Genten_IOtext.h"
#include "Genten_CpAls.h"
#include "Genten_Driver_Utils.h"
#include "Genten_SystemTimer.h"
#include "Genten_MixedFormatOps.h"

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
int run_cpals(const std::string& inputfilename,
              const std::string& outputfilename,
              const ttb_indx index_base,
              const bool gz,
              const ttb_indx rank,
              const unsigned long seed,
              const ttb_indx maxiters,
              const ttb_real tol,
              const ttb_indx printitn,
              const bool no_save,
              const bool debug,
              const bool warmup,
              const SPTENSOR_TYPE tensor_type)
{
  Genten::SystemTimer timer(3);

  // Read in tensor data
  std::string fname(inputfilename);
  timer.start(0);
  Sptensor_type x;
  Genten::import_sptensor(fname, x, index_base, gz, true);
  timer.stop(0);
  printf("Data import took %6.3f seconds\n", timer.getTotalTime(0));
  if (debug) Genten::print_sptensor(x, std::cout, fname);

  // Generate a random starting point
  // Matlab cp_als always sets the weights to one.
  Genten::Ktensor u(rank,x.ndims(),x.size());
  Genten::RandomMT cRMT(seed);
  u.setMatricesScatter(false,cRMT);
  u.setWeights(1.0);
  if (debug) Genten::print_ktensor(u, std::cout, "Initial guess");

  if (warmup)
  {
    // Do a pass through the mttkrp to warm up and make sure the tensor
    // is copied to the device before generating any timings.  Use
    // Sptensor mttkrp and do this before fillComplete() so that
    // fillComplete() timings are not polluted by UVM transfers
    Genten::Ktensor tmp (rank,x.ndims(),x.size());
    Genten::Sptensor& x_tmp = x;
    for (ttb_indx  n = 0; n < x.ndims(); n++)
      Genten::mttkrp(x_tmp, u, n, tmp[n]);
  }

  // Perform any post-processing (e.g., permutation and row ptr generation)
  timer.start(1);
  x.fillComplete();
  timer.stop(1);
  printf ("fillComplete() took %6.3f seconds\n", timer.getTotalTime(1));

  // Run CP-ALS
  ttb_indx iter;
  ttb_real resNorm;
  Genten::cpals_core (x, u, tol, maxiters, -1.0, printitn,
                      iter, resNorm, 0, NULL);

  // Save results to file
  if (!no_save)
  {
    timer.start(2);
    Genten::export_ktensor(outputfilename, u);
    timer.stop(2);
    printf("Data export took %6.3f seconds\n", timer.getTotalTime(2));
  }
  if (debug) Genten::print_ktensor(u, std::cout, "Solution");

  return 0;
}

// forward declarations
void usage(char **argv)
{
  std::cout << "Usage: "<< argv[0]<<" [options]" << std::endl;
  std::cout << "options: " << std::endl;
  std::cout << "  --output <string>  output file name" << std::endl;
  std::cout << "  --input <string>   path to input sptensor data" << std::endl;
  std::cout << "  --index_base <int> starting index for tensor nonzeros" << std::endl;
  std::cout << "  --gz               read tensor in gzip compressed format" << std::endl;
  std::cout << "  --rank <int>       rank of factorization to compute" << std::endl;
  std::cout << "  --maxiters <int>   maximum iterations to perform" << std::endl;
  std::cout << "  --printitn <int>   print every <int>th iteration; 0 for no printing" << std::endl;
  std::cout << "  --tol <float>      stopping tolerance" << std::endl;
  std::cout << "  --seed <int>       seed for random number generator used in initial guess" << std::endl;
  std::cout << "  --save             whether to save the output tensor" << std::endl;
  std::cout << "  --debug            turn on debugging output" << std::endl;
  std::cout << "  --warmup           do an iteration of mttkrp to warmup (useful for generating accurate timing information)" << std::endl;
  std::cout << "  --tensor <type>    Sptensor format: ";
  for (unsigned i=0; i<num_sptensor_types; ++i) {
    std::cout << sptensor_names[i];
    if (i != num_sptensor_types-1)
      std::cout << ", ";
  }
  std::cout << std::endl;
  std::cout << "  --vtune            connect to vtune for Intel-based profiling (assumes vtune profiling tool, amplxe-cl, is in your path)" << std::endl;
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  int ret = 0;

  try {

    ttb_bool help = parse_ttb_bool(argc, argv, "--help", false);
    if ((argc < 2) || (help)) {
      usage(argv);
      Kokkos::finalize();
      return 0;
    }

    // CP-ALS parameters
    // input
    std::string outputfilename =
      parse_string(argc,argv,"--output","output_cpals.txt");
    std::string inputfilename =
      parse_string(argc,argv,"--input","sptensor.dat");
    ttb_indx index_base =
      parse_ttb_indx(argc, argv, "--index_base", 0, 0, INT_MAX);
    ttb_bool gz =
      parse_ttb_bool(argc, argv, "--gz", false);
    ttb_indx rank =
      parse_ttb_indx(argc, argv, "--rank", 1, 1, INT_MAX);
    ttb_indx maxiters =
      parse_ttb_indx(argc, argv, "--maxiters", 1000, 1, INT_MAX);
    ttb_real tol =
      parse_ttb_real(argc, argv, "--tol", 0.0004, 0.0, 1.0);
    ttb_indx printitn =
      parse_ttb_indx(argc, argv, "--printitn", 1, 0, INT_MAX);
    ttb_indx seed =
      parse_ttb_indx(argc, argv, "--seed", 12345, 0, INT_MAX);
    ttb_bool no_save =
      parse_ttb_bool(argc, argv, "--no_save", false);
    ttb_bool debug =
      parse_ttb_bool(argc, argv, "--debug", false);
    ttb_bool warmup =
      parse_ttb_bool(argc, argv, "--warmup", false);
    ttb_bool vtune =
      parse_ttb_bool(argc, argv, "--vtune", false);
    SPTENSOR_TYPE tensor_type =
      parse_ttb_enum(argc, argv, "--tensor", SPTENSOR,
                     num_sptensor_types, sptensor_types, sptensor_names);

    if (vtune)
      Genten::connect_vtune();

    if (debug) {
      std::cout << "PARAMETERS" << std::endl;
      std::cout << "output = " << outputfilename << std::endl;
      std::cout << "input = " << inputfilename << std::endl;
      std::cout << "gz = " << (gz ? "true" : "false") << std::endl;
      std::cout << "index_base = " << index_base << std::endl;
      std::cout << "rank = " << rank << std::endl;
      std::cout << "maxiters = " << maxiters << std::endl;
      std::cout << "printitn = " << printitn << std::endl;
      std::cout << "tol = " << tol << std::endl;
      std::cout << "seed = " << seed << std::endl;
      std::cout << "no_save = " << (no_save ? "true" : "false") << std::endl;
      std::cout << "debug = " << (debug ? "true" : "false") << std::endl;
      std::cout << "warmup = " << (warmup ? "true" : "false") << std::endl;
      std::cout << "tensor type = " << sptensor_names[tensor_type] << std::endl;
      std::cout << std::endl;
    }

    if (tensor_type == SPTENSOR)
      ret = run_cpals<Genten::Sptensor>(
        inputfilename, outputfilename, index_base, gz, rank, seed, maxiters, tol,
        printitn, no_save, debug, warmup, tensor_type);
    else if (tensor_type == SPTENSOR_PERM)
      ret = run_cpals<Genten::Sptensor_perm>(
        inputfilename, outputfilename, index_base, gz, rank, seed, maxiters, tol,
        printitn, no_save, debug, warmup, tensor_type);
    else if (tensor_type == SPTENSOR_ROW)
      ret = run_cpals<Genten::Sptensor_row>(
        inputfilename, outputfilename, index_base, gz, rank, seed, maxiters, tol,
        printitn, no_save, debug, warmup, tensor_type);

  }
  catch(std::string sExc)
  {
    std::cout << "*** Call to cpals_core threw an exception:\n";
    std::cout << "  " << sExc << "\n";
    ret = 0;
  }

  Kokkos::finalize();
  return ret;
}
