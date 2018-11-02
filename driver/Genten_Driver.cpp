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

#include "Genten_Driver.hpp"
#include "Genten_Driver_Utils.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_IOtext.hpp"

void usage(char **argv)
{
  std::cout << "Usage: "<< argv[0]<<" [options]" << std::endl;
  std::cout << "options: " << std::endl;
  std::cout << "  --method <string>  decomposition method" << std::endl;
  std::cout << "  --output <string>  output file name" << std::endl;
  std::cout << "  --input <string>   path to input sptensor data" << std::endl;
  std::cout << "  --rol <string>     path to ROL optimization settings file for GCP method" << std::endl;
  std::cout << "  --index_base <int> starting index for tensor nonzeros" << std::endl;
  std::cout << "  --gz               read tensor in gzip compressed format" << std::endl;
  std::cout << "  --rank <int>       rank of factorization to compute" << std::endl;
  std::cout << "  --type <type>      loss function type for GCP: ";
  for (unsigned i=0; i<Genten::GCP_LossFunction::num_types; ++i) {
    std::cout << Genten::GCP_LossFunction::names[i];
    if (i != Genten::GCP_LossFunction::num_types-1)
      std::cout << ", ";
  }
  std::cout << std::endl;
  std::cout << "  --eps <float>      perturbation of loss functions for entries near 0" << std::endl;
  std::cout << "  --maxiters <int>   maximum iterations to perform" << std::endl;
  std::cout << "  --printitn <int>   print every <int>th iteration; 0 for no printing" << std::endl;
  std::cout << "  --tol <float>      stopping tolerance" << std::endl;
  std::cout << "  --seed <int>       seed for random number generator used in initial guess" << std::endl;
  std::cout << "  --prng             use parallel random number generator (not consistent with Matlab)" << std::endl;
  std::cout << "  --save             whether to save the output tensor" << std::endl;
  std::cout << "  --debug            turn on debugging output" << std::endl;
  std::cout << "  --warmup           do an iteration of mttkrp to warmup (useful for generating accurate timing information)" << std::endl;
  std::cout << "  --mttkrp_method <method> MTTKRP algorithm: ";
  for (unsigned i=0; i<Genten::MTTKRP_Method::num_types; ++i) {
    std::cout << Genten::MTTKRP_Method::names[i];
    if (i != Genten::MTTKRP_Method::num_types-1)
      std::cout << ", ";
  }
  std::cout << std::endl;
  std::cout << "  --mttkrp_tile_size <int> tile size for mttkrp algorithm" << std::endl;
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
    std::string method =
      parse_string(argc,argv,"--method","cpals");
    std::string outputfilename =
      parse_string(argc,argv,"--output","output.txt");
    std::string inputfilename =
      parse_string(argc,argv,"--input","sptensor.dat");
    std::string rolfilename =
      parse_string(argc,argv,"--rol","");
    ttb_indx index_base =
      parse_ttb_indx(argc, argv, "--index_base", 0, 0, INT_MAX);
    ttb_bool gz =
      parse_ttb_bool(argc, argv, "--gz", false);
    ttb_indx rank =
      parse_ttb_indx(argc, argv, "--rank", 1, 1, INT_MAX);
    Genten::GCP_LossFunction::type loss_function_type =
      parse_ttb_enum(argc, argv, "--type",
                     Genten::GCP_LossFunction::Gaussian,
                     Genten::GCP_LossFunction::num_types,
                     Genten::GCP_LossFunction::types,
                     Genten::GCP_LossFunction::names);
    ttb_real eps =
      parse_ttb_real(argc, argv, "--eps", 1.0e-10, 0.0, 1.0);
    ttb_indx maxiters =
      parse_ttb_indx(argc, argv, "--maxiters", 1000, 1, INT_MAX);
    ttb_real tol =
      parse_ttb_real(argc, argv, "--tol", 0.0004, 0.0, 1.0);
    ttb_indx printitn =
      parse_ttb_indx(argc, argv, "--printitn", 1, 0, INT_MAX);
    ttb_indx seed =
      parse_ttb_indx(argc, argv, "--seed", 12345, 0, INT_MAX);
    ttb_bool prng =
      parse_ttb_bool(argc, argv, "--prng", false);
    ttb_bool save =
      parse_ttb_bool(argc, argv, "--save", false);
    ttb_bool debug =
      parse_ttb_bool(argc, argv, "--debug", false);
    ttb_bool warmup =
      parse_ttb_bool(argc, argv, "--warmup", false);
    ttb_bool vtune =
      parse_ttb_bool(argc, argv, "--vtune", false);
    Genten::MTTKRP_Method::type mttkrp_method =
      parse_ttb_enum(argc, argv, "--mttkrp_method",
                     Genten::MTTKRP_Method::Atomic,
                     Genten::MTTKRP_Method::num_types,
                     Genten::MTTKRP_Method::types,
                     Genten::MTTKRP_Method::names);
    ttb_indx mttkrp_tile_size =
      parse_ttb_indx(argc, argv, "--mttkrp_tile_size", 0, 0, INT_MAX);

    if (vtune)
      Genten::connect_vtune();

    if (debug) {
      std::cout << "PARAMETERS" << std::endl;
      std::cout << "method = " << method << std::endl;
      std::cout << "output = " << outputfilename << std::endl;
      std::cout << "input = " << inputfilename << std::endl;
      std::cout << "rol = " << rolfilename << std::endl;
      std::cout << "gz = " << (gz ? "true" : "false") << std::endl;
      std::cout << "index_base = " << index_base << std::endl;
      std::cout << "rank = " << rank << std::endl;
      std::cout << "loss type = " << Genten::GCP_LossFunction::names[loss_function_type] << std::endl;
      std::cout << "eps = " << eps << std::endl;
      std::cout << "maxiters = " << maxiters << std::endl;
      std::cout << "printitn = " << printitn << std::endl;
      std::cout << "tol = " << tol << std::endl;
      std::cout << "seed = " << seed << std::endl;
      std::cout << "parallel rng = " << (prng ? "true" : "false") << std::endl;
      std::cout << "save = " << (save ? "true" : "false") << std::endl;
      std::cout << "debug = " << (debug ? "true" : "false") << std::endl;
      std::cout << "warmup = " << (warmup ? "true" : "false") << std::endl;
      std::cout << "mttkrp_method = " << Genten::MTTKRP_Method::types[mttkrp_method] << std::endl;
      std::cout << "mttkrp_tile_size = " << mttkrp_tile_size << std::endl;
      std::cout << std::endl;
    }

    Genten::AlgParams algParams;
    algParams.mttkrp_method = mttkrp_method;
    algParams.mttkrp_duplicated_factor_matrix_tile_size = mttkrp_tile_size;

    typedef Genten::DefaultExecutionSpace Space;
    typedef Genten::SptensorT<Space> Sptensor_type;
    typedef Genten::SptensorT<Genten::DefaultHostExecutionSpace> Sptensor_host_type;
    typedef Genten::KtensorT<Space> Ktensor_type;
    typedef Genten::KtensorT<Genten::DefaultHostExecutionSpace> Ktensor_host_type;

    Genten::SystemTimer timer(2);

    // Read in tensor data
    std::string fname(inputfilename);
    timer.start(0);
    Sptensor_host_type x_host;
    Genten::import_sptensor(fname, x_host, index_base, gz, true);
    Sptensor_type x = create_mirror_view( Space(), x_host );
    deep_copy( x, x_host );
    timer.stop(0);
    printf("Data import took %6.3f seconds\n", timer.getTotalTime(0));
    if (debug) Genten::print_sptensor(x_host, std::cout, fname);

    // Compute decomposition
    Ktensor_type u_init;
    Ktensor_type u = Genten::driver(x, u_init, method, rank, rolfilename,
                                    loss_function_type, eps, seed, prng,
                                    maxiters, tol, printitn, debug, warmup,
                                    std::cout, algParams);

    // Save results to file
    if (save)
    {
      timer.start(1);
      Ktensor_host_type u_host =
        create_mirror_view(Genten::DefaultHostExecutionSpace(), u);
      deep_copy( u_host, u );
      Genten::export_ktensor(outputfilename, u_host);
      timer.stop(1);
      printf("Data export took %6.3f seconds\n", timer.getTotalTime(2));
    }

  }
  catch(std::string sExc)
  {
    std::cout << "*** Call to genten threw an exception:\n";
    std::cout << "  " << sExc << "\n";
    ret = 0;
  }

  Kokkos::finalize();
  return ret;
}
