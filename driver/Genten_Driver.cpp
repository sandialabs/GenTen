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
#include "Genten_SystemTimer.hpp"
#include "Genten_IOtext.hpp"

void usage(char **argv)
{
  std::cout << "Usage: "<< argv[0]<<" [options]" << std::endl;
  std::cout << "Driver options: " << std::endl;
  std::cout << "  --input <string>   path to input sptensor data" << std::endl;
  std::cout << "  --output <string>  output file name" << std::endl;
  std::cout << "  --index_base <int> starting index for tensor nonzeros" << std::endl;
  std::cout << "  --gz               read tensor in gzip compressed format" << std::endl;
  std::cout << "  --save             whether to save the output tensor" << std::endl;
  std::cout << "  --vtune            connect to vtune for Intel-based profiling (assumes vtune profiling tool, amplxe-cl, is in your path)" << std::endl;
  std::cout << std::endl;
  Genten::AlgParams::print_help(std::cout);
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  int ret = 0;

  try {

    ttb_bool help = Genten::parse_ttb_bool(argc, argv, "--help", false);
    if ((argc < 2) || (help)) {
      usage(argv);
      Kokkos::finalize();
      return 0;
    }

    // Driver options
    std::string inputfilename =
      Genten::parse_string(argc, argv, "--input", "sptensor.dat");
    std::string outputfilename =
      Genten::parse_string(argc, argv, "--output", "");
    ttb_indx index_base =
      Genten::parse_ttb_indx(argc, argv, "--index_base", 0, 0, INT_MAX);
    ttb_bool gz =
      Genten::parse_ttb_bool(argc, argv, "--gz", false);
    ttb_bool vtune =
      Genten::parse_ttb_bool(argc, argv, "--vtune", false);

    // Everything else
    Genten::AlgParams algParams;
    algParams.parse(argc, argv);

    if (algParams.debug) {
      std::cout << "Driver options:" << std::endl;
      std::cout << "  input = " << inputfilename << std::endl;
      std::cout << "  output = " << outputfilename << std::endl;
      std::cout << "  index_base = " << index_base << std::endl;
      std::cout << "  gz = " << (gz ? "true" : "false") << std::endl;
      std::cout << "  vtune = " << (vtune ? "true" : "false") << std::endl;
      algParams.print(std::cout);
    }

    if (vtune)
      Genten::connect_vtune();

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
    if (algParams.debug) Genten::print_sptensor(x_host, std::cout, fname);

    // Compute decomposition
    Ktensor_type u_init;
    Ktensor_type u = Genten::driver(x, u_init, algParams, std::cout);

    // Save results to file
    if (outputfilename != "")
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
