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

#include "Genten_DistContext.hpp"
#include "Genten_DistTensorContext.hpp"
#include "Genten_Driver.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_FacTestSetGenerator.hpp"
#ifdef HAVE_BOOST
#include "Genten_Boost.hpp"
#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/property_tree/json_parser.hpp>
#endif

void usage(char **argv)
{
  std::cout << "Usage: "<< argv[0]<<" [options]" << std::endl;
  std::cout << "Driver options: " << std::endl;
  std::cout << "  --input <string>   path to input sptensor data (leave empty for random tensor)" << std::endl;
  std::cout << "  --dims <array>     random tensor dimensions" << std::endl;
  std::cout << "  --nnz <int>        approximate number of random tensor nonzeros" << std::endl;
  std::cout << "  --index-base <int> starting index for tensor nonzeros" << std::endl;
  std::cout << "  --gz               read tensor in gzip compressed format" << std::endl;
  std::cout << "  --sparse           whether tensor is sparse or dense" << std::endl;
  std::cout << "  --save-tensor <string> filename to save the tensor (leave blank for no save)" << std::endl;
  std::cout << "  --init <string>  file name for reading Ktensor initial guess (leave blank for random initial guess)" << std::endl;
  std::cout << "  --output <string>  output file name for saving Ktensor" << std::endl;
  std::cout << "  --vtune            connect to vtune for Intel-based profiling (assumes vtune profiling tool, amplxe-cl, is in your path)" << std::endl;
  std::cout << std::endl;
  Genten::AlgParams::print_help(std::cout);
}

template <typename Space>
int main_driver(Genten::AlgParams& algParams,
                const std::string& inputfilename,
                const std::string& outputfilename,
                const ttb_bool sparse,
                const std::string& initfilename,
                const ttb_indx index_base,
                const ttb_bool gz,
                const Genten::IndxArray& facDims_h,
                const ttb_indx nnz,
                const std::string& tensor_outputfilename)
{
  int ret = 0;
  Genten::SystemTimer timer(2);

  typedef Genten::SptensorT<Space> Sptensor_type;
  typedef Genten::SptensorT<Genten::DefaultHostExecutionSpace> Sptensor_host_type;
  typedef Genten::TensorT<Space> Tensor_type;
  typedef Genten::TensorT<Genten::DefaultHostExecutionSpace> Tensor_host_type;
  typedef Genten::KtensorT<Space> Ktensor_type;
  typedef Genten::KtensorT<Genten::DefaultHostExecutionSpace> Ktensor_host_type;
  typedef Genten::DistContext DC;

  Genten::DistTensorContext dtc;

  // Read in initial guess if provided
  Ktensor_type u_init;
  if (initfilename != "") {
    u_init = dtc.readInitialGuess<Space>(initfilename);
  }

  Ktensor_type u;
  if (sparse) {
    // Read in tensor data
    Sptensor_host_type x_host;
    Sptensor_type x;
    if (inputfilename != "") {
      timer.start(0);
      x = dtc.readTensorAndInit<Space>(inputfilename, index_base);
      timer.stop(0);
      DC::Barrier();
      if (dtc.gridRank() == 0)
        printf("Data import took %6.3f seconds\n", timer.getTotalTime(0));
    }
    else {
      if (dtc.nprocs() > 1)
        Genten::error("Random tensor not implemented for > 1 MPI procs");

      Genten::IndxArrayT<Space> facDims =
        create_mirror_view( Space(), facDims_h );
      deep_copy( facDims, facDims_h );

      if (dtc.gridRank() == 0) {
        std::cout << "Will construct a random Ktensor/Sptensor pair:\n";
        std::cout << "  Ndims = " << facDims_h.size() << ",  Size = [ ";
        for (ttb_indx  n = 0; n < facDims_h.size(); n++)
          std::cout << facDims_h[n] << ' ';
        std::cout << "]\n";
        std::cout << "  Ncomps = " << algParams.rank << "\n";
        std::cout << "  Maximum nnz = " << nnz << "\n";
      }

      // Generate a random Ktensor, and from it a representative sparse
      // data tensor.
      Genten::RandomMT rng (algParams.seed);
      Ktensor_host_type sol_host;
      timer.start(0);
      Genten::FacTestSetGenerator testGen;
      bool r = testGen.genSpFromRndKtensor(facDims_h, algParams.rank,
                                           nnz, rng, x_host, sol_host);
      if (!r)
        Genten::error("*** Call to genSpFromRndKtensor failed.\n");
      x = create_mirror_view( Space(), x_host );
      deep_copy( x, x_host );
      timer.stop(0);
      DC::Barrier();
      if (dtc.gridRank() == 0) {
        printf ("Data generation took %6.3f seconds\n", timer.getTotalTime(0));
        std::cout << "  Actual nnz  = " << x_host.nnz() << "\n";
      }
    }
    if (algParams.debug) Genten::print_sptensor(x_host, std::cout, "tensor");

    // Compute decomposition
    u = Genten::driver(dtc, x, u_init, algParams, std::cout);

    if (tensor_outputfilename != "") {
      if (dtc.nprocs() > 1)
        Genten::error("Tensor export not implemented for > 1 MPI procs");

      timer.start(1);
      Genten::export_sptensor(tensor_outputfilename, x_host, index_base == 0);
      timer.stop(1);
      DC::Barrier();
      if (dtc.gridRank() == 0)
        printf("Sptensor export took %6.3f seconds\n", timer.getTotalTime(1));
    }
  }
  else {
    if (dtc.nprocs() > 1)
        Genten::error("Dense tensor not implemented for > 1 MPI procs");

    Tensor_host_type x_host;
    Tensor_type x;
    // Read in tensor data
    if (inputfilename != "") {
      timer.start(0);
      Genten::import_tensor(inputfilename, x_host);
      x = create_mirror_view( Space(), x_host );
      deep_copy( x, x_host );
      timer.stop(0);
      printf("Data import took %6.3f seconds\n", timer.getTotalTime(0));
    }
    else {
      timer.start(0);
      Genten::RandomMT rng (algParams.seed);
      Ktensor_host_type sol_host;
      Genten::FacTestSetGenerator testGen;
      testGen.genDnFromRndKtensor(facDims_h, algParams.rank,
                                  rng, x_host, sol_host);
      x = create_mirror_view( Space(), x_host );
      deep_copy( x, x_host );
      timer.stop(0);
      printf ("Data generation took %6.3f seconds\n", timer.getTotalTime(0));
    }

    if (algParams.debug) Genten::print_tensor(x_host, std::cout, "tensor");

    // Compute decomposition
    u = Genten::driver(x, u_init, algParams, std::cout);

    if (tensor_outputfilename != "") {
      timer.start(1);
      Genten::export_tensor(tensor_outputfilename, x_host);
      timer.stop(1);
      printf("Tensor export took %6.3f seconds\n", timer.getTotalTime(1));
    }
  }

  // Save results to file
  if (outputfilename != "")
  {
    timer.start(1);
    dtc.exportToFile(u, outputfilename);
    timer.stop(1);
    DC::Barrier();
    if (dtc.gridRank() == 0)
      printf("Ktensor export took %6.3f seconds\n", timer.getTotalTime(1));
  }

  return ret;
}

int main(int argc, char* argv[])
{
  Genten::InitializeGenten(&argc, &argv);
  int ret = 0;

  try {

    // Convert argc,argv to list of arguments
    auto args = Genten::build_arg_list(argc,argv);

    const ttb_bool help =
      Genten::parse_ttb_bool(args, "--help", "--no-help", false);
    if ((argc < 2) || (help)) {
      usage(argv);
      Kokkos::finalize();
      return 0;
    }

    Genten::AlgParams algParams;

    // Driver options
    ttb_bool vtune = false;
    std::string inputfilename = "";
    ttb_indx index_base = 0;
    ttb_bool gz = false;
    ttb_bool sparse = true;
    std::string tensor_outputfilename = "";
    ttb_indx nnz = 1 * 1000 * 1000;
    Genten::IndxArray facDims_h = { 30, 40, 50 };
    std::string init = "";
    std::string outputfilename = "";

#ifdef HAVE_BOOST
    // Parse a json file if given before command-line arguments, that way
    // command line will override what is in the file
    Genten::ptree json_input;
    const std::string json_file =
      Genten::parse_string(args, "--json", "");
    if (json_file != "") {
      boost::property_tree::read_json(json_file, json_input);
      algParams.parse(json_input);
      Genten::parse_ptree_value(json_input, "vtune", vtune);

      // Tensor
      auto tensor_input_o = json_input.get_child_optional("tensor");
      if (tensor_input_o) {
        auto& tensor_input = *tensor_input_o;
        Genten::parse_ptree_value(tensor_input, "input-file", inputfilename);
        Genten::parse_ptree_value(tensor_input, "index-base", index_base, 0, INT_MAX);
        Genten::parse_ptree_value(tensor_input, "compressed", gz);
        Genten::parse_ptree_value(tensor_input, "sparse", sparse);
        Genten::parse_ptree_value(tensor_input, "output-file", tensor_outputfilename);
        Genten::parse_ptree_value(tensor_input, "rand-nnz", nnz, 1, INT_MAX);
        if (tensor_input.get_child_optional("rand-dims")) {
          std::vector<ttb_real> dims;
          Genten::parse_ptree_value(tensor_input, "rand-dims", dims, 1, INT_MAX);
          facDims_h = Genten::IndxArray(dims.size(), dims.data());
        }
      }

      // K-tensor
      auto ktensor_input_o = json_input.get_child_optional("k-tensor");
      if (ktensor_input_o) {
        auto& ktensor_input = *ktensor_input_o;
        Genten::parse_ptree_value(ktensor_input, "init", init);
        Genten::parse_ptree_value(ktensor_input, "output", outputfilename);
      }
    }
#endif
    vtune =
      Genten::parse_ttb_bool(args, "--vtune", "--no-vtune", vtune);
    inputfilename =
      Genten::parse_string(args, "--input", inputfilename);
    index_base =
      Genten::parse_ttb_indx(args, "--index-base", index_base, 0, INT_MAX);
    gz =
      Genten::parse_ttb_bool(args, "--gz", "--no-gz", gz);
    sparse =
      Genten::parse_ttb_bool(args, "--sparse", "--dense", sparse);
    tensor_outputfilename =
      Genten::parse_string(args, "--save-tensor", "");
    nnz =
      Genten::parse_ttb_indx(args, "--nnz", 1 * 1000 * 1000, 1, INT_MAX);
    facDims_h =
      Genten::parse_ttb_indx_array(args, "--dims", facDims_h, 1, INT_MAX);
    init =
      Genten::parse_string(args, "--init", init);
    outputfilename =
      Genten::parse_string(args, "--output", outputfilename);

    // Everything else
    algParams.parse(args);

    // Check for unrecognized arguments
    if (Genten::check_and_print_unused_args(args, std::cout)) {
      usage(argv);
      // Use throw instead of exit for proper Kokkos shutdown
      throw std::string("Invalid command line arguments.");
    }

    if (algParams.debug) {
      std::cout << "Driver options:" << std::endl;
      if (inputfilename == "")
        std::cout << "  input = " << inputfilename << std::endl;
      else {
        std::cout << "  dims = [";
        for (ttb_indx i=0; i<facDims_h.size(); ++i) {
          std::cout << facDims_h[i];
          if (i != facDims_h.size()-1)
            std::cout << ",";
        }
        std::cout << "]" << std::endl;
        std::cout << "  nnz = " << nnz << std::endl;
      }
      if (init != "")
        std::cout << "  init = " << init << std::endl;
      if (tensor_outputfilename != "")
        std::cout << "  save_tensor = " << tensor_outputfilename << std::endl;
      std::cout << "  output = " << outputfilename << std::endl;
      std::cout << "  sparse = " << (sparse ? "true" : "false") << std::endl;
      std::cout << "  index_base = " << index_base << std::endl;
      std::cout << "  gz = " << (gz ? "true" : "false") << std::endl;
      std::cout << "  vtune = " << (vtune ? "true" : "false") << std::endl;
      algParams.print(std::cout);
    }

    if (vtune)
      Genten::connect_vtune();

    // Parse execution space and run
    if (algParams.exec_space == Genten::Execution_Space::Default)
      ret = main_driver<Genten::DefaultExecutionSpace>(algParams,
                                                       inputfilename,
                                                       outputfilename,
                                                       sparse,
                                                       init,
                                                       index_base,
                                                       gz,
                                                       facDims_h,
                                                       nnz,
                                                       tensor_outputfilename);
#ifdef KOKKOS_ENABLE_CUDA
    else if (algParams.exec_space == Genten::Execution_Space::Cuda)
      ret = main_driver<Kokkos::Cuda>(algParams,
                                      inputfilename,
                                      outputfilename,
                                      sparse,
                                      init,
                                      index_base,
                                      gz,
                                      facDims_h,
                                      nnz,
                                      tensor_outputfilename);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    else if (algParams.exec_space == Genten::Execution_Space::OpenMP)
      ret = main_driver<Kokkos::OpenMP>(algParams,
                                        inputfilename,
                                        outputfilename,
                                        sparse,
                                        init,
                                        index_base,
                                        gz,
                                        facDims_h,
                                        nnz,
                                        tensor_outputfilename);
#endif
#ifdef KOKKOS_ENABLE_THREADS
    else if (algParams.exec_space == Genten::Execution_Space::Threads)
      ret = main_driver<Kokkos::Threads>(algParams,
                                         inputfilename,
                                         outputfilename,
                                         sparse,
                                         init,
                                         index_base,
                                         gz,
                                         facDims_h,
                                         nnz,
                                         tensor_outputfilename);
#endif
#ifdef KOKKOS_ENABLE_SERIAL
    else if (algParams.exec_space == Genten::Execution_Space::Serial)
      ret = main_driver<Kokkos::Serial>(algParams,
                                        inputfilename,
                                        outputfilename,
                                        sparse,
                                        init,
                                        index_base,
                                        gz,
                                        facDims_h,
                                        nnz,
                                        tensor_outputfilename);
#endif
    else
      Genten::error("Invalid execution space: " + std::string(Genten::Execution_Space::names[algParams.exec_space]));

  }
  catch(std::string sExc)
  {
    std::cout << "*** Call to genten threw an exception:\n";
    std::cout << "  " << sExc << "\n";
    ret = 0;
  }

  Genten::FinalizeGenten();
  return ret;
}
