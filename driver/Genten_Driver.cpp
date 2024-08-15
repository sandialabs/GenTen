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
#include "Genten_Ptree.hpp"

#ifdef HAVE_PYTHON_EMBED
#include <pybind11/embed.h>
#endif

void print_banner(std::ostream& out)
{
  std::string banner = R"(
          ___         ___         ___      ___         ___         ___
         /\  \       /\  \       /\__\    /\  \       /\  \       /\__\
        /::\  \     /::\  \     /::|  |   \:\  \     /::\  \     /::|  |
       /:/\:\  \   /:/\:\  \   /:|:|  |    \:\  \   /:/\:\  \   /:|:|  |
      /:/  \:\  \ /::\~\:\  \ /:/|:|  |__  /::\  \ /::\~\:\  \ /:/|:|  |__
     /:/__/_\:\__/:/\:\ \:\__/:/ |:| /\__\/:/\:\__/:/\:\ \:\__/:/ |:| /\__\
     \:\  /\ \/__\:\~\:\ \/__\/__|:|/:/  /:/  \/__\:\~\:\ \/__\/__|:|/:/  /
      \:\ \:\__\  \:\ \:\__\     |:/:/  /:/  /     \:\ \:\__\     |:/:/  /
       \:\/:/  /   \:\ \/__/     |::/  /\/__/       \:\ \/__/     |::/  /
        \::/  /     \:\__\       /:/  /              \:\__\       /:/  /
         \/__/       \/__/       \/__/                \/__/       \/__/

--------------------------------------------------------------------------------
GenTen:  Software for Generalized Canonical Polyadic Tensor Decompositions

Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.
--------------------------------------------------------------------------------
)";

  if (Genten::DistContext::rank() == 0)
    out << banner << std::endl;
}

void usage(char **argv)
{
  if (Genten::DistContext::rank() != 0)
    return;

  std::cout << "Usage: "<< argv[0]<<" [options]" << std::endl;
  std::cout << "Driver options: " << std::endl;
#ifdef HAVE_BOOST
  std::cout << "  --json <string>    Read input paramters from supplied JSON file" << std::endl;
#endif
  std::cout << "  --input <string>   path to input sptensor data (leave empty for random tensor)" << std::endl;
  std::cout << "  --dims <array>     random tensor dimensions" << std::endl;
  std::cout << "  --nnz <int>        approximate number of random tensor nonzeros" << std::endl;
  std::cout << "  --index-base <int> starting index for tensor nonzeros" << std::endl;
  std::cout << "  --gz               read tensor in gzip compressed format" << std::endl;
  std::cout << "  --save-tensor <string> filename to save the tensor (leave blank for no save)" << std::endl;
  std::cout << "  --initial-file <string>  file name for reading Ktensor initial guess (leave blank for random initial guess)" << std::endl;
  std::cout << "  --output-file <string>  output file name for saving Ktensor" << std::endl;
  std::cout << "  --output-dense-reconstruction <string>  output file name for saving the tensor reconstruction as a dense tensor" << std::endl;
  std::cout << "  --output-sparse-reconstruction <string>  output file name for saving the tensor reconstruction as a sparse tensor" << std::endl;
  std::cout << "  --vtune            connect to vtune for Intel-based profiling (assumes vtune profiling tool, amplxe-cl, is in your path)" << std::endl;
  std::cout << "  --history-file     file to save performance history" << std::endl;
  std::cout << std::endl;
  Genten::AlgParams::print_help(std::cout);
}

template <typename Space>
int main_driver(Genten::AlgParams& algParams,
                const Genten::ptree& json_input,
                const std::string& inputfilename,
                const std::string& outputfilename,
                const std::string& initfilename,
                const ttb_indx index_base,
                const ttb_bool gz,
                const Genten::IndxArray& facDims_h,
                const ttb_indx nnz,
                const std::string& tensor_outputfilename,
                const std::string& dense_reconstruction,
                const std::string& sparse_reconstruction,
                const ttb_real sparse_reconstruction_tol,
                const std::string& history_file)
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

  Genten::DistTensorContext<Space> dtc;

  Ktensor_type u;
  Genten::PerfHistory history;
  if (algParams.sparse) {
    Sptensor_host_type x_host;
    Sptensor_type x;
    Tensor_type xd;
    // Read in tensor data
    if (inputfilename != "") {
      timer.start(0);
      auto tensor_input = json_input.get_child_optional("tensor");
      std::tie(x,xd) = dtc.distributeTensor(
        inputfilename, index_base, gz, tensor_input, algParams);
      timer.stop(0);
      DC::Barrier();
      if (dtc.gridRank() == 0)
        printf("  Data import took %6.3f seconds\n", timer.getTotalTime(0));
    }
    else {
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
      timer.stop(0);
      DC::Barrier();
      if (dtc.gridRank() == 0) {
        printf ("  Data generation took %6.3f seconds\n", timer.getTotalTime(0));
        std::cout << "  Actual nnz  = " << x_host.nnz() << "\n";
      }
      x = dtc.distributeTensor(x_host, algParams);
    }

    // Print execution environment
    Genten::print_environment(x, dtc, std::cout);

    if (algParams.debug) Genten::print_sptensor(x_host, std::cout, "tensor");

    // Read in initial guess if provided
    Ktensor_type u_init;
    if (initfilename != "") {
      u_init = dtc.readInitialGuess(initfilename);
    }

    // Compute decomposition
    u = Genten::driver(dtc, x, u_init, algParams, history, std::cout);

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
    Tensor_host_type x_host;
    Tensor_type x;
    Sptensor_type xs;
    // Read in tensor data
    if (inputfilename != "") {
      timer.start(0);
      auto tensor_input = json_input.get_child_optional("tensor");
      std::tie(xs,x) = dtc.distributeTensor(
        inputfilename, index_base, gz, tensor_input, algParams);
      timer.stop(0);
      if (dtc.gridRank() == 0)
        printf("  Data import took %6.3f seconds\n", timer.getTotalTime(0));
    }
    else {
      timer.start(0);
      Genten::RandomMT rng (algParams.seed);
      Ktensor_host_type sol_host;
      Genten::FacTestSetGenerator testGen;
      testGen.genDnFromRndKtensor(facDims_h, algParams.rank,
                                  rng, x_host, sol_host);
      timer.stop(0);
      DC::Barrier();
      if (dtc.gridRank() == 0)
        printf ("Data generation took %6.3f seconds\n", timer.getTotalTime(0));
      x = dtc.distributeTensor(x_host, algParams);
    }

    // Print execution environment
    Genten::print_environment(x, dtc, std::cout);

    if (algParams.debug) Genten::print_tensor(x_host, std::cout, "tensor");

    // Read in initial guess if provided
    Ktensor_type u_init;
    if (initfilename != "") {
      u_init = dtc.readInitialGuess(initfilename);
    }

    // Compute decomposition
    u = Genten::driver(dtc, x, u_init, algParams, history, std::cout);

    if (tensor_outputfilename != "") {
      timer.start(1);
      Genten::export_tensor(tensor_outputfilename, x_host);
      timer.stop(1);
      DC::Barrier();
      if (dtc.gridRank() == 0)
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
      printf("  Ktensor export took %6.3f seconds\n", timer.getTotalTime(1));
  }

  // Save dense/sparse reconstructions
  if (dense_reconstruction != "" || sparse_reconstruction != "") {
    timer.start(1);
    Genten::Ktensor u_root =
      dtc.template importToRoot<Genten::DefaultHostExecutionSpace>(u);
    if (dtc.gridRank() == 0) {
      Genten::Tensor dense_recon(u_root);
      if (dense_reconstruction != "") {
        std::cout << "Writing ktensor dense reconstruction to "
                  << dense_reconstruction << std::endl;
        Genten::export_tensor(dense_reconstruction, dense_recon);
      }
      if (sparse_reconstruction != "") {
        std::cout << "Writing ktensor sparse reconstruction to "
                  << sparse_reconstruction << std::endl;
        Genten::Sptensor sparse_recon(dense_recon, sparse_reconstruction_tol);
        Genten::export_sptensor(sparse_reconstruction, sparse_recon);
      }
    }
    timer.stop(1);
    DC::Barrier();
    if (dtc.gridRank() == 0)
      printf("  Writing reconstruction(s) took %6.3f seconds\n", timer.getTotalTime(1));
  }

  // Save history to file
  if (history_file != "" && dtc.gridRank() == 0) {
    std::cout << "Saving performance history to file " << history_file
              << std::endl;
    history.print(history_file);
  }

  // Testing -- we use int/double on purpose to avoid ambiguities
  if (history.size() > 0) {
    const auto& entry = history.lastEntry();
    auto testing_input_o = json_input.get_child_optional("testing");
    if (testing_input_o) {
      auto testing_input = *testing_input_o;

      // Final fit
      auto fit_input_o = testing_input.get_child_optional("final-fit");
      if (fit_input_o) {
        auto fit_input = *fit_input_o;
        double fit_expected = fit_input.get<double>("value");
        double rtol = fit_input.get("relative-tolerance", 0.0);
        double atol = fit_input.get("absolute-tolerance", 0.0);
        double tol = std::abs(fit_expected)*rtol+atol;
        double fit = entry.fit;
        double diff = std::abs(fit_expected-fit);
        bool passed = diff < tol;
        std::string pass_fail_string = passed ? "Passed!" : "Failed!";
        if (dtc.gridRank() == 0) {
          std::cout << std::endl
                    << "Checking final fit:" << std::endl
                    << "\tValue:      " << fit << std::endl
                    << "\tExpected:   " << fit_expected << std::endl
                    << "\tDifference: " << diff << std::endl
                    << "\tTolerance:  " << tol << std::endl
                    << "\t" << pass_fail_string
                    << std::endl;
        }
        if (!passed) ++ret;
      }

      // Final residual
      auto res_input_o = testing_input.get_child_optional("final-residual");
      if (res_input_o) {
        auto res_input = *res_input_o;
        double res_expected = res_input.get<double>("value");
        double rtol = res_input.get("relative-tolerance", 0.0);
        double atol = res_input.get("absolute-tolerance", 0.0);
        double tol = std::abs(res_expected)*rtol+atol;
        double res = entry.residual;
        double diff = std::abs(res_expected-res);
        bool passed = diff < tol;
        std::string pass_fail_string = passed ? "Passed!" : "Failed!";
        if (dtc.gridRank() == 0) {
          std::cout << std::endl
                    << "Checking final residual:" << std::endl
                    << "\tValue:      " << res << std::endl
                    << "\tExpected:   " << res_expected << std::endl
                    << "\tDifference: " << diff << std::endl
                    << "\tTolerance:  " << tol << std::endl
                    << "\t" << pass_fail_string
                    << std::endl;
        }
        if (!passed) ++ret;
      }

      // Final gradient norm
      auto grad_input_o =
        testing_input.get_child_optional("final-gradient-norm");
      if (grad_input_o) {
        auto grad_input = *grad_input_o;
        double grad_expected = grad_input.get<double>("value");
        double rtol = grad_input.get("relative-tolerance", 0.0);
        double atol = grad_input.get("absolute-tolerance", 0.0);
        double tol = std::abs(grad_expected)*rtol+atol;
        double grad = entry.grad_norm;
        double diff = std::abs(grad_expected-grad);
        bool passed = diff < tol;
        std::string pass_fail_string = passed ? "Passed!" : "Failed!";
        if (dtc.gridRank() == 0) {
          std::cout << std::endl
                    << "Checking final gradient norm:" << std::endl
                    << "\tValue:      " << grad << std::endl
                    << "\tExpected:   " << grad_expected << std::endl
                    << "\tDifference: " << diff << std::endl
                    << "\tTolerance:  " << tol << std::endl
                    << "\t" << pass_fail_string
                    << std::endl;
        }
        if (!passed) ++ret;
      }

      // Number of iterations
      auto itr_input_o = testing_input.get_child_optional("iterations");
      if (itr_input_o) {
        auto itr_input = *itr_input_o;
        int itr_expected = itr_input.get<int>("value");
        int tol = itr_input.get("absolute-tolerance", 0);
        int itr = entry.iteration;
        int diff = std::abs(itr_expected-itr);
        bool passed = diff < tol;
        std::string pass_fail_string = passed ? "Passed!" : "Failed!";
        if (dtc.gridRank() == 0) {
          std::cout << std::endl
                    << "Checking number of iterations:" << std::endl
                    << "\tValue:      " << itr << std::endl
                    << "\tExpected:   " << itr_expected << std::endl
                    << "\tDifference: " << diff << std::endl
                    << "\tTolerance:  " << tol << std::endl
                    << "\t" << pass_fail_string
                    << std::endl;
        }
        if (!passed) ++ret;
      }
    }
  }

  return ret;
}

int main(int argc, char* argv[])
{
  int ret = 0;

  try {
    Genten::InitializeGenten(&argc, &argv);

    print_banner(std::cout);

#ifdef HAVE_PYTHON_EMBED
    // Start up python interpreter
    pybind11::scoped_interpreter guard{};
#endif

    // Convert argc,argv to list of arguments
    auto args = Genten::build_arg_list(argc,argv);

    const ttb_bool help =
      Genten::parse_ttb_bool(args, "--help", "--no-help", false);
    if ((argc < 2) || (help)) {
      usage(argv);
    }
    else {

      Genten::AlgParams algParams;

      // Driver options
      ttb_bool vtune = false;
      std::string inputfilename = "";
      ttb_indx index_base = 0;
      ttb_bool gz = false;
      std::string tensor_outputfilename = "";
      std::string dense_reconstruction = "";
      std::string sparse_reconstruction = "";
      ttb_real sparse_reconstruction_tol = 0.0;
      ttb_indx nnz = 1 * 1000 * 1000;
      Genten::IndxArray facDims_h = { 30, 40, 50 };
      std::string init = "";
      std::string outputfilename = "";
      std::string history_file = "";

      // Parse a json file if given before command-line arguments, that way
      // command line will override what is in the file
      Genten::ptree json_input;
      const std::string json_file =
        Genten::parse_string(args, "--json", "");
      if (json_file != "") {
        read_json(json_file, json_input);
        algParams.parse(json_input);
        Genten::parse_ptree_value(json_input, "vtune", vtune);

        // Tensor
        auto tensor_input_o = json_input.get_child_optional("tensor");
        if (tensor_input_o) {
          auto& tensor_input = *tensor_input_o;
          Genten::parse_ptree_value(tensor_input, "input-file", inputfilename);
          Genten::parse_ptree_value(tensor_input, "index-base", index_base, 0, INT_MAX);
          Genten::parse_ptree_value(tensor_input, "compressed", gz);
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
          Genten::parse_ptree_value(ktensor_input, "initial-file", init);
          Genten::parse_ptree_value(ktensor_input, "output-file", outputfilename);
          Genten::parse_ptree_value(ktensor_input, "dense-reconstruction", dense_reconstruction);
          Genten::parse_ptree_value(ktensor_input, "sparse-reconstruction", sparse_reconstruction);
          Genten::parse_ptree_value(ktensor_input, "sparse-reconstruction-tolerance", sparse_reconstruction_tol, 0.0, DBL_MAX);
        }

        Genten::parse_ptree_value(json_input, "history-file", history_file);
      }

      vtune =
        Genten::parse_ttb_bool(args, "--vtune", "--no-vtune", vtune);
      inputfilename =
        Genten::parse_string(args, "--input", inputfilename);
      index_base =
        Genten::parse_ttb_indx(args, "--index-base", index_base, 0, INT_MAX);
      gz =
        Genten::parse_ttb_bool(args, "--gz", "--no-gz", gz);
      tensor_outputfilename =
        Genten::parse_string(args, "--save-tensor", tensor_outputfilename);
      dense_reconstruction =
        Genten::parse_string(args, "--dense-reconstruction", dense_reconstruction);
      sparse_reconstruction =
        Genten::parse_string(args, "--sparse-reconstruction", sparse_reconstruction);
      sparse_reconstruction_tol =
        Genten::parse_ttb_real(args, "--sparse-reconstruction-tolerance", sparse_reconstruction_tol, 0.0, DBL_MAX);
      nnz =
        Genten::parse_ttb_indx(args, "--nnz", nnz, 1, INT_MAX);
      facDims_h =
        Genten::parse_ttb_indx_array(args, "--dims", facDims_h, 1, INT_MAX);
      init =
        Genten::parse_string(args, "--initial-file", init);
      outputfilename =
        Genten::parse_string(args, "--output-file", outputfilename);
      history_file =
        Genten::parse_string(args, "--history-file", history_file);

      // Everything else
      algParams.parse(args);

      // Check for unrecognized arguments
      if (Genten::check_and_print_unused_args(args, std::cout)) {
        usage(argv);
        // Use throw instead of exit for proper Kokkos shutdown
        throw std::string("Invalid command line arguments.");
      }

      if (algParams.debug && Genten::DistContext::rank() == 0) {
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
          std::cout << "  initial-file = " << init << std::endl;
        if (tensor_outputfilename != "")
          std::cout << "  save-tensor = " << tensor_outputfilename << std::endl;
        if (dense_reconstruction != "")
          std::cout << "  dense-reconstruction = " << dense_reconstruction << std::endl;
        if (sparse_reconstruction != "")
          std::cout << "  sparse-reconstruction = " << sparse_reconstruction << std::endl;
        std::cout << "  output-file = " << outputfilename << std::endl;
        std::cout << "  index_base = " << index_base << std::endl;
        std::cout << "  gz = " << (gz ? "true" : "false") << std::endl;
        std::cout << "  vtune = " << (vtune ? "true" : "false") << std::endl;
        if (history_file != "")
          std::cout << "  history-file = " << history_file << std::endl;
        algParams.print(std::cout);
      }

      if (vtune)
        Genten::connect_vtune();

      // Parse execution space and run
      if (algParams.exec_space == Genten::Execution_Space::Default)
        ret = main_driver<Genten::DefaultExecutionSpace>(algParams,
                                                         json_input,
                                                         inputfilename,
                                                         outputfilename,
                                                         init,
                                                         index_base,
                                                         gz,
                                                         facDims_h,
                                                         nnz,
                                                         tensor_outputfilename,
                                                         dense_reconstruction,
                                                         sparse_reconstruction,
                                                         sparse_reconstruction_tol,
                                                         history_file);
#ifdef HAVE_CUDA
      else if (algParams.exec_space == Genten::Execution_Space::Cuda)
        ret = main_driver<Kokkos::Cuda>(algParams,
                                        json_input,
                                        inputfilename,
                                        outputfilename,
                                        init,
                                        index_base,
                                        gz,
                                        facDims_h,
                                        nnz,
                                        tensor_outputfilename,
                                        dense_reconstruction,
                                        sparse_reconstruction,
                                        sparse_reconstruction_tol,
                                        history_file);
#endif
#ifdef HAVE_HIP
      else if (algParams.exec_space == Genten::Execution_Space::HIP)
        ret = main_driver<Kokkos::Experimental::HIP>(algParams,
                                                     json_input,
                                                     inputfilename,
                                                     outputfilename,
                                                     init,
                                                     index_base,
                                                     gz,
                                                     facDims_h,
                                                     nnz,
                                                     tensor_outputfilename,
                                                     dense_reconstruction,
                                                     sparse_reconstruction,
                                                     sparse_reconstruction_tol,
                                                     history_file);
#endif
#ifdef HAVE_SYCL
      else if (algParams.exec_space == Genten::Execution_Space::SYCL)
        ret = main_driver<Kokkos::Experimental::SYCL>(algParams,
                                                      json_input,
                                                      inputfilename,
                                                      outputfilename,
                                                      init,
                                                      index_base,
                                                      gz,
                                                      facDims_h,
                                                      nnz,
                                                      tensor_outputfilename,
                                                      dense_reconstruction,
                                                      sparse_reconstruction,
                                                      sparse_reconstruction_tol,
                                                      history_file);
#endif
#ifdef HAVE_OPENMP
      else if (algParams.exec_space == Genten::Execution_Space::OpenMP)
        ret = main_driver<Kokkos::OpenMP>(algParams,
                                          json_input,
                                          inputfilename,
                                          outputfilename,
                                          init,
                                          index_base,
                                          gz,
                                          facDims_h,
                                          nnz,
                                          tensor_outputfilename,
                                          dense_reconstruction,
                                          sparse_reconstruction,
                                          sparse_reconstruction_tol,
                                          history_file);
#endif
#ifdef HAVE_THREADS
      else if (algParams.exec_space == Genten::Execution_Space::Threads)
        ret = main_driver<Kokkos::Threads>(algParams,
                                           json_input,
                                           inputfilename,
                                           outputfilename,
                                           init,
                                           index_base,
                                           gz,
                                           facDims_h,
                                           nnz,
                                           tensor_outputfilename,
                                           dense_reconstruction,
                                           sparse_reconstruction,
                                           sparse_reconstruction_tol,
                                           history_file);
#endif
#ifdef HAVE_SERIAL
      else if (algParams.exec_space == Genten::Execution_Space::Serial)
        ret = main_driver<Kokkos::Serial>(algParams,
                                          json_input,
                                          inputfilename,
                                          outputfilename,
                                          init,
                                          index_base,
                                          gz,
                                          facDims_h,
                                          nnz,
                                          tensor_outputfilename,
                                          dense_reconstruction,
                                          sparse_reconstruction,
                                          sparse_reconstruction_tol,
                                          history_file);
#endif
      else
        Genten::error("Invalid execution space: " + std::string(Genten::Execution_Space::names[algParams.exec_space]));
    }
  }
  catch(const std::exception& e)
  {
    if (Genten::DistContext::rank() == 0)
      std::cout << "*** Call to genten threw an exception:" << std::endl
                << "  " << e.what() << std::endl;
    ret = -1;
  }
  catch(const std::string& s)
  {
    if (Genten::DistContext::rank() == 0)
      std::cout << "*** Call to genten threw an exception:" << std::endl
                << "  " << s << std::endl;
    ret = -1;
  }
  catch(...)
  {
    if (Genten::DistContext::rank() == 0)
      std::cout << "*** Call to genten threw an unknown exception"
                << std::endl;
    ret = -1;
  }

  Genten::FinalizeGenten();
  return ret;
}
