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

// Mex Driver for Genten CP decomposition

#include "Genten_Matlab.hpp"
#include "Genten_Driver.hpp"
#include "Genten_Util.hpp"
#include "Genten_SystemTimer.hpp"

template <typename ExecSpace>
void matlab_driver(int nlhs, mxArray *plhs[],
                   int nrhs, const mxArray *prhs[],
                   Genten::AlgParams& algParams)
{
  // Get tensor
  Genten::SystemTimer timer(1, algParams.timings);
  timer.start(0);
  Genten::SptensorT<ExecSpace> X_sparse;
  Genten::TensorT<ExecSpace> X_dense;
  bool sparse = true;
  if (mxIsClass(prhs[0], "sptensor") || mxIsClass(prhs[0], "sptensor_gt")) {
    X_sparse = mxGetSptensor<ExecSpace>(prhs[0], algParams.debug);
    sparse = true;
  }
  else if (mxIsClass(prhs[0], "tensor")) {
    X_dense = mxGetTensor<ExecSpace>(prhs[0], algParams.debug);
    sparse = false;
  }
  else
    Genten::error("First arg is not a tensor or sptensor!");
  timer.stop(1);
  if (algParams.timings)
    std::cout << "Parsing tensor took " << timer.getTotalTime(0)
              << " seconds" << std::endl;

  // Get rank
  algParams.rank = mxGetScalar(prhs[1]);

  // Get initial guess
  Genten::KtensorT<ExecSpace> u_init;
  const mxArray *arg = prhs[2];
  if (mxIsStruct(arg))
    u_init = mxGetKtensor<ExecSpace>(arg, algParams.debug);
  else if (mxIsChar(arg)) {
    std::string str = mxGetStdString(arg);
    // Just check string is a proper value since an empty u_init means a
    // random one in Genten::driver() below
    if (str != "random" && str != "random_gt")
      throw std::string("Invalid random initial guess specification:  ") +
        str;
  }
  else
    throw std::string("Invalid type for initial guess specification.");

  // To do:  initialize this
  Genten::DistTensorContext<ExecSpace> dtc;

  // Call driver
  Genten::PerfHistory history;
  Genten::KtensorT<ExecSpace> u;
  if (sparse)
    u = Genten::driver(dtc, X_sparse, u_init, algParams, history, std::cout);
  else
    u = Genten::driver(dtc, X_dense, u_init, algParams, history, std::cout);

  // Return results
  if (nlhs >= 1)
    plhs[0] = mxSetKtensor(u, algParams.debug);
  if (nlhs >= 2)
    plhs[1] = mxSetKtensor(u_init);
  if (nlhs >= 3)
    plhs[2] = mxSetHistory(history);
}

extern "C" {

DLL_EXPORT_SYM void mexFunction(int nlhs, mxArray *plhs[],
                                int nrhs, const mxArray *prhs[])
{
  GentenInitialize();

  try {
    if (nrhs < 3) {
      std::cout << "Expected at least 3 command line arguments" << std::endl;
      return;
    }

    // Parse inputs
    auto args = mxBuildArgList(nrhs, 3, prhs);
    Genten::AlgParams algParams;
    algParams.parse(args);
    if (Genten::check_and_print_unused_args(args, std::cout)) {
      algParams.print_help(std::cout);
      throw std::string("Invalid command line arguments.");
    }

    // Parse execution space and run
    if (algParams.exec_space == Genten::Execution_Space::Default)
      matlab_driver<Genten::DefaultExecutionSpace>(nlhs, plhs, nrhs, prhs,
                                                   algParams);
#ifdef HAVE_CUDA
    else if (algParams.exec_space == Genten::Execution_Space::Cuda)
      matlab_driver<Kokkos::Cuda>(nlhs, plhs, nrhs, prhs, algParams);
#endif
#ifdef HAVE_HIP
    else if (algParams.exec_space == Genten::Execution_Space::HIP)
      matlab_driver<Kokkos::Experimental::HIP>(nlhs, plhs, nrhs, prhs, algParams);
#endif
#ifdef HAVE_SYCL
    else if (algParams.exec_space == Genten::Execution_Space::SYCL)
      matlab_driver<Kokkos::Experimental::SYCL>(nlhs, plhs, nrhs, prhs, algParams);
#endif
#ifdef HAVE_OPENMP
    else if (algParams.exec_space == Genten::Execution_Space::OpenMP)
      matlab_driver<Kokkos::OpenMP>(nlhs, plhs, nrhs, prhs, algParams);
#endif
#ifdef HAVE_THREADS
    else if (algParams.exec_space == Genten::Execution_Space::Threads)
      matlab_driver<Kokkos::Threads>(nlhs, plhs, nrhs, prhs, algParams);
#endif
#ifdef HAVE_SERIAL
    else if (algParams.exec_space == Genten::Execution_Space::Serial)
      matlab_driver<Kokkos::Serial>(nlhs, plhs, nrhs, prhs, algParams);
#endif
    else
      Genten::error("Invalid execution space: " + std::string(Genten::Execution_Space::names[algParams.exec_space]));
  }

  catch(std::string sExc)
  {
    std::cout << "*** Call to genten threw an exception:\n";
    std::cout << "  " << sExc << "\n";
  }
}

}
