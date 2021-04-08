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
#include "Genten_Online_GCP.hpp"
#include "Genten_Util.hpp"
#include "Genten_SystemTimer.hpp"

extern "C" {

DLL_EXPORT_SYM void mexFunction(int nlhs, mxArray *plhs[],
                                int nrhs, const mxArray *prhs[])
{
  typedef Genten::DefaultExecutionSpace ExecSpace;

  GentenInitialize();

  try {
    if (nrhs < 5) {
      std::cout << "Expected at least 5 command line arguments" << std::endl;
      return;
    }

    // Parse inputs
    auto args = mxBuildArgList(nrhs, 5, prhs);
    Genten::AlgParams algParams;
    algParams.fixup<ExecSpace>(std::cout);
    algParams.parse(args);
    if (Genten::check_and_print_unused_args(args, std::cout)) {
      algParams.print_help(std::cout);
      throw std::string("Invalid command line arguments.");
    }

    // Get tensors
    Genten::SystemTimer timer(1, algParams.timings);
    timer.start(0);
    const mxArray *tensors = prhs[0];
    if (!mxIsCell(tensors))
      Genten::error("First argument must be a cell array of tensors!");
    size_t num_mat = mxGetNumberOfElements(tensors);
    std::vector<Genten::SptensorT<ExecSpace> > X(num_mat);
    for (size_t i=0; i<num_mat; ++i) {
      mxArray *cell = mxGetCell(tensors, i);
      X[i] = mxGetSptensor<ExecSpace>(cell, algParams.debug);
    }
    timer.stop(1);
    if (algParams.timings)
      std::cout << "Parsing tensors took " << timer.getTotalTime(0)
                << " seconds" << std::endl;

    // Get initial guess
    Genten::KtensorT<ExecSpace> u =
      mxGetKtensor<ExecSpace>(prhs[1], algParams.debug);

    // Get temporal and spatial solver parameters
    Genten::AlgParams temporalAlgParams = mxGetAlgParams(prhs[2]);
    Genten::AlgParams spatialAlgParams = mxGetAlgParams(prhs[3]);
    temporalAlgParams.fixup<ExecSpace>(std::cout);
    spatialAlgParams.fixup<ExecSpace>(std::cout);

    // Xinit
    Genten::SptensorT<ExecSpace> Xinit =
      mxGetSptensor<ExecSpace>(prhs[4], algParams.debug);

    // Create OnlineGCP object
    Genten::Array fest, ften;
    Genten::online_gcp(X, Xinit, u, algParams, temporalAlgParams,
                       spatialAlgParams, std::cout, fest, ften);


    // Return results
    if (nlhs >= 1)
      plhs[0] = mxSetKtensor(u, algParams.debug);
    if (nlhs >= 2)
      plhs[1] = mxSetArray(fest);
    if (nlhs >= 3)
      plhs[2] = mxSetArray(ften);
  }

  catch(std::string sExc)
  {
    std::cout << "*** Call to genten threw an exception:\n";
    std::cout << "  " << sExc << "\n";
  }
}

}
