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

extern "C" {

DLL_EXPORT_SYM void mexFunction(int nlhs, mxArray *plhs[],
                                int nrhs, const mxArray *prhs[])
{
  typedef Genten::DefaultExecutionSpace ExecSpace;

  GentenInitialize();

  try {
    if (nrhs < 2) {
      std::cout << "Expected at least 2 command line arguments" << std::endl;
      return;
    }

    // Parse inputs
    Genten::AlgParams algParams;
    Genten::KtensorT<ExecSpace> u_init;
    Genten::SptensorT<ExecSpace> X =
      mxGetSptensor<ExecSpace>(prhs[0], algParams.debug);
    const ttb_indx rank = mxGetScalar(prhs[1]);
    for (int i=2; i<nrhs; i+=2) {
      std::string option = mxGetStdString(prhs[i]);
      const mxArray *ptr = prhs[i+1];
      if (option == "method")
        algParams.method = mxGetStdString(ptr);
      else if (option == "rol")
        algParams.rolfilename = mxGetStdString(ptr);
      else if (option == "type")
        algParams.loss_function_type =
          Genten::parse_enum<Genten::GCP_LossFunction>(mxGetStdString(ptr));
      else if (option == "tol")
        algParams.tol = mxGetScalar(ptr);
      else if (option == "eps")
        algParams.loss_eps = mxGetScalar(ptr);
      else if (option == "seed")
        algParams.seed = mxGetScalar(ptr);
      else if (option == "prng")
        algParams.prng = mxGetScalar(ptr);
      else if (option == "maxiters")
        algParams.maxiters = mxGetScalar(ptr);
      else if (option == "printitn")
        algParams.printitn = mxGetScalar(ptr);
      else if (option == "debug")
        algParams.debug = mxGetScalar(ptr);
      else if (option == "warmup")
        algParams.warmup = mxGetScalar(ptr);
      else if (option == "mttkrp_method")
        algParams.mttkrp_method =
          Genten::parse_enum<Genten::MTTKRP_Method>(mxGetStdString(ptr));
      else if (option == "mttkrp_tile_size")
        algParams.mttkrp_duplicated_factor_matrix_tile_size = mxGetScalar(ptr);
      else if (option == "init")
        u_init = mxGetKtensor<ExecSpace>(ptr, algParams.debug);
      else
        Genten::error("Invalid input option");
    }

    // Call driver
    Genten::KtensorT<ExecSpace> u =
      Genten::driver(X, u_init, algParams, std::cout);

    // Return results
    if (nlhs >= 1)
      plhs[0] = mxSetKtensor(u, algParams.debug);
    if (nlhs >= 2)
      plhs[1] = mxSetKtensor(u_init);
  }

  catch(std::string sExc)
  {
    std::cout << "*** Call to genten threw an exception:\n";
    std::cout << "  " << sExc << "\n";
  }
}

}
