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

// Mex Driver for computing permutation arrays

#include "Genten_Matlab.hpp"
#include "Genten_GCP_ValueKernels.hpp"
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_AlgParams.hpp"

extern "C" {

DLL_EXPORT_SYM void mexFunction(int nlhs, mxArray *plhs[],
                                int nrhs, const mxArray *prhs[])
{
  typedef Genten::DefaultExecutionSpace ExecSpace;
  typedef Genten::SptensorT<ExecSpace> Sptensor_type;
  typedef Genten::KtensorT<ExecSpace> Ktensor_type;
  typedef Genten::ArrayT<ExecSpace> array_type;

  GentenInitialize();

  try {
    if (nrhs < 4 || nlhs != 1) {
      std::string err = "Expected at least 4 input and 1 output arguments";
      throw err;
    }

    Genten::AlgParams algParams;
    algParams.fixup<ExecSpace>(std::cout);

    // Parse inputs
    const Sptensor_type X = mxGetSptensor<ExecSpace>(prhs[0]);
    const Ktensor_type u = mxGetKtensor<ExecSpace>(prhs[1]);
    const array_type w = mxGetArray<ExecSpace>(prhs[2]);
    const std::string loss_type =  mxGetStdString(prhs[3]);
    auto args = mxBuildArgList(nrhs, 4, prhs);
    algParams.parse(args);
    if (Genten::check_and_print_unused_args(args, std::cout)) {
      algParams.print_help(std::cout);
      std::string err = "Invalid command line arguments.";
      throw err;
    }

    ttb_real fest = 0.0;
    if (loss_type == "gaussian" || loss_type == "normal")
      fest = Genten::Impl::gcp_value(
        X, u, w, Genten::GaussianLossFunction(algParams.loss_eps));
    else {
      std::string err = "Unknown loss function " + loss_type;
      throw err;
    }

    // Set output
    mxArray *mat_ptr = mxCreateDoubleMatrix( (mwSize) 1, (mwSize) 1,  mxREAL );
    ttb_real *mat = mxGetDoubles(mat_ptr);
    *mat = fest;
    plhs[0] = mat_ptr;
  }

  catch(std::string sExc)
  {
    std::cout << "*** Call to genten threw an exception:\n";
    std::cout << "  " << sExc << "\n";
  }
}

}
