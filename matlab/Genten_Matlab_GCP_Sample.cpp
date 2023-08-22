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
#include "Genten_GCP_SamplingKernels.hpp"
#include "Genten_GCP_LossFunctions.hpp"

#include "Kokkos_Random.hpp"

extern "C" {

DLL_EXPORT_SYM void mexFunction(int nlhs, mxArray *plhs[],
                                int nrhs, const mxArray *prhs[])
{
  typedef Genten::DefaultExecutionSpace ExecSpace;
  typedef Genten::SptensorT<ExecSpace> Sptensor_type;
  typedef Genten::ArrayT<ExecSpace> array_type;

  GentenInitialize();

  try {
    if (nrhs < 6 || nlhs != 2) {
      std::string err = "Expected at least 6 input and 2 output arguments";
      throw err;
    }

    Genten::AlgParams algParams;
    algParams.fixup<ExecSpace>(std::cout);

    // Parse inputs
    const std::string method = mxGetStdString(prhs[0]);
    Sptensor_type X = mxGetSptensor<ExecSpace>(prhs[1]);
    const ttb_indx num_samples_nonzeros =
      static_cast<ttb_indx>(mxGetScalar(prhs[2]));
    const ttb_indx num_samples_zeros =
      static_cast<ttb_indx>(mxGetScalar(prhs[3]));
    const ttb_real weight_nonzeros =
      static_cast<ttb_real>(mxGetScalar(prhs[4]));
    const ttb_real weight_zeros =
      static_cast<ttb_real>(mxGetScalar(prhs[5]));
    if (nrhs >= 7) {
      auto args = mxBuildArgList(nrhs, 6, prhs);
      algParams.parse(args);
      if (Genten::check_and_print_unused_args(args, std::cout)) {
        algParams.print_help(std::cout);
        std::string err = "Invalid command line arguments.";
        throw err;
      }
    }

    // to do:  figure out how to reuse this across calls, as it is expensive
    // to recreate each time
    // to do:  sort/hash
    Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(rand());
    Genten::KtensorT<ExecSpace> u; // not used
    Genten::GaussianLossFunction loss_func(algParams); // not used
    Sptensor_type Xs;
    array_type w;
    if (method == "stratified") {
      if (!X.isSorted())
        X.sort();
      Genten::Impl::stratified_sample_tensor(
        X, Genten::Impl::SortSearcher<ExecSpace>(X.impl()),
        num_samples_nonzeros, num_samples_zeros,
        weight_nonzeros, weight_zeros,
        u, Genten::Impl::StratifiedGradient<Genten::GaussianLossFunction>(loss_func), false,
        Xs, w, rand_pool, algParams);
    }
    else if (method == "semi-stratified")
      Genten::Impl::stratified_sample_tensor(
        X, Genten::Impl::SemiStratifiedSearcher<ExecSpace>(),
        num_samples_nonzeros, num_samples_zeros,
        weight_nonzeros, weight_zeros,
        u, Genten::Impl::SemiStratifiedGradient<Genten::GaussianLossFunction>(loss_func), false,
        Xs, w, rand_pool, algParams);
    else {
      std::string err = "Unknown method " + method;
      throw err;
    }

    // Set output
    plhs[0] = mxSetSptensor(Xs);
    plhs[1] = mxSetArray(w);
  }

  catch(std::string sExc)
  {
    std::cout << "*** Call to genten threw an exception:\n";
    std::cout << "  " << sExc << "\n";
  }
}

}
