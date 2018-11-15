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
#include "Genten_SystemTimer.hpp"

extern "C" {

DLL_EXPORT_SYM void mexFunction(int nlhs, mxArray *plhs[],
                                int nrhs, const mxArray *prhs[])
{
  typedef Genten::DefaultExecutionSpace ExecSpace;
  typedef Genten::SptensorT<ExecSpace> Sptensor_type;
  typedef Genten::Sptensor Sptensor_host_type;
  typedef typename Sptensor_type::subs_view_type subs_type;
  typedef typename Sptensor_host_type::subs_view_type host_subs_type;

  GentenInitialize();

  try {
    if (nrhs != 1 || nlhs != 1) {
      std::cout << "Expected exactly 1 input and output argument" << std::endl;
      return;
    }

    // Parse inputs
    Sptensor_type X = mxGetSptensor<ExecSpace>(prhs[0]);

    // Create permutation
    Genten::SystemTimer timer(1);
    timer.start(0);
    X.createPermutation();
    timer.stop(0);
    std::cout << "createPermutation() took " << timer.getTotalTime(0)
              << " seconds\n";

    // Create space in Matlab to store perm -- tranpose indices because
    // Matlab uses columnwise layout (LayoutLeft)
    subs_type perm = X.getPerm();
    const mwSize m = perm.extent(0);
    const mwSize n = perm.extent(1);
    plhs[0] = mxCreateNumericMatrix(n, m, mxUINT64_CLASS, mxREAL);

    // Copy into Matlab array
    ttb_indx* ptr = mxGetUint64s(plhs[0]);
    host_subs_type perm_matlab(ptr,m,n);
    deep_copy(perm_matlab, perm);
  }

  catch(std::string sExc)
  {
    std::cout << "*** Call to genten threw an exception:\n";
    std::cout << "  " << sExc << "\n";
  }
}

}
