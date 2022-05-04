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


#include <iostream>
#include <cmath>
#include "Genten_Array.hpp"
#include "Genten_Test_Utils.hpp"
#include "Genten_Util.hpp"
#include "Genten_HigherMoments.hpp"

using namespace Genten::Test;

template <typename ExecSpace>
void Genten_Test_MomentTensorImpl(int infolevel)
{
  //using exec_space = ExecSpace;;
  using host_exec_space = Genten::DefaultHostExecutionSpace;
  //SETUP_DISABLE_CERR;

  std::string space_name = Genten::SpaceProperties<ExecSpace>::name();
  initialize("Tests on Genten::MomentTensor (" + space_name + ")", infolevel);

  const int numRows = 10;
  const int numCols = 4;
  std::vector<ttb_real> a(numRows*numCols);
  int count=0;
  // this loop order matters because the data must be in layoutleft order
  for(int j=0; j<numCols; j++){
    for (int i=0; i<numRows; i++){
      a[count++] = static_cast <double> (i*numCols+j);
    }
  }

  auto r = Genten::create_and_compute_raw_moment_tensor(a.data(), numRows, numCols);

  finalize();
}

void Genten_MomentTensor(int infolevel) {

// #ifdef KOKKOS_ENABLE_CUDA
//   Genten_Test_MomentTensorImpl<Kokkos::Cuda>(infolevel);
// #endif
// #ifdef KOKKOS_ENABLE_HIP
//   Genten_Test_MomentTensorImpl<Kokkos::Experimental::HIP>(infolevel);
// #endif
// #ifdef KOKKOS_ENABLE_OPENMP
  Genten_Test_MomentTensorImpl<Kokkos::OpenMP>(infolevel);
// #endif
// #ifdef KOKKOS_ENABLE_THREADS
//   Genten_Test_MomentTensorImpl<Kokkos::Threads>(infolevel);
// #endif
// #ifdef KOKKOS_ENABLE_SERIAL
//   Genten_Test_MomentTensorImpl<Kokkos::Serial>(infolevel);
// #endif
}
