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

#include <sstream>

#include "Genten_Matlab.hpp"

std::string
mxGetStdString(const mxArray* ptr) {
  const mwSize str_len = mxGetNumberOfElements(ptr);
  char *c_str = new char[str_len+1];
  int ret = mxGetString(ptr, c_str, str_len+1);
  if (ret != 0)
    Genten::error("mxGetString failed!");
  std::string str(c_str);
  delete [] c_str;
  return str;
}

std::vector<std::string>
mxBuildArgList(int nargs, int offset, const mxArray* margs[]) {
  std::vector<std::string> args(nargs-offset);
  for (int i=0; i<nargs-offset; ++i) {
    const mxArray *arg = margs[i+offset];
    if (mxIsScalar(arg))
      args[i] = std::to_string(mxGetScalar(arg));
    else if (mxIsChar(arg))
      args[i] = mxGetStdString(arg);
    else {
      std::stringstream ss;
      ss << "Unknown argument type for argument " << i+offset;
      Genten::error(ss.str());
    }
  }
  return args;
}

Genten::AlgParams
mxGetAlgParams(const mxArray* ptr) {
  std::vector<std::string> args;

  if (mxIsStruct(ptr)) {
    const int num_fields = mxGetNumberOfFields(ptr);
    args = std::vector<std::string>(2*num_fields);
    for (int i=0; i<num_fields; ++i) {
      std::string name = std::string(mxGetFieldNameByNumber(ptr, i));
      args[2*i] = name;
      const mxArray* arg = mxGetFieldByNumber(ptr, 0, i);
      if (mxIsScalar(arg))
        args[2*i+1] = std::to_string(mxGetScalar(arg));
      else if (mxIsChar(arg))
        args[2*i+1] = mxGetStdString(arg);
      else {
        Genten::error(
          std::string("Field ") + std::to_string(i) +
          std::string(" of struct with name ") + args[2*i] +
          std::string(" is not a scalar or string!"));
      }
    }
  }

  else if (mxIsCell(ptr)) {
    const int num_fields = mxGetNumberOfElements(ptr);
    if (num_fields % 2 == 1)
      Genten::error("algParams cell array must have an even length!");
    args = std::vector<std::string>(num_fields);
    for (int i=0; i<num_fields; i+=2) {
      const mxArray* cell = mxGetCell(ptr, i);
      if (!mxIsChar(cell))
        Genten::error(
          std::string("Entry ") + std::to_string(i) +
          std::string(" of algPrams cell array is not a string!"));
      args[i] = mxGetStdString(cell);
      const mxArray* arg = mxGetCell(ptr,i+1);
      if (mxIsScalar(arg))
        args[i+1] = std::to_string(mxGetScalar(arg));
      else if (mxIsChar(arg))
        args[i+1] = mxGetStdString(arg);
      else {
        Genten::error(
          std::string("Entry ") + std::to_string(i+1) +
          std::string(" of algParams cell array is not a scalar or string!"));
      }
    }
  }

  else
    Genten::error("algParams argument is not a struct or cell array!");

  Genten::AlgParams algParams;
  algParams.parse(args);
  return algParams;
}

void GentenInitialize() {
  // Initialize Kokkos
  if (!Kokkos::is_initialized()) {
    Kokkos::initialize();

    // Register at-exit function to finalize Kokkos
    mexAtExit(GentenAtExitFcn);
  }
}

void GentenAtExitFcn() {
  if (Kokkos::is_initialized())
    Kokkos::finalize();
}
