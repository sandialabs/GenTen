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

// Array of structs version
// mxArray* mxSetHistory(const Genten::PerfHistory& h)
// {
//   const char* fieldNames[] = { "iteration", "residual", "fit", "grad_norm", "cum_time", "mttkrp_throughput" };
//   const ttb_indx m = h.size();
//   const ttb_indx n = 1;
//   const ttb_indx nfields = 6;
//   mxArray *struct_ptr = mxCreateStructMatrix(
//     (mwSize) m, (mwSize) n,  (mwSize) nfields, fieldNames );
//   for (ttb_indx i=0; i<m; ++i) {
//     mxArray *it_ptr     = mxCreateDoubleScalar((double) h[i].iteration);
//     mxArray *f_ptr      = mxCreateDoubleScalar((double) h[i].residual);
//     mxArray *fit_ptr    = mxCreateDoubleScalar((double) h[i].fit);
//     mxArray *g_ptr      = mxCreateDoubleScalar((double) h[i].grad_norm);
//     mxArray *t_ptr      = mxCreateDoubleScalar((double) h[i].cum_time);
//     mxArray *mttkrp_ptr = mxCreateDoubleScalar((double) h[i].mttkrp_throughput);
//     mxSetField(struct_ptr, (mwIndex) i, "iteration", it_ptr);
//     mxSetField(struct_ptr, (mwIndex) i, "residual", f_ptr);
//     mxSetField(struct_ptr, (mwIndex) i, "fit", fit_ptr);
//     mxSetField(struct_ptr, (mwIndex) i, "grad_norm", g_ptr);
//     mxSetField(struct_ptr, (mwIndex) i, "cum_time", t_ptr);
//     mxSetField(struct_ptr, (mwIndex) i, "mttkrp_throughput", mttkrp_ptr);
//   }

//   return struct_ptr;
// }

// Struct-of-arrays version
mxArray* mxSetHistory(const Genten::PerfHistory& h)
{
  const char* fieldNames[] = { "iteration", "residual", "fit", "grad_norm", "cum_time", "mttkrp_throughput" };
  const ttb_indx m = h.size();
  const ttb_indx nfields = 6;
  mxArray *struct_ptr = mxCreateStructMatrix(
    (mwSize) 1, (mwSize) 1,  (mwSize) nfields, fieldNames );
  mxArray *it_ptr     = mxCreateDoubleMatrix((mwSize) m, (mwSize) 1, mxREAL);
  mxArray *f_ptr      = mxCreateDoubleMatrix((mwSize) m, (mwSize) 1, mxREAL);
  mxArray *fit_ptr    = mxCreateDoubleMatrix((mwSize) m, (mwSize) 1, mxREAL);
  mxArray *g_ptr      = mxCreateDoubleMatrix((mwSize) m, (mwSize) 1, mxREAL);
  mxArray *t_ptr      = mxCreateDoubleMatrix((mwSize) m, (mwSize) 1, mxREAL);
  mxArray *mttkrp_ptr = mxCreateDoubleMatrix((mwSize) m, (mwSize) 1, mxREAL);
  ttb_real *it     = mxGetDoubles(it_ptr);
  ttb_real *f      = mxGetDoubles(f_ptr);
  ttb_real *fit    = mxGetDoubles(fit_ptr);
  ttb_real *g      = mxGetDoubles(g_ptr);
  ttb_real *t      = mxGetDoubles(t_ptr);
  ttb_real *mttkrp = mxGetDoubles(mttkrp_ptr);
  for (ttb_indx i=0; i<m; ++i) {
    it[i] = h[i].iteration;
    f[i] = h[i].residual;
    fit[i] = h[i].fit;
    g[i] = h[i].grad_norm;
    t[i] = h[i].cum_time;
    mttkrp[i] = h[i].mttkrp_throughput;
  }
  mxSetField(struct_ptr, (mwIndex) 0, "iteration", it_ptr);
  mxSetField(struct_ptr, (mwIndex) 0, "residual", f_ptr);
  mxSetField(struct_ptr, (mwIndex) 0, "fit", fit_ptr);
  mxSetField(struct_ptr, (mwIndex) 0, "grad_norm", g_ptr);
  mxSetField(struct_ptr, (mwIndex) 0, "cum_time", t_ptr);
  mxSetField(struct_ptr, (mwIndex) 0, "mttkrp_throughput", mttkrp_ptr);

  return struct_ptr;
}

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
