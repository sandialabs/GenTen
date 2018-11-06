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

#pragma once

#define DLL_EXPORT_SYM __attribute__ ((visibility("default")))

#include <string>

#include "Genten_Kokkos.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_IOtext.hpp"

extern "C" {
#include "mex.h"
}

template <typename ExecSpace>
Genten::SptensorT<ExecSpace>
mxGetSptensor(const mxArray *ptr, const bool print = false) {
  typedef Genten::SptensorT<ExecSpace> Sptensor_type;
  typedef Genten::Sptensor Sptensor_host_type;
  typedef typename Sptensor_host_type::exec_space host_exec_space;
  typedef typename Sptensor_host_type::subs_view_type subs_type;
  typedef typename Sptensor_host_type::vals_view_type vals_type;

  if (!mxIsClass(ptr, "sptensor") && !mxIsClass(ptr, "sptensor_gt"))
    Genten::error("Arg is not a sptensor or sptensor_gt!");

  Sptensor_host_type X_host;
  mxArray* vals_field = mxGetField(ptr, 0, "vals");
  mxArray* subs_field = mxGetField(ptr, 0, "subs");
  mxArray* size_field = mxGetField(ptr, 0, "size");
  ttb_real* vals = mxGetDoubles(vals_field);
  ttb_real* size = mxGetDoubles(size_field);
  ttb_indx nd = mxGetNumberOfElements(size_field);
  ttb_indx nz = mxGetNumberOfElements(vals_field);

  // Create sparse tensor from Tensor Toolbox sptensor format
  // (subs transposed, 1-based, stored as doubles)
  if (mxIsClass(ptr, "sptensor")) {
    ttb_real* subs = mxGetDoubles(subs_field);
    X_host = Sptensor_host_type(nd, size, nz, vals, subs);
  }

  // Create sparse tensor from Tensor Toolbox sptensor_gt format.
  // Here we just create a view with no copies.
  else if (mxIsClass(ptr, "sptensor_gt")) {
    ttb_indx* subs = mxGetUint64s(subs_field);
    Genten::IndxArrayT<host_exec_space> sz(nd, size);
    subs_type s(subs,nz,nd);
    vals_type v(vals,nz);
    X_host = Sptensor_host_type(sz, v, s);
  }

  Sptensor_type X = create_mirror_view(ExecSpace(), X_host);
  deep_copy(X, X_host);

  if (print)
    Genten::print_sptensor(X_host, std::cout, "X");

  return X;
}

template <typename ExecSpace>
mxArray* mxSetKtensor(const Genten::KtensorT<ExecSpace>& u,
                      const bool print = false) {
  typedef Genten::KtensorT<ExecSpace> Ktensor_type;
  typedef Genten::Ktensor Ktensor_host_type;
  typedef typename Ktensor_host_type::exec_space host_exec_space;

  // Copy u to host
  Ktensor_host_type u_host = create_mirror_view(host_exec_space(), u);
  deep_copy(u_host, u);

  const ttb_indx nd = u.ndims();
  const ttb_indx nc = u.ncomponents();

  if (print)
    Genten::print_ktensor(u_host, std::cout, "Solution");

  // Create factor matrix array (cell array in matlab)
  mxArray *cell_array_ptr = mxCreateCellMatrix( (mwSize) nd, (mwSize) 1 );
  for (ttb_indx i=0; i<nd; ++i) {
    const ttb_indx m = u_host[i].nRows();
    const ttb_indx n = u_host[i].nCols();
    mxArray *mat_ptr = mxCreateDoubleMatrix( (mwSize) m, (mwSize) n,
                                             mxREAL );
    ttb_real *mat = mxGetDoubles(mat_ptr);
    u_host[i].convertToCol(m,n,mat);
    mxSetCell(cell_array_ptr, (mwIndex) i, mat_ptr);
  }

  // Create weights array
  mxArray *lambda_ptr = mxCreateDoubleMatrix( (mwSize) nc, (mwSize) 1, mxREAL );
  ttb_real *lambda = mxGetDoubles(lambda_ptr);
  u_host.weights().copyTo(nc, lambda);

  // // Create Ktensor class
  // const char* fieldNames[] = { "lambda", "u" };
  // mxArray *struct_ptr = mxCreateStructMatrix( (mwSize) 1, (mwSize) 1,
  //                                             (mwSize) 2, fieldNames );
  // mxSetField(struct_ptr, (mwIndex) 0, "lambda", lambda_ptr);
  // mxSetField(struct_ptr, (mwIndex) 0, "u", cell_array_ptr);
  // mxSetClassName(struct_ptr, "Ktensor");

  // Create Ktensor class by calling Ktensor constructor
  mxArray *lhs[1];
  mxArray *rhs[2] = { lambda_ptr, cell_array_ptr };
  mexCallMATLAB(1, lhs, 2, rhs, "ktensor");

  return lhs[0];
}

template <typename ExecSpace>
Genten::KtensorT<ExecSpace>
mxGetKtensor(const mxArray* ptr, const bool print = false) {
  typedef Genten::KtensorT<ExecSpace> Ktensor_type;
  typedef Genten::Ktensor Ktensor_host_type;
  typedef typename Ktensor_host_type::exec_space host_exec_space;

  if (!mxIsStruct(ptr))
    Genten::error("Arg is not a struct!");

  // Weights lambda
  mxArray* lambda_field = mxGetField(ptr, 0, "lambda");
  ttb_real* lambda = mxGetDoubles(lambda_field);
  ttb_indx nc = mxGetNumberOfElements(lambda_field);

  // Cell array of factor matrices
  mxArray* u_field = mxGetField(ptr, 0, "u");
  ttb_indx nd = mxGetNumberOfElements(u_field);

  // Create Ktensor
  Ktensor_type u(nc, nd);
  Ktensor_host_type u_host = create_mirror_view(host_exec_space(), u);

  // Copy lambda
  u_host.weights().copyFrom(nc, lambda);

  // Copy factor matrices
  for (ttb_indx i=0; i<nd; ++i) {
    mxArray *mat = mxGetCell(u_field, (mwIndex) i);
    const ttb_indx m = mxGetM(mat);
    const ttb_indx n = mxGetN(mat);
    ttb_real *data_ptr = mxGetDoubles(mat);
    Genten::FacMatrixT<ExecSpace> fac_mat(m,n);
    Genten::FacMatrixT<host_exec_space> fac_mat_host(m,n,data_ptr);
    u.set_factor(i, fac_mat);
    u_host.set_factor(i, fac_mat_host);
  }

  if (print)
    Genten::print_ktensor(u_host, std::cout, "Initial guess");

  // Copy from host to device
  deep_copy(u, u_host);

  return u;
}

std::string mxGetStdString(const mxArray* ptr);

void GentenInitialize();

void GentenAtExitFcn();
