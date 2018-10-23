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

/*!
  @file Genten_MixedFormatOps.h
  @brief Methods that perform operations between objects of mixed format.
*/

#pragma once

#include "Genten_FacMatrix.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Sptensor_perm.hpp"
#include "Genten_Sptensor_row.hpp"
#include "Genten_Util.hpp"

namespace Genten
{
  //---- Methods for innerprod.

  // Inner product between a sparse tensor and a Ktensor.
  /* Compute the element-wise dot product of all elements.
   */
  template <typename ExecSpace>
  ttb_real innerprod(const Genten::SptensorT<ExecSpace>& s,
                     const Genten::KtensorT<ExecSpace>& u)
  {
    return innerprod(s, u, u.weights());
  }

  // Inner product between a sparse tensor and a Ktensor with weights.
  /* Compute the element-wise dot product of all elements, using an
   * alternate weight vector for the Ktensor.  Used by CpAls, which
   * separates the weights from the Ktensor while iterating.
   */
  template <typename ExecSpace>
  ttb_real innerprod(const Genten::SptensorT<ExecSpace>& s,
                     const Genten::KtensorT<ExecSpace>& u,
                     const Genten::ArrayT<ExecSpace>& lambda);


  //---- Methods for mttkrp.

  // Matricized sparse tensor times Khatri-Rao product.
  /* Same as below except that rather than overwrite u[n],
     the answer is put into v.

     Note: couldn't figure out how to get the ETI right with the
     SptensorT_row overload without ExecSpace as a second template paramter
     (as opposed to being deduced from SparseTensor. */
  template <typename SparseTensor, typename ExecSpace>
  void mttkrp(const SparseTensor& X,
              const Genten::KtensorT<ExecSpace>& u,
              const ttb_indx n,
              const Genten::FacMatrixT<ExecSpace>& v,
              const AlgParams& algParams = AlgParams());
  template <typename ExecSpace>
  void mttkrp(const Genten::SptensorT_row<ExecSpace>& X,
              const Genten::KtensorT<ExecSpace>& u,
              const ttb_indx n,
              const Genten::FacMatrixT<ExecSpace>& v,
              const AlgParams& algParams = AlgParams());

  // Matricized sparse tensor times Khatri-Rao product.
  /* Matricizes the Sptensor X along mode n, and computes the product
     of this with the factor matrices of Ktensor u, excluding mode n.
     Modes are indexed starting with 0.
     The result is scaled by the lambda weights and placed into u[n];
     hence, u[n] is not normalized.

     More specifically, the operation forms X_n * W_n, where X_n is the
     mode n matricization of X, and W_n is the cumulative Khatri-Rao product
     of all factor matrices in u except the nth.
  */
  template <typename SparseTensor>
  void mttkrp(const SparseTensor& X,
              const Genten::KtensorT<typename SparseTensor::exec_space>& u,
              const ttb_indx n,
              const AlgParams& algParams = AlgParams())
  {
    mttkrp (X, u, n, u[n], algParams);
    return;
  }

}     //-- namespace Genten
