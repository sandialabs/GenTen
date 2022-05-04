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

#include "Genten_Sptensor.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_Array.hpp"

namespace Genten {

template <typename Tensor, typename LossFunction,
          unsigned FacBlockSize, unsigned VectorSize>
class GCP_GradTensor : public Tensor
{
public:

  typedef typename Tensor::exec_space exec_space;

  // Create tensor from given data tensor
  GCP_GradTensor(const Tensor& X_, const KtensorT<exec_space>& M_,
                 const ArrayT<exec_space>& w_, const LossFunction& f_) :
    Tensor(X_), M(M_), w(w_), f(f_) {}

  // Default constructor
  GCP_GradTensor() = default;

  // Copy constructor.
  KOKKOS_INLINE_FUNCTION
  GCP_GradTensor(const GCP_GradTensor& arg) = default;

  // Assignment operator.
  KOKKOS_INLINE_FUNCTION
  GCP_GradTensor& operator=(const GCP_GradTensor& arg) = default;

  // Destructor.
  KOKKOS_INLINE_FUNCTION
  ~GCP_GradTensor() = default;

  // Return reference to i-th nonzero
  KOKKOS_INLINE_FUNCTION
  ttb_real value(ttb_indx i) const
  {
    static const bool is_gpu = Genten::is_gpu_space<exec_space>::value;
    static const unsigned WarpSize = is_gpu ? VectorSize : 1;
    const ttb_real m_val =
      compute_Ktensor_value<exec_space, FacBlockSize, WarpSize>(M, *this, i);
    return w[i] * f.deriv(Tensor::value(i), m_val);
  }

protected:

  KtensorT<exec_space> M;
  ArrayT<exec_space> w;
  LossFunction f;
};

}
