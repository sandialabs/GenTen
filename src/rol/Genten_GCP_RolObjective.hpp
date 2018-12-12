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

#include "ROL_Objective.hpp"
#include "Genten_RolKokkosVector.hpp"
#include "Genten_RolKtensorVector.hpp"

#include "Teuchos_TimeMonitor.hpp"

// Choose implementation of ROL::Vector (KtensorVector or KokkosVector)
#define USE_KTENSOR_VECTOR 0

// Whether to copy the ROL::Vector into a new Ktensor before accessing data.
// This adds cost for the copy, but allows mttkrp to use a padded Ktensor
// when using RolKokkosVector.
#define COPY_KTENSOR 0

// Whether to compute gradient tensor "Y" explicitly in GCP kernels
#define COMPUTE_Y 0
#include "Genten_GCP_Kernels.hpp"

namespace Genten {

  //! Implementation of ROL::Objective for GCP problem
  template <typename Tensor, typename LossFunction>
  class GCP_RolObjective : public ROL::Objective<ttb_real> {

  public:

    typedef Tensor tensor_type;
    typedef LossFunction loss_function_type;
    typedef typename tensor_type::exec_space exec_space;
    typedef KtensorT<exec_space> ktensor_type;
#if USE_KTENSOR_VECTOR
    typedef RolKtensorVector<exec_space> vector_type;
#else
    typedef RolKokkosVector<exec_space> vector_type;
#endif

    GCP_RolObjective(const tensor_type& x,
                     const ktensor_type& m,
                     const loss_function_type& func,
                     const AlgParams& algParms) :
      X(x), M(m), f(func), algParams(algParms),
      w(x.nnz(), ttb_real(1.0)/ttb_real(x.nnz()))
    {
#if COPY_KTENSOR
      const unsigned nd = M.ndims();
      const unsigned nc = M.ncomponents();
      G = ktensor_type(nc, nd);
      for (unsigned i=0; i<nd; ++i)
        G.set_factor(i, FacMatrixT<exec_space>(M[i].nRows(), nc));
#endif
    }

    virtual ~GCP_RolObjective() {}

    virtual ttb_real value(const ROL::Vector<ttb_real>& x, ttb_real& tol);

    virtual void gradient(ROL::Vector<ttb_real>& g,
                          const ROL::Vector<ttb_real>& x, ttb_real &tol);

    ROL::Ptr<vector_type> createDesignVector() const
    {
      return ROL::makePtr<vector_type>(M, false);
    }

  protected:

    tensor_type X;
    tensor_type Y;
    ktensor_type M;
    ktensor_type G;
    ArrayT<exec_space> w;
    loss_function_type f;
    AlgParams algParams;

  };

  template <typename Tensor, typename LossFunction>
  ttb_real
  GCP_RolObjective<Tensor,LossFunction>::
  value(const ROL::Vector<ttb_real>& xx, ttb_real& tol)
  {
    TEUCHOS_FUNC_TIME_MONITOR("GCP_RolObjective::value");

    const vector_type& x = dynamic_cast<const vector_type&>(xx);

    // Convert input vector to a Ktensor
    M = x.getKtensor();
#if COPY_KTENSOR
    x.copyToKtensor(M);
#endif

    return Impl::gcp_value(X, M, w, f);
  }

  template <typename Tensor, typename LossFunction>
  void
  GCP_RolObjective<Tensor,LossFunction>::
  gradient(ROL::Vector<ttb_real>& gg, const ROL::Vector<ttb_real>& xx,
           ttb_real &tol)
  {
    TEUCHOS_FUNC_TIME_MONITOR("GCP_RolObjective::gradient");

    const vector_type& x = dynamic_cast<const vector_type&>(xx);
    vector_type& g = dynamic_cast<vector_type&>(gg);

    // Convert input vector to a Ktensor
    M = x.getKtensor();
    G = g.getKtensor();
#if COPY_KTENSOR
    x.copyToKtensor(M);
#endif

    Impl::gcp_gradient(X, Y, M, w, f, G, algParams);

    // Convert Ktensor to vector
#if COPY_KTENSOR
    g.copyFromKtensor(G);
#endif
  }

}
