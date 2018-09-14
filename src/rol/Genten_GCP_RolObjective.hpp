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
#include "Genten_Ktensor.hpp"
#include "Genten_RolKokkosVector.hpp"
#include "Genten_RolKtensorVector.hpp"
#include "Genten_MixedFormatOps.hpp"

#include "Teuchos_TimeMonitor.hpp"

// Choose implementation of ROL::Vector (KtensorVector or KokkosVector)
#define USE_KTENSOR_VECTOR 0

// Whether to copy the ROL::Vector into a new Ktensor before accessing data.
// This adds cost for the copy, but allows mttkrp to use a padded Ktensor
// when using RolKokkosVector.
#define COPY_KTENSOR 0

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
                     const loss_function_type& func) :
      X(x), Y(X.size(), X.getSubscripts()), M(m), f(func)
    {
#if COPY_KTENSOR
      const unsigned nd = M.ndims();
      const unsigned nc = M.ncomponents();
      G = ktensor_type(nc, nd);
      for (unsigned i=0; i<nd; ++i)
        G.set_factor(i, FacMatrixT<exec_space>(M[i].nRows(), nc));
#endif

      // Todo:  maybe do a deep copy instead so we don't have to resort?
      Y.fillComplete();
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
    loss_function_type f;

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

    // value = \sum_i f(X_i, M_i)
    // Todo:  make this a row-based kernel using TinyVec
    const ttb_indx nnz = X.nnz();
    const unsigned nd = M.ndims();
    const unsigned nc = M.ncomponents();
    tensor_type XX = X;  // Can't capture *this
    ktensor_type MM = M;
    loss_function_type ff = f;
    Kokkos::RangePolicy<exec_space> policy(0, nnz);
    ttb_real v = 0.0;
    Kokkos::parallel_reduce(policy, KOKKOS_LAMBDA(const ttb_indx i, ttb_real& d)
    {
      // Compute Ktensor value
      ttb_real m_val = 0.0;
      for (unsigned j=0; j<nc; ++j) {
        ttb_real tmp = MM.weights(j);
        for (unsigned m=0; m<nd; ++m)
          tmp *= MM[m].entry(XX.subscript(i,m),j);
        m_val += tmp;
      }

      // Evaluate link function
      d += ff.value(XX.value(i), m_val);
    }, v);

    v /= nnz;

    return v;
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

    // Compute Y tensor
    // Todo:  make this a row-based kernel using TinyVec
    //   or better yet, do this implicitly in the mttkrp calculation
    const ttb_indx nnz = X.nnz();
    const unsigned nd = M.ndims();
    const unsigned nc = M.ncomponents();

    {
      TEUCHOS_FUNC_TIME_MONITOR("GCP_RolObjective::gradient: Y eval");
      tensor_type XX = X;  // Can't capture *this
      tensor_type YY = Y;
      ktensor_type MM = M;
      loss_function_type ff = f;
      Kokkos::RangePolicy<exec_space> policy(0, nnz);
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ttb_indx i)
      {
        // Compute Ktensor value
        ttb_real m_val = 0.0;
        for (unsigned j=0; j<nc; ++j) {
          ttb_real tmp = MM.weights(j);
          for (unsigned m=0; m<nd; ++m)
            tmp *= MM[m].entry(XX.subscript(i,m),j);
          m_val += tmp;
        }

        // Evaluate link function derivative
        YY.value(i) = ff.deriv(XX.value(i), m_val) / nnz;
      });
    }

    {
      TEUCHOS_FUNC_TIME_MONITOR("GCP_RolObjective::gradient: mttkrp");
      // Compute gradient
      // Todo: new mttkrp kernel that does all nd dimensions
      G.weights() = 1.0;
      for (unsigned m=0; m<nd; ++m)
        mttkrp(Y, M, m, G[m]);
    }

    // Convert Ktensor to vector
#if COPY_KTENSOR
    g.copyFromKtensor(G);
#endif
  }

}
