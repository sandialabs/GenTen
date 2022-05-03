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
#include "Genten_CP_Model.hpp"

#include "Teuchos_TimeMonitor.hpp"

// Choose implementation of ROL::Vector (KtensorVector or KokkosVector)
#define USE_KTENSOR_VECTOR 0

// Whether to copy the ROL::Vector into a new Ktensor before accessing data.
// This adds cost for the copy, but allows mttkrp to use a padded Ktensor
// when using RolKokkosVector.
#define COPY_KTENSOR 0

namespace Genten {

  //! Implementation of ROL::Objective for CP problem
  template <typename Tensor>
  class CP_RolObjective : public ROL::Objective<ttb_real> {

  public:

    typedef Tensor tensor_type;
    typedef typename tensor_type::exec_space exec_space;
    typedef KtensorT<exec_space> ktensor_type;
#if USE_KTENSOR_VECTOR
    typedef RolKtensorVector<exec_space> vector_type;
#else
    typedef RolKokkosVector<exec_space> vector_type;
#endif

    CP_RolObjective(const tensor_type& x, const ktensor_type& m,
                    const AlgParams& algParms);

    virtual ~CP_RolObjective() {}

    virtual void update(const ROL::Vector<ttb_real>& xx, ROL::UpdateType type,
                        int iter) override;

    virtual void update(const ROL::Vector<ttb_real> &xx, bool flag, int iter);

    virtual ttb_real value(const ROL::Vector<ttb_real>& x,
                           ttb_real& tol) override;

    virtual void gradient(ROL::Vector<ttb_real>& g,
                          const ROL::Vector<ttb_real>& x,
                          ttb_real &tol) override;

    virtual void hessVec(ROL::Vector<ttb_real>& hv,
                         const ROL::Vector<ttb_real>& v,
                         const ROL::Vector<ttb_real>& x,
                         ttb_real& tol ) override;

    ROL::Ptr<vector_type> createDesignVector() const
    {
      return ROL::makePtr<vector_type>(M, false);
    }

  protected:

    ktensor_type M, V, G;
    CP_Model<tensor_type> cp_model;

  };

  template <typename Tensor>
  CP_RolObjective<Tensor>::
  CP_RolObjective(const tensor_type& x,
                  const ktensor_type& m,
                  const AlgParams& algParams) : M(m), cp_model(x, m, algParams)
    {
      const ttb_indx nc = M.ncomponents();
      const ttb_indx nd = M.ndims();
#if COPY_KTENSOR
      V = ktensor_type(nc, nd, x.size());
      G = ktensor_type(nc, nd, x.size());
#endif
    }

  // Compute information that is common to both value() and gradient()
  // when a new design vector is computed
  template <typename Tensor>
  void
  CP_RolObjective<Tensor>::
  update(const ROL::Vector<ttb_real>& xx, ROL::UpdateType type, int iter)
  {
    TEUCHOS_FUNC_TIME_MONITOR("CP_RolObjective::update");

    const vector_type& x = dynamic_cast<const vector_type&>(xx);

    // Convert input vector to a Ktensor
    M = x.getKtensor();
#if COPY_KTENSOR
    x.copyToKtensor(M);
#endif

    cp_model.update(M);

  }

  template <typename Tensor>
  void
  CP_RolObjective<Tensor>::
  update(const ROL::Vector<ttb_real>& xx, bool flag, int iter)
  {
    update(xx, ROL::UpdateType::Accept, iter);
  }

  // Compute value of the objective function:
  //      0.5 * ||X-M||^2 = 0.5* (||X||^2 + ||M||^2) - <X,M>
  template <typename Tensor>
  ttb_real
  CP_RolObjective<Tensor>::
  value(const ROL::Vector<ttb_real>& xx, ttb_real& tol)
  {
    TEUCHOS_FUNC_TIME_MONITOR("CP_RolObjective::value");

    const vector_type& x = dynamic_cast<const vector_type&>(xx);

    // Convert input vector to a Ktensor
    M = x.getKtensor();
#if COPY_KTENSOR
    x.copyToKtensor(M);
#endif

    return cp_model.value(M);
  }

  // Compute gradient of objective function:
  //       G[m] = -X_(m)*Z_m*diag(w) + A_m*[diag(w)*Z_m^T*Z_m*diag(w)]
  template <typename Tensor>
  void
  CP_RolObjective<Tensor>::
  gradient(ROL::Vector<ttb_real>& gg, const ROL::Vector<ttb_real>& xx,
           ttb_real &tol)
  {
    TEUCHOS_FUNC_TIME_MONITOR("CP_RolObjective::gradient");

    const vector_type& x = dynamic_cast<const vector_type&>(xx);
    vector_type& g = dynamic_cast<vector_type&>(gg);

    // Convert input vector to a Ktensor
    M = x.getKtensor();
    G = g.getKtensor();
#if COPY_KTENSOR
    x.copyToKtensor(M);
#endif

    cp_model.gradient(G, M);

    // Convert Ktensor to vector
#if COPY_KTENSOR
    g.copyFromKtensor(G);
#endif
  }

  template <typename Tensor>
  void
  CP_RolObjective<Tensor>::
  hessVec(ROL::Vector<ttb_real>& hhv, const ROL::Vector<ttb_real>& vv,
          const ROL::Vector<ttb_real>& xx, ttb_real &tol)
  {
    TEUCHOS_FUNC_TIME_MONITOR("CP_RolObjective::hessVec");

    const vector_type& x = dynamic_cast<const vector_type&>(xx);
    const vector_type& v = dynamic_cast<const vector_type&>(vv);
    vector_type& hv = dynamic_cast<vector_type&>(hhv);

    // Convert input vector to a Ktensor
    M = x.getKtensor();
    V = v.getKtensor();
    G = hv.getKtensor();
#if COPY_KTENSOR
    x.copyToKtensor(M);
    v.copyToKtensor(V);
#endif

    cp_model.hess_vec(G, M, V);

    // Convert Ktensor to vector
#if COPY_KTENSOR
    hv.copyFromKtensor(G);
#endif
  }

}
