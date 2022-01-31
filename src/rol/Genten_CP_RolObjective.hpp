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
#include "Genten_MixedFormatOps.hpp"

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

    ROL::Ptr<vector_type> createDesignVector() const
    {
      return ROL::makePtr<vector_type>(M, false);
    }

  protected:

    tensor_type X;
    ktensor_type M;
    ktensor_type G;
    AlgParams algParams;

    ttb_real nrm_X_sq;
    std::vector< FacMatrixT<exec_space> > gram;
    std::vector< FacMatrixT<exec_space> > hada;
    ArrayT<exec_space> ones;

  };

  template <typename Tensor>
  CP_RolObjective<Tensor>::
  CP_RolObjective(const tensor_type& x,
                  const ktensor_type& m,
                  const AlgParams& algParms) :
      X(x), M(m), algParams(algParms)
    {
      const ttb_indx nc = M.ncomponents();
      const ttb_indx nd = M.ndims();
#if COPY_KTENSOR
      G = ktensor_type(nc, nd);
      for (ttb_indx i=0; i<nd; ++i)
        G.set_factor(i, FacMatrixT<exec_space>(M[i].nRows(), nc));
#endif

      const ttb_real nrm_X = X.norm();
      nrm_X_sq = nrm_X*nrm_X;

      gram.resize(nd);
      hada.resize(nd);
      for (ttb_indx i=0; i<nd; ++i) {
        gram[i] = FacMatrixT<exec_space>(nc,nc);
        hada[i] = FacMatrixT<exec_space>(nc,nc);
      }
      ones = ArrayT<exec_space>(nc, 1.0);
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

    // Gram matrix for each mode
    const ttb_indx nd = M.ndims();
    for (ttb_indx n=0; n<nd; ++n)
      gram[n].gramian(M[n], true, Upper);

    // Hadamard product of gram matrices
    for (ttb_indx n=0; n<nd; ++n) {
      hada[n].oprod(M.weights());
      for (ttb_indx m=0; m<nd; ++m) {
        if (n != m)
          hada[n].times(gram[m]);
      }
    }

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

    // ||M||^2 = <M,M>
    const ttb_indx nd = M.ndims();
    const ttb_real nrm_M_sq = gram[nd-1].innerprod(hada[nd-1], ones);

    // <X,M>.  Unfortunately can't use MTTKRP trick since we don't want to
    // compute MTTKRP in update()
    const ttb_real ip = innerprod(X,M);

    return ttb_real(0.5)*(nrm_X_sq + nrm_M_sq) - ip;
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

    const ttb_indx nd = M.ndims();
    mttkrp_all(X, M, G, algParams);
    for (ttb_indx n=0; n<nd; ++n)
      G[n].gemm(false, false, ttb_real(1.0), M[n], hada[n], ttb_real(-1.0));

    // Convert Ktensor to vector
#if COPY_KTENSOR
    g.copyFromKtensor(G);
#endif
  }

}
