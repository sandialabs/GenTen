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

#include "Genten_Ktensor.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_HessVec.hpp"

namespace Genten {

  // Encapsulation of objective function and derivatives of (Gaussian) CP model
  // optimization problem
  template <typename Tensor>
  class CP_Model {

  public:

    typedef Tensor tensor_type;
    typedef typename tensor_type::exec_space exec_space;
    typedef KtensorT<exec_space> ktensor_type;

    CP_Model(const tensor_type& X, const ktensor_type& M,
             const AlgParams& algParms);

    ~CP_Model() {}

    // Compute information that is common to both value() and gradient()
    // when a new design vector is computed
    void update(const ktensor_type& M);

    // Compute value of the objective function:
    //      0.5 * ||X-M||^2 = 0.5* (||X||^2 + ||M||^2) - <X,M>
    ttb_real value(const ktensor_type& M) const;

    // Compute gradient of objective function:
    //       G[m] = -X_(m)*Z_m*diag(w) + A_m*[diag(w)*Z_m^T*Z_m*diag(w)]
    void gradient(ktensor_type& G, const ktensor_type& M) const;

    // Compute value and gradient together, allowing reuse of some information
    // between the two
    ttb_real value_and_gradient(ktensor_type& G, const ktensor_type& M) const;

    // Compute Hessian-vector product
    /* This method is not const because it may call a finite-difference
       approximation, which requires calling update() */
    void hess_vec(ktensor_type& U, const ktensor_type& M,
                  const ktensor_type& V);

  protected:

    tensor_type X;
    AlgParams algParams;

    ttb_real nrm_X_sq;
    std::vector< FacMatrixT<exec_space> > gram;
    std::vector< FacMatrixT<exec_space> > hada;
    ArrayT<exec_space> ones;

  };

  template <typename Tensor>
  CP_Model<Tensor>::
  CP_Model(const tensor_type& x,
           const ktensor_type& M,
           const AlgParams& algParms) :
      X(x), algParams(algParms)
    {
      const ttb_indx nc = M.ncomponents();
      const ttb_indx nd = M.ndims();
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

  template <typename Tensor>
  void
  CP_Model<Tensor>::
  update(const ktensor_type& M)
  {
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
  ttb_real
  CP_Model<Tensor>::
  value(const ktensor_type& M) const
  {
    // ||M||^2 = <M,M>
    const ttb_indx nd = M.ndims();
    const ttb_real nrm_M_sq = gram[nd-1].innerprod(hada[nd-1], ones);

    // <X,M>.  Unfortunately can't use MTTKRP trick since we don't want to
    // compute MTTKRP in update()
    const ttb_real ip = innerprod(X,M);

    return ttb_real(0.5)*(nrm_X_sq + nrm_M_sq) - ip;
  }

  template <typename Tensor>
  void
  CP_Model<Tensor>::
  gradient(ktensor_type& G, const ktensor_type& M) const
  {
    const ttb_indx nd = M.ndims();
    mttkrp_all(X, M, G, algParams);
    for (ttb_indx n=0; n<nd; ++n)
      G[n].gemm(false, false, ttb_real(1.0), M[n], hada[n], ttb_real(-1.0));
  }

  template <typename Tensor>
  ttb_real
  CP_Model<Tensor>::
  value_and_gradient(ktensor_type& G, const ktensor_type& M) const
  {
    // MTTKRP
    mttkrp_all(X, M, G, algParams);

    // <X,M> using 'MTTKRP' trick
    const ttb_indx nd = M.ndims();
    const ttb_real ip = M[nd-1].innerprod(G[nd-1], M.weights());

    // ||M||^2 = <M,M>
    const ttb_real nrm_M_sq = gram[nd-1].innerprod(hada[nd-1], ones);

    // Compute gradient
    for (ttb_indx n=0; n<nd; ++n)
      G[n].gemm(false, false, ttb_real(1.0), M[n], hada[n], ttb_real(-1.0));

    // Compute objective
    return ttb_real(0.5)*(nrm_X_sq + nrm_M_sq) - ip;
  }

  template <typename Tensor>
  void
  CP_Model<Tensor>::
  hess_vec(ktensor_type& U, const ktensor_type& M, const ktensor_type& V)
  {
    if (algParams.hess_vec_method == Hess_Vec_Method::Full)
      Genten::hess_vec(X, M, V, U, algParams);
    else if (algParams.hess_vec_method == Hess_Vec_Method::GaussNewton)
      Genten::error("Gauss-Newton Hessian approximation not implemented");
    else if (algParams.hess_vec_method == Hess_Vec_Method::FiniteDifference)
    {
      const ttb_real h = 1.0e-7;
      const ttb_indx nc = M.ncomponents();
      const ttb_indx nd = M.ndims();

      KtensorT<exec_space> Mp(nc, nd, X.size()), Up(nc, nd, X.size());
      Mp.setWeights(1.0);
      U.setWeights(1.0);
      for (ttb_indx n=0; n<nd; ++n) {
        deep_copy(Mp[n], M[n]);
        Mp[n].update(h, V[n], 1.0);
      }

      update(M);
      gradient(U, M);
      update(Mp);
      gradient(Up, Mp);

      for (ttb_indx n=0; n<nd; ++n)
        U[n].update(1.0/h, Up[n], -1.0/h);
    }
    else
      Genten::error("Unknown Hessian method");
  }

}
