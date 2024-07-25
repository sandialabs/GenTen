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
#include "Genten_Pmap.hpp"
#include "Genten_DistKtensorUpdate.hpp"

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

    ~CP_Model();

    // Compute information that is common to both value() and gradient()
    // when a new design vector is computed
    void update(const ktensor_type& M);

    // Compute value of the objective function:
    //      ||X-M||^2/||X||^2 = (||X||^2 + ||M||^2 - 2*<X,M>)/||X||^2
    ttb_real value(const ktensor_type& M) const;

    // Compute gradient of objective function:
    //       G[m] = 2*(-X_(m)*Z_m*diag(w) + A_m*[diag(w)*Z_m^T*Z_m*diag(w)])/||X||^2
    void gradient(ktensor_type& G, const ktensor_type& M) const;

    // Compute value and gradient together, allowing reuse of some information
    // between the two
    ttb_real value_and_gradient(ktensor_type& G, const ktensor_type& M) const;

    // Compute Hessian-vector product
    /* This method is not const because it may call a finite-difference
       approximation, which requires calling update() */
    void hess_vec(ktensor_type& U, const ktensor_type& M,
                  const ktensor_type& V);

    // Apply the preconditioner to the given vector
    void prec_vec(ktensor_type& U, const ktensor_type& M,
                  const ktensor_type& V);

    // Whether the ktensor is replicated across sub-grids
    const DistKtensorUpdate<exec_space> *getDistKtensorUpdate() const {
      return dku;
    }

  protected:

    tensor_type X;
    AlgParams algParams;

    ttb_real nrm_X_sq;
    std::vector< FacMatrixT<exec_space> > gram;
    std::vector< FacMatrixT<exec_space> > hada;
    ArrayT<exec_space> ones;

    DistKtensorUpdate<exec_space> *dku;
    mutable ktensor_type M_overlap, G_overlap, V_overlap, U_overlap;

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
    const ttb_real nrm_X = X.global_norm();
    nrm_X_sq = nrm_X*nrm_X;

    // Note gram and hada are not distributed, so we don't set the pmap
    gram.resize(nd);
    hada.resize(nd);
    for (ttb_indx i=0; i<nd; ++i) {
      gram[i] = FacMatrixT<exec_space>(nc,nc);
      hada[i] = FacMatrixT<exec_space>(nc,nc);
    }
    ones = ArrayT<exec_space>(nc, 1.0);

    dku = createKtensorUpdate(x, M, algParams);
    M_overlap = dku->createOverlapKtensor(M);
    G_overlap = dku->createOverlapKtensor(M);
    V_overlap = dku->createOverlapKtensor(M);
    U_overlap = dku->createOverlapKtensor(M);
    for (ttb_indx  i = 0; i < x.ndims(); i++)
    {
      if (x.size(i) != M_overlap[i].nRows())
        Genten::error("Genten::CP_Model - M and x have different size");
    }
  }

  template <typename Tensor>
  CP_Model<Tensor>::
  ~CP_Model()
  {
    delete dku;
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

    if (dku->overlapAliasesArg())
      M_overlap = dku->createOverlapKtensor(M);
    dku->doImport(M_overlap, M);
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
    const ttb_real ip = innerprod(X,M_overlap);

    ttb_real f = (nrm_X_sq + nrm_M_sq - ttb_real(2.0)*ip) / nrm_X_sq;

    if (algParams.penalty != ttb_real(0.0)) {
      for (ttb_indx n=0; n<nd; ++n)
        f += algParams.penalty * M[n].normFsq() / nrm_X_sq;
    }

    return f;
  }

  template <typename Tensor>
  void
  CP_Model<Tensor>::
  gradient(ktensor_type& G, const ktensor_type& M) const
  {
    if (dku->overlapAliasesArg())
      G_overlap = dku->createOverlapKtensor(G);
    mttkrp_all(X, M_overlap, G_overlap, algParams);
    dku->doExport(G, G_overlap);

    const ttb_indx nd = M.ndims();
    for (ttb_indx n=0; n<nd; ++n) {
      G[n].gemm(false, false, ttb_real(2.0)/nrm_X_sq, M[n], hada[n],
                -ttb_real(2.0)/nrm_X_sq);

      if (algParams.penalty != ttb_real(0.0))
        G[n].plus(M[n], algParams.penalty*ttb_real(2.0)/nrm_X_sq);
    }
  }

  template <typename Tensor>
  ttb_real
  CP_Model<Tensor>::
  value_and_gradient(ktensor_type& G, const ktensor_type& M) const
  {
    // MTTKRP
    if (dku->overlapAliasesArg())
      G_overlap = dku->createOverlapKtensor(G);
    mttkrp_all(X, M_overlap, G_overlap, algParams);
    dku->doExport(G, G_overlap);

    // <X,M> using 'MTTKRP' trick
    const ttb_indx nd = M.ndims();
    const ttb_real ip = M[nd-1].innerprod(G[nd-1], M.weights());

    // ||M||^2 = <M,M>
    const ttb_real nrm_M_sq = gram[nd-1].innerprod(hada[nd-1], ones);

    // Compute objective
    ttb_real f = (nrm_X_sq + nrm_M_sq - ttb_real(2.0)*ip) / nrm_X_sq;

    // Compute gradient
    for (ttb_indx n=0; n<nd; ++n) {
      G[n].gemm(false, false, ttb_real(2.0)/nrm_X_sq, M[n], hada[n],
                -ttb_real(2.0)/nrm_X_sq);

      if (algParams.penalty != ttb_real(0.0)) {
        f += algParams.penalty * M[n].normFsq() / nrm_X_sq;
        G[n].plus(M[n], algParams.penalty*ttb_real(2.0)/nrm_X_sq);
      }
    }

    return f;
  }

  template <typename Tensor>
  void
  CP_Model<Tensor>::
  hess_vec(ktensor_type& U, const ktensor_type& M, const ktensor_type& V)
  {
    const ttb_indx nd = M.ndims();

    if (algParams.hess_vec_method == Hess_Vec_Method::Full) {
      if (dku->overlapAliasesArg()) {
        V_overlap = dku->createOverlapKtensor(V);
        U_overlap = dku->createOverlapKtensor(U);
      }
      Genten::hess_vec(X, M, V, U, M_overlap, V_overlap, U_overlap, *dku,
                       algParams);
      for (ttb_indx n=0; n<nd; ++n)
        U[n].times(ttb_real(2.0)/nrm_X_sq);
    }
    else if (algParams.hess_vec_method == Hess_Vec_Method::GaussNewton) {
      gauss_newton_hess_vec(X, M, V, U, algParams);
      for (ttb_indx n=0; n<nd; ++n)
        U[n].times(ttb_real(2.0)/nrm_X_sq);
    }
    else if (algParams.hess_vec_method == Hess_Vec_Method::FiniteDifference)
    {
      const ttb_real h = 1.0e-7;
      const ttb_indx nc = M.ncomponents();

      KtensorT<exec_space> Mp(nc, nd, X.size(), M.getProcessorMap()),
        Up(nc, nd, X.size(), U.getProcessorMap());
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

    template <typename Tensor>
  void
  CP_Model<Tensor>::
  prec_vec(ktensor_type& U, const ktensor_type& M, const ktensor_type& V)
  {
    if (algParams.hess_vec_prec_method == Hess_Vec_Prec_Method::ApproxBlockDiag)
      blk_diag_prec_vec(X, M, V, U, algParams);
    else if (algParams.hess_vec_prec_method == Hess_Vec_Prec_Method::None)
      deep_copy(U, V);
    else
      Genten::error("Unknown hess-vec preconditioner method");
  }

}
