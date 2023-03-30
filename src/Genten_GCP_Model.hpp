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

#include "Genten_Tensor.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_Pmap.hpp"
#include "Genten_DistKtensorUpdate.hpp"
#include "Genten_GCP_ValueKernels.hpp"
#include "Genten_GCP_GradientKernels.hpp"
#include "Genten_GCP_Goal.hpp"

namespace Genten {

  // Encapsulation of objective function and derivatives of GCP model
  // optimization problem
  template <typename ExecSpace, typename LossFunction>
  class GCP_Model {

  public:

    typedef TensorT<ExecSpace> tensor_type;
    typedef LossFunction loss_function_type;
    typedef typename tensor_type::exec_space exec_space;
    typedef KtensorT<exec_space> ktensor_type;

    GCP_Model(const tensor_type& X, const ktensor_type& M,
              const loss_function_type& func, const AlgParams& algParms);

    ~GCP_Model();

    // Compute information that is common to both value() and gradient()
    // when a new design vector is computed
    void update(const ktensor_type& M);

    // Compute value of the objective function:
    ttb_real value(const ktensor_type& M) const;

    // Compute gradient of objective function:
    void gradient(ktensor_type& G, const ktensor_type& M) const;

    // Whether the ktensor is replicated across sub-grids
    const DistKtensorUpdate<exec_space> *getDistKtensorUpdate() const {
      return dku;
    }

    loss_function_type lossFunction() const { return f; }

    ttb_real computeFit(const ktensor_type& M) const;

  protected:

    tensor_type X;
    loss_function_type f;
    AlgParams algParams;
    ttb_real w;
    mutable tensor_type Y;

    DistKtensorUpdate<exec_space> *dku;
    mutable ktensor_type M_overlap, G_overlap;

    ttb_real nrm_X_sq;

    GCP_Goal<ExecSpace> *goal;
    ttb_real s;

  };

  template <typename ExecSpace, typename LossFunction>
  GCP_Model<ExecSpace,LossFunction>::
  GCP_Model(const tensor_type& x,
            const ktensor_type& M,
            const loss_function_type& func,
            const AlgParams& algParms) :
    X(x), f(func), algParams(algParms)
  {
    dku = createKtensorUpdate(x, M, algParams);
    G_overlap = dku->createOverlapKtensor(M);
    M_overlap = dku->createOverlapKtensor(M);
    dku->doImport(M_overlap, M);
    for (ttb_indx  i = 0; i < x.ndims(); i++)
    {
      if (x.size(i) != M_overlap[i].nRows())
        Genten::error("Genten::CP_Model - M and x have different size");
    }

    nrm_X_sq = x.global_norm();
    nrm_X_sq = nrm_X_sq*nrm_X_sq;

    goal = goalFactory(X, M_overlap, algParams);
    if (goal != nullptr) {
      // Scale statistical and goal terms so each is 0.5 at the initial guess
      ttb_real t = Impl::gcp_value(X, M_overlap, 1.0, f);
      if (t == 0.0)
        w = 1.0;
      else
        w = 0.5/t;

      goal->update(M_overlap);
      t = goal->value(M_overlap);
      if (t == 0.0)
        s = 1.0;
      else
        s = 0.5/t;
    }
    else
      // Scale statistical term by inverse of ||X||^2
      w = ttb_real(1.0)/nrm_X_sq;
  }

  template <typename ExecSpace, typename LossFunction>
  GCP_Model<ExecSpace,LossFunction>::
  ~GCP_Model()
  {
    delete dku;
    if (goal != nullptr)
      delete goal;
  }

  template <typename ExecSpace, typename LossFunction>
  void
  GCP_Model<ExecSpace,LossFunction>::
  update(const ktensor_type& M)
  {
    if (dku->overlapAliasesArg())
      M_overlap = dku->createOverlapKtensor(M);
    dku->doImport(M_overlap, M);
    if (goal != nullptr)
      goal->update(M_overlap);
  }

  template <typename ExecSpace, typename LossFunction>
  ttb_real
  GCP_Model<ExecSpace,LossFunction>::
  value(const ktensor_type& M) const
  {
    ttb_real F = Impl::gcp_value(X, M_overlap, w, f);
    if (goal != nullptr)
      F += goal->value(M_overlap)*s;
    return F;
  }

  template <typename ExecSpace, typename LossFunction>
  void
  GCP_Model<ExecSpace,LossFunction>::
  gradient(ktensor_type& G, const ktensor_type& M) const
  {
    if (dku->overlapAliasesArg())
      G_overlap = dku->createOverlapKtensor(G);

    Impl::gcp_gradient(X, Y, M_overlap, w, f, G_overlap, algParams);
    if (goal != nullptr)
      goal->gradient(G_overlap, M_overlap, s);
    dku->doExport(G, G_overlap);
  }

  template <typename ExecSpace, typename LossFunction>
  ttb_real
  GCP_Model<ExecSpace,LossFunction>::
  computeFit(const ktensor_type& M) const
  {
    const ttb_real nrm_M_sq = M.normFsq();
    const ttb_real dot = innerprod(X, M_overlap);
    return ttb_real(1.0) - (nrm_X_sq + nrm_M_sq - ttb_real(2.0)*dot)/nrm_X_sq;
  }

}
