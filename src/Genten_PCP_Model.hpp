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
#include "Genten_Pmap.hpp"
#include "Genten_DistKtensorUpdate.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_GCP_Model.hpp"
#include "Genten_GCP_PoissonLossFunction.hpp"

namespace Genten {

  namespace Impl {

    // Modified Poisson loss function that removes the first (model) term for use with sparse tensors
    class ModifiedPoissonLossFunction {
    public:
      ModifiedPoissonLossFunction(const AlgParams& algParams) : eps(algParams.loss_eps) {}

      std::string name() const { return "Modified Poisson (count)"; }

      KOKKOS_INLINE_FUNCTION
      ttb_real value(const ttb_real& x, const ttb_real& m) const {
  #if defined(__SYCL_DEVICE_ONLY__)
        using sycl::log;
  #else
        using std::log;
  #endif

        return -x*log(m+eps);
      }

      KOKKOS_INLINE_FUNCTION
      ttb_real deriv(const ttb_real& x, const ttb_real& m) const {
        return -x/(m+eps);
      }

      KOKKOS_INLINE_FUNCTION static constexpr bool has_lower_bound() { return true; }
      KOKKOS_INLINE_FUNCTION static constexpr bool has_upper_bound() { return false; }
      KOKKOS_INLINE_FUNCTION static constexpr ttb_real lower_bound() { return 0.0; }
      KOKKOS_INLINE_FUNCTION static constexpr ttb_real upper_bound() { return DOUBLE_MAX; }

    private:
      ttb_real eps;
    };

  }

  // Encapsulation of objective function and derivatives of (Poisson) CP model
  // optimization problem
  template <typename Tensor>
  class PCP_Model {};

  template <typename ExecSpace>
  class PCP_Model< SptensorT<ExecSpace> > {
  public:

    typedef SptensorT<ExecSpace> tensor_type;
    typedef typename tensor_type::exec_space exec_space;
    typedef KtensorT<exec_space> ktensor_type;

    PCP_Model(const tensor_type& X, const ktensor_type& M,
              const AlgParams& algParms);

    ~PCP_Model();

    // Compute information that is common to both value() and gradient()
    // when a new design vector is computed
    void update(const ktensor_type& M);

    // Compute value of the objective function
    ttb_real value(const ktensor_type& M) const;

    // Compute gradient of objective function
    void gradient(ktensor_type& G, const ktensor_type& M) const;

    // Compute value and gradient together, allowing reuse of some information
    // between the two
    ttb_real value_and_gradient(ktensor_type& G, const ktensor_type& M) const;

    // Compute Hessian-vector product
    /* This method is not const because it may call a finite-difference
       approximation, which requires calling update() */
    void hess_vec(ktensor_type& U, const ktensor_type& M, const ktensor_type& V);

    // Apply the preconditioner to the given vector
    void prec_vec(ktensor_type& U, const ktensor_type& M, const ktensor_type& V);

    // Whether the ktensor is replicated across sub-grids
    const DistKtensorUpdate<exec_space> *getDistKtensorUpdate() const {
      return dku;
    }

  protected:

    tensor_type X;
    AlgParams algParams;
    Impl::ModifiedPoissonLossFunction f;

    std::vector< ArrayT<exec_space> > col_sums;
    tensor_type Y;
    ArrayT<exec_space> w;

    DistKtensorUpdate<exec_space> *dku;
    mutable ktensor_type M_overlap, G_overlap;
    //mutable ktensor_type V_overlap, U_overlap;

  };

  template <typename ExecSpace>
  class PCP_Model< TensorT<ExecSpace> > {
  public:

    typedef TensorT<ExecSpace> tensor_type;
    typedef typename tensor_type::exec_space exec_space;
    typedef KtensorT<exec_space> ktensor_type;

    PCP_Model(const tensor_type& X, const ktensor_type& M, const AlgParams& algParms) :
      gcp_model(X,M,PoissonLossFunction(algParms),algParms) {}

    ~PCP_Model() {}

    // Compute information that is common to both value() and gradient()
    // when a new design vector is computed
    void update(const ktensor_type& M) { 
      gcp_model.update(M);
    }

    // Compute value of the objective function
    ttb_real value(const ktensor_type& M) const { 
      return gcp_model.value(M);
    }

    // Compute gradient of objective function
    void gradient(ktensor_type& G, const ktensor_type& M) const { 
      gcp_model.gradient(G,M); 
    }

    // Compute value and gradient together, allowing reuse of some information
    // between the two
    ttb_real value_and_gradient(ktensor_type& G, const ktensor_type& M) const {
      ttb_real F = gcp_model.value(M);
      gcp_model.gradient(G,M);
      return F;
    }

    // Compute Hessian-vector product
    /* This method is not const because it may call a finite-difference
       approximation, which requires calling update() */
    void hess_vec(ktensor_type& U, const ktensor_type& M, const ktensor_type& V) {
      Genten::error("Hessian method not implemented");
    }

    // Apply the preconditioner to the given vector
    void prec_vec(ktensor_type& U, const ktensor_type& M, const ktensor_type& V) {
      Genten::error("Hessian method not implemented");
    }

    // Whether the ktensor is replicated across sub-grids
    const DistKtensorUpdate<exec_space> *getDistKtensorUpdate() const {
      return gcp_model.getDistKtensorUpdate();
    }

  protected:

    GCP_Model< ExecSpace, PoissonLossFunction> gcp_model;

  };

}
