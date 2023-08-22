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

#include "Genten_GCP_RolObjective.hpp"
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_GCP_Model.hpp"
#include "Genten_SystemTimer.hpp"

#include "Teuchos_TimeMonitor.hpp"

namespace Genten {

  template <typename ExecSpace, typename LossFunction>
  class GCP_RolObjective : public GCP_RolObjectiveBase<ExecSpace> {
  public:

    typedef ExecSpace exec_space;
    typedef LossFunction loss_function_type;
    typedef GCP_RolObjectiveBase<exec_space> base_type;
    typedef TensorT<ExecSpace> tensor_type;
    typedef KtensorT<exec_space> ktensor_type;
    typedef typename base_type::vector_type vector_type;

    GCP_RolObjective(const tensor_type& x,
                     const ktensor_type& m,
                     const loss_function_type& func,
                     const AlgParams& algParams,
                     PerfHistory& h);

    virtual ~GCP_RolObjective() {}

    virtual void update(const ROL::Vector<ttb_real>& xx, ROL::UpdateType type,
                          int iter) override;

    virtual void update(const ROL::Vector<ttb_real> &xx, bool flag, int iter) override;

    virtual ttb_real value(const ROL::Vector<ttb_real>& x, ttb_real& tol) override;

    virtual void gradient(ROL::Vector<ttb_real>& g,
                          const ROL::Vector<ttb_real>& x, ttb_real &tol) override;

    ROL::Ptr<vector_type> createDesignVector() const override {
      return ROL::makePtr<vector_type>(
        M, false, gcp_model.getDistKtensorUpdate());
    }

    std::string lossFunctionName() const override {
      return gcp_model.lossFunction().name();
    }

    bool lossFunctionHasLowerBound() const override {
      return gcp_model.lossFunction().has_lower_bound();
    }

    bool lossFunctionHasUpperBound() const override {
      return gcp_model.lossFunction().has_upper_bound();
    }

    ttb_real lossFunctionLowerBound() const override {
      return gcp_model.lossFunction().lower_bound();
    }

    ttb_real lossFunctionUpperBound() const override {
      return gcp_model.lossFunction().upper_bound();
    }

    ttb_real computeFit(const ktensor_type& u) override;

  protected:

    ktensor_type M, G;
    GCP_Model<exec_space, loss_function_type> gcp_model;
    PerfHistory& history;
    SystemTimer timer;
    bool compute_fit;
  };

  template <typename ExecSpace, typename LossFunction>
  GCP_RolObjective<ExecSpace,LossFunction>::
  GCP_RolObjective(const tensor_type& x,
                   const ktensor_type& m,
                   const loss_function_type& func,
                   const AlgParams& algParams,
                   PerfHistory& h) :
    M(m), gcp_model(x, m, func, algParams), history(h), timer(1),
    compute_fit(algParams.compute_fit)
  {
#if COPY_KTENSOR
    const ttb_indx nc = M.ncomponents();
    const ttb_indx nd = M.ndims();
    G = ktensor_type(nc, nd, x.size(), M.getProcessorMap());
#endif
    timer.start(0);

    history.addEmpty();
    history.lastEntry().iteration = 0;
  }

  // Compute information that is common to both value() and gradient()
  // when a new design vector is computed
  template <typename ExecSpace, typename LossFunction>
  void
  GCP_RolObjective<ExecSpace,LossFunction>::
  update(const ROL::Vector<ttb_real>& xx, ROL::UpdateType type, int iter)
  {
    GENTEN_TIME_MONITOR("GCP_RolObjective::update");

    gcp_model.update(M);

    // If the step was accepted, record the time and add a new working entry
    if (type == ROL::UpdateType::Accept) {
      const ttb_indx iter = history.lastEntry().iteration;
      history.lastEntry().cum_time = timer.getTotalTime(0);
      history.addEmpty();
      history.lastEntry().iteration = iter+1;
    }
  }

  template <typename ExecSpace, typename LossFunction>
  void
  GCP_RolObjective<ExecSpace,LossFunction>::
  update(const ROL::Vector<ttb_real>& xx, bool flag, int iter)
  {
    update(xx, ROL::UpdateType::Accept, iter);
  }

  template <typename ExecSpace, typename LossFunction>
  ttb_real
  GCP_RolObjective<ExecSpace,LossFunction>::
  value(const ROL::Vector<ttb_real>& xx, ttb_real& tol)
  {
    GENTEN_TIME_MONITOR("GCP_RolObjective::value");

    const vector_type& x = dynamic_cast<const vector_type&>(xx);

    // Convert input vector to a Ktensor
    M = x.getKtensor();
#if COPY_KTENSOR
    x.copyToKtensor(M);
#endif

    ttb_real res = gcp_model.value(M);
    history.lastEntry().residual = res;
    if (compute_fit)
      history.lastEntry().fit = gcp_model.computeFit(M);

    return res;
  }

  template <typename ExecSpace, typename LossFunction>
  void
  GCP_RolObjective<ExecSpace,LossFunction>::
  gradient(ROL::Vector<ttb_real>& gg, const ROL::Vector<ttb_real>& xx,
           ttb_real &tol)
  {
    GENTEN_TIME_MONITOR("GCP_RolObjective::gradient");

    const vector_type& x = dynamic_cast<const vector_type&>(xx);
    vector_type& g = dynamic_cast<vector_type&>(gg);

    // Convert input vector to a Ktensor
    M = x.getKtensor();
    G = g.getKtensor();
#if COPY_KTENSOR
    x.copyToKtensor(M);
#endif

    gcp_model.gradient(G, M);

    // Convert Ktensor to vector
#if COPY_KTENSOR
    g.copyFromKtensor(G);
#endif

    history.lastEntry().grad_norm = g.normInf();
  }

  template <typename ExecSpace, typename LossFunction>
  ttb_real
  GCP_RolObjective<ExecSpace,LossFunction>::
  computeFit(const ktensor_type& u)
  {
    gcp_model.update(u);
    return gcp_model.computeFit(u);
  }

  template <typename ExecSpace>
  Teuchos::RCP< GCP_RolObjectiveBase<ExecSpace> >
  GCP_createRolObjective(const TensorT<ExecSpace>& x,
                         const KtensorT<ExecSpace>& m,
                         const AlgParams& algParams,
                         PerfHistory& h)
  {
    Teuchos::RCP< GCP_RolObjectiveBase<ExecSpace> > obj;
    dispatch_loss(algParams, [&](const auto& loss)
    {
      using LossType =
        std::remove_cv_t< std::remove_reference_t<decltype(loss)> >;
      obj = Teuchos::rcp(new GCP_RolObjective<ExecSpace,LossType>(
        x, m, loss, algParams, h));
    });
    return obj;
  }

}

#define INST_MACRO(SPACE)                                               \
  template Teuchos::RCP< GCP_RolObjectiveBase<SPACE> >                  \
  GCP_createRolObjective<SPACE>(                                        \
    const TensorT<SPACE>& x,                                            \
    const KtensorT<SPACE>& m,                                           \
    const AlgParams& algParams,                                         \
    PerfHistory& h);

GENTEN_INST(INST_MACRO)
