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

#include <ostream>

#include "Genten_DistTensorContext.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_AlgParams.hpp"
#include "Genten_GCP_SGD_Step.hpp"
#include "Genten_GCP_StreamingHistory.hpp"
#include "Genten_PerfHistory.hpp"

namespace Genten {

  //! Class implementing the generalized CP decomposition using SGD approach
  template <typename TensorType, typename LossFunction>
  class GCPSGD {
  public:
    using exec_space = typename TensorType::exec_space;

  protected:
    const LossFunction loss_func;
    const ttb_indx mode_beg;
    const ttb_indx mode_end;
    const AlgParams algParams;
    Impl::GCP_SGD_Step<exec_space,LossFunction> *stepper;

  public:
    GCPSGD(const KtensorT<exec_space>& u,
           const LossFunction& loss_func,
           const ttb_indx mode_begin,
           const ttb_indx mode_end,
           const AlgParams& algParams);

    GCPSGD(const KtensorT<exec_space>& u,
           const LossFunction& loss_func,
           const AlgParams& algParams);

    ~GCPSGD();

    void reset();

    void solve(TensorType& X,
               KtensorT<exec_space>& u0,
               const ttb_real penalty,
               ttb_indx& numEpochs,
               ttb_real& fest,
               PerfHistory& perfInfo,
               std::ostream& out,
               const bool print_hdr,
               const bool print_ftr,
               const bool print_itn) const;

    void solve(TensorType& X,
               KtensorT<exec_space>& u,
               const StreamingHistory<exec_space>& hist,
               const ttb_real penalty,
               ttb_indx& numEpochs,
               ttb_real& fest,
               ttb_real& ften,
               PerfHistory& perfInfo,
               std::ostream& out,
               const bool print_hdr,
               const bool print_ftr,
               const bool print_itn) const;
  };

  //! Compute the generalized CP decomposition of a tensor using SGD approach
  template<typename TensorType>
  void gcp_sgd (TensorType& x,
                KtensorT<typename TensorType::exec_space>& u,
                const AlgParams& algParams,
                ttb_indx& numIters,
                ttb_real& resNorm,
                PerfHistory& perfInfo,
                std::ostream& out);

}
