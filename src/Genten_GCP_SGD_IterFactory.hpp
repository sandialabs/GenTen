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

#include "Genten_GCP_SGD_Iter.hpp"
#include "Genten_GCP_SGD_Iter_Async.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_AlgParams.hpp"

namespace Genten {
  namespace Impl {

    template <typename LossFunction, typename ExecSpace>
    GCP_SGD_Iter<SptensorT<ExecSpace>,LossFunction>*
    createIter(const SptensorT<ExecSpace>& X,
               const KtensorT<ExecSpace>& u0,
               const StreamingHistory<ExecSpace>& hist,
               const ttb_real penalty,
               const ttb_indx mode_beg,
               const ttb_indx mode_end,
               const AlgParams& algParams)
    {
      using tensor_type = SptensorT<ExecSpace>;
      GCP_SGD_Iter<tensor_type,LossFunction> *itp = nullptr;
      if (algParams.async)
        itp = new GCP_SGD_Iter_Async<ExecSpace,LossFunction>(
          u0, hist, penalty, mode_beg, mode_end, algParams);
      else
        itp = new GCP_SGD_Iter<tensor_type,LossFunction>(
          u0, hist, penalty, mode_beg, mode_end, algParams);
      return itp;
    }

    template <typename LossFunction, typename ExecSpace>
    GCP_SGD_Iter<TensorT<ExecSpace>,LossFunction>*
    createIter(const TensorT<ExecSpace>& X,
               const KtensorT<ExecSpace>& u0,
               const StreamingHistory<ExecSpace>& hist,
               const ttb_real penalty,
               const ttb_indx mode_beg,
               const ttb_indx mode_end,
               const AlgParams& algParams)
    {
      using tensor_type = TensorT<ExecSpace>;
      if (algParams.async)
        Genten::error("Genten::gcp_sgd - cannot use asynchronous iterator with dense tensor!");
      GCP_SGD_Iter<tensor_type,LossFunction> *itp =
        new Impl::GCP_SGD_Iter<tensor_type,LossFunction>(
          u0, hist, penalty, mode_beg, mode_end, algParams);
      return itp;
    }
  }
}
