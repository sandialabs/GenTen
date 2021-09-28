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

#include <vector>
#include <random>

#include "Genten_Ktensor.hpp"
#include "Genten_Array.hpp"
#include "Genten_AlgParams.hpp"

namespace Genten {

  //! Class encapsulating history terms for streaming CP/GCP
  template <typename ExecSpace>
  class StreamingHistory {
  public:

    StreamingHistory();

    StreamingHistory(const KtensorT<ExecSpace>& u, const AlgParams& algParams);

    void updateHistory(const KtensorT<ExecSpace>& u);

    ttb_real objective(const KtensorT<ExecSpace>& u) const;

    void gradient(const KtensorT<ExecSpace>& u,
                  const ttb_indx mode_beg, const ttb_indx mode_end,
                  const KtensorT<ExecSpace>& g) const;

    bool do_gcp_loss() const;

    KtensorT<ExecSpace> up;
    ArrayT<ExecSpace> window_val;
    const ttb_real window_penalty;

  protected:

    const AlgParams algParams;
    std::default_random_engine generator;  // Random number generator
    FacMatrixT<ExecSpace> c1, c2, c3, tmp, tmp2;
    std::vector< FacMatrixT<ExecSpace> > Z1, Z2;
    IndxArray window_idx;
    ttb_indx slice_idx;
    typename ArrayT<ExecSpace>::HostMirror window_val_host;
  };

}
