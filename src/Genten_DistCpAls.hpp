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

#include "Genten_Boost.hpp"
#include "Genten_DistTensorContext.hpp"
#include "Genten_CpAls.hpp"

namespace Genten {

template <typename ExecSpace>
class DistCpAls {
public:
  DistCpAls(const DistTensorContext& dtc,
            const SptensorT<ExecSpace>& spTensor,
            const KtensorT<ExecSpace>& kTensor,
            const ptree& tree);
  ~DistCpAls() = default;

  // For now let's delete these so they aren't accidently used
  // Can come back and define them later if needed
  DistCpAls(DistCpAls const &) = delete;
  DistCpAls &operator=(DistCpAls const &) = delete;

  DistCpAls(DistCpAls &&) = delete;
  DistCpAls &operator=(DistCpAls &&) = delete;

  ttb_real compute();

private:
  void setAlgParams();

  DistTensorContext dtc_;
  SptensorT<ExecSpace> spTensor_;
  KtensorT<ExecSpace> Kfac_;
  ptree input_;
  AlgParams algParams_;
  std::ostream& out;
};

template <typename ExecSpace>
DistCpAls<ExecSpace>::
DistCpAls(const DistTensorContext& dtc,
          const SptensorT<ExecSpace>& spTensor,
          const KtensorT<ExecSpace>& kTensor,
          const ptree& tree) :
  dtc_(dtc),
  spTensor_(spTensor),
  Kfac_(kTensor),
  input_(tree.get_child("cp-als")),
  out(dtc_.pmap().gridRank() == 0 ? std::cout : Genten::bhcout)
{
  setAlgParams();
}

template <typename ExecSpace>
ttb_real DistCpAls<ExecSpace>::compute()
{
  const ProcessorMap* pmap = dtc_.pmap_ptr().get();
  spTensor_.setProcessorMap(pmap);
  Kfac_.setProcessorMap(pmap);

  ttb_indx numIters = 0;
  ttb_real resNorm = 0.0;
  PerfHistory history;
  Genten::cpals_core(spTensor_, Kfac_, algParams_, numIters, resNorm, 1,
                     history, out);

  spTensor_.setProcessorMap(nullptr);
  Kfac_.setProcessorMap(nullptr);

  return resNorm;
}

template <typename ExecSpace>
void
DistCpAls<ExecSpace>::
setAlgParams() {
  algParams_.rank = input_.get<int>("rank", algParams_.rank);
  algParams_.maxiters = input_.get<int>("maxiters", algParams_.maxiters);
  algParams_.maxsecs = input_.get<double>("maxsecs", algParams_.maxsecs);
  algParams_.tol = input_.get<double>("tol", algParams_.tol);
  algParams_.printitn = input_.get<int>("printitn", algParams_.printitn);
  algParams_.timings = input_.get<bool>("timings", algParams_.timings);
  algParams_.full_gram = input_.get<int>("full-gram", algParams_.full_gram);
  algParams_.rank_def_solver = input_.get<int>("rank-def-solver",
                                               algParams_.rank_def_solver);
  algParams_.rcond = input_.get<double>("rcond", algParams_.rcond);
  algParams_.penalty = input_.get<double>("penalty", algParams_.penalty);

  auto mttkrp_input = input_.get_child("mttkrp");
  algParams_.mttkrp_method =
    parse_enum<MTTKRP_Method>(
      mttkrp_input.get<std::string>(
        "method", MTTKRP_Method::names[algParams_.mttkrp_method]));
  algParams_.mttkrp_nnz_tile_size =
    mttkrp_input.get<int>("nnz-tile-size", algParams_.mttkrp_nnz_tile_size);
  algParams_.mttkrp_duplicated_factor_matrix_tile_size =
    mttkrp_input.get<int>("duplicated-tile-size",
                          algParams_.mttkrp_duplicated_factor_matrix_tile_size);
  algParams_.mttkrp_duplicated_threshold =
    mttkrp_input.get<double>("duplicated-threshold",
                             algParams_.mttkrp_duplicated_threshold);

  algParams_.fixup<ExecSpace>(out);
}

} // namespace Genten
