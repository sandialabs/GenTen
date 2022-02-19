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

#include <random>
#include <cmath>
#include <memory>

#include "Genten_Boost.hpp"
#include "Genten_DistContext.hpp"
#include "Genten_DistSpTensor.hpp"
#include "Genten_DistKtensor.hpp"
#include "Genten_Pmap.hpp"
#include "Genten_CpAls.hpp"

namespace Genten {

template <typename ElementType, typename ExecSpace = DefaultExecutionSpace>
class DistCpAls {
  static_assert(std::is_floating_point<ElementType>::value,
                "DistCpAls Requires that the element type be a floating "
                "point type.");

public:
  DistCpAls(const DistSpTensor<ElementType, ExecSpace>& spTensor,
            const DistKtensor<ElementType, ExecSpace>& kTensor,
            const ptree& tree);
  ~DistCpAls() = default;

  // For now let's delete these so they aren't accidently used
  // Can come back and define them later if needed
  DistCpAls(DistCpAls const &) = delete;
  DistCpAls &operator=(DistCpAls const &) = delete;

  DistCpAls(DistCpAls &&) = delete;
  DistCpAls &operator=(DistCpAls &&) = delete;

  ElementType compute();

private:
  void setAlgParams();

  ptree input_;
  DistSpTensor<ElementType, ExecSpace> spTensor_;
  DistKtensor<ElementType, ExecSpace> Kfac_;
  AlgParams algParams_;
  std::ostream& out;
};

template <typename ElementType, typename ExecSpace>
DistCpAls<ElementType, ExecSpace>::
DistCpAls(const DistSpTensor<ElementType, ExecSpace>& spTensor,
          const DistKtensor<ElementType, ExecSpace>& kTensor,
          const ptree& tree) :
  input_(tree.get_child("cp-als")),
  spTensor_(spTensor),
  Kfac_(kTensor),
  out(spTensor_.pmap().gridRank() == 0 ? std::cout : Genten::bhcout)
{
  setAlgParams();
}

template <typename ElementType, typename ExecSpace>
ElementType DistCpAls<ElementType, ExecSpace>::compute()
{
  const ProcessorMap* pmap = spTensor_.pmap_ptr().get();
  SptensorT<ExecSpace>& x = spTensor_.localSpTensor();
  KtensorT<ExecSpace>& u = Kfac_.localKtensor();

  u.setProcessorMap(pmap);
  x.setProcessorMap(pmap);

  ttb_indx numIters = 0;
  ttb_real resNorm = 0.0;
  Genten::cpals_core(x, u, algParams_, numIters, resNorm, 0, nullptr, out);

  u.setProcessorMap(nullptr);
  x.setProcessorMap(nullptr);

  return resNorm;
}

template <typename ElementType, typename ExecSpace>
void
DistCpAls<ElementType, ExecSpace>::
setAlgParams() {
  algParams_.rank = input_.get<int>("rank", algParams_.rank);
  algParams_.maxiters = input_.get<int>("maxiters", algParams_.maxiters);
  algParams_.maxsecs = input_.get<double>("maxsecs", algParams_.maxsecs);
  algParams_.tol = input_.get<double>("tol", algParams_.rank);
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
