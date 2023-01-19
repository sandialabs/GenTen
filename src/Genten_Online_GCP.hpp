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
#include <vector>
#include <random>

#include "Genten_Sptensor.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_Array.hpp"
#include "Genten_AlgParams.hpp"
#include "Genten_GCP_SGD.hpp"
#include "Genten_GCP_StreamingHistory.hpp"

namespace Genten {

  //! Class implementing the generalized CP decomposition using SGD approach
  template <typename TensorT, typename ExecSpace, typename LossFunction>
  class OnlineGCP {
  public:
    OnlineGCP(TensorT& Xinit,
              const KtensorT<ExecSpace>& u0,
              const LossFunction& loss_func,
              const AlgParams& algParams,
              const AlgParams& temporalAlgParams,
              const AlgParams& spatialAlgParams,
              std::ostream& out);

    void processSlice(TensorT& X,
                      KtensorT<ExecSpace>& u,
                      ttb_real& fest,
                      ttb_real& ften,
                      std::ostream& out,
                      const bool print);

    void init(const TensorT& X, KtensorT<ExecSpace>& u);

  protected:

    void leastSquaresSolve(const bool temporal,
                           TensorT& X,
                           KtensorT<ExecSpace>& u,
                           ttb_real& fest,
                           ttb_real& ften,
                           std::ostream& out,
                           const bool print);

    const AlgParams algParams;
    const AlgParams temporalAlgParams;
    const AlgParams spatialAlgParams;
    GCPSGD<TensorT,LossFunction> temporalSolver;
    GCPSGD<TensorT,LossFunction> spatialSolver;
    std::default_random_engine generator;  // Random number generator
    FacMatrixT<ExecSpace> A, tmp; // Temp space needed for least-squares
    std::vector< FacMatrixT<ExecSpace> > P, Q; // Temp space needed for OnlineCP
    StreamingHistory<ExecSpace> hist; // history data
  };

  //! Compute the generalized CP decomposition of a tensor using SGD approach
  template<typename TensorT, typename ExecSpace>
  void online_gcp(std::vector<TensorT>& x,
                  TensorT& x_init,
                  KtensorT<ExecSpace>& u,
                  const AlgParams& algParams,
                  const AlgParams& temporalAlgParams,
                  const AlgParams& spatialAlgParams,
                  std::ostream& out,
                  Array& fest,
                  Array& ften);

}
