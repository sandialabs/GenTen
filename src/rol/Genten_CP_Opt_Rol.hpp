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

#include "Genten_Ktensor.hpp"
#include "Genten_AlgParams.hpp"
#include "Genten_PerfHistory.hpp"

#include "Teuchos_ParameterList.hpp"

namespace Genten {

  //! Compute the CP decomposition of a tensor using ROL.
  /*!
   *  Compute an estimate of the best rank-R CP model of a tensor X
   *  for Gaussian loss.  The input X may be a sparse or dense tensor.
   *  The result is a Ktensor.
   *
   *  An initial guess of factor matrices must be provided, which must be
   *  nonzero.
   *
   *  @param[in] x          Data tensor to be fit by the model.
   *  @param[in,out] u      Input contains an initial guess for the factors.
   *                        The size of each mode must match the corresponding
   *                        mode of x, and the number of components determines
   *                        how many will be in the result.
   *                        Output contains resulting factorization Ktensor.
   *  @param[in] algParams  Genten algorithm/solver parameters
   *  @param[out] history   Performance history for each step of the algorithm
   *  @param[in] rol_params ROL solver parameters
   *  @param[out] stream    ROL output stream (set to null for no output)
   */
  template<typename TensorT, typename ExecSpace>
  void cp_opt_rol(const TensorT& x, KtensorT<ExecSpace>& u,
                  const AlgParams& algParams,
                  PerfHistory& history,
                  Teuchos::ParameterList& rol_params,
                  std::ostream& stream = std::cout);

  //! Compute the CP decomposition of a tensor using ROL.
  /*!
   *  Compute an estimate of the best rank-R CP model of a tensor X
   *  for Gaussian loss.  The input X currently must be a (sparse)
   *  Sptensor.  The result is a Ktensor.
   *
   *  An initial guess of factor matrices must be provided, which must be
   *  nonzero.
   *
   *  This version uses default ROL settings for an algorithm similar to
   *  L-BFGS-B using the supplied stopping criteria.
   *
   *  @param[in] x          Data tensor to be fit by the model.
   *  @param[in,out] u      Input contains an initial guess for the factors.
   *                        The size of each mode must match the corresponding
   *                        mode of x, and the number of components determines
   *                        how many will be in the result.
   *                        Output contains resulting factorization Ktensor.
   *  @param[in] algParams  Genten algorithm/solver parameters
   *  @param[out] history   Performance history for each step of the algorithm
   *  @param[out] stream    ROL output stream (set to null for no output)
   */
  template<typename TensorT, typename ExecSpace>
  void cp_opt_rol(const TensorT& x, KtensorT<ExecSpace>& u,
                  const AlgParams& algParams,
                  PerfHistory& history,
                  std::ostream& stream = std::cout)
  {
    // Setup ROL to do optimization algorithm similar to L-BFGS-B
    Teuchos::ParameterList params;
    Teuchos::ParameterList& rol_params = params.sublist("ROL");
    Teuchos::ParameterList& step_params = rol_params.sublist("Step");
    step_params.set<std::string>("Type", "Line Search");

    // Set stopping criteria
    Teuchos::ParameterList& status_params = rol_params.sublist("Status Test");
    status_params.set<double>("Gradient Tolerance", algParams.gtol);
    status_params.set<double>("Step Tolerance", algParams.ftol);
    status_params.set<int>("Iteration Limit", algParams.maxiters);

    // Print iterations
    if (algParams.printitn > 0) {
      Teuchos::ParameterList& general_params = rol_params.sublist("General");
      general_params.set("Output Level", 1);
    }

    cp_opt_rol(x, u, algParams, history, params, stream);
  }

}
