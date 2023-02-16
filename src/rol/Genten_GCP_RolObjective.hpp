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

#include "ROL_Objective.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_AlgParams.hpp"
#include "Genten_PerfHistory.hpp"
#include "Genten_RolKokkosVector.hpp"
#include "Genten_RolKtensorVector.hpp"

// Choose implementation of ROL::Vector (KtensorVector or KokkosVector)
#define USE_KTENSOR_VECTOR 0

// Whether to copy the ROL::Vector into a new Ktensor before accessing data.
// This adds cost for the copy, but allows mttkrp to use a padded Ktensor
// when using RolKokkosVector.
#define COPY_KTENSOR 0

namespace Genten {

  template <typename ExecSpace>
  class GCP_RolObjectiveBase : public ROL::Objective<ttb_real> {
  public:
    typedef ExecSpace exec_space;
#if USE_KTENSOR_VECTOR
    typedef RolKtensorVector<exec_space> vector_type;
#else
    typedef RolKokkosVector<exec_space> vector_type;
#endif

    GCP_RolObjectiveBase() = default;
    virtual ~GCP_RolObjectiveBase() {}

    virtual ROL::Ptr<vector_type> createDesignVector() const = 0;

    virtual std::string lossFunctionName() const = 0;

    virtual bool lossFunctionHasLowerBound() const = 0;

    virtual bool lossFunctionHasUpperBound() const = 0;

    virtual ttb_real lossFunctionLowerBound() const = 0;

    virtual ttb_real lossFunctionUpperBound() const = 0;

    virtual ttb_real computeFit(const KtensorT<ExecSpace>& u) = 0;
  };

  template <typename ExecSpace>
  Teuchos::RCP< GCP_RolObjectiveBase<ExecSpace> >
  GCP_createRolObjective(const TensorT<ExecSpace>& x,
                         const KtensorT<ExecSpace>& m,
                         const AlgParams& algParams,
                         PerfHistory& h);

}
