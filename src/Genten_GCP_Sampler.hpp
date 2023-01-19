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

#include <iostream>

#include "Genten_Sptensor.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_GCP_Hash.hpp"
#include "Genten_KokkosVector.hpp"
#include "Genten_GCP_StreamingHistory.hpp"

#include "Kokkos_Random.hpp"

namespace Genten {

  template <typename TensorType, typename LossFunction>
  class Sampler {
  public:

    typedef typename TensorType::exec_space exec_space;
    typedef Kokkos::Random_XorShift64_Pool<exec_space> pool_type;

    Sampler() {}

    virtual ~Sampler() {}

    virtual void initialize(const pool_type& rand_pool,
                            const bool print_itn,
                            std::ostream& out) = 0;

    virtual ttb_indx getNumGradSamples() const = 0;

    virtual void print(std::ostream& out) = 0;

    virtual void sampleTensorF(const KtensorT<exec_space>& u,
                               const LossFunction& loss_func) = 0;

    virtual void sampleTensorG(const KtensorT<exec_space>& u,
                               const StreamingHistory<exec_space>& hist,
                               const LossFunction& loss_func) = 0;

    virtual void prepareGradient(const KtensorT<exec_space>& gt) = 0;

    virtual void value(const KtensorT<exec_space>& u,
                       const StreamingHistory<exec_space>& hist,
                       const ttb_real penalty,
                       const LossFunction& loss_func,
                       ttb_real& fest, ttb_real& ften) = 0;

    virtual void gradient(const KtensorT<exec_space>& ut,
                          const StreamingHistory<exec_space>& hist,
                          const ttb_real penalty,
                          const LossFunction& loss_func,
                          KokkosVector<exec_space>& g,
                          const KtensorT<exec_space>& gt,
                          const ttb_indx mode_beg,
                          const ttb_indx mode_end,
                          SystemTimer& timer,
                          const int timer_init,
                          const int timer_nzs,
                          const int timer_zs,
                          const int timer_grad_mttkrp,
                          const int timer_grad_comm,
                          const int timer_grad_update) = 0;
  };

  template <typename ExecSpace, typename LossFunction>
  class Sampler<SptensorT<ExecSpace>,LossFunction> {
  public:

    typedef ExecSpace exec_space;
    typedef Kokkos::Random_XorShift64_Pool<exec_space> pool_type;
    typedef TensorHashMap<exec_space> map_type;

    Sampler() {}

    virtual ~Sampler() {}

    virtual void initialize(const pool_type& rand_pool,
                            const bool print_itn,
                            std::ostream& out) = 0;

    virtual ttb_indx getNumGradSamples() const = 0;

    virtual void print(std::ostream& out) = 0;

    virtual void sampleTensorF(const KtensorT<exec_space>& u,
                               const LossFunction& loss_func) = 0;

    virtual void sampleTensorG(const KtensorT<exec_space>& u,
                               const StreamingHistory<exec_space>& hist,
                               const LossFunction& loss_func) = 0;

    virtual void prepareGradient(const KtensorT<exec_space>& gt) = 0;

    virtual void value(const KtensorT<exec_space>& u,
                       const StreamingHistory<exec_space>& hist,
                       const ttb_real penalty,
                       const LossFunction& loss_func,
                       ttb_real& fest, ttb_real& ften) = 0;

    virtual void gradient(const KtensorT<exec_space>& ut,
                          const StreamingHistory<exec_space>& hist,
                          const ttb_real penalty,
                          const LossFunction& loss_func,
                          KokkosVector<exec_space>& g,
                          const KtensorT<exec_space>& gt,
                          const ttb_indx mode_beg,
                          const ttb_indx mode_end,
                          SystemTimer& timer,
                          const int timer_init,
                          const int timer_nzs,
                          const int timer_zs,
                          const int timer_grad_mttkrp,
                          const int timer_grad_comm,
                          const int timer_grad_update) = 0;

    static map_type buildHashMap(const SptensorT<exec_space>& Xd,
                                 std::ostream& out)
    {
      const auto X = Xd.impl();
      const ttb_indx nnz = X.nnz();
      const ttb_indx nd = X.ndims();
      map_type hash_map(nd, ttb_indx(1.1*nnz));
      Kokkos::parallel_for("Genten::GCP_SGD::hash_kernel",
                           Kokkos::RangePolicy<exec_space>(0,nnz),
                           KOKKOS_LAMBDA(const ttb_indx i)
      {
        auto key = X.getGlobalSubscripts(i);
        hash_map.insert(key, X.value(i));
      });

      const bool print_histogram = false;
      if (print_histogram) {
        hash_map.print_histogram(out);
      }

      return hash_map;
    }
  };

}
