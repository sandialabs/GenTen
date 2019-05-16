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
#include "Genten_GCP_SamplingKernels.hpp"

#include "Kokkos_Random.hpp"
#include "Kokkos_UnorderedMap.hpp"

namespace Genten {

  template <typename ExecSpace, typename LossFunction>
  class Sampler {
  public:

    typedef Kokkos::Random_XorShift64_Pool<ExecSpace> pool_type;
    typedef Impl::Array<ttb_indx, HASH_MAX_TENSOR_DIM> hash_key_type;
    typedef Kokkos::UnorderedMap<hash_key_type, ttb_real, ExecSpace> map_type;

    Sampler() {}

    virtual ~Sampler() {}

    virtual void initialize(const pool_type& rand_pool,
                            std::ostream& out) = 0;

    virtual void print(std::ostream& out) = 0;

    virtual void sampleTensor(const bool gradient,
                              const KtensorT<ExecSpace>& u,
                              const LossFunction& loss_func,
                              SptensorT<ExecSpace>& Xs,
                              ArrayT<ExecSpace>& w) = 0;

    virtual void fusedGradient(const KtensorT<ExecSpace>& u,
                               const LossFunction& loss_func,
                               const KtensorT<ExecSpace>& g,
                               SystemTimer& timer,
                               const int timer_nzs,
                               const int timer_zs) = 0;

    static map_type buildHashMap(const SptensorT<ExecSpace>& X,
                                 std::ostream& out)
    {
      const ttb_indx nnz = X.nnz();
      const ttb_indx nd = X.ndims();
      if (nd != HASH_MAX_TENSOR_DIM)
        Genten::error("Invalid tensor dimension for hash!");
      map_type hash_map(nnz);
      Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,nnz),
                           KOKKOS_LAMBDA(const ttb_indx i)
      {
        hash_key_type key;
        for (ttb_indx j=0; j<nd; ++j)
          key[j] = X.subscript(i,j);
        if (hash_map.insert(key, X.value(i)).failed())
          Kokkos::abort("Hash map insert failed!");
      }, "Genten::GCP_SGD::hash_kernel");

      const bool print_histogram = false;
      if (print_histogram) {
        auto h = hash_map.get_histogram();
        h.calculate();
        out << "length:" << std::endl;
        h.print_length(out);
        out << "distance:" << std::endl;
        h.print_distance(out);
        out << "block distance:" << std::endl;
        h.print_block_distance(out);
      }

      return hash_map;
    }
  };

}
