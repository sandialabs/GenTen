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

#include "Genten_Sptensor.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_Array.hpp"
#include "Genten_AlgParams.hpp"
#include "Genten_SystemTimer.hpp"
#include "Genten_DistKtensorUpdate.hpp"

#include "Kokkos_Random.hpp"

namespace Genten {

  namespace Impl {

    template <typename ExecSpace, typename loss_type>
    void gcp_sgd_ss_grad(
      const SptensorT<ExecSpace>& X,
      const KtensorT<ExecSpace>& M,
      const loss_type& f,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& G,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams,
      SystemTimer& timer,
      const int timer_nzs,
      const int timer_zs);

    template <typename ExecSpace, typename loss_type>
    void gcp_sgd_ss_grad(
      const SptensorT<ExecSpace>& X,
      const KtensorT<ExecSpace>& M,
      const KtensorT<ExecSpace>& Mt,
      const KtensorT<ExecSpace>& Mprev,
      const loss_type& f,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const ArrayT<ExecSpace>& window,
      const ttb_real window_penalty,
      const IndxArrayT<ExecSpace>& modes,
      const KtensorT<ExecSpace>& G,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams,
      SystemTimer& timer,
      const int timer_nzs,
      const int timer_zs);

    template <typename ExecSpace, typename Searcher, typename Gradient>
    void gcp_sgd_ss_grad_onesided(
      const SptensorT<ExecSpace>& Xd,
      const KtensorT<ExecSpace>& ud,
      const Searcher& searcher,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const Gradient& gradient,
      const AlgParams& algParams,
      DistKtensorUpdate<ExecSpace> *dku_ptr,
      SptensorT<ExecSpace>& Yd,
      ArrayT<ExecSpace>& w,
      KtensorT<ExecSpace>& ud_overlapped,
      const KtensorT<ExecSpace>& Gd,
      const KtensorT<ExecSpace>& Gd_overlap,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool);

  }

}
