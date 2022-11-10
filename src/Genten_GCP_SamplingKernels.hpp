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
#include "Genten_AlgParams.hpp"
#include "Genten_GCP_Hash.hpp"

#include "Kokkos_Random.hpp"

namespace Genten {

  namespace Impl {

    template <typename ExecSpace, typename LossFunction>
    void uniform_sample_tensor(
      const SptensorT<ExecSpace>& X,
      const ttb_indx num_samples,
      const ttb_real weight,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);

    template <typename ExecSpace, typename LossFunction>
    void uniform_sample_tensor_hash(
      const SptensorT<ExecSpace>& X,
      const TensorHashMap<ExecSpace>& hash,
      const ttb_indx num_samples,
      const ttb_real weight,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);

    template <typename ExecSpace, typename LossFunction>
    void stratified_sample_tensor(
      const SptensorT<ExecSpace>& X,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);

    template <typename ExecSpace, typename LossFunction>
    void stratified_sample_tensor_hash(
      const SptensorT<ExecSpace>& X,
      const TensorHashMap<ExecSpace>& hash,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);

    template <typename ExecSpace, typename LossFunction>
    void stratified_sample_tensor_hash_tpetra(
      const SptensorT<ExecSpace>& X,
      const TensorHashMap<ExecSpace>& hash,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      KtensorT<ExecSpace>& u_overlap,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);

    template <typename ExecSpace, typename LossFunction>
    void semi_stratified_sample_tensor(
      const SptensorT<ExecSpace>& X,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);

    template <typename ExecSpace, typename LossFunction>
    void sample_tensor_nonzeros(
      const SptensorT<ExecSpace>& X,
      const ttb_indx offset,
      const ttb_indx num_samples,
      const ttb_real weight,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);

    template <typename ExecSpace, typename LossFunction>
    void sample_tensor_nonzeros(
      const SptensorT<ExecSpace>& X,
      const ArrayT<ExecSpace>& w,
      const ttb_indx num_samples,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      SptensorT<ExecSpace>& Y,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);

    template <typename ExecSpace>
    void sample_tensor_nonzeros(
      const SptensorT<ExecSpace>& X,
      const ArrayT<ExecSpace>& w,
      const ttb_indx num_samples,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& z,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);

    template <typename ExecSpace>
    void sample_tensor_zeros(
      const SptensorT<ExecSpace>& X,
      const ttb_indx offset,
      const ttb_indx num_samples,
      SptensorT<ExecSpace>& Y,
      SptensorT<ExecSpace>& Z,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);

    template <typename ExecSpace>
    void merge_sampled_tensors(const SptensorT<ExecSpace>& X_nz,
                               const SptensorT<ExecSpace>& X_z,
                               SptensorT<ExecSpace>& X,
                               const AlgParams& algParams);
  }

}
