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

#include "Genten_Tensor.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_AlgParams.hpp"
#include "Genten_GCP_Hash.hpp"
#include "Genten_DistKtensorUpdate.hpp"

#include "Kokkos_Random.hpp"

namespace Genten {

  namespace Impl {

    template <typename TensorType, typename ExecSpace, typename Searcher, typename LossFunction>
    void uniform_sample_tensor(
      const TensorType& X,
      const Searcher& searcher,
      const ttb_indx num_samples,
      const ttb_real weight,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);

    template <typename TensorType, typename ExecSpace, typename Searcher, typename LossFunction>
    void uniform_sample_tensor_tpetra(
      const TensorType& X,
      const Searcher& searcher,
      const ttb_indx num_samples,
      const ttb_real weight,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      KtensorT<ExecSpace>& u_overlap,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);

    template <typename TensorType, typename ExecSpace, typename Searcher, typename LossFunction>
    void uniform_sample_tensor_onesided(
      const TensorType& X,
      const Searcher& searcher,
      const ttb_indx num_samples,
      const ttb_real weight,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      DistKtensorUpdate<ExecSpace>& dku,
      KtensorT<ExecSpace>& u_overlap,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);

    template <typename ExecSpace, typename Searcher, typename Gradient>
    void stratified_sample_tensor(
      const SptensorT<ExecSpace>& X,
      const Searcher& searcher,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& u,
      const Gradient& gradient,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);

    template <typename ExecSpace, typename Searcher, typename Gradient>
    void stratified_sample_tensor_tpetra(
      const SptensorT<ExecSpace>& X,
      const Searcher& searcher,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& u,
      const Gradient& gradient,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      KtensorT<ExecSpace>& u_overlap,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);

    template <typename ExecSpace, typename Searcher, typename Gradient>
    void stratified_sample_tensor_onesided(
      const SptensorT<ExecSpace>& X,
      const Searcher& searcher,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& u,
      const Gradient& gradient,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      DistKtensorUpdate<ExecSpace>& dku,
      KtensorT<ExecSpace>& u_overlap,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);

    template <typename ExecSpace, typename Layout>
    class DenseSearcher {
    public:
      DenseSearcher(const TensorImpl<ExecSpace,Layout>& X_) : X(X_) {}

      template <typename IndexType>
      KOKKOS_INLINE_FUNCTION
      bool search(const IndexType& ind) const {
        return true;
      }

      template <typename IndexType>
      KOKKOS_INLINE_FUNCTION
      ttb_real value(const IndexType& ind) const {
        const ttb_indx i = X.global_sub2ind(ind);
        return X[i];
      }
    private:
      const TensorImpl<ExecSpace,Layout> X;
    };

    template <typename ExecSpace, typename Layout>
    DenseSearcher<ExecSpace,Layout>
    denseSearcher(const TensorImpl<ExecSpace,Layout>& X) {
      return DenseSearcher<ExecSpace,Layout>(X);
    }

    template <typename ExecSpace>
    class SortSearcher {
    public:
      SortSearcher(const SptensorImpl<ExecSpace>& X_) : X(X_), nnz(X.nnz()) {}

      template <typename IndexType>
      KOKKOS_INLINE_FUNCTION
      bool search(const IndexType& ind) const {
        return (X.index(ind) < nnz);
      }

      template <typename IndexType>
      KOKKOS_INLINE_FUNCTION
      ttb_real value(const IndexType& ind) const {
        const ttb_indx i = X.index(ind);
        if (i < nnz)
          return X.value(i);
        return ttb_real(0.0);
      }
    private:
      const SptensorImpl<ExecSpace> X;
      const ttb_real nnz;
    };

    template <typename ExecSpace>
    class HashSearcher {
    public:
      HashSearcher(const SptensorImpl<ExecSpace>& X_,
                   const TensorHashMap<ExecSpace>& hash_) : X(X_), nnz(X.nnz()), hash(hash_) {}

      template <typename IndexType>
      KOKKOS_INLINE_FUNCTION
      bool search(const IndexType& ind) const {
        return hash.exists(ind);
      }

      template <typename IndexType>
      KOKKOS_INLINE_FUNCTION
      ttb_real value(const IndexType& ind) const {
        const auto hash_index = hash.find(ind);
        if (hash.valid_at(hash_index))
          return hash.value_at(hash_index);
        return ttb_real(0.0);
      }
    private:
      const SptensorImpl<ExecSpace> X;
      const ttb_real nnz;
      const TensorHashMap<ExecSpace> hash;
    };

    template <typename ExecSpace>
    class SemiStratifiedSearcher {
    public:
      SemiStratifiedSearcher() {}

      template <typename IndexType>
      KOKKOS_INLINE_FUNCTION
      bool search(const IndexType& ind) const {
          return false;
      }

      template <typename IndexType>
      KOKKOS_INLINE_FUNCTION
      bool search(const IndexType& ind, ttb_real& x) const {
          return false;
      }
    };

    template <typename LossType>
    class StratifiedGradient {
    public:
      StratifiedGradient(const LossType& loss_) : loss(loss_) {}

      KOKKOS_INLINE_FUNCTION
      ttb_real
      evalNonZero(const ttb_real x, const ttb_real m, const ttb_real w) const {
        return w * loss.deriv(x, m);
      }

      KOKKOS_INLINE_FUNCTION
      ttb_real
      evalZero(const ttb_real m, const ttb_real w) const {
        return w * loss.deriv(ttb_real(0.0), m);
      }

    private:
      const LossType loss;
    };

    template <typename LossType>
    class SemiStratifiedGradient {
    public:
      SemiStratifiedGradient(const LossType& loss_) : loss(loss_) {}

      KOKKOS_INLINE_FUNCTION
      ttb_real
      evalNonZero(const ttb_real x, const ttb_real m, const ttb_real w) const {
        return w * ( loss.deriv(x, m) - loss.deriv(ttb_real(0.0), m) );
      }

      KOKKOS_INLINE_FUNCTION
      ttb_real
      evalZero(const ttb_real m, const ttb_real w) const {
        return w * loss.deriv(ttb_real(0.0), m);
      }

    private:
      const LossType loss;
    };

    template <typename ExecSpace, typename LossFunction>
    void stratified_ktensor_grad(
      const SptensorT<ExecSpace>& X,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& u,
      const KtensorT<ExecSpace>& up,
      const ArrayT<ExecSpace>& window,
      const ttb_real window_penalty,
      const LossFunction& loss_func,
      SptensorT<ExecSpace>& Y,
      const AlgParams& algParams);

    template <typename ExecSpace, typename LossFunction>
    void uniform_ktensor_grad(
      const ttb_indx num_samples,
      const ttb_real weight,
      const KtensorT<ExecSpace>& u,
      const KtensorT<ExecSpace>& up,
      const ArrayT<ExecSpace>& window,
      const ttb_real window_penalty,
      const LossFunction& loss_func,
      SptensorT<ExecSpace>& Y,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams);
  }

}
