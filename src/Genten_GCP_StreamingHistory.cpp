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

#include <algorithm>
#include <random>

#include "Genten_GCP_StreamingHistory.hpp"
#include "Genten_MixedFormatOps.hpp"

namespace Genten {

  template <typename ExecSpace>
  StreamingHistory<ExecSpace>::
  StreamingHistory() : up(), window_val(), window_penalty(0.0) {}

  template <typename ExecSpace>
  StreamingHistory<ExecSpace>::
  StreamingHistory(const KtensorT<ExecSpace>& u,
                   const AlgParams& algParams_) :
    up(), window_val(), window_penalty(algParams_.window_penalty),
    algParams(algParams_),
    generator(algParams.seed)
  {
    // to do:
    //   * make sure u is normalized properly with unit weights

    const ttb_indx nc = u.ncomponents();
    const ttb_indx nd = u.ndims();
    const ttb_indx ws = algParams.window_size;

    if (ws == 0 || window_penalty == ttb_real(0.0))
      return;

    // Always allocate the temporaries needed for ktensor-fro even if not
    // selected in algParams, because other objects (e.g., DenseSampler)
    // may evaluate it regardless of the chosen formulation
    c1   = FacMatrixT<ExecSpace>(nc,nc);
    c2   = FacMatrixT<ExecSpace>(nc,nc);
    c3   = FacMatrixT<ExecSpace>(nc,nc);
    tmp  = FacMatrixT<ExecSpace>(nc,nc);
    tmp2 = FacMatrixT<ExecSpace>(ws,nc);
    Z1   = std::vector< FacMatrixT<ExecSpace> >(nd);
    Z2   = std::vector< FacMatrixT<ExecSpace> >(nd);
    for (ttb_indx k=0; k<nd; ++k) {
      Z1[k] = FacMatrixT<ExecSpace>(nc,nc);
      Z2[k] = FacMatrixT<ExecSpace>(nc,nc);
    }

    // Construct window data -- u contains initial history
    up = KtensorT<ExecSpace>(nc, nd);
    for (ttb_indx i=0; i<nd-1; ++i)
      up.set_factor(i, FacMatrixT<ExecSpace>(u[i].nRows(), nc));
    up.set_factor(nd-1, FacMatrixT<ExecSpace>(ws, nc));
    deep_copy(up.weights(), u.weights());
    window_idx = IndxArray(ws);
    window_val = ArrayT<ExecSpace>(ws);
    window_val_host = create_mirror_view(window_val);
    slice_idx = 0;

    // We rely on unused rows of window_val, up[nd-1] being zero,
    // so explicitly initialize them
    window_val = ttb_real(0.0);
    up[nd-1] = ttb_real(0.0);

    updateHistory(u);
  }

  template <typename ExecSpace>
  void
  StreamingHistory<ExecSpace>::
  updateHistory(const KtensorT<ExecSpace>& u)
  {
    if (window_val.size() == 0 || window_penalty == ttb_real(0.0))
      return;

    const ttb_indx nd = u.ndims();
    ttb_indx num_new = u[nd-1].nRows();

    // Copy spatial modes
    for (ttb_indx i=0; i<nd-1; ++i)
      deep_copy(up[i], u[i]);

    // First copy rows from u[nd-1] into up[nd-1] until history is full
    ttb_indx idx = 0;
    while (idx < num_new && slice_idx+idx < algParams.window_size) {
      auto sub1 = Kokkos::subview(up[nd-1].view(),slice_idx+idx,Kokkos::ALL);
      auto sub2 = Kokkos::subview(u[nd-1].view(),idx,Kokkos::ALL);
      deep_copy(sub1, sub2);
      window_idx[slice_idx+idx] = slice_idx+idx;
      ++idx;
    }

    // Copy in the rest of the rows in u[nd-1] depending on method
    if (idx < num_new) {
      num_new = num_new - idx;
      if (algParams.window_method == GCP_Streaming_Window_Method::Last) {
        // Shift each row of up[nd-1] down by num_new
        for (ttb_indx i=0; i<algParams.window_size-num_new; ++i) {
          auto sub1 = Kokkos::subview(up[nd-1].view(),i,Kokkos::ALL);
          auto sub2 = Kokkos::subview(up[nd-1].view(),i+num_new,Kokkos::ALL);
          deep_copy(sub1, sub2);
          window_idx[i] = window_idx[i+num_new];
        }
        // Copy last num_new rows u[nd-1] into last rows of up[nd-1]
        auto sub1 = Kokkos::subview(
          up[nd-1].view(),
          std::make_pair(algParams.window_size-num_new,algParams.window_size),
          Kokkos::ALL);
        auto sub2 = Kokkos::subview(
          u[nd-1].view(), std::make_pair(idx,idx+num_new), Kokkos::ALL);
        deep_copy(sub1, sub2);
        for (ttb_indx i=0; i<num_new; ++i)
          window_idx[algParams.window_size-num_new+i] = slice_idx+idx+i;
      }
      else if (algParams.window_method ==
               GCP_Streaming_Window_Method::Reservoir) {
        for (ttb_indx i=0; i<num_new; ++i) {
          // Random integer in [0,slice_idx+idx+i)
          std::uniform_int_distribution<ttb_indx> rand(0,slice_idx+idx+i-1);
          const ttb_indx j = rand(generator);

          // Replace entry j in history with slice_idx+i
          if (j < algParams.window_size) {
            auto sub1 = Kokkos::subview(up[nd-1].view(),j,Kokkos::ALL);
            auto sub2 = Kokkos::subview(u[nd-1].view(),idx+i,Kokkos::ALL);
            deep_copy(sub1, sub2);
            window_idx[j] = slice_idx+idx+i;
          }
        }
        // To do:  sort window_idx and permute up[nd-1] appropriately
        // std::sort(window_idx.values().data(),
        //           window_idx.values().data()+algParams.window_size);
      }
      else
        Genten::error("Unsupported window method: ");
    }

    // Update slice index counter
    slice_idx += u[nd-1].nRows();

    // Update weighting values
    for (ttb_indx i=0; i<algParams.window_size; ++i)
      window_val_host[i] =
        std::pow(algParams.window_weight, ttb_real(slice_idx-window_idx[i]));
    deep_copy(window_val, window_val_host);
  }

  template <typename ExecSpace>
  bool
  StreamingHistory<ExecSpace>::
  do_gcp_loss() const {
    return
      algParams.history_method == GCP_Streaming_History_Method::GCP_Loss &&
      up.ndims() != 0 && up.ncomponents() != 0 &&
      window_val.size() != 0 && window_penalty != ttb_real(0.0);
  }

  template <typename ExecSpace>
  ttb_real
  StreamingHistory<ExecSpace>::
  objective(const KtensorT<ExecSpace>& u) const
  {
    if (window_val.size() == 0 || window_penalty == ttb_real(0.0))
      return 0.0;

    ttb_real loss = 0.0;
    if (algParams.history_method == GCP_Streaming_History_Method::Ktensor_Fro)
      loss = ktensor_fro_objective(u);
    else if (algParams.history_method ==
             GCP_Streaming_History_Method::Factor_Fro)
      loss = factor_fro_objective(u);

    return loss;
  }

  template <typename ExecSpace>
  ttb_real
  StreamingHistory<ExecSpace>::
  ktensor_fro_objective(const KtensorT<ExecSpace>& u) const
  {
    if (window_val.size() == 0 || window_penalty == ttb_real(0.0))
      return 0.0;

    const ttb_indx nd = u.ndims();

    // For using Gaussian loss on history term
    // = window_penatly * \sum_h window(h)*\|up - ut\|_F^2 where ut = u
    // with u[nd-1] replaced by up[nd-1].  Use formula
    // \|up - ut\|_F^2 = \|up\|^2 + \|ut\|^2 - 2<up,ut>.

    // c1 = ut[0]'*ut[0] .* ... .* ut[nd-2]*'ut[nd-2] .* z
    // c2 = up[0]'*up[0] .* ... .* up[nd-2]*'up[nd-2] .* z
    // c3 = up[0]'*ut[0] .* ... .* up[nd-2]*'ut[nd-2] .* z
    // z  = up[nd-1]'*diag(window)*up[nd-1]
    c1.oprod(u.weights());
    c2.oprod(up.weights());
    c3.oprod(u.weights(), up.weights());
    for (ttb_indx k=0; k<nd-1; ++k) {
      tmp.gramian(u[k],true);  // Compute full gram
      c1.times(tmp);

      tmp.gramian(up[k],true); // compute full gram
      c2.times(tmp);

      tmp.gemm(true,false,ttb_real(1.0),up[k],u[k],ttb_real(0.0));
      c3.times(tmp);
    }
    deep_copy(tmp2, up[nd-1]);
    tmp2.rowScale(window_val, false);
    tmp.gemm(true,false,ttb_real(1.0),up[nd-1],tmp2,ttb_real(0.0));
    c1.times(tmp);
    c2.times(tmp);
    c3.times(tmp);

    ttb_real t1 = c1.sum(); // t1 = \sum_h window(h)*\|u\|^2
    ttb_real t2 = c2.sum(); // t2 = \sum_h window(h)*\|up\|^2
    ttb_real t3 = c3.sum(); // t3 = \sum_h window(h)*<up,u>

    ttb_real loss = window_penalty * (t1 + t2 - ttb_real(2.0)*t3);
    return loss;
  }

  template <typename ExecSpace>
  ttb_real
  StreamingHistory<ExecSpace>::
  factor_fro_objective(const KtensorT<ExecSpace>& u) const
  {
    if (window_val.size() == 0 || window_penalty == ttb_real(0.0))
      return 0.0;

    const ttb_indx nd = u.ndims();

    ttb_real loss = 0.0;
    for (ttb_indx k=0; k<nd-1; ++k) {
      // Compute \|up[k]-u[k]\|^2 =
      //   \|up[k]\|^2 + \|u[k]\|^2 - 2*<up[k],u[k]>
      ttb_real t1 = up[k].normFsq();
      ttb_real t2 = u[k].normFsq();
      ttb_real t3 = up[k].innerprod(u[k],u.weights());
      loss += window_penalty * ( t1 + t2 - ttb_real(2.0)*t3 );
    }

    return loss;
  }

  template <typename ExecSpace>
  void
  StreamingHistory<ExecSpace>::
  gradient(const KtensorT<ExecSpace>& u,
           const ttb_indx mode_beg, const ttb_indx mode_end,
           const KtensorT<ExecSpace>& g) const
  {
    if (window_val.size() == 0 || window_penalty == ttb_real(0.0))
      return;

    const ttb_indx nd = u.ndims();

    if (mode_end >= nd)
      Genten::error("History term on temporal mode nd-1 is not supported!");

    if (algParams.history_method == GCP_Streaming_History_Method::Ktensor_Fro)
      ktensor_fro_gradient(u, mode_beg, mode_end, g);
    else if (algParams.history_method ==
             GCP_Streaming_History_Method::Factor_Fro)
      factor_fro_gradient(u, mode_beg, mode_end, g);
  }

  template <typename ExecSpace>
  void
  StreamingHistory<ExecSpace>::
  ktensor_fro_gradient(const KtensorT<ExecSpace>& u,
                       const ttb_indx mode_beg, const ttb_indx mode_end,
                       const KtensorT<ExecSpace>& g) const
  {
    if (window_val.size() == 0 || window_penalty == ttb_real(0.0))
      return;

    const ttb_indx nd = u.ndims();
    if (mode_end >= nd)
      Genten::error("History term on temporal mode nd-1 is not supported!");

    // Compute intermediate Gram matricies.  We need contributions for all
    // modes, so use nd as the mode to skip
    prepare_least_squares_contributions(u, nd);

    for (ttb_indx k=mode_beg; k<mode_end; ++k) {
      // Compute c1, c2
      compute_hadamard_products(u, k);

      // g[k] = g[k] + 2*window_penalty * (u[k]*c2 - up[k]*c1)
      g[k].gemm(false,false, ttb_real(2.0)*window_penalty,u[k],c2,
                ttb_real(1.0));
      g[k].gemm(false,false,-ttb_real(2.0)*window_penalty,up[k],c1,
                ttb_real(1.0));
    }
  }

  template <typename ExecSpace>
  void
  StreamingHistory<ExecSpace>::
  factor_fro_gradient(const KtensorT<ExecSpace>& u,
                      const ttb_indx mode_beg, const ttb_indx mode_end,
                      const KtensorT<ExecSpace>& g) const
  {
    if (window_val.size() == 0 || window_penalty == ttb_real(0.0))
      return;

    const ttb_indx nd = u.ndims();

    if (mode_end >= nd)
      Genten::error("History term on temporal mode nd-1 is not supported!");

    for (ttb_indx k=mode_beg; k<mode_end; ++k) {
      // g[k] = g[k] + 2*window_penalty * (u[k]-up[k])
      g[k].plus(u[k],   ttb_real(2.0)*window_penalty);
      g[k].plus(up[k], -ttb_real(2.0)*window_penalty);
    }
  }

  template <typename ExecSpace>
  void
  StreamingHistory<ExecSpace>::
  prepare_least_squares_contributions(const KtensorT<ExecSpace>& u,
                                      const ttb_indx mode) const
  {
    if (window_val.size() == 0 || window_penalty == ttb_real(0.0))
      return;

    // Z1[k] = up[k]'*u[k],  k = 0,...,nd-2
    // Z2[k] =  u[k]'*u[k],  k = 0,...,nd-2
    // Z1[nd-1] = Z2[nd-1] = up[nd-1]'*diag(window)*up[nd-1]
    const ttb_indx nd = u.ndims();
    for (ttb_indx k=0; k<nd-1; ++k) {
      if (k != mode) {
        Z1[k].gemm(true,false,ttb_real(1.0),up[k],u[k],ttb_real(0.0));
        Z2[k].gramian(u[k],true);  // Compute full gram
      }
    }
    deep_copy(tmp2, up[nd-1]);
    tmp2.rowScale(window_val, false);
    Z1[nd-1].gemm(true,false,ttb_real(1.0),up[nd-1],tmp2,ttb_real(0.0));
    deep_copy(Z2[nd-1], Z1[nd-1]);
  }

  template <typename ExecSpace>
  void
  StreamingHistory<ExecSpace>::
  least_squares_contributions(
    const KtensorT<ExecSpace>& u,
    const ttb_indx mode,
    const FacMatrixT<ExecSpace>& lhs,
    const FacMatrixT<ExecSpace>& rhs) const
  {
    // Compute c1, c2
    compute_hadamard_products(u, mode);
    lhs.plus(c2, ttb_real(2.0)*window_penalty);
    rhs.gemm(false,false,ttb_real(2.0)*window_penalty,up[mode],c1,
             ttb_real(1.0));
  }

  template <typename ExecSpace>
  void
  StreamingHistory<ExecSpace>::
  compute_hadamard_products(const KtensorT<ExecSpace>& u,
                            const ttb_indx mode) const
  {
    const ttb_indx nd = u.ndims();
    c1.oprod(u.weights(), up.weights());
    c2.oprod(u.weights());
    for (ttb_indx n=0; n<nd; ++n) {
      if (n != mode) {
        c1.times(Z1[n]);
        c2.times(Z2[n]);
      }
    }
  }

}

#define INST_MACRO(SPACE)                                               \
    template class Genten::StreamingHistory<SPACE>;

GENTEN_INST(INST_MACRO)
