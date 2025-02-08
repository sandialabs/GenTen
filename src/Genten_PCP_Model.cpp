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

#include "Genten_PCP_Model.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_GCP_ValueKernels_Def.hpp" // So gcp_value can be (implicitly) instantiated on a custom loss function

namespace Genten {

  namespace Impl {

    template <typename ExecSpace, typename loss_type>
    struct GCP_Grad_Sptensor {
      typedef SptensorT<ExecSpace> tensor_type;
      typedef KtensorT<ExecSpace> Ktensor_type;

      const tensor_type XX;
      const Ktensor_type MM;
      const ttb_real ww;
      const loss_type ff;
      const tensor_type YY;

      GCP_Grad_Sptensor(const tensor_type& X_, const Ktensor_type& M_,
                        const ttb_real w_, const loss_type& f_,
                        const tensor_type& Y_) :
        XX(X_), MM(M_), ww(w_), ff(f_), YY(Y_) {}

      template <unsigned FBS, unsigned VS>
      void run() const
      {
        typedef typename tensor_type::exec_space exec_space;
        typedef Kokkos::TeamPolicy<exec_space> Policy;
        typedef typename Policy::member_type TeamMember;

        const auto X = XX.impl();
        const auto M = MM.impl();
        const ttb_real w = ww;
        const loss_type f = ff;
        const auto Y = YY.impl();

        static const bool is_gpu = Genten::is_gpu_space<exec_space>::value;
        static const unsigned RowBlockSize = 128;
        static const unsigned FacBlockSize = FBS;
        static const unsigned VectorSize = is_gpu ? VS : 1;
        static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;

        /*const*/ ttb_indx nnz = X.nnz();
        const ttb_indx N = (nnz+RowBlockSize-1)/RowBlockSize;

        Policy policy(N, TeamSize, VectorSize);
        Kokkos::parallel_for("GCP_Sptensor_Gradient: Y eval",
                             policy,
                             KOKKOS_LAMBDA(const TeamMember& team)
        {
          for (ttb_indx ii=team.team_rank(); ii<RowBlockSize; ii+=TeamSize) {
            const ttb_indx i = team.league_rank()*RowBlockSize + ii;
            if (i >= nnz)
              continue;

            // Compute Ktensor value
            ttb_real m_val =
              compute_Ktensor_value<exec_space, FacBlockSize, VectorSize>(
                team, M, X.getSubscripts(i));

            // Evaluate link function derivative
            Kokkos::single( Kokkos::PerThread( team ), [&] ()
            {
              Y.value(i) = w * f.deriv(X.value(i), m_val);
            });
          }
        });
      }
    };

    template <typename ExecSpace, typename loss_type>
    void gcp_gradient(const SptensorT<ExecSpace>& X,
                      const SptensorT<ExecSpace>& Y,
                      const KtensorT<ExecSpace>& M,
                      const ttb_real w,
                      const loss_type& f,
                      const KtensorT<ExecSpace>& G,
                      const AlgParams& algParams)
    {
      // Compute Y tensor
      {
        GENTEN_TIME_MONITOR("GCP_Sptensor_Gradient: Y eval");

        GCP_Grad_Sptensor<ExecSpace,loss_type> kernel(X,M,w,f,Y);
        run_row_simd_kernel(kernel, M.ncomponents());
      }

      // Compute gradient
      {
        GENTEN_TIME_MONITOR("GCP_Sptensor_Gradient: mttkrp");
        G.weights() = 1.0;
        mttkrp_all(Y, M, G, algParams);
      }
    }

  }

  template <typename ExecSpace>
  PCP_Model< SptensorT<ExecSpace> >::
  PCP_Model(const tensor_type& x,
            const ktensor_type& M,
            const AlgParams& algParms) :
      X(x), algParams(algParms), f(algParams)
  {
    const ttb_indx nc = M.ncomponents();
    const ttb_indx nd = M.ndims();

    col_sums.resize(nd);
    for (ttb_indx i=0; i<nd; ++i)
      col_sums[i] = ArrayT<exec_space>(nc);

    // Create Y for gradient, same subscripts and permutation arrays but different values
    ttb_indx nnz = X.nnz();
    auto vals = typename tensor_type::vals_view_type("PCP_Model::gradient::values", nnz);
    Y = tensor_type(X.size(), vals, X.getSubscripts(), X.getPerm(), X.isSorted());
    w = ArrayT<exec_space>(nnz, 1.0);

    dku = createKtensorUpdate(x, M, algParams);
    M_overlap = dku->createOverlapKtensor(M);
    G_overlap = dku->createOverlapKtensor(M);
    //V_overlap = dku->createOverlapKtensor(M);
    //U_overlap = dku->createOverlapKtensor(M);
    for (ttb_indx  i = 0; i < x.ndims(); i++)
    {
      if (x.size(i) != M_overlap[i].nRows())
        Genten::error("Genten::PCP_Model - M and x have different size");
    }
  }

  template <typename ExecSpace>
  PCP_Model< SptensorT<ExecSpace> >::
  ~PCP_Model()
  {
    delete dku;
  }

  template <typename ExecSpace>
  void
  PCP_Model< SptensorT<ExecSpace> >::
  update(const ktensor_type& M)
  {
    // Compute column sums for each factor matrix
    const ttb_indx nd = M.ndims();
    for (ttb_indx n=0; n<nd; ++n)
      M[n].colSums(col_sums[n]);

    if (dku->overlapAliasesArg())
      M_overlap = dku->createOverlapKtensor(M);
    dku->doImport(M_overlap, M);
  }

  template <typename ExecSpace>
  ttb_real
  PCP_Model< SptensorT<ExecSpace> >::
  value(const ktensor_type& M) const
  {
    // This computes the modified Poisson loss on the nonzeros of X, which is the true loss minus the model contribution term
    ttb_real F = Impl::gcp_value(X, M_overlap, w, f);

    // Now add in the model contribution term for nonzeros and zeros
    const ttb_indx nd = M.ndims();
    const ttb_indx nc = M.ncomponents();
    ArrayT<exec_space> s(nc);
    deep_copy(s, M.weights());
    for (ttb_indx i=0; i<nd; ++i)
      s.times(col_sums[i]);
    return F+s.sum();
  }

  template <typename ExecSpace>
  void
  PCP_Model< SptensorT<ExecSpace> >::
  gradient(ktensor_type& G, const ktensor_type& M) const
  {
    // Compute gradient contribution for nonzeros with modified Poisson loss not including the 1's
    if (dku->overlapAliasesArg())
      G_overlap = dku->createOverlapKtensor(G);
    Impl::gcp_gradient(X, Y, M_overlap, 1.0, f, G_overlap, algParams);
    dku->doExport(G, G_overlap);

    // Now add in the 1's for all entries
    const ttb_indx nd = M.ndims();
    const ttb_indx nc = M.ncomponents();
    ArrayT<exec_space> s(nc);
    for (ttb_indx i=0; i<nd; ++i) {
      deep_copy(s, M.weights());
      for (ttb_indx j=0; j<nd; ++j) {
        if (i != j)
          s.times(col_sums[j]);
      }
      auto Gi = G[i];
      Gi.apply_func(KOKKOS_LAMBDA(const ttb_indx col, const ttb_indx row) { Gi.entry(row,col) += s[col]; });
    }
  }

  template <typename ExecSpace>
  ttb_real
  PCP_Model< SptensorT<ExecSpace> >::
  value_and_gradient(ktensor_type& G, const ktensor_type& M) const
  {
    ttb_real F = value(M);
    gradient(G,M);
    return F;
  }

  template <typename ExecSpace>
  void
  PCP_Model< SptensorT<ExecSpace> >::
  hess_vec(ktensor_type& U, const ktensor_type& M, const ktensor_type& V)
  {
    const ttb_indx nd = M.ndims();

    if (algParams.hess_vec_method == Hess_Vec_Method::FiniteDifference)
    {
      const ttb_real h = 1.0e-7;
      const ttb_indx nc = M.ncomponents();

      KtensorT<exec_space> Mp(nc, nd, X.size(), M.getProcessorMap()),
        Up(nc, nd, X.size(), U.getProcessorMap());
      Mp.setWeights(1.0);
      U.setWeights(1.0);
      for (ttb_indx n=0; n<nd; ++n) {
        deep_copy(Mp[n], M[n]);
        Mp[n].update(h, V[n], 1.0);
      }

      update(M);
      gradient(U, M);
      update(Mp);
      gradient(Up, Mp);

      for (ttb_indx n=0; n<nd; ++n)
        U[n].update(1.0/h, Up[n], -1.0/h);
    }
    else
      Genten::error("Unknown Hessian method");
  }

  template <typename ExecSpace>
  void
  PCP_Model< SptensorT<ExecSpace> >::
  prec_vec(ktensor_type& U, const ktensor_type& M, const ktensor_type& V)
  {
    if (algParams.hess_vec_prec_method == Hess_Vec_Prec_Method::None)
      deep_copy(U, V);
    else
      Genten::error("Unknown hess-vec preconditioner method");
  }

}

#define INST_MACRO(SPACE) template class Genten::PCP_Model< Genten::SptensorT<SPACE> >;
GENTEN_INST(INST_MACRO)
