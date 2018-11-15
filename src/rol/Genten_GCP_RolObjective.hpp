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
#include "Genten_Ktensor.hpp"
#include "Genten_RolKokkosVector.hpp"
#include "Genten_RolKtensorVector.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_TinyVec.hpp"

#include "Genten_MTTKRP.hpp"
#include "Genten_GCP_GradTensor.hpp"

#include "Teuchos_TimeMonitor.hpp"

// Choose implementation of ROL::Vector (KtensorVector or KokkosVector)
#define USE_KTENSOR_VECTOR 0

// Whether to copy the ROL::Vector into a new Ktensor before accessing data.
// This adds cost for the copy, but allows mttkrp to use a padded Ktensor
// when using RolKokkosVector.
#define COPY_KTENSOR 0

// Whether to compute gradient tensor "Y" explicitly
#define COMPUTE_Y 0

namespace Genten {

  //! Implementation of ROL::Objective for GCP problem
  template <typename Tensor, typename LossFunction>
  class GCP_RolObjective : public ROL::Objective<ttb_real> {

  public:

    typedef Tensor tensor_type;
    typedef LossFunction loss_function_type;
    typedef typename tensor_type::exec_space exec_space;
    typedef KtensorT<exec_space> ktensor_type;
#if USE_KTENSOR_VECTOR
    typedef RolKtensorVector<exec_space> vector_type;
#else
    typedef RolKokkosVector<exec_space> vector_type;
#endif

    GCP_RolObjective(const tensor_type& x,
                     const ktensor_type& m,
                     const loss_function_type& func,
                     const AlgParams& algParms) :
      X(x),
      M(m), f(func), algParams(algParms)
    {
#if COPY_KTENSOR
      const unsigned nd = M.ndims();
      const unsigned nc = M.ncomponents();
      G = ktensor_type(nc, nd);
      for (unsigned i=0; i<nd; ++i)
        G.set_factor(i, FacMatrixT<exec_space>(M[i].nRows(), nc));
#endif

#if COMPUTE_Y
      Y = tensor_type(X.size(), X.getSubscripts());
      Y.fillComplete(); // Todo:  Deep copy instead so we don't have to re-sort?
#endif
    }

    virtual ~GCP_RolObjective() {}

    virtual ttb_real value(const ROL::Vector<ttb_real>& x, ttb_real& tol);

    virtual void gradient(ROL::Vector<ttb_real>& g,
                          const ROL::Vector<ttb_real>& x, ttb_real &tol);

    ROL::Ptr<vector_type> createDesignVector() const
    {
      return ROL::makePtr<vector_type>(M, false);
    }

  protected:

    tensor_type X;
    tensor_type Y;
    ktensor_type M;
    ktensor_type G;
    loss_function_type f;
    AlgParams algParams;

  };

  namespace Impl {

    template <typename ExecSpace, typename loss_type>
    struct GCP_Grad_Tensor {
      typedef SptensorT<ExecSpace> tensor_type;
      typedef KtensorT<ExecSpace> Ktensor_type;

      const tensor_type XX;
      const Ktensor_type MM;
      const loss_type ff;
      const tensor_type YY;

      GCP_Grad_Tensor(const tensor_type& X_, const Ktensor_type& M_,
                      const loss_type& f_, const tensor_type& Y_) :
        XX(X_), MM(M_), ff(f_), YY(Y_) {}

      template <unsigned FBS, unsigned VS>
      void run() const
      {
        typedef typename tensor_type::exec_space exec_space;
        typedef Kokkos::TeamPolicy<exec_space> Policy;
        typedef typename Policy::member_type TeamMember;

        const tensor_type X = XX;
        const Ktensor_type M = MM;
        const loss_type f = ff;
        const tensor_type Y = YY;

        static const bool is_cuda = Genten::is_cuda_space<exec_space>::value;
        static const unsigned RowBlockSize = 128;
        static const unsigned FacBlockSize = FBS;
        static const unsigned VectorSize = is_cuda ? VS : 1;
        static const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;

        const ttb_indx nnz = X.nnz();
        const ttb_indx N = (nnz+RowBlockSize-1)/RowBlockSize;

        Policy policy(N, TeamSize, VectorSize);
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team)
        {
          for (ttb_indx ii=team.team_rank(); ii<RowBlockSize; ii+=TeamSize) {
            const ttb_indx i = team.league_rank()*RowBlockSize + ii;
            if (i >= nnz)
              continue;

            // Compute Ktensor value
            ttb_real m_val =
              compute_Ktensor_value<exec_space, FacBlockSize, VectorSize>(
                M, X, i);

            // Evaluate link function derivative
            Y.value(i) = f.deriv(X.value(i), m_val) / nnz;
          }
        }, "GCP_RolObjective::gradient: Y eval");
      }
    };

    template <typename ExecSpace, typename loss_type>
    struct GCP_Value {
      typedef SptensorT<ExecSpace> tensor_type;
      typedef KtensorT<ExecSpace> Ktensor_type;

      const tensor_type XX;
      const Ktensor_type MM;
      const loss_type ff;

      ttb_real value;

      GCP_Value(const tensor_type& X_, const Ktensor_type& M_,
                const loss_type& f_) :
        XX(X_), MM(M_), ff(f_) {}

      template <unsigned FBS, unsigned VS>
      void run()
      {
        typedef typename tensor_type::exec_space exec_space;
        typedef Kokkos::TeamPolicy<exec_space> Policy;
        typedef typename Policy::member_type TeamMember;

        const tensor_type X = XX;
        const Ktensor_type M = MM;
        const loss_type f = ff;

        static const bool is_cuda = Genten::is_cuda_space<exec_space>::value;
        static const unsigned RowBlockSize = 128;
        static const unsigned FacBlockSize = FBS;
        static const unsigned VectorSize = is_cuda ? VS : 1;
        static const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;

        const ttb_indx nnz = X.nnz();
        const ttb_indx N = (nnz+RowBlockSize-1)/RowBlockSize;

        Policy policy(N, TeamSize, VectorSize);
        ttb_real v = 0.0;
        Kokkos::parallel_reduce("GCP_RolObjective::value",
                                policy, KOKKOS_LAMBDA(const TeamMember& team,
                                                      ttb_real& d)
        {
          for (ttb_indx ii=team.team_rank(); ii<RowBlockSize; ii+=TeamSize) {
            const ttb_indx i = team.league_rank()*RowBlockSize + ii;
            if (i >= nnz)
              continue;

            // Compute Ktensor value
            ttb_real m_val =
              compute_Ktensor_value<exec_space, FacBlockSize, VectorSize>(
                M, X, i);

            // Evaluate link function
            d += f.value(X.value(i), m_val);
          }
        }, v);
        Kokkos::fence();  // ensure v is updated before using it
        value = v / nnz;
      }
    };

#if 1
    template <typename tensor_type, typename loss_type>
    struct GCP_Grad {
      typedef typename tensor_type::exec_space exec_space;
      typedef KtensorT<exec_space> Ktensor_type;

      const tensor_type X;
      const Ktensor_type M;
      const loss_type f;
      const Ktensor_type G;
      const AlgParams algParams;

      GCP_Grad(const tensor_type& X_, const Ktensor_type& M_,
               const loss_type& f_, const Ktensor_type& G_,
               const AlgParams& algParams_) :
        X(X_), M(M_), f(f_), G(G_), algParams(algParams_) {}

      template <unsigned FBS, unsigned VS>
      void run() const {
        using Kokkos::Experimental::ScatterDuplicated;
        using Kokkos::Experimental::ScatterNonDuplicated;
        using Kokkos::Experimental::ScatterAtomic;
        using Kokkos::Experimental::ScatterNonAtomic;

        typedef SpaceProperties<exec_space> space_prop;

        MTTKRP_Method::type method = algParams.mttkrp_method;

        // Compute default MTTKRP method
        if (method == MTTKRP_Method::Default)
          method = MTTKRP_Method::computeDefault<exec_space>();

        // Check if Perm is selected, that perm is computed
        if (method == MTTKRP_Method::Perm && !X.havePerm())
          Genten::error("Perm MTTKRP method selected, but permutation array not computed!");

        // Never use Duplicated or Atomic for Serial, use Single instead
        if (space_prop::is_serial && (method == MTTKRP_Method::Duplicated ||
                                      method == MTTKRP_Method::Atomic))
          method = MTTKRP_Method::Single;

        // Never use Duplicated for Cuda, use Atomic instead
        if (space_prop::is_cuda && method == MTTKRP_Method::Duplicated)
          method = MTTKRP_Method::Atomic;

        GCP_GradTensor<tensor_type,loss_type,FBS,VS> XX(X, M, f);
        const unsigned nd = M.ndims();
        for (unsigned n=0; n<nd; ++n) {
          if (method == MTTKRP_Method::Single ||
              Genten::is_serial_space<exec_space>::value)
            mttkrp_kernel<ScatterNonDuplicated,ScatterNonAtomic,FBS,VS>(
              XX,M,n,G[n],algParams);
          else if (method == MTTKRP_Method::Atomic)
            mttkrp_kernel<ScatterNonDuplicated,ScatterAtomic,FBS,VS>(
              XX,M,n,G[n],algParams);
          else if (method == MTTKRP_Method::Duplicated)
            mttkrp_kernel<ScatterDuplicated,ScatterNonAtomic,FBS,VS>(
              XX,M,n,G[n],algParams);
          else if (method == MTTKRP_Method::Perm)
            mttkrp_kernel_perm<FBS,VS>(XX,M,n,G[n],algParams);
        }
      }
    };
#else
    template <int Dupl, int Cont, unsigned FBS, unsigned VS,
              typename SparseTensor, typename ExecSpace, typename loss_type>
    void
    gcp_mttkrp_kernel(const SparseTensor& X,
                      const Genten::KtensorT<ExecSpace>& M,
                      const loss_type& f,
                      const unsigned n,
                      const Genten::FacMatrixT<ExecSpace>& Gn,
                      const AlgParams& algParams)
    {
      Gn = ttb_real(0.0);

      using Kokkos::Experimental::create_scatter_view;
      using Kokkos::Experimental::ScatterView;
      using Kokkos::Experimental::ScatterSum;

      static const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;
      static const unsigned RowBlockSize = 128;
      static const unsigned FacBlockSize = FBS;
      static const unsigned VectorSize = is_cuda ? VS : 1;
      static const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;
      static const unsigned RowsPerTeam = TeamSize * RowBlockSize;

      /*const*/ unsigned nd = M.ndims();
      /*const*/ unsigned nc_total = M.ncomponents();
      const ttb_indx nnz = X.nnz();
      const ttb_indx N = (nnz+RowsPerTeam-1)/RowsPerTeam;

      typedef Kokkos::TeamPolicy<ExecSpace> Policy;
      typedef typename Policy::member_type TeamMember;
      Policy policy(N, TeamSize, VectorSize);

      // Use factor matrix tile size as requested by the user, or all columns if
      // unspecified
      const unsigned FacTileSize =
        algParams.mttkrp_duplicated_factor_matrix_tile_size > 0 ? algParams.mttkrp_duplicated_factor_matrix_tile_size : nc_total;
      for (unsigned nc_beg=0; nc_beg<nc_total; nc_beg += FacTileSize) {
        const unsigned nc =
          nc_beg+FacTileSize <= nc_total ? FacTileSize : nc_total-nc_beg;
        const unsigned nc_end = nc_beg+nc;
        auto vv = Kokkos::subview(Gn.view(),Kokkos::ALL,
                                  std::make_pair(nc_beg,nc_end));
        auto sv = create_scatter_view<ScatterSum,Dupl,Cont>(vv);
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team)
        {
          auto va = sv.access();

          // Loop over tensor non-zeros with a large stride on the GPU to
          // reduce atomic contention when the non-zeros are in a nearly sorted
          // order (often the first dimension of the tensor).  This is similar
          // to an approach used in ParTi (https://github.com/hpcgarage/ParTI)
          // by Jaijai Li.
          ttb_indx offset;
          ttb_indx stride;
          if (is_cuda) {
            offset = team.league_rank()*TeamSize+team.team_rank();
            stride = team.league_size()*TeamSize;
          }
          else {
            offset =
              (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
            stride = 1;
          }
          for (unsigned ii=0; ii<RowBlockSize; ++ii) {
            const ttb_indx i = offset + ii*stride;
            if (i >= nnz)
              continue;

            const ttb_indx k = X.subscript(i,n);

            // Compute Ktensor value
            const ttb_real m_val =
              compute_Ktensor_value<ExecSpace, FacBlockSize, VectorSize>(
                M, X, i);

            // Compute Y value
            const ttb_real y_val = f.deriv(X.value(i), m_val) / nnz;

            auto row_func = [&](auto j, auto nj, auto Nj) {
              typedef TinyVec<exec_space, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TV;

              TV tmp(nj, y_val);
              tmp *= &(M.weights(mc_beg+j));
              for (unsigned m=0; m<nd; ++m) {
                if (m != n)
                  tmp *= &(M[m].entry(X.subscript(i,m),nc_beg+j));
              }
              va(k,j) += tmp;
            };

            for (unsigned j=0; j<nc; j+=FacBlockSize) {
              if (j+FacBlockSize < nc) {
                const unsigned nj = FacBlockSize;
                row_func(j, nj, std::integral_constant<unsigned,nj>());
              }
              else {
                const unsigned nj = nc-j;
                row_func(j, nj, std::integral_constant<unsigned,0>());
              }
            }
          }
        }, "mttkrp_kernel");

        sv.contribute_into(vv);
      }
    }

    template <unsigned FBS, unsigned VS,
              typename SparseTensor, typename ExecSpace, typename loss_type>
    void
    gcp_mttkrp_kernel_perm(const SparseTensor& X,
                           const Genten::KtensorT<ExecSpace>& M,
                           const loss_type& f,
                           const unsigned n,
                           const Genten::FacMatrixT<ExecSpace>& Gn,
                           const AlgParams& algParams)
    {
      Gn = ttb_real(0.0);

      static const bool is_cuda = Genten::is_cuda_space<exec_space>::value;
      static const unsigned RowBlockSize = 128;
      static const unsigned FacBlockSize = FBS;
      static const unsigned VectorSize = is_cuda ? VS : 1;
      static const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;
      static const unsigned RowsPerTeam = TeamSize * RowBlockSize;

      /*const*/ unsigned nd = M.ndims();
      /*const*/ unsigned nc = M.ncomponents();
      /*const*/ ttb_indx nnz = X.nnz();
      const ttb_indx N = (nnz+RowsPerTeam-1)/RowsPerTeam;

      typedef Kokkos::TeamPolicy<exec_space> Policy;
      typedef typename Policy::member_type TeamMember;
      Policy policy(N, TeamSize, VectorSize);

      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team)
      {
        /*const*/ ttb_indx invalid_row = ttb_indx(-1);
        /*const*/ ttb_indx i_block =
          (team.league_rank()*TeamSize + team.team_rank())*RowBlockSize;

        auto row_func = [&](auto j, auto nj, auto Nj) {
          typedef TinyVec<exec_space, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TV;
          TV val(nj, 0.0), tmp(nj, 0.0);

          ttb_indx row_prev = invalid_row;
          ttb_indx row = invalid_row;
          ttb_indx first_row = invalid_row;
          ttb_indx p = invalid_row;
          ttb_real y_val = 0.0;

          for (unsigned ii=0; ii<RowBlockSize; ++ii) {
            /*const*/ ttb_indx i = i_block+ii;
            if (i >= nnz)
              row = invalid_row;
            else {
              p = X.getPerm(i,n);
              row = X.subscript(p,n);

              // Compute Ktensor value
              const ttb_real m_val =
                compute_Ktensor_value<exec_space, FacBlockSize, VectorSize>(
                  M, X, p);

              // Compute Y value
              y_val = f.deriv(X.value(p), m_val) / nnz;
            }

            if (ii == 0)
              first_row = row;

            // If we got a different row index, add in result
            if (row != row_prev) {
              if (row_prev != invalid_row) {
                if (row_prev == first_row)
                  Kokkos::atomic_add(&(G[n].entry(row_prev,j)), val);
                else
                  val.store_plus(&(G[n].entry(row_prev,j)));
                val.broadcast(0.0);
              }
              row_prev = row;
            }

            if (row != invalid_row) {
              tmp.load(&(M.weights(j)));
              tmp *= y_val;
              for (unsigned m=0; m<nd; ++m) {
                if (m != n)
                  tmp *= &(M[m].entry(X.subscript(p,m),j));
              }
              val += tmp;
            }
          }

          // Sum in last row
          if (row != invalid_row) {
            Kokkos::atomic_add(&(Gn.entry(row,j)), val);
          }
        };

        for (unsigned j=0; j<nc; j+=FacBlockSize) {
          if (j+FacBlockSize < nc) {
            const unsigned nj = FacBlockSize;
            row_func(j, nj, std::integral_constant<unsigned,nj>());
          }
          else {
            const unsigned nj = nc-j;
            row_func(j, nj, std::integral_constant<unsigned,0>());
          }
        }
      });
    }

    template <typename tensor_type, typename loss_type>
    struct GCP_Grad {
      typedef typename tensor_type::exec_space exec_space;
      typedef KtensorT<exec_space> Ktensor_type;

      const tensor_type X;
      const Ktensor_type M;
      const loss_type f;
      const Ktensor_type G;
      const AlgParams algParams;

      GCP_Grad(const tensor_type& X_, const Ktensor_type& M_,
               const loss_type& f_, const Ktensor_type& G_,
               const AlgParams& algParams_) :
        X(X_), M(M_), f(f_), G(G_), algParams(algParams_) {}

      template <unsigned FBS, unsigned VS>
      void run() const {
        using Kokkos::Experimental::ScatterDuplicated;
        using Kokkos::Experimental::ScatterNonDuplicated;
        using Kokkos::Experimental::ScatterAtomic;
        using Kokkos::Experimental::ScatterNonAtomic;

        typedef SpaceProperties<exec_space> space_prop;

        MTTKRP_Method::type method = algParams.mttkrp_method;

        // Compute default MTTKRP method
        if (method == MTTKRP_Method::Default)
          method = MTTKRP_Method::computeDefault<exec_space>();

        // Check if Perm is selected, that perm is computed
        if (method == MTTKRP_Method::Perm && !X.havePerm())
          Genten::error("Perm MTTKRP method selected, but permutation array not computed!");

        // Never use Duplicated or Atomic for Serial, use Single instead
        if (space_prop::is_serial && (method == MTTKRP_Method::Duplicated ||
                          method == MTTKRP_Method::Atomic))
          method = MTTKRP_Method::Single;

        // Never use Duplicated for Cuda, use Atomic instead
        if (space_prop::is_cuda && method == MTTKRP_Duplicated)
          method = MTTKRP_Method::Atomic;

        GCP_GradTensor<tensor_type,loss_type,FBS,VS> XX(X, M, f);
        const unsigned nd = M.ndims();
        for (unsigned n=0; n<nd; ++n) {
          if (method == MTTKRP_Method::Single ||
              Genten::is_serial_space<exec_space>::value)
            gcp_mttkrp_kernel<ScatterNonDuplicated,ScatterNonAtomic,FBS,VS>(
              XX,M,n,G[n],algParams);
          else if (method == MTTKRP_Method::Atomic)
            gcp_mttkrp_kernel<ScatterNonDuplicated,ScatterAtomic,FBS,VS>(
              XX,M,n,G[n],algParams);
          else if (method == MTTKRP_Method::Duplicated)
            gcp_mttkrp_kernel<ScatterDuplicated,ScatterNonAtomic,FBS,VS>(
              XX,M,n,G[n],algParams);
          else if (method == MTTKRP_Method::Perm)
            gcp_mttkrp_kernel_perm<FBS,VS>(XX,M,n,G[n],algParams);
        }
      }
    };
#endif

    template <typename ExecSpace, typename loss_type>
    void gcp_grad_tensor(const SptensorT<ExecSpace>& X,
                         const KtensorT<ExecSpace>& M,
                         const loss_type& f,
                         const SptensorT<ExecSpace>& Y)
    {
#if 1
      GCP_Grad_Tensor<ExecSpace,loss_type> kernel(X,M,f,Y);
      run_row_simd_kernel(kernel, M.ncomponents());
#else
      const ttb_indx nnz = X.nnz();
      const unsigned nd = M.ndims();
      const unsigned nc = M.ncomponents();

      Kokkos::RangePolicy<ExecSpace> policy(0, nnz);
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ttb_indx i)
      {
        // Compute Ktensor value
        ttb_real m_val = 0.0;
        for (unsigned j=0; j<nc; ++j) {
          ttb_real tmp = M.weights(j);
          for (unsigned m=0; m<nd; ++m)
            tmp *= M[m].entry(X.subscript(i,m),j);
          m_val += tmp;
        }

        // Evaluate link function derivative
        Y.value(i) = f.deriv(X.value(i), m_val) / nnz;
      }, "GCP_RolObjective::gradient: Y eval");
#endif
    }

    template <typename Tensor, typename loss_type>
    void gcp_gradient(const Tensor& X,
                      const Tensor& Y,
                      const KtensorT<typename Tensor::exec_space>& M,
                      const loss_type& f,
                      const KtensorT<typename Tensor::exec_space>& G,
                      const AlgParams& algParams)
    {
#if !COMPUTE_Y
      // Compute gradient evaluating Y tensor implicitly
      {
        TEUCHOS_FUNC_TIME_MONITOR("GCP_RolObjective::gradient: mttkrp");
        G.weights() = 1.0;
        GCP_Grad<Tensor,loss_type> kernel(X,M,f,G,algParams);
        run_row_simd_kernel(kernel, M.ncomponents());
      }
#else
      // Compute Y tensor
      {
        TEUCHOS_FUNC_TIME_MONITOR("GCP_RolObjective::gradient: Y eval");
        Impl::gcp_grad_tensor(X, M, f, Y);
      }

      // Compute gradient
      {
        TEUCHOS_FUNC_TIME_MONITOR("GCP_RolObjective::gradient: mttkrp");
        G.weights() = 1.0;
        const unsigned nd = M.ndims();
        for (unsigned m=0; m<nd; ++m)
          mttkrp(Y, M, m, G[m], algParams);
      }
#endif
    }

    template <typename ExecSpace, typename loss_type>
    ttb_real gcp_value(const SptensorT<ExecSpace>& X,
                       const KtensorT<ExecSpace>& M,
                       const loss_type& f)
    {
#if 1
      GCP_Value<ExecSpace,loss_type> kernel(X,M,f);
      run_row_simd_kernel(kernel, M.ncomponents());
      return kernel.value;
#else
      const ttb_indx nnz = X.nnz();
      const unsigned nd = M.ndims();
      const unsigned nc = M.ncomponents();
      Kokkos::RangePolicy<ExecSpace> policy(0, nnz);
      ttb_real v = 0.0;
      Kokkos::parallel_reduce("GCP_RolObjective::value", policy,
                              KOKKOS_LAMBDA(const ttb_indx i, ttb_real& d)
      {
        // Compute Ktensor value
        ttb_real m_val = 0.0;
        for (unsigned j=0; j<nc; ++j) {
          ttb_real tmp = MM.weights(j);
          for (unsigned m=0; m<nd; ++m)
            tmp *= MM[m].entry(XX.subscript(i,m),j);
          m_val += tmp;
        }

        // Evaluate link function
        d += ff.value(XX.value(i), m_val);
      }, v);
      Kokkos::fence();  // ensure v is updated before using it
      return v / nnz;
#endif
    }

  }

  template <typename Tensor, typename LossFunction>
  ttb_real
  GCP_RolObjective<Tensor,LossFunction>::
  value(const ROL::Vector<ttb_real>& xx, ttb_real& tol)
  {
    TEUCHOS_FUNC_TIME_MONITOR("GCP_RolObjective::value");

    const vector_type& x = dynamic_cast<const vector_type&>(xx);

    // Convert input vector to a Ktensor
    M = x.getKtensor();
#if COPY_KTENSOR
    x.copyToKtensor(M);
#endif

    return Impl::gcp_value(X, M, f);
  }

  template <typename Tensor, typename LossFunction>
  void
  GCP_RolObjective<Tensor,LossFunction>::
  gradient(ROL::Vector<ttb_real>& gg, const ROL::Vector<ttb_real>& xx,
           ttb_real &tol)
  {
    TEUCHOS_FUNC_TIME_MONITOR("GCP_RolObjective::gradient");

    const vector_type& x = dynamic_cast<const vector_type&>(xx);
    vector_type& g = dynamic_cast<vector_type&>(gg);

    // Convert input vector to a Ktensor
    M = x.getKtensor();
    G = g.getKtensor();
#if COPY_KTENSOR
    x.copyToKtensor(M);
#endif

    Impl::gcp_gradient(X, Y, M, f, G, algParams);

    // Convert Ktensor to vector
#if COPY_KTENSOR
    g.copyFromKtensor(G);
#endif
  }

}
