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

#include "Teuchos_TimeMonitor.hpp"

// Choose implementation of ROL::Vector (KtensorVector or KokkosVector)
#define USE_KTENSOR_VECTOR 0

// Whether to copy the ROL::Vector into a new Ktensor before accessing data.
// This adds cost for the copy, but allows mttkrp to use a padded Ktensor
// when using RolKokkosVector.
#define COPY_KTENSOR 0

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
                     const loss_function_type& func) :
      X(x), Y(X.size(), X.getSubscripts()), M(m), f(func)
    {
#if COPY_KTENSOR
      const unsigned nd = M.ndims();
      const unsigned nc = M.ncomponents();
      G = ktensor_type(nc, nd);
      for (unsigned i=0; i<nd; ++i)
        G.set_factor(i, FacMatrixT<exec_space>(M[i].nRows(), nc));
#endif

      // Todo:  maybe do a deep copy instead so we don't have to resort?
      Y.fillComplete();
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

  };

  namespace Impl {

    template <unsigned VS, typename Func>
    void run_row_simd_kernel_impl(Func& f, const unsigned nc)
    {
      static const unsigned VS4 = 4*VS;
      static const unsigned VS3 = 3*VS;
      static const unsigned VS2 = 2*VS;
      static const unsigned VS1 = 1*VS;

      if (nc > VS3)
        f.template run<VS4,VS>();
      else if (nc > VS2)
        f.template run<VS3,VS>();
      else if (nc > VS1)
        f.template run<VS2,VS>();
      else
        f.template run<VS1,VS>();
    }

    template <typename Func>
    void run_row_simd_kernel(Func& f, const unsigned nc)
    {
      if (nc >= 96)
        run_row_simd_kernel_impl<32>(f, nc);
      else if (nc >= 48)
        run_row_simd_kernel_impl<16>(f, nc);
      else if (nc >= 8)
        run_row_simd_kernel_impl<8>(f, nc);
      else if (nc >= 4)
        run_row_simd_kernel_impl<4>(f, nc);
      else if (nc >= 2)
        run_row_simd_kernel_impl<2>(f, nc);
      else
        run_row_simd_kernel_impl<1>(f, nc);
    }

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

        /*const*/ ttb_indx nnz = X.nnz();
        /*const*/ unsigned nd = M.ndims();
        /*const*/ unsigned nc = M.ncomponents();
        const ttb_indx N = (nnz+RowBlockSize-1)/RowBlockSize;

        Policy policy(N, TeamSize, VectorSize);
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team)
        {
          for (ttb_indx ii=team.team_rank(); ii<RowBlockSize; ii+=TeamSize) {
            const ttb_indx i = team.league_rank()*RowBlockSize + ii;
            if (i >= nnz)
              continue;

            // Compute Ktensor value
            typedef TinyVec<exec_space, ttb_real, unsigned, FacBlockSize, FacBlockSize, VectorSize> TV1;

            TV1 m_val(FacBlockSize,0.0);

            auto row_func = [&](auto j, auto nj, auto Nj) {
              typedef TinyVec<exec_space, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TV2;
              TV2 tmp(nj, 0.0);
              tmp.load(&(M.weights(j)));
              for (unsigned m=0; m<nd; ++m)
                tmp *= &(M[m].entry(X.subscript(i,m),j));
              m_val += tmp;
            };

            for (unsigned j=0; j<nc; j+=FacBlockSize) {
              if (j+FacBlockSize < nc) {
                const unsigned nj = FacBlockSize;
                row_func(j, nj, std::integral_constant<unsigned,FacBlockSize>());
              }
              else {
                const unsigned nj = nc-j;
                row_func(j, nj, std::integral_constant<unsigned,0>());
              }
            }

            // Evaluate link function derivative
            Y.value(i) = f.deriv(X.value(i), m_val.sum()) / nnz;
          }
        });
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

        /*const*/ ttb_indx nnz = X.nnz();
        /*const*/ unsigned nd = M.ndims();
        /*const*/ unsigned nc = M.ncomponents();
        const ttb_indx N = (nnz+RowBlockSize-1)/RowBlockSize;

        Policy policy(N, TeamSize, VectorSize);
        ttb_real v = 0.0;
        Kokkos::parallel_reduce(policy, KOKKOS_LAMBDA(const TeamMember& team,
                                                      ttb_real& d)
        {
          for (ttb_indx ii=team.team_rank(); ii<RowBlockSize; ii+=TeamSize) {
            const ttb_indx i = team.league_rank()*RowBlockSize + ii;
            if (i >= nnz)
              continue;

            // Compute Ktensor value
            typedef TinyVec<exec_space, ttb_real, unsigned, FacBlockSize, FacBlockSize, VectorSize> TV1;

            TV1 m_val(FacBlockSize,0.0);

            auto row_func = [&](auto j, auto nj, auto Nj) {
              typedef TinyVec<exec_space, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TV2;
              TV2 tmp(nj, 0.0);
              tmp.load(&(M.weights(j)));
              for (unsigned m=0; m<nd; ++m)
                tmp *= &(M[m].entry(X.subscript(i,m),j));
              m_val += tmp;
            };

            for (unsigned j=0; j<nc; j+=FacBlockSize) {
              if (j+FacBlockSize < nc) {
                const unsigned nj = FacBlockSize;
                row_func(j, nj, std::integral_constant<unsigned,FacBlockSize>());
              }
              else {
                const unsigned nj = nc-j;
                row_func(j, nj, std::integral_constant<unsigned,0>());
              }
            }

            // Evaluate link function
            d += f.value(X.value(i), m_val.sum());
          }
        }, v);
        Kokkos::fence();  // ensure v is updated before using it
        value = v / nnz;
      }
    };

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
      });
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
      Kokkos::parallel_reduce(policy,
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

    // Compute Y tensor
    {
      TEUCHOS_FUNC_TIME_MONITOR("GCP_RolObjective::gradient: Y eval");
      Impl::gcp_grad_tensor(X, M, f, Y);
    }

    {
      TEUCHOS_FUNC_TIME_MONITOR("GCP_RolObjective::gradient: mttkrp");
      // Compute gradient
      // Todo: new mttkrp kernel that does all nd dimensions
      G.weights() = 1.0;
      const unsigned nd = M.ndims();
      for (unsigned m=0; m<nd; ++m)
        mttkrp(Y, M, m, G[m]);
    }

    // Convert Ktensor to vector
#if COPY_KTENSOR
    g.copyFromKtensor(G);
#endif
  }

}
