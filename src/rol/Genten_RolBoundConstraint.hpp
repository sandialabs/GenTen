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

#include "ROL_BoundConstraint.hpp"
#include "Genten_Kokkos.hpp"
#include "Genten_RolKokkosVector.hpp"
#include "Genten_RolKtensorVector.hpp"

namespace Genten {

  template <typename VectorType>
  class RolBoundConstraint {};

  //! Implementation of ROL::BoundConstraint using RolKokkosVector
  template <typename ExecSpace>
  class RolBoundConstraint< RolKokkosVector<ExecSpace> > :
    public ROL::BoundConstraint<ttb_real> {
  public:

    typedef ExecSpace exec_space;
    typedef RolKokkosVector<exec_space> vector_type;
    typedef typename vector_type::view_type view_type;

    RolBoundConstraint(const ROL::Ptr<vector_type>& lower,
                       const ROL::Ptr<vector_type>& upper,
                       const ttb_real& scale = 1.0) :
      l(lower->getView()),
      u(upper->getView()),
      s(scale),
      policy(0,l.extent(0))
    {
      compute_gap();
    }

    void compute_gap() // Can't have host/device lambda in constructor
    {
      view_type L = l;
      view_type U = u;
      ttb_real gap;
      Kokkos::parallel_reduce(
        policy,
        KOKKOS_LAMBDA(const ttb_indx i, ttb_real& min)
        {
          ttb_real gap = U(i)-L(i);
          if (gap < min)
            min = gap;
        }, Kokkos::Min<ttb_real>(gap));
      min_diff = 0.5*gap;
    }

    virtual ~RolBoundConstraint() {}

    bool isFeasible(const ROL::Vector<ttb_real>& xx)
    {
      view_type X = dynamic_cast<const vector_type&>(xx).getView();
      view_type L = l;
      view_type U = u;
      unsigned feasible;
      Kokkos::parallel_reduce(
        policy,
        KOKKOS_LAMBDA(const ttb_indx i, unsigned& fsbl)
        {
          if ( (X(i) < L(i)) || (X(i) > U(i)) )
            fsbl = 0;
          else
            fsbl = 1;
        }, Kokkos::LAnd<unsigned>(feasible));
      return feasible == 1 ? true : false;
    }

    void project(ROL::Vector<ttb_real>& x)
    {
      view_type X = dynamic_cast<vector_type&>(x).getView();
      view_type L = l;
      view_type U = u;
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ttb_indx i)
      {
        if (X(i) < L(i))
          X(i) = L(i);
        else if (X(i) > U(i))
          X(i) = U(i);
      });
    }

    void pruneLowerActive(ROL::Vector<ttb_real>& v,
                          const ROL::Vector<ttb_real>& x,
                          ttb_real eps)
    {
      view_type V = dynamic_cast<vector_type&>(v).getView();
      view_type X = dynamic_cast<const vector_type&>(x).getView();
      view_type L = l;
      ttb_real epsn = std::min(s*eps, min_diff);
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ttb_indx i)
      {
        if (X(i) <= L(i)+epsn)
          V(i) = 0.0;
      });
    }

    void pruneLowerActive(ROL::Vector<ttb_real>& v,
                          const ROL::Vector<ttb_real>& g,
                          const ROL::Vector<ttb_real>& x,
                          ttb_real xeps,
                          ttb_real geps)
    {
      view_type V = dynamic_cast<vector_type&>(v).getView();
      view_type G = dynamic_cast<const vector_type&>(g).getView();
      view_type X = dynamic_cast<const vector_type&>(x).getView();
      view_type U = u;
      ttb_real epsn = std::min(s*xeps, min_diff);
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ttb_indx i)
      {
        if (X(i) <= U(i)-epsn && G(i) > geps)
          V(i) = 0.0;
      });
    }

    void pruneUpperActive(ROL::Vector<ttb_real>& v,
                          const ROL::Vector<ttb_real>& x,
                          ttb_real eps)
    {
      view_type V = dynamic_cast<vector_type&>(v).getView();
      view_type X = dynamic_cast<const vector_type&>(x).getView();
      view_type U = u;
      ttb_real epsn = std::min(s*eps, min_diff);
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ttb_indx i)
      {
        if (X(i) >= U(i)-epsn)
          V(i) = 0.0;
      });
    }

    void pruneUpperActive(ROL::Vector<ttb_real>& v,
                          const ROL::Vector<ttb_real>& g,
                          const ROL::Vector<ttb_real>& x,
                          ttb_real xeps,
                          ttb_real geps)
    {
      view_type V = dynamic_cast<vector_type&>(v).getView();
      view_type G = dynamic_cast<const vector_type&>(g).getView();
      view_type X = dynamic_cast<const vector_type&>(x).getView();
      view_type U = u;
      ttb_real epsn = std::min(s*xeps, min_diff);
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ttb_indx i)
      {
        if (X(i) >= U(i)-epsn && G(i) < -geps)
          V(i) = 0.0;
      });
    }

    void pruneActive(ROL::Vector<ttb_real>& v,
                     const ROL::Vector<ttb_real>& x,
                     ttb_real eps)
    {
      view_type V = dynamic_cast<vector_type&>(v).getView();
      view_type X = dynamic_cast<const vector_type&>(x).getView();
      view_type L = l;
      view_type U = u;
      ttb_real epsn = std::min(s*eps, min_diff);
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ttb_indx i)
      {
        if (X(i) <= L(i)+epsn || X(i) >= U(i)-epsn)
          V(i) = 0.0;
      });
    }

    void pruneLowerActive(ROL::Vector<ttb_real>& v,
                          const ROL::Vector<ttb_real>& g,
                          const ROL::Vector<ttb_real>& x,
                          ttb_real eps)
    {
      view_type V = dynamic_cast<vector_type&>(v).getView();
      view_type G = dynamic_cast<const vector_type&>(g).getView();
      view_type X = dynamic_cast<const vector_type&>(x).getView();
      view_type L = l;
      ttb_real epsn = std::min(s*eps, min_diff);
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ttb_indx i)
      {
        if (X(i) <= L(i)+epsn && G(i) > 0.0)
          V(i) = 0.0;
      });
    }

    void pruneUpperActive(ROL::Vector<ttb_real>& v,
                          const ROL::Vector<ttb_real>& g,
                          const ROL::Vector<ttb_real>& x,
                          ttb_real eps)
    {
      view_type V = dynamic_cast<vector_type&>(v).getView();
      view_type G = dynamic_cast<const vector_type&>(g).getView();
      view_type X = dynamic_cast<const vector_type&>(x).getView();
      view_type U = u;
      ttb_real epsn = std::min(s*eps, min_diff);
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ttb_indx i)
      {
        if (X(i) >= U(i)-epsn && G(i) < 0.0)
          V(i) = 0.0;
      });
    }

    void pruneActive(ROL::Vector<ttb_real>& v,
                     const ROL::Vector<ttb_real>& g,
                     const ROL::Vector<ttb_real>& x,
                     ttb_real eps)
    {
      view_type V = dynamic_cast<vector_type&>(v).getView();
      view_type G = dynamic_cast<const vector_type&>(g).getView();
      view_type X = dynamic_cast<const vector_type&>(x).getView();
      view_type L = l;
      view_type U = u;
      ttb_real epsn = std::min(s*eps, min_diff);
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ttb_indx i)
      {
        if ( (X(i) <= L(i)+epsn && G(i) > 0.0) ||
             (X(i) >= U(i)-epsn && G(i) < 0.0) )
          V(i) = 0.0;
      });
    }

  protected:

    view_type l;
    view_type u;
    ttb_real s;
    Kokkos::RangePolicy<exec_space> policy;
    ttb_real min_diff;

  };

  //! Implementation of ROL::BoundConstraint using RolKtensorVector
  template <typename ExecSpace>
  class RolBoundConstraint< RolKtensorVector<ExecSpace> > :
    public ROL::BoundConstraint<ttb_real> {
  public:

    typedef ExecSpace exec_space;
    typedef RolKtensorVector<exec_space> vector_type;

    RolBoundConstraint(const ROL::Ptr<vector_type>& lower,
                       const ROL::Ptr<vector_type>& upper,
                       const ttb_real& scale = 1.0) :
      l(lower->getKtensor()),
      u(upper->getKtensor()),
      s(scale),
      nd(l.ndims())
    {
      compute_gap();
    }

    void compute_gap()  // Can't have host/device lambda in constructor
    {
      ttb_real gap = DOUBLE_MAX;
      for (unsigned k=0; k<nd; ++k) {
        view_type L = l[k].view();
        view_type U = u[k].view();
        ttb_real local_gap;
        Kokkos::Min<ttb_real> min_reducer(local_gap);
        l[k].reduce_func(
          KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i, ttb_real& min)
          {
            ttb_real gap = U(i,j)-L(i,j);
            if (gap < min)
              min = gap;
          }, min_reducer);
        min_reducer.join(gap, local_gap);
      }
      min_diff = 0.5*gap;
    }

    virtual ~RolBoundConstraint() {}

    bool isFeasible(const ROL::Vector<ttb_real>& xx)
    {
      Ktensor_type x = dynamic_cast<const vector_type&>(xx).getKtensor();
      unsigned feasible = 1;
      for (unsigned k=0; k<nd; ++k) {
        view_type L = l[k].view();
        view_type U = u[k].view();
        view_type X = x[k].view();
        unsigned local_feasible;
        Kokkos::LAnd<unsigned> and_reducer(local_feasible);
        x[k].reduce_func(
          KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i, unsigned& fsbl)
          {
            if ( (X(i,j) < L(i,j)) || (X(i,j) > U(i,j)) )
              fsbl = 0;
            else
              fsbl = 1;
          }, and_reducer);
        and_reducer.join(feasible, local_feasible);
      }
      return feasible == 1 ? true : false;
    }

    void project(ROL::Vector<ttb_real>& xx)
    {
      Ktensor_type x = dynamic_cast<vector_type&>(xx).getKtensor();
      for (unsigned k=0; k<nd; ++k) {
        view_type L = l[k].view();
        view_type U = u[k].view();
        view_type X = x[k].view();
        x[k].apply_func(KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i)
        {
          if (X(i,j) < L(i,j))
            X(i,j) = L(i,j);
          else if (X(i,j) > U(i,j))
            X(i,j) = U(i,j);
        });
      }
    }

    void pruneLowerActive(ROL::Vector<ttb_real>& vv,
                          const ROL::Vector<ttb_real>& xx,
                          ttb_real eps)
    {
      Ktensor_type v = dynamic_cast<vector_type&>(vv).getKtensor();
      Ktensor_type x = dynamic_cast<const vector_type&>(xx).getKtensor();
      ttb_real epsn = std::min(s*eps, min_diff);
      for (unsigned k=0; k<nd; ++k) {
        view_type L = l[k].view();
        view_type V = v[k].view();
        view_type X = x[k].view();
        v[k].apply_func(KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i)
        {
          if (X(i,j) <= L(i,j)+epsn)
            V(i,j) = 0.0;
        });
      }
    }

    void pruneUpperActive(ROL::Vector<ttb_real>& vv,
                          const ROL::Vector<ttb_real>& xx,
                          ttb_real eps)
    {
      Ktensor_type v = dynamic_cast<vector_type&>(vv).getKtensor();
      Ktensor_type x = dynamic_cast<const vector_type&>(xx).getKtensor();
      ttb_real epsn = std::min(s*eps, min_diff);
      for (unsigned k=0; k<nd; ++k) {
        view_type U = u[k].view();
        view_type V = v[k].view();
        view_type X = x[k].view();
        v[k].apply_func(KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i)
        {
          if (X(i,j) >= U(i,j)-epsn)
            V(i,j) = 0.0;
        });
      }
    }

    void pruneActive(ROL::Vector<ttb_real>& vv,
                     const ROL::Vector<ttb_real>& xx,
                     ttb_real eps)
    {
      Ktensor_type v = dynamic_cast<vector_type&>(vv).getKtensor();
      Ktensor_type x = dynamic_cast<const vector_type&>(xx).getKtensor();
      ttb_real epsn = std::min(s*eps, min_diff);
      for (unsigned k=0; k<nd; ++k) {
        view_type L = l[k].view();
        view_type U = u[k].view();
        view_type V = v[k].view();
        view_type X = x[k].view();
        v[k].apply_func(KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i)
        {
          if (X(i,j) <= L(i,j)+epsn || X(i,j) >= U(i,j)-epsn)
            V(i,j) = 0.0;
        });
      }
    }

    void pruneLowerActive(ROL::Vector<ttb_real>& vv,
                          const ROL::Vector<ttb_real>& gg,
                          const ROL::Vector<ttb_real>& xx,
                          ttb_real eps)
    {
      Ktensor_type v = dynamic_cast<vector_type&>(vv).getKtensor();
      Ktensor_type g = dynamic_cast<const vector_type&>(gg).getKtensor();
      Ktensor_type x = dynamic_cast<const vector_type&>(xx).getKtensor();
      ttb_real epsn = std::min(s*eps, min_diff);
      for (unsigned k=0; k<nd; ++k) {
        view_type L = l[k].view();
        view_type V = v[k].view();
        view_type G = g[k].view();
        view_type X = x[k].view();
        v[k].apply_func(KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i)
        {
          if (X(i,j) <= L(i,j)+epsn && G(i,j) > 0.0)
            V(i,j) = 0.0;
        });
      }
    }

    void pruneUpperActive(ROL::Vector<ttb_real>& vv,
                          const ROL::Vector<ttb_real>& gg,
                          const ROL::Vector<ttb_real>& xx,
                          ttb_real eps)
    {
      Ktensor_type v = dynamic_cast<vector_type&>(vv).getKtensor();
      Ktensor_type g = dynamic_cast<const vector_type&>(gg).getKtensor();
      Ktensor_type x = dynamic_cast<const vector_type&>(xx).getKtensor();
      ttb_real epsn = std::min(s*eps, min_diff);
      for (unsigned k=0; k<nd; ++k) {
        view_type U = u[k].view();
        view_type V = v[k].view();
        view_type G = g[k].view();
        view_type X = x[k].view();
        v[k].apply_func(KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i)
        {
          if (X(i,j) >= U(i,j)-epsn && G(i,j) < 0.0)
            V(i,j) = 0.0;
        });
      }
    }

    void pruneActive(ROL::Vector<ttb_real>& vv,
                     const ROL::Vector<ttb_real>& gg,
                     const ROL::Vector<ttb_real>& xx,
                     ttb_real eps)
    {
      Ktensor_type v = dynamic_cast<vector_type&>(vv).getKtensor();
      Ktensor_type g = dynamic_cast<const vector_type&>(gg).getKtensor();
      Ktensor_type x = dynamic_cast<const vector_type&>(xx).getKtensor();
      ttb_real epsn = std::min(s*eps, min_diff);
      for (unsigned k=0; k<nd; ++k) {
        view_type L = l[k].view();
        view_type U = u[k].view();
        view_type V = v[k].view();
        view_type G = g[k].view();
        view_type X = x[k].view();
        v[k].apply_func(KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i)
        {
          if ( (X(i,j) <= L(i,j)+epsn && G(i,j) > 0.0) ||
               (X(i,j) >= U(i,j)-epsn && G(i,j) < 0.0) )
            V(i,j) = 0.0;
        });
      }
    }

  protected:

    typedef typename vector_type::Ktensor_type Ktensor_type;
    typedef FacMatrixT<ExecSpace> fac_matrix_type;
    typedef typename fac_matrix_type::view_type view_type;

    Ktensor_type l;
    Ktensor_type u;
    ttb_real s;
    ttb_real min_diff;
    unsigned nd;

  };

}
