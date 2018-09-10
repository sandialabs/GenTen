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

namespace Genten {

  namespace Impl {

    // Find the minimum u_i-l_i
    template <typename ViewType>
    struct MinGap {
      typedef typename ViewType::execution_space execution_space;
      ViewType L_; // Lower bounds
      ViewType U_; // Upper bounds

      MinGap(const ViewType& L, const ViewType& U) : L_(L), U_(U) {}

      KOKKOS_INLINE_FUNCTION
      void operator() (const ttb_indx i, ttb_real& min) const {
        ttb_real gap = U_(i)-L_(i);
        if (gap < min)
          min = gap;
      }

      KOKKOS_INLINE_FUNCTION
      void init(ttb_real& min) const {
        min = DOUBLE_MAX;
      }

      KOKKOS_INLINE_FUNCTION
      void join(volatile ttb_real& globalMin,
                const volatile ttb_real& localMin) const {
        if (localMin < globalMin)
          globalMin = localMin;
      }
    };

    // Determine if every l_i<=x_i<=u_i
    template <typename ViewType>
    struct Feasible {
      typedef typename ViewType::execution_space execution_space;
      ViewType X_; // Optimization variable
      ViewType L_; // Lower bounds
      ViewType U_; // Upper bounds

      Feasible(const ViewType& X, const ViewType& L, const ViewType& U) :
        X_(X), L_(L), U_(U) {}

      KOKKOS_INLINE_FUNCTION
      void operator() (const ttb_indx i, unsigned& feasible) const {
        if ( (X_(i) < L_(i)) || (X_(i) > U_(i)) )
          feasible = 0;
        else
          feasible = 1;
      }

      KOKKOS_INLINE_FUNCTION
      void init(unsigned& feasible) const {
        feasible = 1;
      }

      KOKKOS_INLINE_FUNCTION
      void join(volatile unsigned& globalFeasible,
                const volatile unsigned& localFeasible) const {
        globalFeasible *= localFeasible;
      }
    };
  }

  //! Implementation of ROL::BoundConstraint using Kokkos::View
  template <typename ExecSpace>
  class RolKokkosBoundConstraint : public ROL::BoundConstraint<ttb_real> {
  public:

    typedef ExecSpace exec_space;
    typedef RolKokkosVector<exec_space> vector_type;
    typedef typename vector_type::view_type view_type;

    RolKokkosBoundConstraint(const ROL::Ptr<vector_type>& lower,
                             const ROL::Ptr<vector_type>& upper,
                             const ttb_real& scale = 1.0) :
      l(lower->getView()),
      u(upper->getView()),
      policy(0,l.extent(0)),
      s(scale)
    {
      Impl::MinGap<view_type> findmin(l,u);
      ttb_real gap = 0.0;
      Kokkos::parallel_reduce(policy,findmin,gap);
      min_diff = 0.5*gap;
    }

    virtual ~RolKokkosBoundConstraint() {}

    bool isFeasible(const ROL::Vector<ttb_real>& xx)
    {
      view_type x = dynamic_cast<const vector_type&>(xx).getView();

      unsigned feasible = 1;
      Impl::Feasible<view_type> check(x, l, u);
      Kokkos::parallel_reduce(policy,check,feasible);

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

}
