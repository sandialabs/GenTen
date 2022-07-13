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

#include <cstdlib>
#include <cassert>

#include "Genten_Kokkos.hpp"
#include "Genten_Ktensor.hpp"
#include "Kokkos_Random.hpp"

namespace Genten {

  // Treats a Ktensor as a vector
  template <typename ExecSpace>
  class KokkosVector {
  public:

    typedef ExecSpace exec_space;
    typedef Kokkos::View<ttb_real*,exec_space> view_type;
    typedef KtensorT<exec_space> Ktensor_type;

    KokkosVector() : nc(0), nd(0), sz(0) {}

    KokkosVector(const Ktensor_type& V_) :
      nc(V_.ncomponents()), nd(V_.ndims()), sz(nd)
    {
      for (unsigned j=0; j<nd; ++j)
        sz[j] = V_[j].nRows();
      initialize();
    }

    template <typename Space>
    KokkosVector(const unsigned nc_, const unsigned nd_,
                 const IndxArrayT<Space> & sz_) :
      nc(nc_), nd(nd_), sz(sz_.size())
    {
      deep_copy(sz, sz_);
      initialize();
    }

    KokkosVector(const KokkosVector& x, const ttb_indx mode_beg,
                 const ttb_indx mode_end) :
      nc(x.nc), nd(mode_end-mode_beg)
    {
      auto sub =
        Kokkos::subview(x.sz.values(), std::make_pair(mode_beg,mode_end));
      sz = IndxArray(sub);
      ttb_indx nb = 0;
      ttb_indx ne = 0;
      for (unsigned i=0; i<mode_beg; ++i) {
        nb += x.sz[i]*nc;
        ne += x.sz[i]*nc;
      }
      for (unsigned i=mode_beg; i<mode_end; ++i)
        ne += x.sz[i]*nc;
      v = Kokkos::subview(x.v, std::make_pair(nb, ne));
    }

    ~KokkosVector() {}

    view_type getView() const { return v; }

    // Create and return a Ktensor that is a view of the vector data
    Ktensor_type getKtensor() const
    {
      // Create Ktensor from subviews of 1-D data
      typedef FacMatrixT<exec_space> fac_matrix_type;
      Ktensor_type V(nc, nd);
      ttb_real *d = v.data();
      ttb_indx offset = 0;
      for (unsigned i=0; i<nd; ++i) {
        const unsigned nr = sz[i];
        typename fac_matrix_type::view_type s(d+offset, nr, nc);
        fac_matrix_type A(s);
        V.set_factor(i, A);
        offset += nr*nc;
      }
      V.weights() = 1.0;
      return V;
    }

    KokkosVector clone() const
    {
      return KokkosVector<exec_space>(nc,nd,sz);
    }

    KokkosVector clone(const ttb_indx mode_beg, const ttb_indx mode_end) const
    {
      const ttb_indx nm = mode_end-mode_beg;
      auto sub =
        Kokkos::subview(sz.values(), std::make_pair(mode_beg,mode_end));
      return KokkosVector(nc,nm,IndxArray(sub));
    }

    KokkosVector subview(const ttb_indx mode_beg, const ttb_indx mode_end) const
    {
      return KokkosVector(*this, mode_beg, mode_end);
    }

    void copyToKtensor(const Ktensor_type& Kt) const {
      deep_copy(Kt, getKtensor());
      Kt.weights() = 1.0;
    }

    void copyFromKtensor(const Ktensor_type& Kt) const {
      deep_copy(getKtensor(), Kt);
    }

    void plus(const KokkosVector& x)
    {
      view_type my_v = v;
      view_type xv = x.v;
      apply_func(KOKKOS_LAMBDA(const ttb_indx i)
      {
        my_v(i) += xv(i);
      }, "Genten::KokkosVector::plus");
    }

    void plus(const KokkosVector& x, const ttb_real alpha)
    {
      view_type my_v = v;
      view_type xv = x.v;
      apply_func(KOKKOS_LAMBDA(const ttb_indx i)
      {
        my_v(i) += alpha * xv(i);
      }, "Genten::KokkosVector::plus_alpha");
    }

    void elastic_difference(KokkosVector &diff, 
                            const KokkosVector& center, 
                            const ttb_real alpha)
    {

      view_type my_v = v;
      view_type diff_v = diff.v;
      view_type c_v = center.v;
      apply_func(KOKKOS_LAMBDA(const ttb_indx i)
      {
        diff_v(i) = alpha * (my_v(i) - c_v(i));
      }, "Genten::KokkosVector::elastic_difference");
    }

    void scale(const ttb_real alpha)
    {
      view_type my_v = v;
      apply_func(KOKKOS_LAMBDA(const ttb_indx i)
      {
        my_v(i) *= alpha;
      }, "Genten::KokkosVector::scale");
    }

    ttb_real dot(const KokkosVector& x) const
    {
      view_type my_v = v;
      view_type xv = x.v;
      ttb_real result = 0.0;
      reduce_func(KOKKOS_LAMBDA(const ttb_indx i, ttb_real& d)
      {
        d += my_v(i)*xv(i);
      }, result, "Genten::KokkosVector::dot");
      return result;
    }

    ttb_real norm() const
    {
      return std::sqrt(dot(*this));
    }

    ttb_real normFsq() const
    {
      return dot(*this);
    }

    ttb_real normInf() const
    {
      view_type my_v = v;
      ttb_real result = 0.0;
      Kokkos::parallel_reduce("Genten::KokkosVector::normInf",
                              Kokkos::RangePolicy<exec_space>(0,v.extent(0)),
                              KOKKOS_LAMBDA(const ttb_indx i, ttb_real& d)
      {
        using std::abs;
        if (abs(my_v(i)) > d)
          d = abs(my_v(i));
      }, Kokkos::Max<ttb_real>(result));
      return result;
    }

    void axpy(const ttb_real alpha, const KokkosVector& x)
    {
      view_type my_v = v;
      view_type xv = x.v;
      apply_func(KOKKOS_LAMBDA(const ttb_indx i)
      {
        my_v(i) += alpha*xv(i);
      }, "Genten::KokkosVector::axpy");
    }

    void zero()
    {
      view_type my_v = v;
      apply_func(KOKKOS_LAMBDA(const ttb_indx i)
      {
        my_v(i) = 0.0;
      }, "Genten::KokkosVector::zero");
    }

    int dimension() const
    {
      return v.extent(0);
    }

    void set(const KokkosVector& x)
    {
      view_type my_v = v;
      view_type xv = x.v;
      apply_func(KOKKOS_LAMBDA(const ttb_indx i)
      {
        my_v(i) = xv(i);
      }, "Genten::KokkosVector::set");
    }

    void print(std::ostream& outStream) const
    {
      host_view_type h_v = Kokkos::create_mirror_view(v);
      Kokkos::deep_copy(h_v, v);
      const ttb_indx n = h_v.extent(0);
      outStream << "v = [" << std::endl;
      for (ttb_indx i=0; i<n; ++i)
        outStream << "\t" << h_v(i) << std::endl;
      outStream << std::endl;
    }

    void setScalar(const ttb_real C) {
      view_type my_v = v;
      apply_func(KOKKOS_LAMBDA(const ttb_indx i)
      {
        my_v(i) = C;
      }, "Genten::KokkosVector::setScalar");
    }

    void randomize(const ttb_real l = 0.0, const ttb_real u = 1.0)
    {
      const ttb_indx seed = std::rand();
      Kokkos::Random_XorShift64_Pool<exec_space> rand_pool(seed);
      Kokkos::fill_random(v, rand_pool, l, u);
    }

    template <typename Func>
    void apply_func(const Func& f, const std::string& name = "") const
    {
      const ttb_indx n = v.extent(0);
      Kokkos::RangePolicy<exec_space> policy(0,n);
      Kokkos::parallel_for(policy, f, name);
    }

    template <typename Func>
    void reduce_func(const Func& f, ttb_real& d, const std::string& name = "") const
    {
      const ttb_indx n = v.extent(0);
      Kokkos::RangePolicy<exec_space> policy(0,n);
      Kokkos::parallel_reduce(name, policy, f, d);
    }

  protected:

    typedef typename view_type::HostMirror host_view_type;
    typedef typename host_view_type::execution_space host_exec_space;

    void initialize()
    {
      // Form 1-D array of data
      ttb_indx n = 0;
      for (unsigned i=0; i<nd; ++i)
        n += sz[i]*nc;
      v = view_type(Kokkos::view_alloc(Kokkos::WithoutInitializing, "v"), n);
    }

    unsigned nc;
    unsigned nd;
    IndxArray sz;  // this is on the host
    view_type v;

  };

}
