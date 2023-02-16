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
#include "Genten_DistKtensorUpdate.hpp"
#include "Kokkos_Random.hpp"

namespace Genten {

  // Treats a Ktensor as a vector
  template <typename ExecSpace>
  class KokkosVector {
  public:

    typedef ExecSpace exec_space;
    typedef Kokkos::View<ttb_real*,exec_space> view_type;
    typedef KtensorT<exec_space> Ktensor_type;

    KokkosVector() : nc(0), nd(0), sz(0), pmap(nullptr), dku(nullptr) {}

    KokkosVector(const Ktensor_type& V_,
                 const DistKtensorUpdate<ExecSpace> *dku_ = nullptr) :
      nc(V_.ncomponents()), nd(V_.ndims()), sz(nd), pmap(V_.getProcessorMap()),
      dku(dku_)
    {
      for (unsigned j=0; j<nd; ++j)
        sz[j] = V_[j].nRows();
      initialize();
    }

    template <typename Space>
    KokkosVector(const unsigned nc_, const unsigned nd_,
                 const IndxArrayT<Space> & sz_,
                 const ProcessorMap* pmap_ = nullptr,
                 const DistKtensorUpdate<ExecSpace> *dku_ = nullptr) :
      nc(nc_), nd(nd_), sz(sz_.size()), pmap(pmap_), dku(dku_)
    {
      deep_copy(sz, sz_);
      initialize();
    }

    KokkosVector(const KokkosVector& x, const ttb_indx mode_beg,
                 const ttb_indx mode_end) :
      nc(x.nc), nd(mode_end-mode_beg), pmap(x.pmap), dku(x.dku)
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

    KokkosVector(const KokkosVector&) = default;
    KokkosVector(KokkosVector&&) = default;
    ~KokkosVector() = default;
    KokkosVector& operator=(const KokkosVector&) = default;
    KokkosVector& operator=(KokkosVector&&) = default;

    view_type getView() const { return v; }

    // Create and return a Ktensor that is a view of the vector data
    Ktensor_type getKtensor() const
    {
      // Create Ktensor from subviews of 1-D data
      typedef FacMatrixT<exec_space> fac_matrix_type;
      Ktensor_type V(nc, nd, pmap);
      ttb_real *d = v.data();
      ttb_indx offset = 0;
      for (unsigned i=0; i<nd; ++i) {
        const unsigned nr = sz[i];
        typename fac_matrix_type::view_type s(d+offset, nr, nc);
        fac_matrix_type A(s);
        if (pmap != nullptr)
          A.setProcessorMap(pmap->facMap(i));
        V.set_factor(i, A);
        offset += nr*nc;
      }
      V.weights() = 1.0;
      return V;
    }

    KokkosVector clone() const
    {
      return KokkosVector<exec_space>(nc,nd,sz,pmap,dku);
    }

    KokkosVector clone(const ttb_indx mode_beg, const ttb_indx mode_end) const
    {
      const ttb_indx nm = mode_end-mode_beg;
      auto sub =
        Kokkos::subview(sz.values(), std::make_pair(mode_beg,mode_end));
      return KokkosVector(nc,nm,IndxArray(sub),pmap,dku);
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
      if (pmap == nullptr || pmap->gridSize() == 1) {
        reduce_func(KOKKOS_LAMBDA(const ttb_indx i, ttb_real& d)
        {
          d += my_v(i)*xv(i);
        }, result, "Genten::KokkosVector::dot");
      }
      else {
        ttb_indx n_beg = 0;
        ttb_indx n_end = 0;
        for (ttb_indx i=0; i<nd; ++i) {
          n_beg = n_end;
          n_end = n_end + sz[i]*nc;
          ttb_real r = 0.0;
          Kokkos::parallel_reduce("Genten::KokkosVector::dot",
                                  Kokkos::RangePolicy<exec_space>(n_beg,n_end),
                                  KOKKOS_LAMBDA(const ttb_indx j, ttb_real& d)
          {
            d += my_v(j)*xv(j);
          }, r);
          Kokkos::fence();
          r = pmap->facMap(i)->allReduce(r);
          result = result + r;
        }
      }

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
      ttb_real result = -DBL_MAX;
      if (pmap == nullptr || pmap->gridSize() == 1) {
        Kokkos::parallel_reduce("Genten::KokkosVector::normInf",
                                Kokkos::RangePolicy<exec_space>(0,v.extent(0)),
                                KOKKOS_LAMBDA(const ttb_indx i, ttb_real& d)
        {
          using std::abs;
          if (abs(my_v(i)) > d)
            d = abs(my_v(i));
        }, Kokkos::Max<ttb_real>(result));
      }
      else {
        ttb_indx n_beg = 0;
        ttb_indx n_end = 0;
        for (ttb_indx i=0; i<nd; ++i) {
          n_beg = n_end;
          n_end = n_end + sz[i]*nc;
          ttb_real r = -DBL_MAX;
          Kokkos::parallel_reduce("Genten::KokkosVector::normInf",
                                  Kokkos::RangePolicy<exec_space>(n_beg,n_end),
                                  KOKKOS_LAMBDA(const ttb_indx j, ttb_real& d)
          {
            using std::abs;
            if (abs(my_v(j)) > d)
              d = abs(my_v(j));
          }, Kokkos::Max<ttb_real>(r));
          Kokkos::fence();
          r = pmap->facMap(i)->allReduce(r, ProcessorMap::Max);
          if (r > result)
            result = r;
        }
      }

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

    void axpby(const ttb_real alpha, const KokkosVector& x, const ttb_real beta, const KokkosVector& y)
    {
      view_type my_v = v;
      view_type xv = x.v;
      view_type yv = y.v;
      apply_func(KOKKOS_LAMBDA(const ttb_indx i)
      {
        my_v(i) = alpha*xv(i) + beta*yv(i);
      }, "Genten::KokkosVector::axpby");
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

      // Broadcast values across each sub-grid from sub-grid root to ensure
      // consistency
      if (pmap != nullptr) {
        if (dku == nullptr)
          Genten::error("KokkosVector:randomize() called in distributed setting with null DistKtensorUpdate!");
        if (dku->isReplicated()) {
          ttb_indx n_beg = 0;
          ttb_indx n_end = 0;
          for (ttb_indx i=0; i<nd; ++i) {
            n_beg = n_end;
            n_end = n_end + sz[i]*nc;
            pmap->subGridBcast(i, v.data()+n_beg, n_end-n_beg, 0);
          }
        }
      }
    }

    template <typename Func>
    void apply_func(const Func& f, const std::string& name = "") const
    {
      const ttb_indx n = v.extent(0);
      Kokkos::RangePolicy<exec_space> policy(0,n);
      Kokkos::parallel_for(name, policy, f);
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
    const ProcessorMap* pmap;
    const DistKtensorUpdate<exec_space> *dku;

  };

}
