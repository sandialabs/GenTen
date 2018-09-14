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

#include "ROL_Vector.hpp"
#include "Genten_Kokkos.hpp"
#include "Genten_Ktensor.hpp"
#include "Kokkos_Random.hpp"

#include "Teuchos_TimeMonitor.hpp"

namespace Genten {

  //! Implementation of ROL::Vector using Kokkos::View
  // Note:  Variation from RolKtensorVector that creates an unmanaged
  // Ktensor from the underlying view.  This appears to be more efficient
  // for the vector operations, with the tradeoff of not creating a Ktensor
  // with padded data.
  template <typename ExecSpace>
  class RolKokkosVector : public ROL::Vector<ttb_real> {
  public:

    typedef ExecSpace exec_space;
    typedef Kokkos::View<ttb_real*,exec_space> view_type;
    typedef KtensorT<exec_space> Ktensor_type;

    RolKokkosVector(const Ktensor_type& V_, const bool make_view) :
      nc(V_.ncomponents()), nd(V_.ndims()), sz(nd)
    {
      assert(make_view == false);

      for (unsigned j=0; j<nd; ++j)
        sz[j] = V_[j].nRows();
      initialize();
    }

    RolKokkosVector(const unsigned nc_, const unsigned nd_,
                     const IndxArrayT<ExecSpace> & sz_) :
      nc(nc_), nd(nd_), sz(sz_)
    {
      initialize();
    }

    virtual ~RolKokkosVector() {}

    view_type getView() const { return v; }

    // Create and return a Ktensor that is a view of the vector data
    Ktensor_type getKtensor() const
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::getKtensor");

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

    ROL::Ptr<RolKokkosVector> clone_vector() const
    {
      return ROL::makePtr< RolKokkosVector<exec_space> >(nc,nd,sz);
    }

    void copyToKtensor(const Ktensor_type& Kt) const {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::copyToKtensor");
      deep_copy(Kt, getKtensor());
      Kt.weights() = 1.0;
    }

    void copyFromKtensor(const Ktensor_type& Kt) const {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::copyFromKtensor");
      deep_copy(getKtensor(), Kt);
    }

    virtual void plus(const ROL::Vector<ttb_real>& xx)
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::plus");
      const RolKokkosVector& x = dynamic_cast<const RolKokkosVector&>(xx);
      view_type my_v = v;
      view_type xv = x.v;
      apply_func<exec_space>(KOKKOS_LAMBDA(const ttb_indx i)
      {
        my_v(i) += xv(i);
      });
    }

    virtual void scale(const ttb_real alpha)
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::scale");
      view_type my_v = v;
      apply_func<exec_space>(KOKKOS_LAMBDA(const ttb_indx i)
      {
        my_v(i) *= alpha;
      });
    }

    virtual ttb_real dot(const ROL::Vector<ttb_real>& xx) const
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::dot");
      const RolKokkosVector& x = dynamic_cast<const RolKokkosVector&>(xx);
      view_type my_v = v;
      view_type xv = x.v;
      ttb_real result = 0.0;
      reduce_func<exec_space>(KOKKOS_LAMBDA(const ttb_indx i, ttb_real& d)
      {
        d += my_v(i)*xv(i);
      }, result);
      return result;
    }

    virtual ttb_real norm() const
    {
      return std::sqrt(dot(*this));
    }

    virtual ROL::Ptr< ROL::Vector<ttb_real> > clone() const
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::clone");
      return clone_vector();
    }

    virtual void axpy(const ttb_real alpha, const ROL::Vector<ttb_real>& xx)
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::axpy");
      const RolKokkosVector& x = dynamic_cast<const RolKokkosVector&>(xx);
      view_type my_v = v;
      view_type xv = x.v;
      apply_func<exec_space>(KOKKOS_LAMBDA(const ttb_indx i)
      {
        my_v(i) += alpha*xv(i);
      });
    }

    virtual void zero()
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::zero");
      view_type my_v = v;
      apply_func<exec_space>(KOKKOS_LAMBDA(const ttb_indx i)
      {
        my_v(i) = 0.0;
      });
    }

    virtual ROL::Ptr< ROL::Vector<ttb_real> > basis(const int i) const
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::basis");
      ROL::Ptr<RolKokkosVector> x = clone_vector();
      view_type xv = x->v;
      Kokkos::parallel_for(Kokkos::RangePolicy<exec_space>(0,1),
                           KOKKOS_LAMBDA(const ttb_indx)
      {
        xv(i) = 1.0;
      });
      return x;
    }

    virtual int dimension() const
    {
      return v.extent(0);
    }

    virtual void set(const ROL::Vector<ttb_real>& xx)
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::set");
      const RolKokkosVector& x = dynamic_cast<const RolKokkosVector&>(xx);
      view_type my_v = v;
      view_type xv = x.v;
      apply_func<exec_space>(KOKKOS_LAMBDA(const ttb_indx i)
      {
        my_v(i) = xv(i);
      });
    }

    virtual const Vector<ttb_real>& dual() const
    {
      return *this;
    }

    // Comment out applyUnary/applyBinary/reduce since they are not efficient
    // on the GPU.  These shouldn't be needed for the optimization methods
    // we're using, and this way an exception will be thrown if they are
    // called.

    /*
    virtual void applyUnary(
      const ROL::Elementwise::UnaryFunction<ttb_real>& f)
    {
      // Have to transfer to the host because these functions classes
      // use virtual functions
      host_view_type h_v = Kokkos::create_mirror_view(v);
      Kokkos::deep_copy(h_v, v);
      apply_func<host_exec_space>([&](const ttb_indx i)
      {
        h_v(i) = f.apply(h_v(i));
      });
      Kokkos::deep_copy(v, h_v);
    }

    virtual void applyBinary(
      const ROL::Elementwise::BinaryFunction<ttb_real>& f,
      const ROL::Vector<ttb_real>& xx)
    {
      const RolKokkosVector& x = dynamic_cast<const RolKokkosVector&>(xx);

      // Have to transfer to the host because these functions classes
      // use virtual functions
      host_view_type h_v = Kokkos::create_mirror_view(v);
      host_view_type h_x = Kokkos::create_mirror_view(x.v);
      Kokkos::deep_copy(h_v, v);
      Kokkos::deep_copy(h_x, x.v);
      apply_func<host_exec_space>([&](const ttb_indx i)
      {
        h_v(i) = f.apply(h_v(i), h_x(i));
      });
      Kokkos::deep_copy(v, h_v);
    }

    virtual ttb_real reduce(
      const ROL::Elementwise::ReductionOp<ttb_real>& r) const
    {
      // Have to transfer to the host because these functions classes
      // use virtual functions
      host_view_type h_v = Kokkos::create_mirror_view(v);
      Kokkos::deep_copy(h_v, v);
      ttb_real result = 0.0;
      reduce_func<host_exec_space>([&](const ttb_indx i, ttb_real& d)
      {
        r.reduce(h_v(i), d);
      }, result);
      return result;
    }
    */

    virtual void print(std::ostream& outStream) const
    {
      host_view_type h_v = Kokkos::create_mirror_view(v);
      Kokkos::deep_copy(h_v, v);
      const ttb_indx n = h_v.extent(0);
      outStream << "v = [" << std::endl;
      for (ttb_indx i=0; i<n; ++i)
        outStream << "\t" << h_v(i) << std::endl;
      outStream << std::endl;
    }

    virtual void setScalar(const ttb_real C) {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::setScalar");
      view_type my_v = v;
      apply_func<exec_space>(KOKKOS_LAMBDA(const ttb_indx i)
      {
        my_v(i) = C;
      });
    }

    virtual void randomize(const ttb_real l = 0.0, const ttb_real u = 1.0)
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::randomize");
      const ttb_indx seed = std::rand();
      Kokkos::Random_XorShift64_Pool<exec_space> rand_pool(seed);
      Kokkos::fill_random(v, rand_pool, l, u);
    }

  protected:

    typedef typename view_type::HostMirror host_view_type;
    typedef typename host_view_type::execution_space host_exec_space;

    template <typename Space, typename Func>
    void apply_func(const Func& f) const
    {
      const ttb_indx n = v.extent(0);
      Kokkos::RangePolicy<Space> policy(0,n);
      Kokkos::parallel_for(policy, f);
    }

    template <typename Space, typename Func>
    void reduce_func(const Func& f, ttb_real& d) const
    {
      const ttb_indx n = v.extent(0);
      Kokkos::RangePolicy<Space> policy(0,n);
      Kokkos::parallel_reduce(policy, f, d);
    }

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
    IndxArrayT<exec_space> sz;
    view_type v;

  };

}
