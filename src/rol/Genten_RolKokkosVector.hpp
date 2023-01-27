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

#include "ROL_Vector.hpp"
#include "Genten_Util.hpp"
#include "Genten_KokkosVector.hpp"

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

    typedef KokkosVector<ExecSpace> kokkos_vector;
    typedef ExecSpace exec_space;
    typedef typename kokkos_vector::view_type view_type;
    typedef typename kokkos_vector::Ktensor_type Ktensor_type;

    // This just copies the shape of V, not the values
    RolKokkosVector(const Ktensor_type& V, const bool make_view,
                    const DistKtensorUpdate<ExecSpace> *dku) : kv(V, dku)
    {
      gt_assert(make_view == false);
    }

    RolKokkosVector(const unsigned nc, const unsigned nd, const IndxArray& sz,
                    const ProcessorMap* pmap,
                    const DistKtensorUpdate<ExecSpace> *dku) :
      kv(nc,nd,sz,pmap,dku)
    {
    }

    // Constructor taking a kokkos_vector.  This is a shallow-copy, unlike
    // the other constructors which just copy the shape, so use carefully!
    RolKokkosVector(const kokkos_vector& kv_) : kv(kv_) {}

    virtual ~RolKokkosVector() {}

    view_type getView() const { return kv.getView(); }

    // Create and return a Ktensor that is a view of the vector data
    Ktensor_type getKtensor() const
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::getKtensor");
      return kv.getKtensor();
    }

    ROL::Ptr<RolKokkosVector> clone_vector() const
    {
      return ROL::makePtr< RolKokkosVector<exec_space> >(kv.clone());
    }

    void copyToKtensor(const Ktensor_type& Kt) const {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::copyToKtensor");
      kv.copyToKtensor(Kt);
    }

    void copyFromKtensor(const Ktensor_type& Kt) const {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::copyFromKtensor");
      kv.copyFromKtensor(Kt);
    }

    virtual void plus(const ROL::Vector<ttb_real>& xx)
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::plus");
      const RolKokkosVector& x = dynamic_cast<const RolKokkosVector&>(xx);
      kv.plus(x.kv);
    }

    virtual void scale(const ttb_real alpha)
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::scale");
      kv.scale(alpha);
    }

    virtual ttb_real dot(const ROL::Vector<ttb_real>& xx) const
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::dot");
      const RolKokkosVector& x = dynamic_cast<const RolKokkosVector&>(xx);
      return kv.dot(x.kv);
    }

    virtual ttb_real norm() const
    {
      return kv.norm();
    }

    ttb_real normInf() const
    {
      return kv.normInf();
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
      kv.axpy(alpha,x.kv);
    }

    virtual void zero()
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::zero");
      kv.zero();
    }

    virtual ROL::Ptr< ROL::Vector<ttb_real> > basis(const int i) const
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::basis");
      ROL::Ptr<RolKokkosVector> x = clone_vector();
      view_type xv = x->kv.getView();
      Kokkos::parallel_for(Kokkos::RangePolicy<exec_space>(0,1),
                           KOKKOS_LAMBDA(const ttb_indx)
      {
        xv(i) = 1.0;
      });
      return x;
    }

    virtual int dimension() const
    {
      return kv.dimension();
    }

    virtual void set(const ROL::Vector<ttb_real>& xx)
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::set");
      const RolKokkosVector& x = dynamic_cast<const RolKokkosVector&>(xx);
      kv.set(x.kv);
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
      host_view_type h_v = Kokkos::create_mirror_view(kv.getView());
      Kokkos::deep_copy(h_v, kv.getView());
      kv.apply_func<host_exec_space>([&](const ttb_indx i)
      {
        h_v(i) = f.apply(h_v(i));
      });
      Kokkos::deep_copy(kv.getView(), h_v);
    }

    virtual void applyBinary(
      const ROL::Elementwise::BinaryFunction<ttb_real>& f,
      const ROL::Vector<ttb_real>& xx)
    {
      const RolKokkosVector& x = dynamic_cast<const RolKokkosVector&>(xx);

      // Have to transfer to the host because these functions classes
      // use virtual functions
      host_view_type h_v = Kokkos::create_mirror_view(kv.getView());
      host_view_type h_x = Kokkos::create_mirror_view(x.kv.getView());
      Kokkos::deep_copy(h_v, kv.getView());
      Kokkos::deep_copy(h_x, x.kv.getView());
      kv.apply_func<host_exec_space>([&](const ttb_indx i)
      {
        h_v(i) = f.apply(h_v(i), h_x(i));
      });
      Kokkos::deep_copy(kv.getView(), h_v);
    }

    virtual ttb_real reduce(
      const ROL::Elementwise::ReductionOp<ttb_real>& r) const
    {
      // Have to transfer to the host because these functions classes
      // use virtual functions
      host_view_type h_v = Kokkos::create_mirror_view(kv.getView());
      Kokkos::deep_copy(h_v, kv.getView());
      ttb_real result = 0.0;
      kv.reduce_func<host_exec_space>([&](const ttb_indx i, ttb_real& d)
      {
        r.reduce(h_v(i), d);
      }, result);
      return result;
    }
    */

    virtual void print(std::ostream& outStream) const
    {
      kv.print(outStream);
    }

    virtual void setScalar(const ttb_real C) {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::setScalar");
      kv.setScalar(C);
    }

    virtual void randomize(const ttb_real l = 0.0, const ttb_real u = 1.0)
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::randomize");
      kv.randomize(l,u);
    }

  protected:

    kokkos_vector kv;

  };

}
