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
#include "Genten_IOtext.hpp"
#include "Kokkos_Random.hpp"

#include "Teuchos_TimeMonitor.hpp"

namespace Genten {

  //! Implementation of ROL::Vector using Genten::Ktensor
  template <typename ExecSpace>
  class RolKtensorVector : public ROL::Vector<ttb_real> {
  public:

    typedef ExecSpace exec_space;
    typedef KtensorT<ExecSpace> Ktensor_type;
    typedef FacMatrixT<ExecSpace> fac_matrix_type;
    typedef typename fac_matrix_type::view_type view_type;

    RolKtensorVector(const Ktensor_type& V_, const bool make_view) :
      nc(V_.ncomponents()), nd(V_.ndims())
    {
      if (make_view)
        V = V_;
      else {
        V = Ktensor_type(nc,nd);
        for (unsigned i=0; i<nd; ++i)
          V.set_factor(i, FacMatrixT<exec_space>(V_[i].nRows(), nc));
      }
    }

    RolKtensorVector(const unsigned nc_, const unsigned nd_,
                     const IndxArrayT<ExecSpace> & sz_) :
      V(nc_,nd_,sz_), nc(nc_), nd(nd_) {}

    virtual ~RolKtensorVector() {}

    Ktensor_type getKtensor() const { return V; }

    ROL::Ptr<RolKtensorVector> clone_vector() const
    {
      IndxArrayT<exec_space> sz(nd);
      for (unsigned j=0; j<nd; ++j)
        sz[j] = V[j].nRows();
      return ROL::makePtr< RolKtensorVector<exec_space> >(nc,nd,sz);
    }

    void copyToKtensor(const Ktensor_type& Kt) const {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::copyToKtensor");
      deep_copy(Kt, V);
      Kt.weights() = 1.0;
    }

    void copyFromKtensor(const Ktensor_type& Kt) const {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::copyFromKtensor");
      deep_copy(V, Kt);
    }

    virtual void plus(const ROL::Vector<ttb_real>& xx)
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::plus");
      const RolKtensorVector& x = dynamic_cast<const RolKtensorVector&>(xx);
      for (unsigned k=0; k<nd; ++k) {
        view_type v = V[k].view();
        view_type xv = x.V[k].view();
        V[k].apply_func(KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i)
        {
          v(i,j) += xv(i,j);
        });
      }
    }

    virtual void scale(const ttb_real alpha)
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::scale");
      for (unsigned k=0; k<nd; ++k) {
        view_type v = V[k].view();
        V[k].apply_func(KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i)
        {
          v(i,j) *= alpha;
        });
      }
    }

    virtual ttb_real dot(const ROL::Vector<ttb_real>& xx) const
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::dot");
      const RolKtensorVector& x = dynamic_cast<const RolKtensorVector&>(xx);
      ttb_real result = 0.0;
      for (unsigned k=0; k<nd; ++k) {
        view_type v = V[k].view();
        view_type xv = x.V[k].view();
        ttb_real local_result = 0.0;
        V[k].reduce_func(KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i,
                                       ttb_real& d)
        {
          d += v(i,j)*xv(i,j);
        }, Kokkos::Sum<ttb_real>(local_result));
        result += local_result;
      }
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
      const RolKtensorVector& x = dynamic_cast<const RolKtensorVector&>(xx);
      for (unsigned k=0; k<nd; ++k) {
        view_type v = V[k].view();
        view_type xv = x.V[k].view();
        V[k].apply_func(KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i)
        {
          v(i,j) += alpha*xv(i,j);
        });
      }
    }

    virtual void zero()
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::zero");
      for (unsigned k=0; k<nd; ++k) {
        view_type v = V[k].view();
        V[k].apply_func(KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i)
        {
          v(i,j) = 0.0;
        });
      }
    }

    virtual ROL::Ptr< ROL::Vector<ttb_real> > basis(const int idx) const
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::basis");
      unsigned k = 0;
      ttb_indx n = 0;
      while (k < nd && n+V[k].nRows()*nc < idx) {
        n += V[k].nRows()*nc;
        ++k;
      }
      const unsigned i = (idx-n) / nc;
      const unsigned j = (idx-n) - i*nc;
      assert(k < nd);
      assert(i < V[k].nRows());
      assert(j < nc);

      ROL::Ptr<RolKtensorVector> x = clone_vector();
      view_type xv = x->V[k].view();
      Kokkos::parallel_for(Kokkos::RangePolicy<exec_space>(0,1),
                           KOKKOS_LAMBDA(const ttb_indx)
      {
        xv(i,j) = 1.0;
      });
      return x;
    }

    virtual int dimension() const
    {
      ttb_indx n = 0;
      for (unsigned k=0; k<nd; ++k)
        n += V[k].nRows()*nc;
      return n;
    }

    virtual void set(const ROL::Vector<ttb_real>& xx)
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::set");
      const RolKtensorVector& x = dynamic_cast<const RolKtensorVector&>(xx);
      for (unsigned k=0; k<nd; ++k) {
        view_type v = V[k].view();
        view_type xv = x.V[k].view();
        V[k].apply_func(KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i)
        {
          v(i,j) = xv(i,j);
        });
      }
    }

    virtual const Vector<ttb_real>& dual() const
    {
      return *this;
    }

    virtual void print(std::ostream& outStream) const
    {
      Ktensor_host_type h_V =
        create_mirror_view(Genten::DefaultHostExecutionSpace(), V);
      deep_copy(h_V, V);
      print_ktensor(h_V, outStream, "Genten::RolKtensorVector");
    }

    virtual void setScalar(const ttb_real C) {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::setScalar");
      for (unsigned k=0; k<nd; ++k) {
        view_type v = V[k].view();
        V[k].apply_func(KOKKOS_LAMBDA(const ttb_indx j, const ttb_indx i)
        {
          v(i,j) = C;
        });
      }
    }

    virtual void randomize(const ttb_real l = 0.0, const ttb_real u = 1.0)
    {
      TEUCHOS_FUNC_TIME_MONITOR("ROL::Vector::randomize");
      const ttb_indx seed = std::rand();
      Kokkos::Random_XorShift64_Pool<exec_space> rand_pool(seed);
      for (unsigned k=0; k<nd; ++k) {
        view_type v = V[k].view();
        Kokkos::fill_random(v, rand_pool, l, u);
      }
    }

protected:

    typedef Genten::KtensorT<Genten::DefaultHostExecutionSpace> Ktensor_host_type;

    Ktensor_type V;
    unsigned nc;
    unsigned nd;

  };

}
