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

/*!
  @file Genten_FacMatArray.h
  @brief Container class for collections of Genten::FacMatrix objects.
 */

#pragma once
#include "Genten_Util.hpp"
#include "Genten_IndxArray.hpp"
#include "Genten_FacMatrix.hpp"
#include "Genten_Pmap.hpp"

namespace Genten
{

template <typename ExecSpace> class FacMatArrayT;
typedef FacMatArrayT<DefaultHostExecutionSpace> FacMatArray;

namespace Impl {

// Free function to assign a factor matrix in a non-host-accessible space
template <typename ViewType, typename MatrixType>
void assign_factor_matrix_view(const ttb_indx i, const ViewType& v,
                               const MatrixType& mat)
{
  Kokkos::RangePolicy<typename ViewType::execution_space> policy(0,1);
  Kokkos::parallel_for(
    "Genten::Impl::assign_factor_matrix_view",
    policy, KOKKOS_LAMBDA(const ttb_indx j)
  {
    v[i].assign_view(mat);
  });
}

}

template <typename ExecSpace>
class FacMatArrayT
{
public:

  typedef ExecSpace exec_space;
  typedef Kokkos::View<FacMatrixT<ExecSpace>*,Kokkos::LayoutRight,ExecSpace> view_type;
  typedef Kokkos::View<FacMatrixT<ExecSpace>*,typename view_type::array_layout,DefaultHostExecutionSpace> host_view_type;
  typedef typename view_type::host_mirror_space::execution_space host_mirror_space;
  typedef FacMatArrayT<host_mirror_space> HostMirror;

  // ----- CREATER & DESTROY -----

  // Empty constructor
  KOKKOS_INLINE_FUNCTION
  FacMatArrayT() : ref_count(nullptr) {}

  // Construct an array to hold n factor matrices.
  FacMatArrayT(ttb_indx n) : data("Genten::FacMatArray::data",n)
  {
    if (Kokkos::Impl::MemorySpaceAccess< typename DefaultHostExecutionSpace::memory_space, typename ExecSpace::memory_space >::accessible)
      host_data = host_view_type(data.data(), n);
    else
      host_data = host_view_type("Genten::FacMatArray::host_data",n);
    ref_count = new int(1);
  }

  // Construct an array to hold n factor matrices.
  FacMatArrayT(ttb_indx n, const IndxArrayT<ExecSpace> & nrow,
               ttb_indx ncol, const ProcessorMap* pmap = nullptr) :
    FacMatArrayT(n)
  {
    auto nrow_host = create_mirror_view(nrow);
    deep_copy(nrow_host, nrow);
    for (ttb_indx i=0; i<n; ++i) {
      if (pmap != nullptr)
        set_factor( i, FacMatrixT<ExecSpace>(nrow_host[i],ncol,pmap->facMap(i)) );
      else
        set_factor( i, FacMatrixT<ExecSpace>(nrow_host[i],ncol) );
    }
  }

  // Copy constructor
  KOKKOS_INLINE_FUNCTION
  FacMatArrayT(const FacMatArrayT & src) :
    data(src.data), host_data(src.host_data), ref_count(src.ref_count) {
    if (ref_count != nullptr)
      *ref_count += 1;
  }

  // Destructor.
  KOKKOS_INLINE_FUNCTION
  ~FacMatArrayT() {
    KOKKOS_IF_ON_HOST(destroyHost();)
  }

  // Set all entries of all matrices to val
  void operator=(ttb_real val) const
  {
    ttb_indx sz = size();
    for (ttb_indx i=0; i<sz; ++i)
      host_data[i] = val;
  }

  // Make a copy of an existing array.
  KOKKOS_INLINE_FUNCTION
  FacMatArrayT & operator=(const FacMatArrayT & src) {
    KOKKOS_IF_ON_HOST(copyHost(src);)
    KOKKOS_IF_ON_DEVICE(copyDevice(src);)
    return *this;
  }

  // Set a factor matrix
  void set_factor(const ttb_indx i, const FacMatrixT<ExecSpace>& src) const
  {
    gt_assert(i < size());
    host_data[i] = src;
    if (!Kokkos::Impl::MemorySpaceAccess< typename DefaultHostExecutionSpace::memory_space, typename ExecSpace::memory_space >::accessible)
      Genten::Impl::assign_factor_matrix_view(i, data, src);
  }

  // ----- PROPERTIES -----

  // Return the number of factor matrices.
  KOKKOS_INLINE_FUNCTION
  ttb_indx size() const
  {
    return data.extent(0);
  }

  // count the total ttb_reals currently stored here for any purpose
  ttb_indx reals() const;

  // ----- ELEMENT ACCESS -----

  // Return n-th factor matrix
  KOKKOS_INLINE_FUNCTION
  const FacMatrixT<ExecSpace>& operator[](ttb_indx n) const
  {
    KOKKOS_IF_ON_DEVICE(return data[n];)
    KOKKOS_IF_ON_HOST(return host_data[n];)
  }

  // Return n-th factor matrix
  KOKKOS_INLINE_FUNCTION
  FacMatrixT<ExecSpace>& operator[](ttb_indx n)
  {
    KOKKOS_IF_ON_DEVICE(return data[n];)
    KOKKOS_IF_ON_HOST(return host_data[n];)
  }

  // Return whether FacMatArrays point to same data
  template <typename E>
  bool is_same(const FacMatArrayT<E>& x) {
    return static_cast<void*>(data.data()) == static_cast<void*>(x.data.data());
  }

private:

  template <typename E> friend class FacMatArrayT;

  // Array of factor matrices, each factor matrix on device
  view_type data;

  // Host view of array of factor matrices
  host_view_type host_data;

  // To avoid deadlocks, Kokkos now requires us to deallocate host_view ourselves
  int *ref_count;

  void copyHost(const FacMatArrayT & src) {
    if (this != &src) {
      destroyHost();
      data = src.data;
      host_data = src.host_data;
      ref_count = src.ref_count;
      if (ref_count != nullptr)
        *ref_count += 1;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void copyDevice(const FacMatArrayT & src) {
    if (this != &src) {
      data = src.data;
    }
  }

  void destroyHost() {
    if (ref_count != nullptr) {
      *ref_count -= 1;
      if (*ref_count <= 0) {
        ttb_indx sz = size();
        for (ttb_indx i=0; i<sz; ++i)
          host_data[i] = FacMatrixT<ExecSpace>();
        delete ref_count;
      }
    }
  }
};

// Overload of create_mirror_view when spaces are the same
template <typename ExecSpace>
FacMatArrayT<ExecSpace>
create_mirror_view(const ExecSpace& s, const FacMatArrayT<ExecSpace>& a)
{
  return a;
}

template <typename Space, typename ExecSpace>
FacMatArrayT<Space>
create_mirror_view(const Space& s, const FacMatArrayT<ExecSpace>& src)
{
  const ttb_indx n = src.size();
  FacMatArrayT<Space> dst(n);

  for (ttb_indx i=0; i<n; ++i)
    dst.set_factor( i, create_mirror_view(s, src[i]) );

  return dst;
}

template <typename ExecSpace>
typename FacMatArrayT<ExecSpace>::HostMirror
create_mirror_view(const FacMatArrayT<ExecSpace>& a)
{
  return create_mirror_view(typename FacMatArrayT<ExecSpace>::HostMirror::exec_space(), a);
}

template <typename E1, typename E2>
void deep_copy(const FacMatArrayT<E1>& dst, const FacMatArrayT<E2>& src)
{
  gt_assert( dst.size() == src.size() );
  const ttb_indx n = dst.size();

  for (ttb_indx i=0; i<n; ++i)
    deep_copy( dst[i], src[i] );
}

}
