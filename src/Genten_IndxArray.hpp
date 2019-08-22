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

#include <assert.h>
#include <ostream>
#include <initializer_list>

#include "Genten_Util.hpp"
#include "Genten_IndxArray.hpp"

namespace Genten {

  /* The Genten::IndxArray class is used to store size information for tensors.
   * This is a refactored version to make better use of Kokkos, in particular
   * it uses view semantics instead of value semantics. */

template <typename ExecSpace> class IndxArrayT;
typedef IndxArrayT<DefaultHostExecutionSpace> IndxArray;

template <typename ExecSpace>
class IndxArrayT
{
public:

  typedef ExecSpace exec_space;
  typedef Kokkos::View<ttb_indx*,Kokkos::LayoutRight,ExecSpace> view_type;
  typedef Kokkos::View<ttb_indx*,typename view_type::array_layout,DefaultHostExecutionSpace> host_view_type;
  typedef typename view_type::host_mirror_space::execution_space host_mirror_space;
  typedef IndxArrayT<host_mirror_space> HostMirror;

  // ----- CREATE & DESTROY -----

  // Empty Constructor.
  /* Creates an empty size array of length 0. */
  KOKKOS_INLINE_FUNCTION
  IndxArrayT() = default;

  // Constructor of length n
  /* Not necessarily initializes to zero. */
  IndxArrayT(ttb_indx n);

  // Constructor of length n and initialize all entries to given value
  IndxArrayT(ttb_indx n, ttb_indx val);

  // Constructor from initializer list
  template <typename T>
  IndxArrayT(const std::initializer_list<T>& v) :
    IndxArrayT(v.size())
  {
    ttb_indx i = 0;
    for (const T& x : v)
    {
      host_data[i++] = (ttb_indx) x;
    }
    if (!Kokkos::Impl::MemorySpaceAccess< typename DefaultHostExecutionSpace::memory_space, typename ExecSpace::memory_space >::accessible)
      deep_copy( data, host_data );
  }

  // Copy Constructor.
  /* Creates a size array of length n, making a (deep) copy of the elements
     of v. */
  IndxArrayT(ttb_indx n, ttb_indx * v);

  // Copy Constructor (converts from double).
  /* Creates a size array of length n, copying and converting the entries of v.
     This is needed for MATLAB MEX compatibility because it passes some integer
     arrays as doubles. */
  IndxArrayT(ttb_indx n, const ttb_real * v);

  //! @brief Create array from supplied view
  IndxArrayT(const view_type& v) : data(v) {
    if (Kokkos::Impl::MemorySpaceAccess< typename DefaultHostExecutionSpace::memory_space, typename ExecSpace::memory_space >::accessible)
      host_data = host_view_type(data.data(), data.extent(0));
    else {
      host_data = host_view_type("Genten::IndxArray::host_data",
                                 data.extent(0));
      deep_copy( host_data, data );
    }
  }

  // Copy Constructor.
  /* Does a (deep) copy of the data in src. The reserved size (rsz)
     is set to be equal to the length and may not be the same as for src. */
  KOKKOS_INLINE_FUNCTION
  IndxArrayT(const IndxArrayT & src) = default;

  // Leave-one-out Copy Constructor.
  /* Does a (deep) copy of the data in src except for the n-th element.
     The reserved size (rsz) is set to be equal to the length */
  IndxArrayT(const IndxArrayT & src, ttb_indx n);

  // Destructor.
  KOKKOS_INLINE_FUNCTION
  ~IndxArrayT() = default;

  // ----- MODIFY & RESET -----

  // Copies data from src, automatically resizing if necessary.
  KOKKOS_INLINE_FUNCTION
  IndxArrayT& operator=(const IndxArrayT & src) = default;

  template <typename IndxType>
  KOKKOS_INLINE_FUNCTION
  IndxArrayT& operator=(const IndxType& indx)
  { data = indx;
    return *this;
  }

  // Zero out all the data.
  void zero();

  // ----- PROPERTIES -----

  // Returns true if size is zero, false otherwise.
  KOKKOS_INLINE_FUNCTION
  ttb_bool empty() const
  {
    return (data.extent(0) == 0);
  }

  // Returns array length.
  KOKKOS_INLINE_FUNCTION
  ttb_indx size() const
  {
    return data.extent(0);
  }

  // ----- ELEMENT ACCESS -----

  // Return reference to value at position i (no out-of-bounds check).
  // ETP 10/26/17:  This is very dangerous with the simulatneous host-
  // device data and should be fixed.
  KOKKOS_INLINE_FUNCTION
  ttb_indx & operator[](ttb_indx i) const
  {
    assert(i < data.extent(0));
    if (Kokkos::Impl::MemorySpaceAccess< typename Kokkos::Impl::ActiveExecutionMemorySpace::memory_space, typename ExecSpace::memory_space >::accessible)
      return data[i];
    else
      return host_data[i];
  }

  // Return reference to value at position i (out-of-bounds check).
  ttb_indx & at(ttb_indx i) const;

  // ----- FUNCTIONS -----

  // Comparison operators (according to lexicographic order)
  ttb_bool operator==(const IndxArrayT & a) const;
  ttb_bool operator<=(const IndxArrayT & a) const;

  // Returns product of all the entries (total size).
  /* Returns dflt for an emtpy array. */
  KOKKOS_INLINE_FUNCTION
  ttb_indx prod(ttb_indx dflt = 0) const
  {
    const ttb_indx sz = host_data.extent(0);
    if (sz == 0)
    {
      return dflt;
    }

    ttb_indx p = 1;
    for (ttb_indx i = 0; i < sz; i ++)
    {
      p = p * host_data[i];
    }
    return p;
  }

  // Compute the product of entries i through j-1.
  /* Returns dflt if j <= i. */
  ttb_indx prod(ttb_indx i, ttb_indx j, ttb_indx dflt = 0) const;

  // Compute cummulative product of entries in src and store in this object.
  /* Computes the cummulative product of the entries in src. The result is the same size as the src array. Entry 0 is 1, and
     the i-th entry is the product of the first (i-1) entries of the src array. */
  void cumprod(const IndxArrayT & src);

  // Return true if this is a valid permutation, false otherwise.
  ttb_bool isPermutation() const;

  // Increments entries in lexicographic order with respect to dims
  void increment(const IndxArrayT & dims);

  KOKKOS_INLINE_FUNCTION
  view_type values() const { return data; }

  KOKKOS_INLINE_FUNCTION
  host_view_type host_values() const { return host_data; }

  IndxArrayT clone() const {
    IndxArrayT v(data.size());
    deep_copy(v.data, data);
     if (!Kokkos::Impl::MemorySpaceAccess< typename Kokkos::Impl::ActiveExecutionMemorySpace::memory_space, typename ExecSpace::memory_space >::accessible)
       deep_copy(v.host_data, host_data);
     return v;
  }

private:

  typedef Kokkos::View<ttb_indx*,typename view_type::array_layout,ExecSpace,Kokkos::MemoryUnmanaged> unmanaged_view_type;
  typedef Kokkos::View<const ttb_indx*,typename view_type::array_layout,ExecSpace,Kokkos::MemoryUnmanaged> unmanaged_const_view_type;

  // Pointer to the actual data. managed with new/delete[]
  view_type data;
  host_view_type host_data;
};

template <typename ExecSpace>
typename IndxArrayT<ExecSpace>::HostMirror
create_mirror_view(const IndxArrayT<ExecSpace>& a)
{
  typedef typename IndxArrayT<ExecSpace>::HostMirror HostMirror;
  return HostMirror( create_mirror_view(a.values()) );
}

template <typename Space, typename ExecSpace>
IndxArrayT<Space>
create_mirror_view(const Space& s, const IndxArrayT<ExecSpace>& a)
{
  return IndxArrayT<Space>( create_mirror_view(s, a.values()) );
}

template <typename E1, typename E2>
void deep_copy(const IndxArrayT<E1>& dst, const IndxArrayT<E2>& src)
{
  deep_copy(dst.values(), src.values());
  if (!Kokkos::Impl::MemorySpaceAccess< typename DefaultHostExecutionSpace::memory_space, typename E1::memory_space >::accessible)
    deep_copy(dst.host_values(), src.host_values());
}

// Allow printing of IndxArray to screen
template <typename E>
std::ostream& operator << (std::ostream& os, const IndxArrayT<E>& a)
{
  os << "[ ";
  for (ttb_indx i=0; i<a.size(); ++i)
    os << a[i] << " ";
  os << "]";
  return os;
}

}
