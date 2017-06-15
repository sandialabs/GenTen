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
// ************************************************************************
//@HEADER

#pragma once

#include <assert.h>
#include <initializer_list>

#include "Genten_Util.h"
#include "Genten_IndxArray.h"

#include "Kokkos_Core.hpp"

namespace Genten {

  /* The Genten::IndxArray class is used to store size information for tensors.
   * This is a refactored version to make better use of Kokkos, in particular
   * it uses view semantics instead of value semantics. */

class IndxArray
{
public:
  // ----- CREATE & DESTROY -----

  // Empty Constructor.
  /* Creates an empty size array of length 0. */
  KOKKOS_INLINE_FUNCTION
  IndxArray() = default;

  // Constructor of length n
  /* Not necessarily initializes to zero. */
  IndxArray(ttb_indx n);

  // Constructor of length n and initialize all entries to given value
  IndxArray(ttb_indx n, ttb_indx val);

  // Constructor from initializer list
  template <typename T>
  IndxArray(const std::initializer_list<T>& v) :
    data("Genten::IndxArray::data", v.size())
  {
    ttb_indx i = 0;
    for (const T& x : v)
    {
      data[i++] = (ttb_indx) x;
    }
  }

  // Copy Constructor.
  /* Creates a size array of length n, making a (deep) copy of the elements
     of v. */
  IndxArray(ttb_indx n, ttb_indx * v);

  // Copy Constructor (converts from double).
  /* Creates a size array of length n, copying and converting the entries of v.
     This is needed for MATLAB MEX compatibility because it passes some integer
     arrays as doubles. */
  IndxArray(ttb_indx n, const ttb_real * v);

  // Copy Constructor.
  /* Does a (deep) copy of the data in src. The reserved size (rsz)
     is set to be equal to the length and may not be the same as for src. */
  KOKKOS_INLINE_FUNCTION
  IndxArray(const IndxArray & src) = default;

  // Leave-one-out Copy Constructor.
  /* Does a (deep) copy of the data in src except for the n-th element.
     The reserved size (rsz) is set to be equal to the length */
  IndxArray(const IndxArray & src, ttb_indx n);

  // Destructor.
  KOKKOS_INLINE_FUNCTION
  ~IndxArray() = default;

  // ----- MODIFY & RESET -----

  // Copies data from src, automatically resizing if necessary.
  KOKKOS_INLINE_FUNCTION
  IndxArray& operator=(const IndxArray & src) = default;

  template <typename IndxType>
  KOKKOS_INLINE_FUNCTION
  IndxArray& operator=(const IndxType& indx)
  { data = indx;
    return *this;
  }

  // Deep copy
  void deep_copy(const IndxArray & src) {
    assert(data.dimension_0() == src.data.dimension_0());
    Kokkos::deep_copy(data, src.data);
  }

  // Zero out all the data.
  void zero();

  // ----- PROPERTIES -----

  // Returns true if size is zero, false otherwise.
  KOKKOS_INLINE_FUNCTION
  ttb_bool empty() const
  {
    return (data.dimension_0() == 0);
  }

  // Returns array length.
  KOKKOS_INLINE_FUNCTION
  ttb_indx size() const
  {
    return data.dimension_0();
  }

  // ----- ELEMENT ACCESS -----

  // Return reference to value at position i (no out-of-bounds check).
  KOKKOS_INLINE_FUNCTION
  ttb_indx & operator[](ttb_indx i) const
  {
    assert(i < data.dimension_0());
    return(data[i]);
  }

  // Return reference to value at position i (out-of-bounds check).
  ttb_indx & at(ttb_indx i) const;

  // ----- FUNCTIONS -----

  // Comparison operators (according to lexicographic order)
  ttb_bool operator==(const IndxArray & a) const;
  ttb_bool operator<=(const IndxArray & a) const;

  // Returns product of all the entries (total size).
  /* Returns dflt for an emtpy array. */
  KOKKOS_INLINE_FUNCTION
  ttb_indx prod(ttb_indx dflt = 0) const
  {
    const ttb_indx sz = data.dimension_0();
    if (sz == 0)
    {
      return dflt;
    }

    ttb_indx p = 1;
    for (ttb_indx i = 0; i < sz; i ++)
    {
      p = p * data[i];
    }
    return p;
  }

  // Compute the product of entries i through j-1.
  /* Returns dflt if j <= i. */
  ttb_indx prod(ttb_indx i, ttb_indx j, ttb_indx dflt = 0) const;

  // Compute cummulative product of entries in src and store in this object.
  /* Computes the cummulative product of the entries in src. The result is the same size as the src array. Entry 0 is 1, and
     the i-th entry is the product of the first (i-1) entries of the src array. */
  void cumprod(const IndxArray & src);

  // Return true if this is a valid permutation, false otherwise.
  ttb_bool isPermutation() const;

  // Increments entries in lexicographic order with respect to dims
  void increment(const IndxArray & dims);

private:

  typedef Kokkos::View<ttb_indx*> view_type;
  typedef Kokkos::View<ttb_indx*,Kokkos::MemoryUnmanaged> unmanaged_view_type;
  typedef Kokkos::View<const ttb_indx*,Kokkos::MemoryUnmanaged> unmanaged_const_view_type;

  // Pointer to the actual data. managed with new/delete[]
  view_type data;
};
}
