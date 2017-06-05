//@HEADER
// ************************************************************************
//     Genten Tensor Toolbox
//     Software package for tensor math by Sandia National Laboratories
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

/*!
  @file Genten_Array.h
  @brief Data class for "flat" versions of vectors, matrices and tensors.
*/

#pragma once
#include <vector>
#include <assert.h>

#include "Genten_RandomMT.h"
#include "Genten_Util.h"

#include "Kokkos_Core.hpp"

namespace Genten {

class FacMatrix; // Forward declaration

 /*! @class Genten::Array
 *  @brief  Data class for "flat" versions of vectors, matrices and tensors.
 *
 *  The Genten::Array is similar to the
 *  std::vector<double> class. It will be used to serve "flat" versions
 *  of vectors, matrices, and tensors. It uses several typedefs defined
 *  in Genten_Util.h in order to increase future portability.
 *
 *  MKL has a vector library called VML which could be used for many
 *  of these functions (e.g., times could be a wrapper for
 *  "vdmul");
 *  this would ensure use of SIMD capabilities of Intel processors
 */
  class Array
  {
  public:
    // ----- CREATE & DESTROY -----

    //! @name Constructor/Destructor
    //@{

    //! @brief Empty constructor.
    //!
    //! Creates an empty array of length zero.
    KOKKOS_INLINE_FUNCTION
    Array() = default;

    //! @brief Size constructor.
    //!
    //! Creates an un-initialized array of length n.
    Array(ttb_indx n, bool parallel=false);

    //! @brief Size and initial value constructor.
    Array(ttb_indx n, ttb_real val);

    //! @brief (Shadow) Copy constructor.
    //!
    //! Creates an array of length n. If shdw is true, creates a shadow
    //! copy of the data passed in via the pointer d. This means that
    //! changes to this array will also change d. A shadow array can be
    //!  used as usual except that it cannot be resized beyond its original
    //!  length. If shdw is false, then the data in d is (deep) copied and
    //!  the original data is never modified.
    //!
    //!  The ability for shadow copies is used for compatibility with MATLAB.
    //!  We do not want to do deep copies of large arrays.
    Array(ttb_indx n, ttb_real * d, bool shdw = true);

    //! @brief Copy constructor.
    //!
    //! Does a (deep) copy of the data in src. The reserved size (rsz)
    //! is set to be equal to the length and may not be the same as for src.
    KOKKOS_INLINE_FUNCTION
    Array(const Array & src) = default;

    //! @brief Destructor.
    KOKKOS_INLINE_FUNCTION
    ~Array() = default;
    //@}

    //! @name Modify/Reset
    //@{

    //! @brief Copy from another Genten::Array.
    //!
    //! Does a shallow copy
    KOKKOS_INLINE_FUNCTION
    Array & operator= (const Array & src) = default;

    //! @brief Copy from another Genten::Array.
    //!
    //! Does a deep copy. Does not modify rsz unless it needs to be enlarged.
    void deep_copy(const Array & src)
    {
      assert(data.dimension_0() == src.data.dimension_0());
      Kokkos::deep_copy(data, src.data);
    }

    //! @brief Copy from a double
    void copyFrom(ttb_indx n, const ttb_real * src);

    //! @brief Copy to a double. Assumes src has n elements allocated.
    void copyTo(ttb_indx n, ttb_real * dest) const;

    //! @brief Set all entries to the specified value.
    void operator=(ttb_real val);

    //! @brief Set all entries to random values drawn uniformly from [0,1).
    //!
    //!  A new stream of Mersenne twister random numbers is generated, starting
    //!  from an arbitrary seed value.  Use scatter() for reproducibility.
    void rand();

    //! @brief Set all entries to reproducible random values drawn uniformly from [0,1).
    //!
    //!  @param[in] bUseMatlabRNG  If true, then generate random samples
    //!                            consistent with Matlab (costs twice as much
    //!                            compared with no Matlab consistency).
    //!  @param[in] cRMT           Mersenne Twister random number generator.
    //!                            The seed should already be set.
    void scatter (const bool        bUseMatlabRNG,
                  RandomMT &  cRMT);
    //@}

    //! @name Properties
    //@{

    //! @brief Returns true if size is zero, false otherwise.
    KOKKOS_INLINE_FUNCTION
    ttb_bool empty() const
    {
      return (data.dimension_0()==0);
    }

    //! @brief Tell where the inf/nan is first seen.
    bool hasNonFinite(ttb_indx &where) const;

    //! @brief Return nnz
    ttb_indx nnz() const;

    //! @brief Returns array size.
    KOKKOS_INLINE_FUNCTION
    ttb_indx size() const
    {
      return data.dimension_0();
    }

    //@}

    //! @name Element Access
    //@{

    //! @brief Return reference to value at position i (out-of-bounds checked).
    ttb_real & at(ttb_indx i) const;

    //! @brief Return reference to value at position i (no out-of-bounds check).
    KOKKOS_INLINE_FUNCTION
    ttb_real & operator[](ttb_indx i) const
    {
      assert(i < data.dimension_0());
      return(data[i]);
    }

    //@}

    //! @name Mathematical Operations
    //@{

    //! @brief Return 2-norm
    ttb_real norm(Genten::NormType ntype) const;

    //! @brief Return dot product x'*y
    ttb_real dot(const Array & y) const;

    //! @brief Return true if the two arrays are exactly equal.
    bool operator==(const Array & a) const;

    //! @brief Return true if this matrix is equal to b within the specified tolerance
    //!
    //! Being equal means that the two matrices are the same size and
    //!
    //!         fabs(x(i) - y(i))
    //!   ----------------------------   < TOL
    //!   max(1, fabs(x(i)), fabs(y(i))
    //!
    //!   for all i.
    bool isEqual(const Array & y, ttb_real tol) const;

    //! @brief x = a * x
    void times(ttb_real a);

    //! @brief x = a * y
    void times(ttb_real a, const Array & y);

    //! @brief x = a / x
    void invert(ttb_real a);

    // x = a / y
    void invert(ttb_real a, const Array & y);

    //! @brief x = x^a
    void power(ttb_real a);

    //! @brief x = y^a
    void power(ttb_real a, const Array & y);

    //! @brief x = a + x
    void shift(ttb_real a);

    //! @brief x = a + y
    void shift(ttb_real a, const Array & y);

    //! @brief x = x + y
    void plus(const Array & y);

    //! @brief x = x + sum(y[i])
    void plusVec(std::vector< const Array * > y);

    //! @brief x = y + z
    void plus(const Array & y, const Array & z);

    //! @brief x = x - y
    void minus(const Array & y);

    //! @brief x = y - z
    void minus(const Array & y, const Array & z);

    //! @brief x = x .* y (elementwise product)
    void times(const Array & y);

    //! @brief x = x .* y (elementwise product)
    //void times(const ttb_real * y, ttb_indx incy = 1);

    //! @brief x = y .* z (elementwise product)
    void times(const Array & y, const Array & z);

    //! @brief x = x ./ y (elementwise divide)
    void divide(const Array & y);

    //! @brief x = y ./ z (elementwise divide)
    void divide(const Array & y, const Array & z);

    //! @brief Returns sum of all the entries
    ttb_real sum() const;
    //@}

    void print(std::ostream& os) const {
      const ttb_indx sz = data.dimension_0();
      os << std::endl;
      for (ttb_indx i=0; i<sz; ++i)
        os << data[i] << " ";
      os << std::endl;
    }

  private:

    typedef Kokkos::View<ttb_real*> view_type;
    typedef Kokkos::View<ttb_real*,Kokkos::MemoryUnmanaged> unmanaged_view_type;
    typedef Kokkos::View<const ttb_real*,Kokkos::MemoryUnmanaged> unmanaged_const_view_type;

    //! Pointer to the actual data.
    view_type data;

    // ----- Special Functions for Friends -----

    // Return pointer to data.
    inline const ttb_real * ptr() const { return data.data(); }

    // Return pointer to data.
    inline ttb_real * ptr() { return data.data(); }

    friend class FacMatrix;

  };
}
