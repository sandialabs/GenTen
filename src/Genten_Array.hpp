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
  @file Genten_Array.h
  @brief Data class for "flat" versions of vectors, matrices and tensors.
*/

#pragma once
#include <vector>
#include <cassert>

#include "Genten_Util.hpp"
#include "Genten_RandomMT.hpp"

namespace Genten {

  template <typename ExecSpace> class ArrayT;
  typedef ArrayT<DefaultHostExecutionSpace> Array;

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
  template <typename ExecSpace>
  class ArrayT
  {
  public:
    typedef ExecSpace exec_space;
    typedef Kokkos::View<ttb_real*,Kokkos::LayoutRight,exec_space> view_type;
    typedef typename view_type::host_mirror_space::execution_space host_mirror_space;
    typedef ArrayT<host_mirror_space> HostMirror;

    // ----- CREATE & DESTROY -----

    //! @name Constructor/Destructor
    //@{

    //! @brief Empty constructor.
    //!
    //! Creates an empty array of length zero.
    KOKKOS_DEFAULTED_FUNCTION
    ArrayT() = default;

    //! @brief Size constructor.
    //!
    //! Creates an un-initialized array of length n.
    ArrayT(ttb_indx n, bool parallel=false);

    //! @brief Size and initial value constructor.
    ArrayT(ttb_indx n, ttb_real val);

    //! @brief Create array from supplied view
    KOKKOS_INLINE_FUNCTION
    ArrayT(const view_type& v) : data(v) {}

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
    ArrayT(ttb_indx n, ttb_real * d, bool shdw = true);

    //! @brief Copy constructor.
    //!
    //! Does a (deep) copy of the data in src. The reserved size (rsz)
    //! is set to be equal to the length and may not be the same as for src.
    KOKKOS_DEFAULTED_FUNCTION
    ArrayT(const ArrayT & src) = default;

    //! @brief Destructor.
    KOKKOS_DEFAULTED_FUNCTION
    ~ArrayT() = default;
    //@}

    //! @name Modify/Reset
    //@{

    //! @brief Copy from another Genten::ArrayT.
    //!
    //! Does a shallow copy
    KOKKOS_DEFAULTED_FUNCTION
    ArrayT & operator= (const ArrayT & src) = default;

    //! @brief Copy from a double
    void copyFrom(ttb_indx n, const ttb_real * src) const;

    //! @brief Copy to a double. Assumes src has n elements allocated.
    void copyTo(ttb_indx n, ttb_real * dest) const;

    //! @brief Set all entries to the specified value.
    void operator=(ttb_real val) const;

    //! @brief Set all entries to the specified value.
    void operator=(ttb_real val);

    //! @brief Set all entries to random values drawn uniformly from [0,1).
    //!
    //!  A new stream of Mersenne twister random numbers is generated, starting
    //!  from an arbitrary seed value.  Use scatter() for reproducibility.
    void rand() const;

    //! @brief Set all entries to reproducible random values drawn uniformly from [0,1).
    //!
    //!  @param[in] bUseMatlabRNG  If true, then generate random samples
    //!                            consistent with Matlab (costs twice as much
    //!                            compared with no Matlab consistency).
    //! @param[in] bUseParallelRNG If true, then generate random samples in
    //!                            parallel (resulting random number sequence
    //!                            will depend on number of threads and
    //!                            architecture).
    //!  @param[in] cRMT           Mersenne Twister random number generator.
    //!                            The seed should already be set.
    void scatter (const bool        bUseMatlabRNG,
                  const bool bUseParallelRNG,
                  RandomMT &  cRMT) const;
    //@}

    //! @name Properties
    //@{

    //! @brief Returns true if size is zero, false otherwise.
    KOKKOS_INLINE_FUNCTION
    ttb_bool empty() const
    {
      return (data.extent(0)==0);
    }

    //! @brief Return nnz
    ttb_indx nnz() const;

    //! @brief Returns array size.
    KOKKOS_INLINE_FUNCTION
    ttb_indx size() const
    {
      return data.extent(0);
    }

    //@}

    //! @name Element Access
    //@{

    //! @brief Return reference to value at position i (no out-of-bounds check).
    KOKKOS_INLINE_FUNCTION
    ttb_real & operator[](ttb_indx i) const
    {
      assert(i < data.extent(0));
      return(data[i]);
    }

    //@}

    //! @name Mathematical Operations
    //@{

    //! @brief Return 2-norm
    ttb_real norm(Genten::NormType ntype) const;

    //! @brief Return dot product x'*y
    ttb_real dot(const ArrayT & y) const;

    //! @brief Return true if the two arrays are exactly equal.
    bool operator==(const ArrayT & a) const;

    //! @brief Return true if this matrix is equal to b within the specified tolerance
    //!
    //! Being equal means that the two matrices are the same size and
    //!
    //!         fabs(x(i) - y(i))
    //!   ----------------------------   < TOL
    //!   max(1, fabs(x(i)), fabs(y(i))
    //!
    //!   for all i.
    bool isEqual(const ArrayT & y, ttb_real tol) const;

    //! @brief x = a * x
    void times(ttb_real a) const;

    //! @brief x = a * y
    void times(ttb_real a, const ArrayT & y) const;

    //! @brief x = a / x
    void invert(ttb_real a) const;

    // x = a / y
    void invert(ttb_real a, const ArrayT & y) const;

    //! @brief x = x^a
    void power(ttb_real a) const;

    //! @brief x = y^a
    void power(ttb_real a, const ArrayT & y) const;

    //! @brief x = a + x
    void shift(ttb_real a) const;

    //! @brief x = a + y
    void shift(ttb_real a, const ArrayT & y) const;

    //! @brief x = x + s*y
    void plus(const ArrayT & y, const ttb_real s = ttb_real(1.0)) const;

    //! @brief x = a*y + b*x
    void update(const ttb_real a, const ArrayT & y, const ttb_real b) const;

    //! @brief x = y + z
    void plus(const ArrayT & y, const ArrayT & z) const;

    //! @brief x = x - y
    void minus(const ArrayT & y) const;

    //! @brief x = y - z
    void minus(const ArrayT & y, const ArrayT & z) const;

    //! @brief x = x .* y (elementwise product)
    void times(const ArrayT & y) const;

    //! @brief x = x .* y (elementwise product)
    //void times(const ttb_real * y, ttb_indx incy = 1) const;

    //! @brief x = y .* z (elementwise product)
    void times(const ArrayT & y, const ArrayT & z) const;

    //! @brief x = x ./ y (elementwise divide)
    void divide(const ArrayT & y) const;

    //! @brief x = y ./ z (elementwise divide)
    void divide(const ArrayT & y, const ArrayT & z) const;

    //! @brief Returns sum of all the entries
    ttb_real sum() const;
    //@}

    void print(std::ostream& os) const {
      const ttb_indx sz = data.extent(0);
      os << std::endl;
      for (ttb_indx i=0; i<sz; ++i)
        os << data[i] << " ";
      os << std::endl;
    }

    KOKKOS_INLINE_FUNCTION
    view_type values() const { return data; }

    // Return pointer to data.
    inline ttb_real * ptr() const { return data.data(); }

  private:

    typedef Kokkos::View<ttb_real*,typename view_type::array_layout,exec_space,Kokkos::MemoryUnmanaged> unmanaged_view_type;
    typedef Kokkos::View<const ttb_real*,typename view_type::array_layout,exec_space,Kokkos::MemoryUnmanaged> unmanaged_const_view_type;

    //! Pointer to the actual data.
    view_type data;
  };

  template <typename ExecSpace>
  typename ArrayT<ExecSpace>::HostMirror
  create_mirror_view(const ArrayT<ExecSpace>& a)
  {
    typedef typename ArrayT<ExecSpace>::HostMirror HostMirror;
    return HostMirror( create_mirror_view(a.values()) );
  }

  template <typename Space, typename ExecSpace>
  ArrayT<Space>
  create_mirror_view(const Space& s, const ArrayT<ExecSpace>& a)
  {
    return ArrayT<Space>( create_mirror_view(s, a.values()) );
  }

  template <typename E1, typename E2>
  void deep_copy(const ArrayT<E1>& dst, const ArrayT<E2>& src)
  {
    deep_copy( dst.values(), src.values() );
  }
}
