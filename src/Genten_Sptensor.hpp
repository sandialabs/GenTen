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

#include "Genten_Array.hpp"
#include "Genten_IndxArray.hpp"
#include "Genten_Ktensor.hpp"

namespace Genten
{

  /* The Genten::Sptensor class stores sparse tensors.
   * This is a refactored version to make better use of Kokkos, in particular
   * it uses view semantics instead of value semantics.
   */

template <typename ExecSpace> class SptensorT;
typedef SptensorT<DefaultHostExecutionSpace> Sptensor;

template <typename ExecSpace>
class SptensorT
{

public:

  typedef ExecSpace exec_space;
  typedef Kokkos::View<ttb_indx**,Kokkos::LayoutRight,ExecSpace> subs_view_type;
  typedef Kokkos::View<ttb_real*,Kokkos::LayoutRight,ExecSpace> vals_view_type;
  typedef typename ArrayT<ExecSpace>::host_mirror_space host_mirror_space;
  typedef SptensorT<host_mirror_space> HostMirror;

  // Empty construtor.
  /* Creates an empty tensor with an empty size. */
  KOKKOS_INLINE_FUNCTION
  SptensorT() : siz(),nNumDims(0),values(),subs(),perm() {}

  // Constructor for a given size and number of nonzeros
  SptensorT(const IndxArrayT<ExecSpace>& sz, ttb_indx nz) :
    siz(sz), nNumDims(sz.size()), values(nz),
    subs("Genten::Sptensor::subs",nz,sz.size()), perm() {}

  /* Constructor from complete raw data indexed C-wise in C types.
     All input are deep copied.
     @param nd number of dimensions.
     @param dims length of each dimension.
     @param nz number of nonzeros.
     @param vals values [nz] of nonzeros.
     @param subscripts [nz*nd] coordinates of each nonzero, grouped by indices of each nonzero adjacent.
  */
  SptensorT(ttb_indx nd, ttb_indx *dims, ttb_indx nz, ttb_real *vals, ttb_indx *subscripts);

  // Constructor (for data from MATLAB).
  /* a) Copies everything locally.
     b) There are no checks for duplicate entries. Call sort() to dedup.
     c) It is assumed that sbs starts numbering at one,
     and so one is subtracted to make it start at zero. */
  SptensorT(ttb_indx nd, ttb_real * sz, ttb_indx nz, ttb_real * vls, ttb_real * sbs);

  /* Constructor from complete raw data indexed C-wise using STL types.
     All input are deep copied.
     @param dims length of each dimension.
     @param vals nonzero values.
     @param subscripts 2-d array of subscripts.
  */
  SptensorT(const std::vector<ttb_indx>& dims,
            const std::vector<ttb_real>& vals,
            const std::vector< std::vector<ttb_indx> >& subscripts);

  // Create tensor from supplied dimensions, values, and subscripts
  KOKKOS_INLINE_FUNCTION
  SptensorT(const IndxArrayT<ExecSpace>& d, const vals_view_type& vals,
            const subs_view_type& s,
            const subs_view_type& p = subs_view_type()) :
    siz(d), nNumDims(d.size()), values(vals), subs(s), perm(p) {}

  // Create tensor from supplied dimensions and subscripts, zero values
  SptensorT(const IndxArrayT<ExecSpace>& d, const subs_view_type& s) :
    siz(d), nNumDims(d.size()), values(s.extent(0),ttb_real(0.0)), subs(s), perm() {}

  // Copy constructor.
  KOKKOS_INLINE_FUNCTION
  SptensorT (const SptensorT & arg) = default;

  // Assignment operator.
  KOKKOS_INLINE_FUNCTION
  SptensorT & operator= (const SptensorT & arg) = default;

  // Destructor.
  KOKKOS_INLINE_FUNCTION
  ~SptensorT() = default;

  // Return the number of dimensions (i.e., the order).
  KOKKOS_INLINE_FUNCTION
  ttb_indx ndims() const
  {
    return nNumDims;
  }

  // Return size of dimension i.
  KOKKOS_INLINE_FUNCTION
  ttb_indx size(ttb_indx i) const
  {
    return siz[i];
  }

  // Return the entire size array.
  KOKKOS_INLINE_FUNCTION
  const IndxArrayT<ExecSpace> & size() const
  {
    return siz;
  }

  // Return the total number of (zero and nonzero) elements in the tensor.
  KOKKOS_INLINE_FUNCTION
  ttb_indx numel() const
  {
    return siz.prod();
  }

  // Return the number of structural nonzeros.
  KOKKOS_INLINE_FUNCTION
  ttb_indx nnz() const
  {
    return values.size();
  }

  // get count of ints and reals stored by implementation
  void words(ttb_indx& icount, ttb_indx& rcount) const;

  // Return true if this Sptensor is equal to b within a specified tolerance.
  /* Being equal means that the two Sptensors are the same size, same number
   * of nonzeros, and all nonzero elements satisfy

             fabs(a(i,j) - b(i,j))
        ---------------------------------   < TOL .
        max(1, fabs(a(i,j)), fabs(b(i,j))
  */
  bool isEqual(const SptensorT & b, ttb_real tol) const;

  // Return reference to i-th nonzero
  KOKKOS_INLINE_FUNCTION
  ttb_real & value(ttb_indx i) const
  {
    assert(i < values.size());
    return values[i];
  }

  // Get whole values array
  KOKKOS_INLINE_FUNCTION
  vals_view_type getValues() const { return values.values(); }

  // Return reference to n-th subscript of i-th nonzero
  template <typename IType, typename NType>
  KOKKOS_INLINE_FUNCTION
  ttb_indx & subscript(IType i, NType n) const
  {
    assert((i < values.size()) && (n < nNumDims));
    return subs(i,n);
  }

  // Get subscripts of i-th nonzero, place into IndxArray object
  KOKKOS_INLINE_FUNCTION
  void getSubscripts(ttb_indx i,  const IndxArrayT<ExecSpace> & sub) const
  {
    assert(i < values.size());

    // This can be accomplished using subview() as below, but is a fair
    // amount slower than just manually copying into the given index array
    //sub = Kokkos::subview( subs, i, Kokkos::ALL() );

    assert(sub.size() == nNumDims);
    for (ttb_indx n = 0; n < nNumDims; n++)
    {
      sub[n] = subs(i,n);
    }
  }

  // Get whole subscripts array
  KOKKOS_INLINE_FUNCTION
  subs_view_type getSubscripts() const { return subs; }

  // Return the norm (sqrt of the sum of the squares of all entries).
  ttb_real norm() const
  {
    return values.norm(NormTwo);
  }

  // Return the i-th linearly indexed element.
  KOKKOS_INLINE_FUNCTION
  ttb_real & operator[](ttb_indx i) const
  {
    return values[i];
  }

  /* Result stored in this tensor */
  void times(const KtensorT<ExecSpace> & K, const SptensorT & X);

  // Elementwise division of input tensor X and Ktensor K.
  /* Result stored in this tensor. The argument epsilon is the minimum value allowed for the division. */
  void divide(const KtensorT<ExecSpace> & K, const SptensorT & X, ttb_real epsilon);

  KOKKOS_INLINE_FUNCTION
  ttb_indx getPerm(ttb_indx i, ttb_indx n) const
  {
    assert((i < perm.extent(0)) && (n < perm.extent(1)));
    return perm(i,n);
  }

  // Get whole perm array
  KOKKOS_INLINE_FUNCTION
  subs_view_type getPerm() const { return perm; }

  // Create permutation array by sorting each column of subs
  void createPermutation();

  // Whether permutation array is computed
  KOKKOS_INLINE_FUNCTION
  bool havePerm() const { return perm.span() == subs.span(); }

protected:

  // Size of the tensor
  IndxArrayT<ExecSpace> siz;

  // Number of dimensions, from siz.size(), but faster to store it.
  ttb_indx nNumDims;

  // Data array (an array of nonzero values)
  ArrayT<ExecSpace> values;

  // Subscript array of nonzero elements.  This vector is treated as a 2D array
  // of size nnz by nNumDims.
  subs_view_type subs;

  // Permutation array for iterating over subs in non-decreasing fashion
  subs_view_type perm;

};

template <typename ExecSpace>
typename SptensorT<ExecSpace>::HostMirror
create_mirror_view(const SptensorT<ExecSpace>& a)
{
  typedef typename SptensorT<ExecSpace>::HostMirror HostMirror;
  return HostMirror( create_mirror_view(a.size()),
                     create_mirror_view(a.getValues()),
                     create_mirror_view(a.getSubscripts()),
                     create_mirror_view(a.getPerm()) );
}

template <typename Space, typename ExecSpace>
SptensorT<Space>
create_mirror_view(const Space& s, const SptensorT<ExecSpace>& a)
{
  return SptensorT<Space>( create_mirror_view(s, a.size()),
                           create_mirror_view(s, a.getValues()),
                           create_mirror_view(s, a.getSubscripts()),
                           create_mirror_view(s, a.getPerm()) );
}

template <typename E1, typename E2>
void deep_copy(SptensorT<E1>& dst, const SptensorT<E2>& src)
{
  deep_copy( dst.size(), src.size() );
  deep_copy( dst.getValues(), src.getValues() );
  deep_copy( dst.getSubscripts(), src.getSubscripts() );
  deep_copy( dst.getPerm(), src.getPerm() );
}

}
