//@HEADER
// ************************************************************************
//     Genten Tensor Toolbox
//     Software package for tensor math by Sandia National Laboratories
//
// Sandia National Laboratories is a multimission laboratory managed
// and operated by National Technology and Engineering Solutions of Sandia,
// LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
// U.S. Department of Energy’s National Nuclear Security Administration under
// contract DE-NA0003525.
//
// Copyright 20XX, Sandia Corporation.
// ************************************************************************
//@HEADER

#pragma once

#include <assert.h>

#include "Genten_Array.h"
#include "Genten_IndxArray.h"
#include "Genten_Ktensor.h"

#include "Kokkos_Core.hpp"

namespace Genten
{

  /* The Genten::Sptensor class stores sparse tensors.
   * This is a refactored version to make better use of Kokkos, in particular
   * it uses view semantics instead of value semantics.
   */

class Sptensor
{

public:

  // Empty construtor.
  /* Creates an empty tensor with an empty size. */
  KOKKOS_INLINE_FUNCTION
  Sptensor() : siz(),nNumDims(0),values(),subs() {}

  // Constructor for a given size and number of nonzeros
  Sptensor(const IndxArray& sz, ttb_indx nz) :
    siz(sz), nNumDims(sz.size()), values(nz),
    subs("Genten::Sptensor::subs",nz,sz.size()) {}

  /* Constructor from complete raw data indexed C-wise in C types.
     All input are deep copied.
     @param nd number of dimensions.
     @param dims length of each dimension.
     @param nz number of nonzeros.
     @param vals values [nz] of nonzeros.
     @param subscripts [nz*nd] coordinates of each nonzero, grouped by indices of each nonzero adjacent.
  */
  Sptensor(ttb_indx nd, ttb_indx *dims, ttb_indx nz, ttb_real *vals, ttb_indx *subscripts);

  // Constructor (for data from MATLAB).
  /* a) Copies everything locally.
     b) There are no checks for duplicate entries. Call sort() to dedup.
     c) It is assumed that sbs starts numbering at one,
     and so one is subtracted to make it start at zero. */
  Sptensor(ttb_indx nd, ttb_real * sz, ttb_indx nz, ttb_real * vls, ttb_real * sbs);

  /* Constructor from complete raw data indexed C-wise using STL types.
     All input are deep copied.
     @param dims length of each dimension.
     @param vals nonzero values.
     @param subscripts 2-d array of subscripts.
  */
  Sptensor(const std::vector<ttb_indx>& dims,
                  const std::vector<ttb_real>& vals,
                  const std::vector< std::vector<ttb_indx> >& subscripts);

  // Copy constructor.
  KOKKOS_INLINE_FUNCTION
  Sptensor (const Sptensor & arg) = default;

  // Assignment operator.
  KOKKOS_INLINE_FUNCTION
  Sptensor & operator= (const Sptensor & arg) = default;

  // Destructor.
  KOKKOS_INLINE_FUNCTION
  ~Sptensor() = default;

  // Deep copy into tensor
  void deep_copy(const Sptensor& X) {
    values.deep_copy(X.values);
    Kokkos::deep_copy(subs, X.subs);
  }

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
  const Genten::IndxArray & size() const
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
  bool isEqual(const Sptensor & b, ttb_real tol) const;

  // Return reference to i-th nonzero
  KOKKOS_INLINE_FUNCTION
  ttb_real & value(ttb_indx i) const
  {
    assert(i < values.size());
    return values[i];
  }

  // Return reference to n-th subscript of i-th nonzero
  KOKKOS_INLINE_FUNCTION
  ttb_indx & subscript(ttb_indx i, ttb_indx n) const
  {
    assert((i < values.size()) && (n < nNumDims));
    return subs(i,n);
  }

  // Get subscripts of i-th nonzero, place into IndxArray object
  KOKKOS_INLINE_FUNCTION
  void getSubscripts(ttb_indx i,  const IndxArray & sub) const
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

  // Return the norm (sqrt of the sum of the squares of all entries).
  ttb_real norm() const
  {
    return values.norm(Genten::NormTwo);
  }

  // Return the i-th linearly indexed element.
  KOKKOS_INLINE_FUNCTION
  ttb_real & operator[](ttb_indx i) const
  {
    return values[i];
  }

  // Finalize any setup of the tensor after all entries have been added
  void fillComplete() {}

  /* Result stored in this tensor */
  void times(const Genten::Ktensor & K, const Genten::Sptensor & X);

  // Elementwise division of input tensor X and Ktensor K.
  /* Result stored in this tensor. The argument epsilon is the minimum value allowed for the division. */
  void divide(const Genten::Ktensor & K, const Genten::Sptensor & X, ttb_real epsilon);

protected:

  // Size of the tensor
  IndxArray siz;

  // Number of dimensions, from siz.size(), but faster to store it.
  ttb_indx nNumDims;

  // Data array (an array of nonzero values)
  Array values;

  // Subscript array of nonzero elements.  This vector is treated as a 2D array
  // of size nnz by nNumDims.
  typedef Kokkos::View<ttb_indx**> subs_view_type;
  subs_view_type subs;

};

}
