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

#pragma once

#include <assert.h>

#include "Genten_Sptensor.h"

namespace Genten
{

  /* The Genten::Sptensor_perm class stores sparse tensors.
   * This version is derived from Sptensor and adds a permutation
   * array for each dimension where the indices for each nonzero in that
   * dimension are non-decreasing.
   */

class Sptensor_perm : public Sptensor
{

public:

  // Empty construtor.
  /* Creates an empty tensor with an empty size. */
  KOKKOS_INLINE_FUNCTION
    Sptensor_perm() : Sptensor(), perm() {}

  // Constructor for a given size and number of nonzeros
  Sptensor_perm(const IndxArray& sz, ttb_indx nz) :
    Sptensor(sz,nz), perm() {}

  /* Constructor from complete raw data indexed C-wise in C types.
     All input are deep copied.
     @param nd number of dimensions.
     @param dims length of each dimension.
     @param nz number of nonzeros.
     @param vals values [nz] of nonzeros.
     @param subscripts [nz*nd] coordinates of each nonzero, grouped by indices of each nonzero adjacent.
  */
  Sptensor_perm(ttb_indx nd, ttb_indx *dims, ttb_indx nz, ttb_real *vals, ttb_indx *subscripts) :
    Sptensor(nd,dims,nz,vals,subscripts), perm() {}

  // Constructor (for data from MATLAB).
  /* a) Copies everything locally.
     b) There are no checks for duplicate entries. Call sort() to dedup.
     c) It is assumed that sbs starts numbering at one,
     and so one is subtracted to make it start at zero. */
  Sptensor_perm(ttb_indx nd, ttb_real * sz, ttb_indx nz, ttb_real * vls, ttb_real * sbs) :
    Sptensor(nd,sz,nz,vls,sbs), perm() {}

  /* Constructor from complete raw data indexed C-wise using STL types.
     All input are deep copied.
     @param dims length of each dimension.
     @param vals nonzero values.
     @param subscripts 2-d array of subscripts.
  */
  Sptensor_perm(const std::vector<ttb_indx>& dims,
               const std::vector<ttb_real>& vals,
               const std::vector< std::vector<ttb_indx> >& subscripts) :
    Sptensor(dims, vals, subscripts) {}

  // Copy constructor.
  KOKKOS_INLINE_FUNCTION
  Sptensor_perm (const Sptensor_perm & arg) = default;

  // Assignment operator.
  KOKKOS_INLINE_FUNCTION
  Sptensor_perm & operator= (const Sptensor_perm & arg) = default;

  // Destructor.
  KOKKOS_INLINE_FUNCTION
  ~Sptensor_perm() = default;

  KOKKOS_INLINE_FUNCTION
  ttb_indx getPerm(ttb_indx i, ttb_indx n) const
  {
    assert((i < values.size()) && (n < nNumDims));
    return perm(i,n);
  }

  // Finalize any setup of the tensor after all entries have been added
  void fillComplete() { createPermutation(); }

  // Create permutation array by sorting each column of subs
  // Currently must be public for Cuda-lambda
  void createPermutation();

protected:

  typedef Sptensor::subs_view_type subs_view_type;
  subs_view_type perm;

};

}
