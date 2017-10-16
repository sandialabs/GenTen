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

#include "Genten_Sptensor_perm.hpp"

namespace Genten
{

  /* The Genten::Sptensor class stores sparse tensors.
   * This version is derived from Sptensor_perm and adds a row pointer (a la
   * Crs) into the permutation array where each row starts and stops.
   */

class Sptensor_row : public Sptensor_perm
{

public:

  // Empty construtor.
  /* Creates an empty tensor with an empty size. */
  KOKKOS_INLINE_FUNCTION
  Sptensor_row() : Sptensor_perm(), rowptr() {}

  // Constructor for a given size and number of nonzeros
  Sptensor_row(const IndxArray& sz, ttb_indx nz) :
    Sptensor_perm(sz,nz), rowptr() {}

  /* Constructor from complete raw data indexed C-wise in C types.
     All input are deep copied.
     @param nd number of dimensions.
     @param dims length of each dimension.
     @param nz number of nonzeros.
     @param vals values [nz] of nonzeros.
     @param subscripts [nz*nd] coordinates of each nonzero, grouped by indices of each nonzero adjacent.
  */
  Sptensor_row(ttb_indx nd, ttb_indx *dims, ttb_indx nz, ttb_real *vals, ttb_indx *subscripts) :
    Sptensor_perm(nd,dims,nz,vals,subscripts), rowptr() {}

  // Constructor (for data from MATLAB).
  /* a) Copies everything locally.
     b) There are no checks for duplicate entries. Call sort() to dedup.
     c) It is assumed that sbs starts numbering at one,
     and so one is subtracted to make it start at zero. */
  Sptensor_row(ttb_indx nd, ttb_real * sz, ttb_indx nz, ttb_real * vls, ttb_real * sbs) :
    Sptensor_perm(nd,sz,nz,vls,sbs), rowptr() {}

  /* Constructor from complete raw data indexed C-wise using STL types.
     All input are deep copied.
     @param dims length of each dimension.
     @param vals nonzero values.
     @param subscripts 2-d array of subscripts.
  */
  Sptensor_row(const std::vector<ttb_indx>& dims,
               const std::vector<ttb_real>& vals,
               const std::vector< std::vector<ttb_indx> >& subscripts) :
    Sptensor_perm(dims, vals, subscripts) {}

  // Copy constructor.
  KOKKOS_INLINE_FUNCTION
  Sptensor_row (const Sptensor_row & arg) = default;

  // Assignment operator.
  KOKKOS_INLINE_FUNCTION
  Sptensor_row & operator= (const Sptensor_row & arg) = default;

  // Destructor.
  KOKKOS_INLINE_FUNCTION
  ~Sptensor_row() = default;

  // Create permutation array by sorting each column of subs
  void createPermutation();

  KOKKOS_INLINE_FUNCTION
  ttb_indx getPermRowBegin(ttb_indx i, ttb_indx n) const
  {
    assert((n < nNumDims) && (i < rowptr(n).size()));
    return rowptr(n)(i);
  }

  // Finalize any setup of the tensor after all entries have been added
  void fillComplete() {
    Sptensor_perm::createPermutation();
    createRowPtr();
  }

  // Fill rowptr array with row offsets
  void createRowPtr();

protected:

  typedef Kokkos::View< Kokkos::View<ttb_indx*>* > row_ptr_type;
  row_ptr_type rowptr;

};

}
