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

template <typename ExecSpace> class SptensorT_row;
typedef SptensorT_row<DefaultHostExecutionSpace> Sptensor_row;

template <typename ExecSpace>
class SptensorT_row : public SptensorT_perm<ExecSpace>
{

public:

  typedef ExecSpace exec_space;
  typedef typename SptensorT_perm<ExecSpace>::host_mirror_space host_mirror_space;
  typedef SptensorT_row<host_mirror_space> HostMirror;
  typedef typename SptensorT_perm<ExecSpace>::subs_view_type subs_view_type;
  typedef typename SptensorT_perm<ExecSpace>::vals_view_type vals_view_type;
  typedef Kokkos::View< Kokkos::View<ttb_indx*,Kokkos::LayoutRight,ExecSpace>*,Kokkos::LayoutRight,ExecSpace > row_ptr_type;
  typedef Kokkos::View< Kokkos::View<ttb_indx*,Kokkos::LayoutRight,ExecSpace>*,Kokkos::LayoutRight,DefaultHostExecutionSpace > host_row_ptr_type;

  // Empty construtor.
  /* Creates an empty tensor with an empty size. */
  KOKKOS_INLINE_FUNCTION
  SptensorT_row() : SptensorT_perm<ExecSpace>(), rowptr(), host_rowptr() {}

  // Constructor for a given size and number of nonzeros
  SptensorT_row(const IndxArrayT<ExecSpace>& sz, ttb_indx nz) :
    SptensorT_perm<ExecSpace>(sz,nz), rowptr(), host_rowptr() {}

  /* Constructor from complete raw data indexed C-wise in C types.
     All input are deep copied.
     @param nd number of dimensions.
     @param dims length of each dimension.
     @param nz number of nonzeros.
     @param vals values [nz] of nonzeros.
     @param subscripts [nz*nd] coordinates of each nonzero, grouped by indices of each nonzero adjacent.
  */
  SptensorT_row(ttb_indx nd, ttb_indx *dims, ttb_indx nz, ttb_real *vals, ttb_indx *subscripts) :
    SptensorT_perm<ExecSpace>(nd,dims,nz,vals,subscripts),
    rowptr(), host_rowptr() {}

  // Constructor (for data from MATLAB).
  /* a) Copies everything locally.
     b) There are no checks for duplicate entries. Call sort() to dedup.
     c) It is assumed that sbs starts numbering at one,
     and so one is subtracted to make it start at zero. */
  SptensorT_row(ttb_indx nd, ttb_real * sz, ttb_indx nz, ttb_real * vls, ttb_real * sbs) :
    SptensorT_perm<ExecSpace>(nd,sz,nz,vls,sbs), rowptr(), host_rowptr() {}

  /* Constructor from complete raw data indexed C-wise using STL types.
     All input are deep copied.
     @param dims length of each dimension.
     @param vals nonzero values.
     @param subscripts 2-d array of subscripts.
  */
  SptensorT_row(const std::vector<ttb_indx>& dims,
                const std::vector<ttb_real>& vals,
                const std::vector< std::vector<ttb_indx> >& subscripts) :
    SptensorT_perm<ExecSpace>(dims, vals, subscripts),
    rowptr(), host_rowptr() {}

  // Create tensor from supplied dimensions, values, and subscripts
  KOKKOS_INLINE_FUNCTION
  SptensorT_row(const IndxArrayT<ExecSpace>& d, const vals_view_type& vals,
                const subs_view_type& s) :
    SptensorT_perm<ExecSpace>(d, vals, s), rowptr(), host_rowptr() {}

  // Create tensor from supplied dimensions and subscripts, zero values
  SptensorT_row(const IndxArrayT<ExecSpace>& d, const subs_view_type& s) :
    SptensorT_perm<ExecSpace>(d, s), rowptr(), host_rowptr() {}

  // Copy constructor.
  KOKKOS_INLINE_FUNCTION
  SptensorT_row (const SptensorT_row & arg) = default;

  // Assignment operator.
  KOKKOS_INLINE_FUNCTION
  SptensorT_row & operator= (const SptensorT_row & arg) = default;

  // Destructor.
  KOKKOS_INLINE_FUNCTION
  ~SptensorT_row() = default;

  // Create permutation array by sorting each column of subs
  void createPermutation();

  KOKKOS_INLINE_FUNCTION
  ttb_indx getPermRowBegin(ttb_indx i, ttb_indx n) const
  {
    assert((n < this->nNumDims) && (i < rowptr(n).size()));
    return rowptr(n)(i);
  }

  // Get whole perm array
  KOKKOS_INLINE_FUNCTION
  row_ptr_type getRowPtr() const { return rowptr; }

  // Finalize any setup of the tensor after all entries have been added
  void fillComplete() {
    SptensorT_perm<ExecSpace>::createPermutation();
    createRowPtr();
  }

  // Fill rowptr array with row offsets
  void createRowPtr();

protected:

  row_ptr_type rowptr;
  host_row_ptr_type host_rowptr;

};

// You must call fillComplete() after creating the mirror and deep_copying
template <typename ExecSpace>
typename SptensorT_row<ExecSpace>::HostMirror
create_mirror_view(const SptensorT<ExecSpace>& a)
{
  typedef typename SptensorT_row<ExecSpace>::HostMirror HostMirror;
  return HostMirror( create_mirror_view(a.size()),
                     create_mirror_view(a.getValues()),
                     create_mirror_view(a.getSubscripts()) );
}

template <typename Space, typename ExecSpace>
SptensorT_row<Space>
create_mirror_view(const Space& s, const SptensorT_row<ExecSpace>& a)
{
  return SptensorT_row<Space>( create_mirror_view(s, a.size()),
                               create_mirror_view(s, a.getValues()),
                               create_mirror_view(s, a.getSubscripts()) );
}

template <typename E1, typename E2>
void deep_copy(const SptensorT_row<E1>& dst, const SptensorT_row<E2>& src)
{
  deep_copy( dst.size(), src.size() );
  deep_copy( dst.getValues(), src.getValues() );
  deep_copy( dst.getSubscripts(), src.getSubscripts() );
}

}
