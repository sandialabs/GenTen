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

#include "Genten_Sptensor.hpp"

namespace Genten
{

  /* The Genten::Sptensor_perm class stores sparse tensors.
   * This version is derived from Sptensor and adds a permutation
   * array for each dimension where the indices for each nonzero in that
   * dimension are non-decreasing.
   */

template <typename ExecSpace> class SptensorT_perm;
typedef SptensorT_perm<DefaultHostExecutionSpace> Sptensor_perm;

template <typename ExecSpace>
class SptensorT_perm : public SptensorT<ExecSpace>
{
public:

  typedef ExecSpace exec_space;
  typedef typename SptensorT<ExecSpace>::host_mirror_space host_mirror_space;
  typedef SptensorT_perm<host_mirror_space> HostMirror;
  typedef typename SptensorT<ExecSpace>::subs_view_type subs_view_type;
  typedef typename SptensorT<ExecSpace>::vals_view_type vals_view_type;

  // Empty construtor.
  /* Creates an empty tensor with an empty size. */
  KOKKOS_INLINE_FUNCTION
  SptensorT_perm() : SptensorT<ExecSpace>(), perm() {}

  // Constructor for a given size and number of nonzeros
  SptensorT_perm(const IndxArrayT<ExecSpace>& sz, ttb_indx nz) :
  SptensorT<ExecSpace>(sz,nz), perm() {}

  /* Constructor from complete raw data indexed C-wise in C types.
     All input are deep copied.
     @param nd number of dimensions.
     @param dims length of each dimension.
     @param nz number of nonzeros.
     @param vals values [nz] of nonzeros.
     @param subscripts [nz*nd] coordinates of each nonzero, grouped by indices of each nonzero adjacent.
  */
  SptensorT_perm(ttb_indx nd, ttb_indx *dims, ttb_indx nz, ttb_real *vals, ttb_indx *subscripts) :
    SptensorT<ExecSpace>(nd,dims,nz,vals,subscripts), perm() {}

  // Constructor (for data from MATLAB).
  /* a) Copies everything locally.
     b) There are no checks for duplicate entries. Call sort() to dedup.
     c) It is assumed that sbs starts numbering at one,
     and so one is subtracted to make it start at zero. */
  SptensorT_perm(ttb_indx nd, ttb_real * sz, ttb_indx nz, ttb_real * vls, ttb_real * sbs) :
    SptensorT<ExecSpace>(nd,sz,nz,vls,sbs), perm() {}

  /* Constructor from complete raw data indexed C-wise using STL types.
     All input are deep copied.
     @param dims length of each dimension.
     @param vals nonzero values.
     @param subscripts 2-d array of subscripts.
  */
  SptensorT_perm(const std::vector<ttb_indx>& dims,
                 const std::vector<ttb_real>& vals,
                 const std::vector< std::vector<ttb_indx> >& subscripts) :
    SptensorT<ExecSpace>(dims, vals, subscripts) {}

  // Create tensor from supplied dimensions, values, and subscripts
  KOKKOS_INLINE_FUNCTION
  SptensorT_perm(const IndxArrayT<ExecSpace>& d, const vals_view_type& vals,
                 const subs_view_type& s) :
    SptensorT<ExecSpace>(d, vals, s),
    perm() {}

  // Create tensor from supplied dimensions and subscripts, zero values
  SptensorT_perm(const IndxArrayT<ExecSpace>& d, const subs_view_type& s) :
    SptensorT<ExecSpace>(d, s),
    perm() {}

  // Copy constructor.
  KOKKOS_INLINE_FUNCTION
  SptensorT_perm (const SptensorT_perm & arg) = default;

  // Assignment operator.
  KOKKOS_INLINE_FUNCTION
  SptensorT_perm & operator= (const SptensorT_perm & arg) = default;

  // Destructor.
  KOKKOS_INLINE_FUNCTION
  ~SptensorT_perm() = default;

  KOKKOS_INLINE_FUNCTION
  ttb_indx getPerm(ttb_indx i, ttb_indx n) const
  {
    assert((i < this->values.size()) && (n < this->nNumDims));
    return perm(i,n);
  }

  // Get whole perm array
  KOKKOS_INLINE_FUNCTION
  subs_view_type getPerm() const { return perm; }

  // Finalize any setup of the tensor after all entries have been added
  void fillComplete();

  // Create permutation array by sorting each column of subs
  // Currently must be public for Cuda-lambda
  void createPermutation();

protected:

  subs_view_type perm;

};

// You must call fillComplete() after creating the mirror and deep_copying
template <typename ExecSpace>
typename SptensorT_perm<ExecSpace>::HostMirror
create_mirror_view(const SptensorT<ExecSpace>& a)
{
  typedef typename SptensorT_perm<ExecSpace>::HostMirror HostMirror;
  return HostMirror( create_mirror_view(a.size()),
                     create_mirror_view(a.getValues()),
                     create_mirror_view(a.getSubscripts()) );
}

template <typename Space, typename ExecSpace>
SptensorT_perm<Space>
create_mirror_view(const Space& s, const SptensorT_perm<ExecSpace>& a)
{
  return SptensorT_perm<Space>( create_mirror_view(s, a.size()),
                                create_mirror_view(s, a.getValues()),
                                create_mirror_view(s, a.getSubscripts()) );
}

template <typename E1, typename E2>
void deep_copy(const SptensorT_perm<E1>& dst, const SptensorT_perm<E2>& src)
{
  deep_copy( dst.size(), src.size() );
  deep_copy( dst.getValues(), src.getValues() );
  deep_copy( dst.getSubscripts(), src.getSubscripts() );
}

}
