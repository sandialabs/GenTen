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

#include "CMakeInclude.h"
#include "Genten_Array.hpp"
#include "Genten_IndxArray.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_Pmap.hpp"
#include "Genten_Tpetra.hpp"

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
  SptensorT() : siz(),siz_host(),nNumDims(0),values(),subs(),subs_gids(),perm(),
                is_sorted(false), pmap(nullptr), lower_bound(), upper_bound() {}

  // Constructor for a given size and number of nonzeros
  SptensorT(const IndxArrayT<ExecSpace>& sz, ttb_indx nz) :
    siz(sz.clone()), nNumDims(sz.size()), values(nz),
    subs("Genten::Sptensor::subs",nz,sz.size()), subs_gids(subs), perm(),
    is_sorted(false), pmap(nullptr),
    lower_bound(nNumDims,ttb_indx(0)), upper_bound(siz.clone()) {
    siz_host = create_mirror_view(siz);
    deep_copy(siz_host, siz);
#ifdef HAVE_TPETRA
    factorMaps.resize(nNumDims);
    tensorMaps.resize(nNumDims);
    importers.resize(nNumDims);
#endif
  }

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

  /* Constructor from complete raw data indexed C-wise using STL types.
     All input are deep copied.
     @param dims length of each dimension.
     @param vals nonzero values.
     @param subscripts 2-d array of subscripts.
     @param global_subscripts 2-d array of subscript global IDs
     @param global_lower_bound Lower bound of global subscripts
     @param global_upper_bound Upper bound of global subscripts
  */
  SptensorT(const std::vector<ttb_indx>& dims,
            const std::vector<ttb_real>& vals,
            const std::vector< std::vector<ttb_indx> >& subscripts,
            const std::vector< std::vector<ttb_indx> >& global_subscripts,
            const std::vector<ttb_indx>& global_lower_bound,
            const std::vector<ttb_indx>& global_upper_bound);

  // Create tensor from supplied dimensions, values, and subscripts
  SptensorT(const IndxArrayT<ExecSpace>& d, const vals_view_type& vals,
            const subs_view_type& s,
            const subs_view_type& p = subs_view_type(),
            const bool sorted = false) :
    siz(d), nNumDims(d.size()), values(vals), subs(s), subs_gids(s), perm(p),
    is_sorted(sorted), pmap(nullptr),
    lower_bound(nNumDims,ttb_indx(0)), upper_bound(siz.clone()) {
    siz_host = create_mirror_view(siz);
    deep_copy(siz_host, siz);
#ifdef HAVE_TPETRA
    factorMaps.resize(nNumDims);
    tensorMaps.resize(nNumDims);
    importers.resize(nNumDims);
#endif
  }

  // Create tensor from supplied dimensions, values, and subscripts
  SptensorT(const IndxArrayT<ExecSpace>& d,
            const vals_view_type& vals,
            const subs_view_type& s,
            const subs_view_type& p,
            const bool sorted,
            const subs_view_type& s_g,
            const IndxArrayT<ExecSpace>& l,
            const IndxArrayT<ExecSpace>& u) :
    siz(d), nNumDims(d.size()), values(vals), subs(s), subs_gids(s_g), perm(p),
    is_sorted(sorted), pmap(nullptr),
    lower_bound(l), upper_bound(u) {
    siz_host = create_mirror_view(siz);
    deep_copy(siz_host, siz);
#ifdef HAVE_TPETRA
    factorMaps.resize(nNumDims);
    tensorMaps.resize(nNumDims);
    importers.resize(nNumDims);
#endif
  }

  // Create tensor from supplied dimensions and subscripts, zero values
  SptensorT(const IndxArrayT<ExecSpace>& d, const subs_view_type& s) :
    siz(d), nNumDims(d.size()), values(s.extent(0),ttb_real(0.0)), subs(s),
    subs_gids(subs), perm(), is_sorted(false), pmap(nullptr),
    lower_bound(nNumDims,ttb_indx(0)), upper_bound(siz.clone()) {
    siz_host = create_mirror_view(siz);
    deep_copy(siz_host, siz);
#ifdef HAVE_TPETRA
    factorMaps.resize(nNumDims);
    tensorMaps.resize(nNumDims);
    importers.resize(nNumDims);
#endif
  }

  // Copy constructor.
  KOKKOS_DEFAULTED_FUNCTION
  SptensorT (const SptensorT & arg) = default;

  // Assignment operator.
  KOKKOS_DEFAULTED_FUNCTION
  SptensorT & operator= (const SptensorT & arg) = default;

  // Destructor.
  KOKKOS_DEFAULTED_FUNCTION
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
    KOKKOS_IF_ON_DEVICE(return siz[i];)
    KOKKOS_IF_ON_HOST(return siz_host[i];)
  }

  // Return the entire size array.
  KOKKOS_INLINE_FUNCTION
  const IndxArrayT<ExecSpace> & size() const
  {
    return siz;
  }

  // Return the entire size array.
  const IndxArrayT<host_mirror_space>& size_host() const { return siz_host; }

  // Return the total number of (zero and nonzero) elements in the tensor.
  ttb_indx numel() const
  {
    return siz_host.prod();
  }
  ttb_indx global_numel() const
  {
    ttb_indx numel = siz_host.prod();
    if (pmap != nullptr)
      numel = pmap->gridAllReduce(numel);
    return numel;
  }

  // Return the total number of (zero and nonzero) elements in the tensor as
  // a float (to avoid overflow for large tensors)
  ttb_real numel_float() const
  {
    return siz_host.prod_float();
  }
  ttb_real global_numel_float() const
  {
    ttb_real numel = siz_host.prod_float();
    if (pmap != nullptr)
      numel = pmap->gridAllReduce(numel);
    return numel;
  }

  // Return the number of structural nonzeros.
  KOKKOS_INLINE_FUNCTION
  ttb_indx nnz() const
  {
    return values.size();
  }
  ttb_indx global_nnz() const
  {
    ttb_indx nnz = values.size();
    if (pmap != nullptr)
      nnz = pmap->gridAllReduce(nnz);
    return nnz;
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

  KOKKOS_INLINE_FUNCTION
  ArrayT<ExecSpace> const& getValArray() const { return values; }

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

  // Get subscripts of i-th nonzero
  KOKKOS_INLINE_FUNCTION
  auto getSubscripts(ttb_indx i) const
  {
    assert(i < values.size());
    return Kokkos::subview( subs, i, Kokkos::ALL() );
  }

  // Get whole subscripts array
  KOKKOS_INLINE_FUNCTION
  subs_view_type getSubscripts() const { return subs; }

  // Allocate global subscripts array
  void allocGlobalSubscripts() {
    subs_gids = subs_view_type(
      "Genten::Sptensor::subs_gids", subs.extent(0), subs.extent(1));
  }

  // Return reference to n-th subscript of i-th nonzero
  template <typename IType, typename NType>
  KOKKOS_INLINE_FUNCTION
  ttb_indx & globalSubscript(IType i, NType n) const
  {
    assert((i < values.size()) && (n < nNumDims));
    return subs_gids(i,n);
  }

  // Get subscripts of i-th nonzero, place into IndxArray object
  KOKKOS_INLINE_FUNCTION
  void getGlobalSubscripts(ttb_indx i,  const IndxArrayT<ExecSpace> & sub) const
  {
    assert(i < values.size());

    // This can be accomplished using subview() as below, but is a fair
    // amount slower than just manually copying into the given index array
    //sub = Kokkos::subview( subs, i, Kokkos::ALL() );

    assert(sub.size() == nNumDims);
    for (ttb_indx n = 0; n < nNumDims; n++)
    {
      sub[n] = subs_gids(i,n);
    }
  }

  // Get subscripts of i-th nonzero
  KOKKOS_INLINE_FUNCTION
  auto getGlobalSubscripts(ttb_indx i) const
  {
    assert(i < values.size());
    return Kokkos::subview( subs_gids, i, Kokkos::ALL() );
  }

  // Get whole subscripts array
  KOKKOS_INLINE_FUNCTION
  subs_view_type getGlobalSubscripts() const { return subs_gids; }

  // Return the norm (sqrt of the sum of the squares of all entries).
  ttb_real norm() const
  {
    return values.norm(NormTwo);
  }
  ttb_real global_norm() const
  {
    ttb_real nrm_sqrd = values.dot(values);
    if (pmap != nullptr)
      nrm_sqrd = pmap->gridAllReduce(nrm_sqrd);
    return std::sqrt(nrm_sqrd);
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

  // Sort tensor lexicographically
  void sort();

  // Is tensor sorted
  bool isSorted() const { return is_sorted; }

  // Set sorted flag
  void setIsSorted(bool sorted) { is_sorted = sorted; }

  // Return index for given subscript array.  Returns nnz() if not found.
  template <typename sub_type>
  KOKKOS_INLINE_FUNCTION
  ttb_indx index(const sub_type& sub) const;

  // Return index for given subscripts.  Returns nnz() if not found.
  // Allows syntax of the form index(i1,i2,...,id) for any ordinal types
  // i1,...,id.
  template <typename... P>
  KOKKOS_INLINE_FUNCTION
  ttb_indx index(P... args) const
  {
    const ttb_indx ind[] = { ttb_indx(args)... };
    return index(ind);
  }

  // Return smallest index i such that subs(i,:) >= sub for given subscript
  // array sub.  Requires tensor to be sorted.  start is a hint as to where
  // to start the search.
  template <typename sub_type>
  KOKKOS_INLINE_FUNCTION
  ttb_indx sorted_lower_bound(const sub_type& sub,
                              const ttb_indx start = 0) const;

  // Return whether subscript at given index is equal to sub
  template <typename sub_type>
  KOKKOS_INLINE_FUNCTION
  bool isSubscriptEqual(const ttb_indx i, const sub_type& sub) const;

  void setProcessorMap(const ProcessorMap* pmap_) { pmap = pmap_; }
  const ProcessorMap* getProcessorMap() const { return pmap; }

#ifdef HAVE_TPETRA
  Teuchos::RCP<tpetra_map_type<ExecSpace> >
  factorMap(const unsigned n) const {
    assert(n < factorMaps.size());
    return factorMaps[n];
  }
  Teuchos::RCP<tpetra_map_type<ExecSpace> >
  tensorMap(const unsigned n) const {
    assert(n < tensorMaps.size());
    return tensorMaps[n];
  }
  Teuchos::RCP<tpetra_import_type<ExecSpace> >
  importer(const unsigned n) const {
    assert(n < importers.size());
    return importers[n];
  }
  Teuchos::RCP<tpetra_map_type<ExecSpace> >&
  factorMap(const unsigned n) {
    assert(n < factorMaps.size());
    return factorMaps[n];
  }
  Teuchos::RCP<tpetra_map_type<ExecSpace> >&
  tensorMap(const unsigned n) {
    assert(n < tensorMaps.size());
    return tensorMaps[n];
  }
  Teuchos::RCP<tpetra_import_type<ExecSpace> >&
  importer(const unsigned n) {
    assert(n < importers.size());
    return importers[n];
  }
#endif

  KOKKOS_INLINE_FUNCTION
  ttb_indx lowerBound(const unsigned n) const {
    assert(n < lower_bound.size());
    return lower_bound[n];
  }
  KOKKOS_INLINE_FUNCTION
  ttb_indx upperBound(const unsigned n) const {
    assert(n < upper_bound.size());
    return upper_bound[n];
  }
  KOKKOS_INLINE_FUNCTION
  IndxArrayT<ExecSpace> getLowerBounds() const { return lower_bound; }
  KOKKOS_INLINE_FUNCTION
  IndxArrayT<ExecSpace> getUpperBounds() const { return upper_bound; }

protected:

  // Size of the tensor
  IndxArrayT<ExecSpace> siz;
  IndxArrayT<host_mirror_space> siz_host;

  // Number of dimensions, from siz.size(), but faster to store it.
  ttb_indx nNumDims;

  // Data array (an array of nonzero values)
  ArrayT<ExecSpace> values;

  // Subscript array of nonzero elements.  This vector is treated as a 2D array
  // of size nnz by nNumDims.  In MPI-parallel contexts, subs stores the
  // local indices of each nonzero (which is usually what you want)
  subs_view_type subs;

  // Global indices for nonzeros, which is assumed equal to subs unless
  // otherwise specified
  subs_view_type subs_gids;

  // Permutation array for iterating over subs in non-decreasing fashion
  subs_view_type perm;

  // Whether tensor has been sorted
  bool is_sorted;

  const ProcessorMap* pmap;

  IndxArrayT<ExecSpace> lower_bound;
  IndxArrayT<ExecSpace> upper_bound;
#ifdef HAVE_TPETRA
  std::vector< Teuchos::RCP< tpetra_map_type<ExecSpace> > > factorMaps;
  std::vector< Teuchos::RCP< tpetra_map_type<ExecSpace> > > tensorMaps;
  std::vector< Teuchos::RCP< tpetra_import_type<ExecSpace> > > importers;
#endif

};

template <typename ExecSpace>
typename SptensorT<ExecSpace>::HostMirror
create_mirror_view(const SptensorT<ExecSpace>& a)
{
  typedef typename SptensorT<ExecSpace>::HostMirror HostMirror;
  return HostMirror( create_mirror_view(a.size()),
                     create_mirror_view(a.getValues()),
                     create_mirror_view(a.getSubscripts()),
                     create_mirror_view(a.getPerm()),
                     a.isSorted() );
}

template <typename Space, typename ExecSpace>
SptensorT<Space>
create_mirror_view(const Space& s, const SptensorT<ExecSpace>& a)
{
  SptensorT<Space> v;
  if (a.getGlobalSubscripts().data() == a.getSubscripts().data()) {
    // When subs_gids aliases gids, don't create a new view for subs_gids
    auto sv = create_mirror_view(s, a.getSubscripts());
    v = SptensorT<Space>( create_mirror_view(s, a.size()),
                          create_mirror_view(s, a.getValues()),
                          sv,
                          create_mirror_view(s, a.getPerm()),
                          a.isSorted(),
                          sv,
                          create_mirror_view(s, a.getLowerBounds()),
                          create_mirror_view(s, a.getUpperBounds()) );
  }
  else
    v = SptensorT<Space>( create_mirror_view(s, a.size()),
                          create_mirror_view(s, a.getValues()),
                          create_mirror_view(s, a.getSubscripts()),
                          create_mirror_view(s, a.getPerm()),
                          a.isSorted(),
                          create_mirror_view(s, a.getGlobalSubscripts()),
                          create_mirror_view(s, a.getLowerBounds()),
                          create_mirror_view(s, a.getUpperBounds()) );
  return v;
}

template <typename E1, typename E2>
void deep_copy(SptensorT<E1>& dst, const SptensorT<E2>& src)
{
  deep_copy( dst.size(), src.size() );
  deep_copy( dst.size_host(), src.size_host() );
  deep_copy( dst.getValues(), src.getValues() );
  deep_copy( dst.getSubscripts(), src.getSubscripts() );
  deep_copy( dst.getPerm(), src.getPerm() );
  dst.setIsSorted( src.isSorted() );

  // Only deep copy subs_gids if it points to unique data
  if (dst.getGlobalSubscripts().data() != dst.getSubscripts().data())
    deep_copy( dst.getGlobalSubscripts(), src.getGlobalSubscripts() );

  deep_copy( dst.getLowerBounds(), src.getLowerBounds() );
  deep_copy( dst.getUpperBounds(), src.getUpperBounds() );
}

template <typename ExecSpace>
template <typename ind_type>
KOKKOS_INLINE_FUNCTION
ttb_indx SptensorT<ExecSpace>::index(const ind_type& ind) const
{
  const ttb_indx nz = subs.extent(0);
  const ttb_indx nd = subs.extent(1);

  // For unsorted, have to do linear search
  if (!is_sorted) {
    ttb_indx i = 0;
    for (; i<nz; ++i) {
      bool t = true;
      for (ttb_indx j=0; j<nd; ++j) {
        t = t && (ind[j] == subs(i,j));
        if (!t)
          break;
      }
      if (t)
        break;
    }
    return i;
  }

  // If sorted, do binary search
  const ttb_indx idx = sorted_lower_bound(ind);
  if (isSubscriptEqual(idx,ind))
    return idx;

  return nz;
}

template <typename ExecSpace>
template <typename ind_type>
KOKKOS_INLINE_FUNCTION
ttb_indx SptensorT<ExecSpace>::sorted_lower_bound(const ind_type& ind,
                                                  const ttb_indx start) const
{
  const ttb_indx nz = subs.extent(0);
  /*const*/ ttb_indx nd = subs.extent(1);

  if (!is_sorted) {
    Kokkos::abort("Cannot call sorted_lower_bound() on unsorted tensor");
    return nz;
  }

  if (start >= nz)
    return start;

  // Find index "first" such that subs(first,:) >= ind using binary search
  auto less = [&](const ttb_indx& i, const ind_type& b)
  {
    unsigned n = 0;
    while ((n < nd) && (subs(i,n) == b[n])) ++n;
    if (n == nd || subs(i,n) >= b[n]) return false;
    return true;
  };
  ttb_indx i = 0;
  ttb_indx first = start;
  ttb_indx last = nz;
  ttb_indx count = last-first;
  ttb_indx step = 0;
  while (count > 0) {
    i = first;
    step = count / 2;
    i += step;
    if (less(i, ind)) {
      first = ++i;
      count -= step + 1;
    }
    else
      count = step;
  }
  return first;
}

template <typename ExecSpace>
template <typename sub_type>
KOKKOS_INLINE_FUNCTION
bool SptensorT<ExecSpace>::isSubscriptEqual(const ttb_indx i,
                                            const sub_type& sub) const
{
  if (i >= subs.extent(0))
    return false;

  const unsigned nd = subs.extent(1);
  unsigned n = 0;
  while ((n < nd) && (subs(i,n) == sub[n])) ++n;
  if (n == nd)
    return true;

  return false;
}

}
