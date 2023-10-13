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

#include <cmath>

#include "Genten_Sptensor.hpp"
#include "Genten_Tensor.hpp"

#include "Kokkos_Sort.hpp"

#ifdef KOKKOS_ENABLE_OPENMP
#include "parallel_stable_sort.hpp"
#endif

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#endif

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

//#if defined(KOKKOS_ENABLE_SYCL)
//#include <execution>
//#endif

namespace Genten {
namespace Impl {

template <typename SubsViewType, typename T>
void init_subs(const SubsViewType& subs, const T* sbs, const ttb_indx shift)
{
  typedef typename SubsViewType::execution_space exec_space;
  const ttb_indx nz = subs.extent(0);
  const ttb_indx nd = subs.extent(1);
  Kokkos::parallel_for(Kokkos::RangePolicy<exec_space>(0,nz),
                       KOKKOS_LAMBDA(const ttb_indx i)
  {
    for (ttb_indx j=0; j<nd; ++j)
    {
      subs(i,j) = (ttb_indx) sbs[i+j*nz] - shift;
    }
  });
}

template <typename ExecSpace, typename Layout>
ttb_indx countNonzeros(const TensorImpl<ExecSpace,Layout>& x, const ttb_real tol)
{
  const ttb_indx ne = x.numel();
  ttb_indx nz = 0;
  Kokkos::parallel_reduce("countNonzeros",
                         Kokkos::RangePolicy<ExecSpace>(0,ne),
                         KOKKOS_LAMBDA(const ttb_indx i, ttb_indx& n)
  {
    if (std::abs(x[i]) > tol) ++n;
  }, nz);
  return nz;
}

template <typename ExecSpace, typename Layout, typename SubsViewType, typename ValsViewType>
void copyFromTensor(const TensorImpl<ExecSpace,Layout>& x,
                    const SubsViewType& subs, const ValsViewType& vals,
                    const ttb_real tol)
{
  Kokkos::View<ttb_indx,ExecSpace> nonzero_index("nonzero_index");
  const ttb_indx ne = x.numel();
  Kokkos::parallel_for("copyFromTensor",
                       Kokkos::RangePolicy<ExecSpace>(0,ne),
                       KOKKOS_LAMBDA(const ttb_indx i)
  {
    const ttb_real xv = x[i];
    if (std::abs(xv) > tol) {
      const ttb_indx ind = Kokkos::atomic_fetch_add(&(nonzero_index()), 1);
      vals[ind] = xv;
      auto sub = Kokkos::subview(subs, ind, Kokkos::ALL);
      x.ind2sub(sub, i);
    }
  });
}

}
}

template <typename ExecSpace>
Genten::SptensorImpl<ExecSpace>::
SptensorImpl(ttb_indx nd, ttb_real * sz, ttb_indx nz, ttb_real * vls,
             ttb_real * sbs):
  siz(nd,sz), nNumDims(nd), values(nz,vls,false),
  subs(Kokkos::view_alloc("Genten::Sptensor::subs",
                          Kokkos::WithoutInitializing),nz,nd),
  subs_gids(subs), perm(), is_sorted(false),
  lower_bound(nNumDims,ttb_indx(0)), upper_bound(siz.clone())
{
  siz_host = create_mirror_view(siz);
  deep_copy(siz_host, siz);

  // convert subscripts to ttb_indx with zero indexing and transpose subs array
  // to store each nonzero's subscripts contiguously
  Impl::init_subs(subs, sbs, 1);
}

template <typename ExecSpace>
Genten::SptensorImpl<ExecSpace>::
SptensorImpl(ttb_indx nd, ttb_indx *dims, ttb_indx nz, ttb_real *vals,
             ttb_indx *subscripts):
  siz(nd,dims), nNumDims(nd), values(nz,vals,false),
  subs(Kokkos::view_alloc("Genten::Sptensor::subs",
                          Kokkos::WithoutInitializing),nd,nz),
  subs_gids(subs), perm(), is_sorted(false),
  lower_bound(nNumDims,ttb_indx(0)), upper_bound(siz.clone())
{
  siz_host = create_mirror_view(siz);
  deep_copy(siz_host, siz);

  // Copy subscripts into subs.  Because of polymorphic layout, we can't
  // assume subs and subscripts are ordered in the same way
  Impl::init_subs(subs, subscripts, 0);
}

template <typename ExecSpace>
Genten::SptensorImpl<ExecSpace>::
SptensorImpl(const std::vector<ttb_indx>& dims,
             const std::vector<ttb_real>& vals,
             const std::vector< std::vector<ttb_indx> >& subscripts):
  siz(ttb_indx(dims.size()),const_cast<ttb_indx*>(dims.data())),
  nNumDims(dims.size()),
  values(vals.size(),const_cast<ttb_real*>(vals.data()),false),
  subs("Genten::Sptensor::subs",subscripts.size(),dims.size()),
  subs_gids(subs), perm(), is_sorted(false),
  lower_bound(nNumDims,ttb_indx(0)), upper_bound(siz.clone())
{
  gt_assert(vals.size() == subscripts.size());

  siz_host = create_mirror_view(siz);
  deep_copy(siz_host, siz);

  auto subs_host = create_mirror_view(subs);
  for (ttb_indx i = 0; i < vals.size(); i ++)
  {
    for (ttb_indx j = 0; j < dims.size(); j ++)
    {
      subs_host(i,j) = subscripts[i][j];
    }
  }
  deep_copy(subs, subs_host);
}

template <typename ExecSpace>
Genten::SptensorImpl<ExecSpace>::
SptensorImpl(const std::vector<ttb_indx>& dims,
             const std::vector<ttb_real>& vals,
             const std::vector< std::vector<ttb_indx> >& subscripts,
             const std::vector< std::vector<ttb_indx> >& global_subscripts,
             const std::vector<ttb_indx>& global_lower_bound,
             const std::vector<ttb_indx>& global_upper_bound):
  siz(ttb_indx(dims.size()),const_cast<ttb_indx*>(dims.data())),
  nNumDims(dims.size()),
  values(vals.size(),const_cast<ttb_real*>(vals.data()),false),
  subs("Genten::Sptensor::subs",subscripts.size(),dims.size()),
  subs_gids("Genten::Sptensor::subs_gids",global_subscripts.size(),dims.size()),
  perm(), is_sorted(false),
  lower_bound(ttb_indx(dims.size()),const_cast<ttb_indx*>(global_lower_bound.data())),
  upper_bound(ttb_indx(dims.size()),const_cast<ttb_indx*>(global_upper_bound.data()))
{
  gt_assert(vals.size() == subscripts.size());
  gt_assert(vals.size() == global_subscripts.size());
  gt_assert(global_lower_bound.size() == dims.size());
  gt_assert(global_upper_bound.size() == dims.size());

  siz_host = create_mirror_view(siz);
  deep_copy(siz_host, siz);

  auto subs_host = create_mirror_view(subs);
  auto subs_gids_host = create_mirror_view(subs_gids);
  for (ttb_indx i = 0; i < vals.size(); i ++)
  {
    for (ttb_indx j = 0; j < dims.size(); j ++)
    {
      subs_host(i,j) = subscripts[i][j];
      subs_gids_host(i,j) = global_subscripts[i][j];
    }
  }
  deep_copy(subs, subs_host);
  deep_copy(subs_gids, subs_gids_host);
}

template <typename ExecSpace>
Genten::SptensorImpl<ExecSpace>::
SptensorImpl(const TensorT<ExecSpace>& x,
             const ttb_real tol) :
  siz(x.size().clone()), nNumDims(x.ndims()), values(),
  subs(), subs_gids(), perm(),
  is_sorted(false),
  lower_bound(nNumDims,ttb_indx(0)), upper_bound(siz.clone())
{
  siz_host = create_mirror_view(siz);
  deep_copy(siz_host, siz);

  // Compute number of nonzeros
  if (x.has_left_impl()) {
    const ttb_indx nz = Impl::countNonzeros(x.left_impl(), tol);

    // Compute nonzero subscripts and copy values
    subs = subs_view_type("Genten::Sptensor::subs",nz,siz.size());
    values = ArrayT<ExecSpace>(nz);
    Impl::copyFromTensor(x.left_impl(), subs, values.values(), tol);
  }
  else {
    const ttb_indx nz = Impl::countNonzeros(x.right_impl(), tol);

    // Compute nonzero subscripts and copy values
    subs = subs_view_type("Genten::Sptensor::subs",nz,siz.size());
    values = ArrayT<ExecSpace>(nz);
    Impl::copyFromTensor(x.right_impl(), subs, values.values(), tol);
  }
  subs_gids = subs;
}

template <typename ExecSpace>
void Genten::SptensorImpl<ExecSpace>::
words(ttb_indx& iw, ttb_indx& rw) const
{
  rw = values.size();
  iw = subs.size() + nNumDims;
}

template <typename ExecSpace>
bool Genten::SptensorImpl<ExecSpace>::
isEqual(const Genten::SptensorT<ExecSpace> & b, ttb_real tol) const
{
  // Check for equal sizes.
  if (this->ndims() != b.ndims())
  {
    return( false );
  }
  for (ttb_indx  i = 0; i < ndims(); i++)
  {
    if (this->size(i) != b.size(i))
    {
      return( false );
    }
  }
  if (this->nnz() != b.nnz())
  {
    return( false );
  }

  // Check that elements are equal.
  for (ttb_indx  i = 0; i < nnz(); i++)
  {
    if (Genten::isEqualToTol(this->value(i), b.value(i), tol) == false)
    {
      return( false );
    }
  }

  return( true );
}

template <typename ExecSpace>
void Genten::SptensorImpl<ExecSpace>::
times(const Genten::KtensorT<ExecSpace> & K,
      const Genten::SptensorT<ExecSpace> & X)
{
  // Copy X into this (including its size array)
  deep_copy(*this, X);

  // Check sizes
  gt_assert(K.isConsistent(siz));

  // Stream through nonzeros
  Genten::IndxArrayT<ExecSpace> subs(nNumDims);
  ttb_indx nz = this->nnz();
  for (ttb_indx i = 0; i < nz; i ++)
  {
    this->getSubscripts(i,subs);
    values[i] *= K.entry(subs);
  }

  //TODO: Check for any zeros!
}

template <typename ExecSpace>
void Genten::SptensorImpl<ExecSpace>::
divide(const Genten::KtensorT<ExecSpace> & K,
       const Genten::SptensorT<ExecSpace> & X, ttb_real epsilon)
{
  // Copy X into this (including its size array)
  deep_copy(*this, X);

  // Check sizes
  gt_assert(K.isConsistent(siz));

  // Stream through nonzeros
  Genten::IndxArrayT<ExecSpace> subs(nNumDims);
  ttb_indx nz = this->nnz();
  for (ttb_indx i = 0; i < nz; i ++)
  {
    this->getSubscripts(i,subs);
    ttb_real val = K.entry(subs);
    if (fabs(val) < epsilon)
    {
      values[i] /= epsilon;
    }
    else
    {
      values[i] /= val;
    }
  }
}

namespace Genten {
namespace Impl {
// Implementation of createPermutation().  Has to be done as a
// non-member function because lambda capture of *this doesn't work on Cuda.
template <typename ExecSpace, typename subs_view_type, typename siz_type>
void
createPermutationImpl(const subs_view_type& perm, const subs_view_type& subs,
                      const siz_type& siz)
{
  const ttb_indx sz = subs.extent(0);
  const ttb_indx nNumDims = subs.extent(1);

  // Neither std::sort or the Kokkos sort will work with non-contiguous views,
  // so we need to sort into temporary views
  typedef Kokkos::View<ttb_indx*,typename subs_view_type::array_layout,ExecSpace> ViewType;
  ViewType tmp(Kokkos::view_alloc(Kokkos::WithoutInitializing,"tmp_perm"),sz);

  for (ttb_indx n = 0; n < nNumDims; ++n) {

// Whether to sort using Kokkos or device specific approaches
// The Kokkos sort seems slower
#define USE_KOKKOS_SORT 0
#if USE_KOKKOS_SORT

    typedef Kokkos::BinOp1D<ViewType> CompType;
    // Kokkos::sort doesn't allow supplying a custom comparator, so we
    // have to use its implementation to get the permutation vector
    deep_copy( tmp, Kokkos::subview(subs, Kokkos::ALL(), n));
    Kokkos::BinSort<ViewType, CompType> bin_sort(
      tmp,CompType(sz/2,0,siz[n]),true);
    bin_sort.create_permute_vector();
    deep_copy( Kokkos::subview(perm, Kokkos::ALL(), n),
               bin_sort.get_permute_vector() );

#else

    // Sort tmp=[1:sz] using subs(:,n) as a comparator
    Kokkos::parallel_for("Genten::Sptensor::createPermutationImpl_init_kernel",
                         Kokkos::RangePolicy<ExecSpace>(0,sz),
                         KOKKOS_LAMBDA(const ttb_indx i)
    {
      tmp(i) = i;
    });

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if (is_gpu_space<ExecSpace>::value) {
      thrust::stable_sort(thrust::device_ptr<ttb_indx>(tmp.data()),
                          thrust::device_ptr<ttb_indx>(tmp.data()+sz),
                          KOKKOS_LAMBDA(const ttb_indx& a, const ttb_indx& b)
      {
        return (subs(a,n) < subs(b,n));
      });
    }
    else
#endif

#if defined(KOKKOS_ENABLE_SYCL)
  if (is_sycl_space<ExecSpace>::value) {
    auto subs_mir = create_mirror_view(subs);
    deep_copy(subs_mir, subs);

    auto tmp_mir = create_mirror_view(tmp);
    deep_copy(tmp_mir, tmp);

    std::stable_sort(
      /*std::execution::par,*/ tmp_mir.data(), tmp_mir.data() + sz,
      KOKKOS_LAMBDA(const ttb_indx& a, const ttb_indx& b) {
        return (subs_mir(a,n) < subs_mir(b,n));
      }
    );

    deep_copy(tmp, tmp_mir);
  }
  else
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
    if (std::is_same<ExecSpace, Kokkos::OpenMP>::value) {
      pss::parallel_stable_sort(tmp.data(), tmp.data()+sz,
                                [&](const ttb_indx& a, const ttb_indx& b)
      {
        return (subs(a,n) < subs(b,n));
      });
    }
    else
#endif

    std::stable_sort(tmp.data(), tmp.data()+sz,
                     [&](const ttb_indx& a, const ttb_indx& b)
    {
      return (subs(a,n) < subs(b,n));
    });


    deep_copy( Kokkos::subview(perm, Kokkos::ALL(), n), tmp );

#endif

  }

#undef USE_KOKKOS_SORT

  const bool check = false;
  if (check) {
    for (ttb_indx n = 0; n < nNumDims; ++n) {
      for (ttb_indx i = 1; i < sz; ++i) {
        if (subs(perm(i,n),n) < subs(perm(i-1,n),n)) {
          std::cout << "Check failed: " << std::endl
                    << "\t" << "i = " << i << std::endl
                    << "\t" << "n = " << n << std::endl
                    << "\t" << "perm(i,n) = " << perm(i,n) << std::endl
                    << "\t" << "perm(i-1,n) = " << perm(i-1,n) << std::endl
                    << "\t" << "subs(perm(i,n),n) = " << subs(perm(i,n),n) << std::endl
                    << "\t" << "subs(perm(i-1,n),n)) = " << subs(perm(i-1,n),n) << std::endl
                    << std::endl;
        }
      }
    }
  }
}

// Implementation of sort().  Has to be done as a
// non-member function because lambda capture of *this doesn't work on Cuda.
template <typename ExecSpace, typename vals_type, typename subs_type>
void
sortImpl(vals_type& vals, subs_type& subs, subs_type& subs_gids)
{
  const ttb_indx sz = subs_gids.extent(0);
  const unsigned nd = subs_gids.extent(1);

  // Neither std::sort or the Kokkos sort will work with non-contiguous views,
  // so we need to sort into temporary views
  Kokkos::View<ttb_indx*,typename subs_type::array_layout,ExecSpace>
    tmp(Kokkos::view_alloc(Kokkos::WithoutInitializing,"tmp_perm"),sz);

  // Sort tmp=[1:sz] using subs_gids as a comparator to compute permutation
  // array to lexicographically sorted order

  // Initialize tmp to unsorted order
  Kokkos::parallel_for("Genten::Sptensor::sortImpl_init_kernel",
                       Kokkos::RangePolicy<ExecSpace>(0,sz),
                       KOKKOS_LAMBDA(const ttb_indx i)
  {
    tmp(i) = i;
  });

  // Comparison functor for lexicographic sorting
  auto cmp =
    KOKKOS_LAMBDA(const ttb_indx& a, const ttb_indx& b)
    {
      unsigned n = 0;
      while ((n < nd) && (subs_gids(a,n) == subs_gids(b,n))) ++n;
      if (n == nd || subs_gids(a,n) >= subs_gids(b,n)) return false;
      return true;
    };

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  if (is_gpu_space<ExecSpace>::value) {
    thrust::stable_sort(thrust::device_ptr<ttb_indx>(tmp.data()),
                        thrust::device_ptr<ttb_indx>(tmp.data()+sz),
                        cmp);
  }
  else
#endif

#if defined(KOKKOS_ENABLE_SYCL)
  if (is_sycl_space<ExecSpace>::value) {
    auto subs_gids_mir = create_mirror_view(subs_gids);
    deep_copy(subs_gids_mir, subs_gids);

    auto tmp_mir = create_mirror_view(tmp);
    deep_copy(tmp_mir, tmp);

    std::stable_sort(
      /*std::execution::par,*/ tmp_mir.data(), tmp_mir.data() + sz,
      KOKKOS_LAMBDA(const ttb_indx& a, const ttb_indx& b) {
        unsigned n = 0;
        while ((n < nd) && (subs_gids_mir(a,n) == subs_gids_mir(b,n))) ++n;
        if (n == nd || subs_gids_mir(a,n) >= subs_gids_mir(b,n)) return false;
        return true;
      }
    );

    deep_copy(tmp, tmp_mir);
  }
  else
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
  if (std::is_same<ExecSpace, Kokkos::OpenMP>::value) {
    pss::parallel_stable_sort(tmp.data(), tmp.data()+sz, cmp);
  }
  else
#endif
  std::stable_sort(tmp.data(), tmp.data() + sz, cmp);

  // Now copy vals and subs to sorted order
  typename vals_type::view_type sorted_vals(
    Kokkos::view_alloc("Genten::Sptensor::vals", Kokkos::WithoutInitializing),
    sz);
  subs_type sorted_subs(
    Kokkos::view_alloc("Genten::Sptensor::subs", Kokkos::WithoutInitializing),
    sz, nd);
  subs_type sorted_subs_gids;
  const bool has_gids = subs.data() != subs_gids.data();
  if (has_gids)
    sorted_subs_gids = subs_type(
      Kokkos::view_alloc("Genten::Sptensor::subs_gids",
                         Kokkos::WithoutInitializing),
      sz, nd);
  else
    sorted_subs_gids = sorted_subs;
  Kokkos::parallel_for("Genten::Sptensor::sortImpl_copy_kernel",
                       Kokkos::RangePolicy<ExecSpace>(0,sz),
                       KOKKOS_LAMBDA(const ttb_indx i)
  {
    sorted_vals(i) = vals[tmp(i)];
    for (unsigned n=0; n<nd; ++n) {
      sorted_subs(i,n) = subs(tmp(i),n);
      if (has_gids)
        sorted_subs_gids(i,n) = subs_gids(tmp(i),n);
    }
  });

  vals = vals_type(sorted_vals);
  subs = sorted_subs;
  subs_gids = sorted_subs_gids;
}
}
}

template <typename ExecSpace>
void Genten::SptensorImpl<ExecSpace>::
createPermutation()
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::Sptensor::createPermutation()");
#endif

  // Only create permutation array if it hasn't already been created
  const ttb_indx sz = subs.extent(0);
  const ttb_indx nNumDims = subs.extent(1);
  if ((perm.extent(0) != sz) || (perm.extent(1) != nNumDims)) {
    perm = subs_view_type(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                             "Genten::Sptensor_kokkos::perm"),
                          sz, nNumDims);
  }
  Genten::Impl::createPermutationImpl<ExecSpace>(perm, subs, siz);
}

template <typename ExecSpace>
void Genten::SptensorImpl<ExecSpace>::
sort()
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::Sptensor::sort()");
#endif

  if (!is_sorted) {
    Genten::Impl::sortImpl<ExecSpace>(values, subs, subs_gids);
    is_sorted = true;
  }
}

#define INST_MACRO(SPACE) template class Genten::SptensorImpl<SPACE>;
GENTEN_INST(INST_MACRO)
