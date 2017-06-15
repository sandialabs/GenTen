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
// ************************************************************************
//@HEADER

#include "Genten_Sptensor_perm.h"
#include "Kokkos_Sort.hpp"

#ifdef KOKKOS_HAVE_OPENMP
#include "parallel_stable_sort.hpp"
#endif

#ifdef KOKKOS_HAVE_CUDA
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#endif


// Implementation of createPermutation().  Has to be done as a
// non-member function because lambda capture of *this doesn't work on Cuda.
template <typename subs_view_type, typename siz_type>
subs_view_type
createPermutationImpl(const subs_view_type& subs, const siz_type& siz)
{
  typedef typename subs_view_type::execution_space ExecSpace;

  const ttb_indx sz = subs.dimension_0();
  const ttb_indx nNumDims = subs.dimension_1();
  subs_view_type perm(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                         "Genten::Sptensor_kokkos::perm"),
                      sz,nNumDims);

  // Neither std::sort or the Kokkos sort will work with non-contiguous views,
  // so we need to sort into temporary views
  typedef Kokkos::View<ttb_indx*,ExecSpace> ViewType;
  ViewType tmp(Kokkos::view_alloc(Kokkos::WithoutInitializing,"tmp_perm"),sz);

  for (ttb_indx n = 0; n < nNumDims; ++n) {

// Whether to sort using Kokkos or device specific approaches
// The Kokkos sort seems slower
#define USE_KOKKOS_SORT 0
#if USE_KOKKOS_SORT

    typedef Kokkos::BinOp1D<ViewType> CompType;
    // Kokkos::sort doesn't allow supplying a custom comparator, so we
    // have to use its implementation to get the permutation vector
    Kokkos::deep_copy( tmp, Kokkos::subview(subs, Kokkos::ALL(), n));
    Kokkos::BinSort<ViewType, CompType> bin_sort(
      tmp,CompType(sz/2,0,siz[n]),true);
    bin_sort.create_permute_vector();
    Kokkos::deep_copy( Kokkos::subview(perm, Kokkos::ALL(), n),
                       bin_sort.get_permute_vector() );

#else

    // Sort tmp=[1:sz] using subs(:,n) as a comparator
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,sz),
                         KOKKOS_LAMBDA(const ttb_indx i)
    {
      tmp(i) = i;
    });

#if defined(KOKKOS_HAVE_CUDA)
    if (std::is_same<ExecSpace, Kokkos::Cuda>::value) {
      thrust::stable_sort(thrust::device_ptr<ttb_indx>(tmp.data()),
                          thrust::device_ptr<ttb_indx>(tmp.data()+sz),
                          KOKKOS_LAMBDA(const ttb_indx& a, const ttb_indx& b)
      {
        return (subs(a,n) < subs(b,n));
      });
    }
    else
#endif

#if defined(KOKKOS_HAVE_OPENMP)
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


    Kokkos::deep_copy( Kokkos::subview(perm, Kokkos::ALL(), n), tmp );

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

  return perm;

}

void Genten::Sptensor_perm::
createPermutation()
{
  perm = createPermutationImpl(subs, siz);
}
