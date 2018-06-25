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

#include "Genten_Sptensor_perm.hpp"
#include "Kokkos_Sort.hpp"

#ifdef KOKKOS_HAVE_OPENMP
#include "parallel_stable_sort.hpp"
#endif

#ifdef KOKKOS_HAVE_CUDA
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#endif

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

namespace Genten {
namespace Impl {
// Implementation of createPermutation().  Has to be done as a
// non-member function because lambda capture of *this doesn't work on Cuda.
template <typename ExecSpace, typename subs_view_type, typename siz_type>
subs_view_type
createPermutationImpl(const subs_view_type& subs, const siz_type& siz)
{
  const ttb_indx sz = subs.extent(0);
  const ttb_indx nNumDims = subs.extent(1);
  subs_view_type perm(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                         "Genten::Sptensor_kokkos::perm"),
                      sz,nNumDims);

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
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,sz),
                         KOKKOS_LAMBDA(const ttb_indx i)
    {
      tmp(i) = i;
    }, "Genten::Sptensor_perm::createPermutationImpl_init_kernel");

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

  return perm;

}
}
}

template <typename ExecSpace>
void Genten::SptensorT_perm<ExecSpace>::
fillComplete()
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::Sptensor_perm::fillComplete()");
#endif

  createPermutation();
}

template <typename ExecSpace>
void Genten::SptensorT_perm<ExecSpace>::
createPermutation()
{
  perm = Genten::Impl::createPermutationImpl<ExecSpace>(this->subs, this->siz);
}

#define INST_MACRO(SPACE) template class Genten::SptensorT_perm<SPACE>;
GENTEN_INST(INST_MACRO)
