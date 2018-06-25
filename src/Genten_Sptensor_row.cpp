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

#include "Genten_Sptensor_row.hpp"

namespace Genten {
namespace Impl {
// Implementation of createRow().  Has to be done as a non-member function
// because lambda capture of *this doesn't work on Cuda.
template <typename ExecSpace,
          typename row_ptr_type,
          typename perm_type,
          typename subs_type,
          typename siz_type>
row_ptr_type
createRowPtrImpl(const perm_type& perm,
                 const subs_type& subs,
                 const siz_type& siz)
{
  // Create rowptr array with the starting index of each row
  const ttb_indx sz = perm.extent(0);
  const ttb_indx nNumDims = perm.extent(1);
  row_ptr_type rowptr("Genten::Sptensor_kokkos::host_rowptr",nNumDims);

  for (ttb_indx n = 0; n < nNumDims; ++n) {
    Kokkos::View<ttb_indx*,Kokkos::LayoutRight,ExecSpace> rowptr_n(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,"rowptr_n"),siz[n]+1);

    Kokkos::parallel_for( Kokkos::RangePolicy<ExecSpace>(0,sz+1),
                          KOKKOS_LAMBDA(const ttb_indx i)
    {
      if (i == 0) {
        // The first nonzero row is subs(perm(0,n),n).  The for-loop handles
        // the case when this row != 0.
        const ttb_indx s = subs(perm(0,n),n);
        for (ttb_indx k=0; k<=s; ++k)
          rowptr_n(k) = 0;
      }
      else if (i == sz) {
        // The last nonzero row is subs(perm(sz-1,n),n).  The for-loop handles
        // the case when this row != siz[n]-1.
        const ttb_indx sm = subs(perm(sz-1,n),n);
        for (ttb_indx k=sm+1; k<=siz[n]; ++k)
          rowptr_n(k) = sz;
      }
      else {
        // A row starts when subs(perm(i,n),n) != subs(perm(i-1,n),n).
        // The inner for-loop handles the case when
        // subs(perm(i,n),n) != subs(perm(i-1,n),n)+1
        const ttb_indx s  = subs(perm(i,n),n);
        const ttb_indx sm = subs(perm(i-1,n),n);
        if (s != sm) {
          for (ttb_indx k=sm+1; k<=s; ++k)
            rowptr_n(k) = i;
        }
      }
    });

    rowptr(n) = rowptr_n;
  }


  // Check
  const bool check = false;
  if (check) {
    for (ttb_indx n=0; n<nNumDims; ++n) {
      for (ttb_indx i=0; i<siz[n]; ++i) {
        const ttb_indx r_beg = rowptr(n)(i);
        const ttb_indx r_end = rowptr(n)(i+1);
        for (ttb_indx r=r_beg; r<r_end; ++r) {
          if (subs(perm(r,n),n) != i)
            std::cout << "Check failed:  Expected " << i
                      << " got " << subs(perm(r,n),n) << std::endl;
        }
      }
    }
  }

  return rowptr;
}
}
}

template <typename ExecSpace>
void Genten::SptensorT_row<ExecSpace>::
createRowPtr()
{
  host_rowptr =
    Genten::Impl::createRowPtrImpl<ExecSpace,host_row_ptr_type>(this->perm,
                                                                this->subs,
                                                                this->siz);
  const ttb_indx n = host_rowptr.extent(0);
  row_ptr_type r("Genten::Sptensor_kokkos::rowptr", n);
  for (ttb_indx i=0; i<n; ++i) {
    auto h = host_rowptr(i);
    Kokkos::RangePolicy<ExecSpace> policy(0,1);
    Kokkos::parallel_for( policy, KOKKOS_LAMBDA(const ttb_indx j)
    {
      r(i) = Kokkos::View<ttb_indx*,Kokkos::LayoutRight,ExecSpace>(h.data(), h.extent(0));
    });
  }
  rowptr = r;
}

#define INST_MACRO(SPACE) template class Genten::SptensorT_row<SPACE>;
GENTEN_INST(INST_MACRO)
