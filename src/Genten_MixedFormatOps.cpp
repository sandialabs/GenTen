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

/*!
 * Methods in this file perform operations between objects of mixed formats.
 * In most cases the method could be moved into a particular class, but
 * then similar methods become disconnected, and it could force a
 * fundamental class like Tensor to include knowledge of a derived class
 * like Ktensor.
 */

#include <assert.h>

#include "Genten_Util.hpp"
#include "Genten_FacMatrix.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_Sptensor.hpp"

#include "Genten_MTTKRP.hpp"

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

//-----------------------------------------------------------------------------
//  Method:  innerprod, Sptensor and Ktensor with alternate weights
//-----------------------------------------------------------------------------

namespace Genten {
namespace Impl {

template <typename ExecSpace,
          unsigned RowBlockSize, unsigned FacBlockSize,
          unsigned TeamSize, unsigned VectorSize>
struct InnerProductKernel {

  typedef Kokkos::TeamPolicy<ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  typedef Kokkos::View< ttb_real**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

  const Genten::SptensorT<ExecSpace>& s;
  const Genten::KtensorT<ExecSpace>& u;
  const Genten::ArrayT<ExecSpace>& lambda;
  const ttb_indx nnz;
  const unsigned nd;

  const TeamMember& team;
  const unsigned team_index;
  const unsigned team_size;
  TmpScratchSpace tmp;
  const ttb_indx i_block;

  static inline Policy policy(const ttb_indx nnz_) {
    const ttb_indx N = (nnz_+RowBlockSize-1)/RowBlockSize;
    Policy policy(N,TeamSize,VectorSize);
    size_t bytes = TmpScratchSpace::shmem_size(RowBlockSize,FacBlockSize);
    return policy.set_scratch_size(0,Kokkos::PerTeam(bytes));
  }

  KOKKOS_INLINE_FUNCTION
  InnerProductKernel(const Genten::SptensorT<ExecSpace>& s_,
                     const Genten::KtensorT<ExecSpace>& u_,
                     const Genten::ArrayT<ExecSpace>& lambda_,
                     const TeamMember& team_) :
    s(s_), u(u_), lambda(lambda_),
    nnz(s.nnz()), nd(u.ndims()),
    team(team_), team_index(team.team_rank()), team_size(team.team_size()),
    tmp(team.team_scratch(0), RowBlockSize, FacBlockSize),
    i_block(team.league_rank()*RowBlockSize)
    {}

  template <unsigned Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const unsigned j, const unsigned nj_, ttb_real& d)
  {
    // nj.value == Nj_ if Nj_ > 0 and nj_ otherwise
    Kokkos::Impl::integral_nonzero_constant<unsigned, Nj> nj(nj_);

    const ttb_real *l = &lambda[j];

    for (unsigned ii=team_index; ii<RowBlockSize; ii+=team_size) {
      const ttb_indx i = i_block + ii;
      const ttb_real s_val = i < nnz ? s.value(i) : 0.0;

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,unsigned(nj.value)),
                           [&] (const unsigned& jj)
      {
        tmp(ii,jj) = s_val * l[jj];
      });

      if (i < nnz) {
        for (unsigned m=0; m<nd; ++m) {
          const ttb_real *row = &(u[m].entry(s.subscript(i,m),j));
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,unsigned(nj.value)),
                               [&] (const unsigned& jj)
          {
            tmp(ii,jj) *= row[jj];
          });
        }
      }

    }

    // Do the inner product with 3 levels of parallelism
    ttb_real t = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange( team, RowBlockSize ),
                            [&] ( const unsigned& k, ttb_real& t_outer )
    {
      ttb_real update_outer = 0.0;

      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,unsigned(nj.value)),
                              [&] (const unsigned& jj, ttb_real& t_inner)
      {
        t_inner += tmp(k,jj);
      }, update_outer);

      Kokkos::single( Kokkos::PerThread( team ), [&] ()
      {
        t_outer += update_outer;
      });

    }, t);

    Kokkos::single( Kokkos::PerTeam( team ), [&] ()
    {
      d += t;
    });
  }
};

// Specialization of InnerProductKernel to TeamSize == VectorSize == 1
// (for, e.g., KNL).  Overall this is about 10% faster on KNL.  We could use a
// RangePolicy here, but the TeamPolicy seems to be about 25% faster on KNL.
template <typename ExecSpace,
          unsigned RowBlockSize, unsigned FacBlockSize>
struct InnerProductKernel<ExecSpace,RowBlockSize,FacBlockSize,1,1> {

  typedef Kokkos::TeamPolicy<ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;

  const Genten::SptensorT<ExecSpace>& s;
  const Genten::KtensorT<ExecSpace>& u;
  const Genten::ArrayT<ExecSpace>& lambda;
  const ttb_indx nnz;
  const unsigned nd;
  const ttb_indx i_block;

  alignas(64) ttb_real val[FacBlockSize];
  alignas(64) ttb_real tmp[FacBlockSize];

  static inline Policy policy(const ttb_indx nnz_) {
    const ttb_indx N = (nnz_+RowBlockSize-1)/RowBlockSize;
    Policy policy(N,1,1);
    return policy;
  }

  KOKKOS_INLINE_FUNCTION
  InnerProductKernel(const Genten::SptensorT<ExecSpace>& s_,
                     const Genten::KtensorT<ExecSpace>& u_,
                     const Genten::ArrayT<ExecSpace>& lambda_,
                     const TeamMember& team_) :
    s(s_), u(u_), lambda(lambda_), nnz(s.nnz()), nd(u.ndims()),
    i_block(team_.league_rank()*RowBlockSize)
    {}

  template <unsigned Nj>
  KOKKOS_INLINE_FUNCTION
  void run(const unsigned j, const unsigned nj_, ttb_real& d)
  {
    // nj.value == Nj if Nj > 0 and nj_ otherwise
    Kokkos::Impl::integral_nonzero_constant<unsigned, Nj> nj(nj_);

    const ttb_real *l = &lambda[j];

    for (ttb_indx jj=0; jj<nj.value; ++jj)
      val[jj] = 0.0;

    for (unsigned ii=0; ii<RowBlockSize; ++ii) {
      const ttb_indx i = i_block + ii;

      if (i < nnz) {
        const ttb_real s_val = s.value(i);
        for (ttb_indx jj=0; jj<nj.value; ++jj)
          tmp[jj] = s_val * l[jj];

        for (unsigned m=0; m<nd; ++m) {
          const ttb_real *row = &(u[m].entry(s.subscript(i,m),j));
          for (ttb_indx jj=0; jj<nj.value; ++jj)
            tmp[jj] *= row[jj];
        }

        for (ttb_indx jj=0; jj<nj.value; ++jj)
          val[jj] += tmp[jj];
      }
    }

    for (ttb_indx jj=0; jj<nj.value; ++jj)
      d += val[jj];
  }
};

template <typename ExecSpace, unsigned FacBlockSize>
ttb_real innerprod_kernel(const Genten::SptensorT<ExecSpace>& s,
                          const Genten::KtensorT<ExecSpace>& u,
                          const Genten::ArrayT<ExecSpace>& lambda)
{
  // Compute team and vector sizes, depending on the architecture
  const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;

  const unsigned VectorSize =
    is_cuda ? (FacBlockSize <= 16 ? FacBlockSize : 16) : 1;
  const unsigned TeamSize =
    is_cuda ? 128/VectorSize : 1;
  const unsigned RowBlockSize =
    is_cuda ? TeamSize : 32;

  typedef InnerProductKernel<ExecSpace,RowBlockSize,FacBlockSize,TeamSize,VectorSize> Kernel;
  typedef typename Kernel::TeamMember TeamMember;

  // Do the inner product
  ttb_real dTotal = 0.0;
  Kokkos::parallel_reduce("Genten::innerprod_kernel",
                          Kernel::policy(s.nnz()),
                          KOKKOS_LAMBDA(TeamMember team, ttb_real& d)
  {
    // For some reason using the above typedef causes a really strange
    // compiler error with NVCC 8.0 + GCC 4.9.2
    InnerProductKernel<ExecSpace,RowBlockSize,FacBlockSize,TeamSize,VectorSize> kernel(s,u,lambda,team);

    const unsigned nc = u.ncomponents();
    for (unsigned j=0; j<nc; j+=FacBlockSize) {
      if (j+FacBlockSize <= nc)
        kernel.template run<FacBlockSize>(j, FacBlockSize, d);
      else
        kernel.template run<0>(j, nc-j, d);
    }

  }, dTotal);
  Kokkos::fence();

  return dTotal;
}

}
}

template <typename ExecSpace>
ttb_real Genten::innerprod(const Genten::SptensorT<ExecSpace>& s,
                           const Genten::KtensorT<ExecSpace>& u,
                           const Genten::ArrayT<ExecSpace>& lambda)
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::innerprod");
#endif

  const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;

  const ttb_indx nc = u.ncomponents();               // Number of components
  const ttb_indx nd = u.ndims();                     // Number of dimensions

  // Check on sizes
  assert(nd == s.ndims());
  assert(u.isConsistent(s.size()));
  assert(nc == lambda.size());

  // Call kernel with factor block size determined from nc
  ttb_real d = 0.0;
  if (nc == 1)
    d = Impl::innerprod_kernel<ExecSpace,1>(s,u,lambda);
  else if (nc == 2)
    d = Impl::innerprod_kernel<ExecSpace,2>(s,u,lambda);
  else if (nc <= 4)
    d = Impl::innerprod_kernel<ExecSpace,4>(s,u,lambda);
  else if (nc <= 8)
    d = Impl::innerprod_kernel<ExecSpace,8>(s,u,lambda);
  else if (nc <= 16)
    d = Impl::innerprod_kernel<ExecSpace,16>(s,u,lambda);
  else if (nc < 64 || !is_cuda)
    d = Impl::innerprod_kernel<ExecSpace,32>(s,u,lambda);
  else
    d = Impl::innerprod_kernel<ExecSpace,64>(s,u,lambda);

  return d;
}

#define INST_MACRO(SPACE)                                               \
  template                                                              \
  ttb_real innerprod<>(const Genten::SptensorT<SPACE>& s,               \
                       const Genten::KtensorT<SPACE>& u,                \
                       const Genten::ArrayT<SPACE>& lambda);            \
  template                                                              \
  void mttkrp<>(const Genten::SptensorT<SPACE>& X,                      \
                const Genten::KtensorT<SPACE>& u,                       \
                const ttb_indx n,                                       \
                const Genten::FacMatrixT<SPACE>& v,                     \
                const AlgParams& algParams);
GENTEN_INST(INST_MACRO)
