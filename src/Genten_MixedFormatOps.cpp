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

  const Genten::SptensorImpl<ExecSpace>& s;
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
  InnerProductKernel(const Genten::SptensorImpl<ExecSpace>& s_,
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

  const Genten::SptensorImpl<ExecSpace>& s;
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
  InnerProductKernel(const Genten::SptensorImpl<ExecSpace>& s_,
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
ttb_real innerprod_kernel(const Genten::SptensorImpl<ExecSpace>& s,
                          const Genten::KtensorT<ExecSpace>& u,
                          const Genten::ArrayT<ExecSpace>& lambda)
{
  // Compute team and vector sizes, depending on the architecture
  const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;

  const unsigned VectorSize =
    is_gpu ? (FacBlockSize <= 16 ? FacBlockSize : 16) : 1;
  const unsigned TeamSize =
    is_gpu ? 128/VectorSize : 1;
  const unsigned RowBlockSize =
    is_gpu ? TeamSize : 32;

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

  const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;

  const ttb_indx nc = u.ncomponents();               // Number of components

  // Check on sizes
  assert(u.ndims() == s.ndims());
  assert(u.isConsistent(s.size()));
  assert(nc == lambda.size());

  // Call kernel with factor block size determined from nc
  ttb_real d = 0.0;
  if (nc == 1)
    d = Impl::innerprod_kernel<ExecSpace,1>(s.impl(),u,lambda);
  else if (nc == 2)
    d = Impl::innerprod_kernel<ExecSpace,2>(s.impl(),u,lambda);
  else if (nc <= 4)
    d = Impl::innerprod_kernel<ExecSpace,4>(s.impl(),u,lambda);
  else if (nc <= 8)
    d = Impl::innerprod_kernel<ExecSpace,8>(s.impl(),u,lambda);
  else if (nc <= 16)
    d = Impl::innerprod_kernel<ExecSpace,16>(s.impl(),u,lambda);
  else if (nc < 64 || !is_gpu)
    d = Impl::innerprod_kernel<ExecSpace,32>(s.impl(),u,lambda);
  else
    d = Impl::innerprod_kernel<ExecSpace,64>(s.impl(),u,lambda);

  if (u.getProcessorMap() != nullptr) {
    Kokkos::fence();
    d = u.getProcessorMap()->gridAllReduce(d);
  }

  return d;
}

template <typename ExecSpace>
ttb_real Genten::innerprod(const Genten::TensorT<ExecSpace>& xd,
                           const Genten::KtensorT<ExecSpace>& u,
                           const Genten::ArrayT<ExecSpace>& lambda)
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::innerprod");
#endif

  const auto x = xd.impl();

  typedef Kokkos::TeamPolicy<ExecSpace> Policy;
  typedef typename Policy::member_type TeamMember;
  typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

  /*const*/ ttb_indx ne = x.numel();
  /*const*/ unsigned nd = u.ndims();
  /*const*/ unsigned nc = u.ncomponents();

  // Make VectorSize*TeamSize ~= 256 on Cuda, HIP or SYCL
  static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
  const unsigned VectorSize = is_gpu ? nc : 1;
  const unsigned TeamSize = is_gpu ? (256+nc-1)/nc : 1;
  const ttb_indx N = (ne+TeamSize-1)/TeamSize;

  // Check on sizes
  assert(nd == x.ndims());
  assert(u.isConsistent(x.size()));
  assert(nc == lambda.size());

  ttb_real d = 0.0;
  const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);
  Policy policy(N, TeamSize, VectorSize);
  Kokkos::parallel_reduce("Genten::innerprod",
                          policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
                          KOKKOS_LAMBDA(const TeamMember& team, ttb_real& t)
  {
    // Term in inner product we compute
    const unsigned team_rank = team.team_rank();
    const unsigned team_size = team.team_size();
    const ttb_indx i = team.league_rank()*team_size+team_rank;

    ttb_real u_val = 0.0;
    if (i < ne) {
      // Compute subscript for entry i
      TmpScratchSpace scratch(team.team_scratch(0), team_size, nd);
      ttb_indx *sub = &scratch(team_rank, 0);
      Kokkos::single(Kokkos::PerThread(team), [&]()
      {
        x.ind2sub(sub, i);
      });

      // Compute Ktensor value for given indices
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, nc),
                              [&](const unsigned j, ttb_real& v)
      {
        ttb_real tmp = lambda[j];
        for (unsigned m=0; m<nd; ++m) {
          tmp *= u[m].entry(sub[m],j);
        }
        v += tmp;
      }, u_val);
      u_val *= x[i];
    }

    // Reduce inner-product contributions across team
    ttb_real dt = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, team_size),
                            [&](const unsigned, ttb_real& dt_team)
    {
      Kokkos::single(Kokkos::PerThread(team), [&]() { dt_team += u_val; });
    }, dt);

    // Add in team contribution to inner-product
    Kokkos::single(Kokkos::PerTeam(team), [&]() { t += dt; });

  }, d);

  if (u.getProcessorMap() != nullptr) {
    Kokkos::fence();
    d = u.getProcessorMap()->gridAllReduce(d);
  }

  return d;
}

namespace Genten {
namespace Impl {

template <typename ExecSpace>
struct MTTKRP_Dense_Kernel {
  const TensorImpl<ExecSpace> XX;
  const KtensorT<ExecSpace> uu;
  const ttb_indx nn;
  const FacMatrixT<ExecSpace> vv;
  const AlgParams algParams;

  MTTKRP_Dense_Kernel(const TensorImpl<ExecSpace>& X_,
                      const KtensorT<ExecSpace>& u_,
                      const ttb_indx n_,
                      const FacMatrixT<ExecSpace>& v_,
                      const AlgParams& algParams_) :
    XX(X_), uu(u_), nn(n_), vv(v_), algParams(algParams_) {}

  template <unsigned FBS, unsigned VS>
  void run() const
  {
    const TensorImpl<ExecSpace> X = XX;
    const KtensorT<ExecSpace> u = uu;
    const ttb_indx n = nn;
    const FacMatrixT<ExecSpace> v = vv;

    typedef Kokkos::TeamPolicy<ExecSpace> Policy;
    typedef typename Policy::member_type TeamMember;
    typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

    static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
    static const unsigned FacBlockSize = FBS;
    static const unsigned VectorSize = is_gpu ? VS : 1;
    static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;

    /*const*/ unsigned nd = u.ndims();
    /*const*/ unsigned nc = u.ncomponents();
    /*const*/ ttb_indx ns = X.size(n);
    const ttb_indx N = (ns+TeamSize-1)/TeamSize;

    const size_t bytes = TmpScratchSpace::shmem_size(TeamSize, nd);
    Policy policy(N, TeamSize, VectorSize);
    Kokkos::parallel_for("mttkrp_kernel",
                         policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
                         KOKKOS_LAMBDA(const TeamMember& team)
    {
      // Row of v we write to
      const unsigned team_rank = team.team_rank();
      const unsigned team_size = team.team_size();
      /*const*/ ttb_indx i = team.league_rank()*team_size + team_rank;
      if (i >= ns)
        return;

      // Scratch space for storing tensor subscripts
      TmpScratchSpace scratch(team.team_scratch(0), team_size, nd);
      ttb_indx *sub = &scratch(team_rank, 0);

      // lambda function for MTTKRP for block of size nj
      auto row_func = [&](auto j, auto nj, auto Nj) {
        typedef TinyVecMaker<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TVM;

        // Work around internal-compiler errors in recent Intel compilers
        unsigned nd_ = nd;
        unsigned n_ = n;
        TensorImpl<ExecSpace> X_ = X;

        // Initialize our subscript array for row i of mode n
        Kokkos::single(Kokkos::PerThread(team), [&]()
        {
          for (unsigned l=0; l<nd_; ++l)
            sub[l] = 0;
          sub[n_] = i;
        });

        auto val = TVM::make(team, nj, 0.0);
        int done = 0;
        while (!done) {  // Would like to get some parallelism in this loop
          const ttb_indx k = X.sub2ind(sub);
          const ttb_real x_val = X[k];
          auto tmp = TVM::make(team, nj, x_val);
          tmp *= &(u.weights(j));
          for (unsigned m=0; m<nd_; ++m) {
            if (m != n_)
              tmp *= &(u[m].entry(sub[m],j));
          }
          val += tmp;

          Kokkos::single(Kokkos::PerThread(team), [&](int& dn)
          {
            dn = !X_.increment_sub(sub,n_);
          }, done);
        };
        val.store_plus(&v.entry(i,j));
      };

      // Do MTTKRP in blocks of size FacBlockSize
      for (unsigned j=0; j<nc; j+=FacBlockSize) {
        if (j+FacBlockSize <= nc) {
          const unsigned nj = FacBlockSize;
          row_func(j, nj, std::integral_constant<unsigned,nj>());
        }
        else {
          const unsigned nj = nc-j;
          row_func(j, nj, std::integral_constant<unsigned,0>());
        }
      }
    });
  }

};

}
}

template <typename ExecSpace>
void Genten::mttkrp(const Genten::TensorT<ExecSpace>& X,
                    const Genten::KtensorT<ExecSpace>& u,
                    const ttb_indx n,
                    const Genten::FacMatrixT<ExecSpace>& v,
                    const Genten::AlgParams& algParams,
                    const bool zero_v)
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::mttkrp");
#endif

  const ttb_indx nc = u.ncomponents();     // Number of components
  const ttb_indx nd = u.ndims();           // Number of dimensions

  assert(X.ndims() == nd);
  assert(u.isConsistent());
  for (ttb_indx i = 0; i < nd; i++)
  {
    if (i != n)
      assert(u[i].nRows() == X.size(i));
  }
  assert( v.nRows() == X.size(n) );
  assert( v.nCols() == nc );

  if (zero_v)
    v = ttb_real(0.0);

  Genten::Impl::MTTKRP_Dense_Kernel<ExecSpace> kernel(X.impl(),u,n,v,algParams);
  Genten::Impl::run_row_simd_kernel(kernel, nc);
}

#define INST_MACRO(SPACE)                                               \
  template                                                              \
  ttb_real innerprod<>(const Genten::SptensorT<SPACE>& s,               \
                       const Genten::KtensorT<SPACE>& u,                \
                       const Genten::ArrayT<SPACE>& lambda);            \
                                                                        \
  template                                                              \
  ttb_real innerprod<>(const Genten::TensorT<SPACE>& s,                 \
                       const Genten::KtensorT<SPACE>& u,                \
                       const Genten::ArrayT<SPACE>& lambda);            \
                                                                        \
  template                                                              \
  void mttkrp<>(const Genten::SptensorT<SPACE>& X,                      \
                const Genten::KtensorT<SPACE>& u,                       \
                const ttb_indx n,                                       \
                const Genten::FacMatrixT<SPACE>& v,                     \
                const AlgParams& algParams,                             \
                const bool zero_v);                                     \
                                                                        \
  template                                                              \
  void mttkrp<>(const Genten::TensorT<SPACE>& X,                        \
                const Genten::KtensorT<SPACE>& u,                       \
                const ttb_indx n,                                       \
                const Genten::FacMatrixT<SPACE>& v,                     \
                const AlgParams& algParams,                             \
                const bool zero_v);                                     \
                                                                        \
  template                                                              \
  void mttkrp_all<>(const Genten::SptensorT<SPACE>& X,                  \
                    const Genten::KtensorT<SPACE>& u,                   \
                    const Genten::KtensorT<SPACE>& v,                   \
                    const ttb_indx mode_beg,                            \
                    const ttb_indx mode_end,                            \
                    const AlgParams& algParams,                         \
                    const bool zero_v);
GENTEN_INST(INST_MACRO)
