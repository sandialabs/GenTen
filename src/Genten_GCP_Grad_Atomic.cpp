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

#include "Genten_GCP_HogWild.hpp"
#include "Genten_GCP_LossFunctions.hpp"
#include "Genten_SimdKernel.hpp"
#include "Genten_Util.hpp"
#include "Genten_RandomMT.hpp"

#include "Kokkos_Random.hpp"

namespace Genten {

  namespace {

    // Helper class for generating a single random number and broadcasting
    // across a Cuda warp
    template <typename ExecSpace, typename Rand, unsigned VectorSize>
    struct VectorRng {
      template <typename Gen>
      KOKKOS_INLINE_FUNCTION
      static ttb_indx eval(Gen& gen, const ttb_indx nnz) {
        return Rand::draw(gen,0,nnz);
      }
    };

#if defined(KOKKOS_HAVE_CUDA) && defined(__CUDA_ARCH__)
    template <typename Rand, unsigned VectorSize>
    struct VectorRng<Kokkos::Cuda,Rand,VectorSize> {
      template <typename Gen>
      __device__
      static ttb_indx eval(Gen& gen, const ttb_indx nnz) {
        ttb_indx i = 0;
        if (threadIdx.x == 0)
          i = Rand::draw(gen,0,nnz);
        if (VectorSize > 1)
          i = __shfl_sync(0xffffffff, i, 0, VectorSize);
        return i;
      }
    };
#endif

  }

  namespace Impl {

    // Gradient kernel for gcp_sgd3 that combines sampling and mttkrp for
    // computing the gradient.  It also does each mode in the MTTKRP for each
    // nonzero, rather than doing a full MTTKRP for each mode.  This speeds it
    // up significantly.  Obviously it only works with atomic writes.
    template <unsigned FBS, unsigned VS,
              typename ExecSpace, typename loss_type>
    void gcp_sgd_grad_atomic_kernel(const SptensorT<ExecSpace>& X,
                                    const KtensorT<ExecSpace>& M,
                                    const ArrayT<ExecSpace>& w,
                                    const loss_type& f,
                                    const ttb_indx num_samples,
                                    const KtensorT<ExecSpace>& G,
                                    RandomMT& rng,
                                    const AlgParams& algParams)
    {
      G.setMatrices(0.0);

      static const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;
      static const unsigned RowBlockSize = 128;
      static const unsigned FacBlockSize = FBS;
      static const unsigned VectorSize = is_cuda ? VS : 1;
      static const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;
      static const unsigned RowsPerTeam = TeamSize * RowBlockSize;

      /*const*/ unsigned nd = M.ndims();
      /*const*/ unsigned nc = M.ncomponents();
      /*const*/ ttb_indx ns = num_samples;
      /*const*/ ttb_indx nnz = X.nnz();
      const ttb_indx N = (ns+RowsPerTeam-1)/RowsPerTeam;

      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      RandomPool rand_pool(rng.genrnd_int32());

      typedef Kokkos::TeamPolicy<ExecSpace> Policy;
      typedef typename Policy::member_type TeamMember;
      Policy policy(N, TeamSize, VectorSize);
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx idx = offset + ii;
          if (idx >= ns)
            continue;

          // Generate random tensor index
          /*const*/ ttb_indx i =
            VectorRng<ExecSpace,Rand,VectorSize>::eval(gen,nnz);

          // Compute Ktensor value
          const ttb_real m_val =
            compute_Ktensor_value<ExecSpace, FacBlockSize, VectorSize>(
              M, X, i);

          // Compute Y value
          const ttb_real y_val = w[i] * f.deriv(X.value(i), m_val);

          auto row_func = [&](auto j, auto nj, auto Nj, auto k, auto n) {
            typedef TinyVec<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TV;

            TV tmp(nj, y_val);
            for (unsigned m=0; m<nd; ++m) {
              if (m != n)
                tmp *= &(M[m].entry(X.subscript(i,m),j));
            }
            Kokkos::atomic_add(&(G[n].entry(k,j)), tmp);
          };

          for (unsigned n=0; n<nd; ++n) {
            const ttb_indx k = X.subscript(i,n);
            for (unsigned j=0; j<nc; j+=FacBlockSize) {
              if (j+FacBlockSize <= nc) {
                const unsigned nj = FacBlockSize;
                row_func(j, nj, std::integral_constant<unsigned,nj>(), k, n);
              }
              else {
                const unsigned nj = nc-j;
                row_func(j, nj, std::integral_constant<unsigned,0>(), k, n);
              }
            } // j
          } // n
        } // i
        rand_pool.free_state(gen);
      }, "gcp_sgd_grad_atomic_kernel");
    }

    template <typename ExecSpace, typename loss_type>
    struct GCP_Grad_Atomic {
      typedef ExecSpace exec_space;
      typedef SptensorT<exec_space> tensor_type;
      typedef KtensorT<exec_space> Ktensor_type;
      typedef ArrayT<ExecSpace> weights_type;

      const tensor_type X;
      const Ktensor_type M;
      const weights_type w;
      const loss_type f;
      const ttb_indx num_samples;
      const Ktensor_type G;
      RandomMT& rng;
      const AlgParams algParams;

      GCP_Grad_Atomic(const tensor_type& X_, const Ktensor_type& M_,
                      const weights_type w_, const loss_type& f_,
                      const ttb_indx num_samples_,
                      const Ktensor_type& G_,
                      RandomMT& rng_,
                      const AlgParams& algParams_) :
        X(X_), M(M_), w(w_), f(f_), num_samples(num_samples_), G(G_), rng(rng_),
        algParams(algParams_) {}

      template <unsigned FBS, unsigned VS>
      void run() const {
        gcp_sgd_grad_atomic_kernel<FBS,VS>(X,M,w,f,num_samples,G,rng,algParams);
      }
    };

    template <typename ExecSpace, typename loss_type>
    void gcp_sgd_grad_atomic(const SptensorT<ExecSpace>& X,
                             const KtensorT<ExecSpace>& M,
                             const ArrayT<ExecSpace>& w,
                             const loss_type& f,
                             const ttb_indx num_samples,
                             const KtensorT<ExecSpace>& G,
                             RandomMT& rng,
                             const AlgParams& algParams)
    {
      GCP_Grad_Atomic<ExecSpace,loss_type> kernel(X,M,w,f,num_samples,G,rng,
                                                  algParams);
      run_row_simd_kernel(kernel, M.ncomponents());
    }

  }

}

#define LOSS_INST_MACRO(SPACE,LOSS)                                     \
  template void Impl::gcp_sgd_grad_atomic(                              \
    const SptensorT<SPACE>& X,                                          \
    const KtensorT<SPACE>& M,                                           \
    const ArrayT<SPACE>& w,                                             \
    const LOSS& f,                                                      \
    const ttb_indx num_samples,                                         \
    const KtensorT<SPACE>& G,                                           \
    RandomMT& rng,                                                      \
    const AlgParams& algParams);

// #define INST_MACRO(SPACE)                                               \
//   LOSS_INST_MACRO(SPACE,GaussianLossFunction)                           \
//   LOSS_INST_MACRO(SPACE,RayleighLossFunction)                           \
//   LOSS_INST_MACRO(SPACE,GammaLossFunction)                              \
//   LOSS_INST_MACRO(SPACE,BernoulliLossFunction)                          \
//   LOSS_INST_MACRO(SPACE,PoissonLossFunction)


#define INST_MACRO(SPACE)                                               \
  LOSS_INST_MACRO(SPACE,GaussianLossFunction)                           \
  LOSS_INST_MACRO(SPACE,PoissonLossFunction)

GENTEN_INST(INST_MACRO)
