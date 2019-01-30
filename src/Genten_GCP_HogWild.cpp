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

    // The combined sampling, MTTKRP, and update kernel for GCP_SGD using
    // HogWild!-style parallelism over SGD iterations.  It isn't clear if
    // the ADAM update is correct, or even possible to do in a consistent
    // manner with multiple threads updating the same row.
    template <unsigned FBS, unsigned VS,
              typename ExecSpace, typename loss_type>
    void gcp_sgd_hogwild_kernel(const SptensorT<ExecSpace>& X,
                                const KtensorT<ExecSpace>& u,
                                const ArrayT<ExecSpace>& w,
                                const loss_type& f,
                                const ttb_indx num_samples,
                                const ttb_real step,
                                const ttb_real beta1,
                                const ttb_real beta2,
                                const ttb_real beta1t,
                                const ttb_real beta2t,
                                const ttb_real eps,
                                const bool use_adam,
                                KtensorT<ExecSpace>& g,
                                KtensorT<ExecSpace>& v,
                                RandomMT& rng,
                                const AlgParams& algParams)
    {
      if (use_adam) {
        g.setMatrices(0.0);
        v.setMatrices(0.0);
      }

      static const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;
      static const unsigned RowBlockSize = 128;
      static const unsigned FacBlockSize = FBS;
      static const unsigned VectorSize = is_cuda ? VS : 1;
      static const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;
      static const unsigned RowsPerTeam = TeamSize * RowBlockSize;

      /*const*/ unsigned nd = u.ndims();
      /*const*/ unsigned nc = u.ncomponents();
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
        using std::sqrt;

        generator_type gen = rand_pool.get_state();
        ttb_real my_beta1t = beta1t;
        ttb_real my_beta2t = beta2t;
        ttb_real adam_step = 0.0;

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx idx = offset + ii;
          if (idx >= ns)
            continue;

          // ADAM step size
          if (use_adam) {
            my_beta1t = beta1 * my_beta1t;
            my_beta2t = beta2 * my_beta2t;
            adam_step = step*sqrt(1.0-my_beta2t) / (1.0-my_beta1t);
          }

          // Generate random tensor index, broadcast across vector
          /*const*/ ttb_indx i =
            VectorRng<ExecSpace,Rand,VectorSize>::eval(gen,nnz);

          // Compute Ktensor value
          const ttb_real m_val =
            compute_Ktensor_value<ExecSpace, FacBlockSize, VectorSize>(
              u, X, i);

          // Compute Y value
          const ttb_real y_val = w[i] * f.deriv(X.value(i), m_val);

          // MTTKRP and gradient update
          auto row_func = [&](auto j, auto nj, auto Nj, auto k, auto n) {
            typedef TinyVec<ExecSpace, ttb_real, unsigned, FacBlockSize, Nj(), VectorSize> TV;

            // MTTKRP
            TV tmp(nj, y_val);
            for (unsigned m=0; m<nd; ++m) {
              if (m != n)
                tmp *= &(u[m].entry(X.subscript(i,m),j));
            }

            // Update
            if (use_adam) {
              TV gt = Genten::atomic_oper_fetch([&](auto gt, auto tt)
              {
                return beta1*gt + (1.0-beta1)*tt;
              }, &g[n].entry(k,j), tmp);

              TV vt = Genten::atomic_oper_fetch([&](auto vt, auto tt)
              {
                return beta2*vt + (1.0-beta2)*tt*tt;
              }, &v[n].entry(k,j), tmp);
              Kokkos::atomic_add(&u[n].entry(k,j),(-adam_step)*gt/sqrt(vt+eps));
            }
            else {
              Kokkos::atomic_add(&u[n].entry(k,j), (-step)*tmp);
            }

            // Clip
            if (f.has_lower_bound() || f.has_upper_bound()) {
              TV lb(nj, f.lower_bound());
              TV ub(nj, f.upper_bound());
              Kokkos::atomic_fetch_max(&u[n].entry(k,j), lb);
              Kokkos::atomic_fetch_min(&u[n].entry(k,j), ub);
            }
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
      }, "gcp_sgd_hogwild_kernel");
    }

    template <typename ExecSpace, typename loss_type>
    struct GCP_Grad_HogWild {
      typedef ExecSpace exec_space;
      typedef SptensorT<exec_space> tensor_type;
      typedef KtensorT<exec_space> Ktensor_type;
      typedef ArrayT<ExecSpace> weights_type;

      const tensor_type& X;
      const Ktensor_type& u;
      const weights_type& w;
      const loss_type& f;
      const ttb_indx num_samples;
      const ttb_real step;
      const ttb_real beta1;
      const ttb_real beta2;
      const ttb_real beta1t;
      const ttb_real beta2t;
      const ttb_real eps;
      const bool use_adam;
      Ktensor_type& g;
      Ktensor_type& v;
      RandomMT& rng;
      const AlgParams& algParams;

      GCP_Grad_HogWild(const tensor_type& X_,
                       const Ktensor_type& u_,
                       const weights_type& w_,
                       const loss_type& f_,
                       const ttb_indx num_samples_,
                       const ttb_real step_,
                       const ttb_real beta1_,
                       const ttb_real beta2_,
                       const ttb_real beta1t_,
                       const ttb_real beta2t_,
                       const ttb_real eps_,
                       const bool use_adam_,
                       KtensorT<ExecSpace>& g_,
                       KtensorT<ExecSpace>& v_,
                       RandomMT& rng_,
                       const AlgParams& algParams_) :
        X(X_), u(u_), w(w_), f(f_), num_samples(num_samples_), step(step_),
        beta1(beta1_), beta2(beta2_), beta1t(beta1_), beta2t(beta2t_),
        eps(eps_), use_adam(use_adam_),
        g(g_), v(v_), rng(rng_), algParams(algParams_)
        {}

      template <unsigned FBS, unsigned VS>
      void run() const {
        gcp_sgd_hogwild_kernel<FBS,VS>(
          X,u,w,f,num_samples,step,beta1,beta2,beta1t,beta2t,eps,use_adam,g,v,
          rng,algParams);
      }
    };

    template <typename ExecSpace, typename loss_type>
    void gcp_sgd_hogwild(const SptensorT<ExecSpace>& X,
                         const KtensorT<ExecSpace>& u,
                         const ArrayT<ExecSpace>& w,
                         const loss_type& f,
                         const ttb_indx num_samples,
                         const ttb_real step,
                         const ttb_real beta1,
                         const ttb_real beta2,
                         const ttb_real beta1t,
                         const ttb_real beta2t,
                         const ttb_real eps,
                         const bool use_adam,
                         KtensorT<ExecSpace>& g,
                         KtensorT<ExecSpace>& v,
                         RandomMT& rng,
                         const AlgParams& algParams)
    {
      const ttb_indx nd = u.ndims();
      GCP_Grad_HogWild<ExecSpace,loss_type> kernel(
        X,u,w,f,num_samples,step,beta1,beta2,beta1t,beta2t,eps,use_adam,g,v,
        rng,algParams);
      run_row_simd_kernel(kernel, u.ncomponents());
    }

  }

}

#define LOSS_INST_MACRO(SPACE,LOSS)                                     \
  template void Impl::gcp_sgd_hogwild(                                  \
    const SptensorT<SPACE>& X,                                          \
    const KtensorT<SPACE>& u,                                           \
    const ArrayT<SPACE>& w,                                             \
    const LOSS& f,                                                      \
    const ttb_indx num_samples,                                         \
    const ttb_real step,                                                \
    const ttb_real beta1,                                               \
    const ttb_real beta2,                                               \
    const ttb_real beta1t,                                              \
    const ttb_real beta2t,                                              \
    const ttb_real eps,                                                 \
    const bool use_adam,                                                \
    KtensorT<SPACE>& g,                                                 \
    KtensorT<SPACE>& v,                                                 \
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
