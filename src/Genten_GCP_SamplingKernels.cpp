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

#include "Genten_GCP_SamplingKernels.hpp"

#include "Kokkos_Random.hpp"

namespace Genten {

  namespace Impl {

    template <typename ExecSpace, typename LossFunction>
    void stratified_sample_tensor(const SptensorT<ExecSpace>& X,
                                  const ttb_indx num_samples_nonzeros,
                                  const ttb_indx num_samples_zeros,
                                  const ttb_real weight_nonzeros,
                                  const ttb_real weight_zeros,
                                  const KtensorT<ExecSpace>& u,
                                  const LossFunction& loss_func,
                                  const bool compute_gradient,
                                  SptensorT<ExecSpace>& Y,
                                  ArrayT<ExecSpace>& w,
                                  RandomMT& rng,
                                  const AlgParams& algParams)
    {
      const ttb_indx nnz = X.nnz();
      const ttb_indx nd = X.ndims();
      const ttb_indx tsz = X.numel();

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples_nonzeros + num_samples_zeros;
      if (Y.nnz() < total_samples) {
        Y = SptensorT<ExecSpace>(X.size(), total_samples);
        w = ArrayT<ExecSpace>(total_samples);
      }

#if 1
      // Parallel sampling on the device
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      RandomPool rand_pool(rng.genrnd_int32());
      const ttb_indx nloops = algParams.rng_iters;

      // Generate samples of nonzeros
      const ttb_indx N_nonzeros = (num_samples_nonzeros+nloops-1)/nloops;
      Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,N_nonzeros),
                           KOKKOS_LAMBDA(const ttb_indx k)
      {
        generator_type gen = rand_pool.get_state();
        for (ttb_indx l=0; l<nloops; ++l) {
          const ttb_indx i = k*nloops+l;
          if (i<num_samples_nonzeros) {
            const ttb_indx idx = Rand::draw(gen,0,nnz);
            const auto& ind = X.getSubscripts(idx);
            if (compute_gradient) {
              const ttb_real m_val = compute_Ktensor_value(u, ind);
              Y.value(i) =
                weight_nonzeros * loss_func.deriv(X.value(idx), m_val);
            }
            else {
              Y.value(i) = X.value(idx);
              w[i] = weight_nonzeros;
            }
            for (ttb_indx j=0; j<nd; ++j)
              Y.subscript(i,j) = ind[j];
          }
        }
        rand_pool.free_state(gen);
      }, "Genten::GCP_SGD::Uniform_Sample_Nonzeros");

      // Generate samples of zeros
      typedef Kokkos::View< ttb_indx*, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;
      typedef Kokkos::TeamPolicy<ExecSpace> Policy;
      typedef typename Policy::member_type TeamMember;
      const bool is_cuda = is_cuda_space<ExecSpace>::value;
      const ttb_indx vector_size = 1;
      const ttb_indx team_size = is_cuda ? 128/vector_size : 1;
      const ttb_indx loops_per_team = nloops*team_size;
      const ttb_indx N_zeros =
        (num_samples_zeros+loops_per_team-1)/loops_per_team;
      const size_t bytes = TmpScratchSpace::shmem_size(nd);
      Policy policy(N_zeros,team_size,vector_size);
      Kokkos::parallel_for(policy.set_scratch_size(0,Kokkos::PerThread(bytes)),
                           KOKKOS_LAMBDA(const TeamMember& team)
      {
        const ttb_indx k = team.league_rank()*team_size + team.team_rank();
        TmpScratchSpace ind(team.thread_scratch(0), nd);

        generator_type gen = rand_pool.get_state();
        for (ttb_indx l=0; l<nloops; ++l) {
          const ttb_indx i = k*nloops+l;
          if (i<num_samples_zeros) {

            // Keep generating samples until we get one not in the tensor
            bool found = true;
            while (found) {
              // Generate index
              for (ttb_indx j=0; j<nd; ++j)
                ind[j] = Rand::draw(gen,0,X.size(j));

              // Search for index
              found = (X.index(ind) < nnz);

              // If not found, add it
              if (!found) {
                const ttb_indx idx = num_samples_nonzeros + i;
                for (ttb_indx j=0; j<nd; ++j)
                  Y.subscript(idx,j) = ind[j];
                if (compute_gradient) {
                  const ttb_real m_val = compute_Ktensor_value(u, ind);
                  Y.value(idx) =
                    weight_zeros * loss_func.deriv(ttb_real(0.0), m_val);
                }
                else {
                  Y.value(idx) = 0.0;
                  w[idx] = weight_zeros;
                }
              }
            }

          }
        }
        rand_pool.free_state(gen);
      }, "Genten::GCP_SGD::Uniform_Sample_Zeros");
#else
      // Serial sampling on the host
      typedef typename SptensorT<ExecSpace>::HostMirror Sptensor_host_type;
      typedef typename Sptensor_host_type::exec_space host_exec_space;
      typedef typename ArrayT<ExecSpace>::HostMirror Array_host_type;
      typedef typename KtensorT<ExecSpace>::HostMirror Ktensor_host_type;

      // Copy to host
      Sptensor_host_type X_host = create_mirror_view(host_exec_space(), X);
      Sptensor_host_type Y_host = create_mirror_view(host_exec_space(), Y);
      Array_host_type w_host;
      Ktensor_host_type u_host;
      if (compute_gradient) {
        u_host = create_mirror_view(host_exec_space(), u);
        deep_copy(u_host, u);
      }
      else
        w_host = create_mirror_view(host_exec_space(), w);
      deep_copy(X_host, X);

      // Geneate samples of nonzeros
      for (ttb_indx i=0; i<num_samples_nonzeros; ++i) {
        const ttb_indx idx = ttb_indx(rng.genrnd_double() * nnz);
        const auto& ind = X_host.getSubscripts(idx);
        if (compute_gradient) {
          const ttb_real m_val = compute_Ktensor_value(u_host, ind);
          Y_host.value(i) =
            weight_nonzeros * loss_func.deriv(X.value(idx), m_val);
        }
        else {
          Y_host.value(i) = X_host.value(idx);
          w_host[i] = weight_nonzeros;
        }
        for (ttb_indx j=0; j<nd; ++j)
          Y_host.subscript(i,j) = ind[j];
      }

      // Generate samples of zeros
      IndxArrayT<host_exec_space> ind(nd);
      ttb_indx i=0;
      while (i<num_samples_zeros) {

        // Generate index
        for (ttb_indx j=0; j<nd; ++j)
          ind[j] = ttb_indx(rng.genrnd_double() * X_host.size(j));

        // Search for index
        const bool found = (X_host.index(ind) < nnz);

        // If not found, add it
        if (!found) {
          const ttb_indx idx = num_samples_nonzeros + i;
          for (ttb_indx j=0; j<nd; ++j)
            Y_host.subscript(idx,j) = ind[j];
          if (compute_gradient) {
            const ttb_real m_val = compute_Ktensor_value(u_host, ind);
            Y_host.value(idx) =
              weight_zeros * loss_func.deriv(ttb_real(0.0), m_val);
          }
          else {
            Y_host.value(idx) = 0.0;
            w_host[idx] = weight_zeros;
          }
          ++i;
        }

      }

      // Copy to device
      deep_copy(Y, Y_host);
      if (!compute_gradient)
        deep_copy(w, w_host);
#endif

      if (algParams.mttkrp_method == MTTKRP_Method::Perm) {
        Y.createPermutation();
      }
    }

  }

}

#include "Genten_GCP_LossFunctions.hpp"

#define LOSS_INST_MACRO(SPACE,LOSS)                                     \
  template void                                                         \
  Impl::stratified_sample_tensor(const SptensorT<SPACE>& X,         \
                                 const ttb_indx num_samples_nonzeros,   \
                                 const ttb_indx num_samples_zeros,      \
                                 const ttb_real weight_nonzeros,        \
                                 const ttb_real weight_zeros,           \
                                 const KtensorT<SPACE>& u,              \
                                 const LOSS& loss_func,                 \
                                 const bool compute_gradient,           \
                                 SptensorT<SPACE>& Y,                   \
                                 ArrayT<SPACE>& w,                      \
                                 RandomMT& rng,                         \
                                 const AlgParams& algParams);

#define INST_MACRO(SPACE)                                               \
  LOSS_INST_MACRO(SPACE,GaussianLossFunction)                           \
  LOSS_INST_MACRO(SPACE,RayleighLossFunction)                           \
  LOSS_INST_MACRO(SPACE,GammaLossFunction)                              \
  LOSS_INST_MACRO(SPACE,BernoulliLossFunction)                          \
  LOSS_INST_MACRO(SPACE,PoissonLossFunction)

GENTEN_INST(INST_MACRO)
