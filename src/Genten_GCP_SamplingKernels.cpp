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
#include "Kokkos_UnorderedMap.hpp"

#include "Genten_IOtext.hpp"

namespace Genten {

  namespace Impl {

    template <typename ExecSpace, typename LossFunction>
    void uniform_sample_tensor(
      const SptensorT<ExecSpace>& X,
      const ttb_indx num_samples,
      const ttb_real weight,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams)
    {
      typedef Kokkos::TeamPolicy<ExecSpace> Policy;
      typedef typename Policy::member_type TeamMember;
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

      static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
      static const unsigned RowBlockSize = 1;
      static const unsigned FacBlockSize = 16; // FIXME
      static const unsigned VectorSize = is_gpu ? 16 : 1; //FIXME
      static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;
      static const unsigned RowsPerTeam = TeamSize * RowBlockSize;

      /*const*/ ttb_indx nnz = X.nnz();
      /*const*/ unsigned nd = u.ndims();
      /*const*/ ttb_indx ns = num_samples;
      const ttb_indx N = (ns+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples;
      if (Y.nnz() < total_samples) {
        Y = SptensorT<ExecSpace>(X.size(), total_samples);
        w = ArrayT<ExecSpace>(total_samples);
      }

      // Generate samples of tensor
      Policy policy(N, TeamSize, VectorSize);
      Kokkos::parallel_for(
        policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx idx = offset + ii;
          if (idx >= ns)
            continue;

          // Sample tensor and get value
          ttb_real x_val = 0.0;
          Kokkos::single( Kokkos::PerThread( team ), [&] (ttb_real& xv)
          {
            for (ttb_indx m=0; m<nd; ++m)
              ind[m] = Rand::draw(gen,0,X.size(m));
            const ttb_indx i = X.index(ind);
            if (i < nnz)
              xv = X.value(i);
            else
              xv = 0.0;
          }, x_val);

          // Compute Ktensor value
          ttb_real m_val = 0.0;
          if (compute_gradient)
            m_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(u, ind);

          // Add new entry
          const ttb_indx row = idx;
          Kokkos::single( Kokkos::PerThread( team ), [&] ()
          {
            for (ttb_indx m=0; m<nd; ++m)
              Y.subscript(row,m) = ind[m];
            if (compute_gradient) {
              Y.value(row) =
                weight * loss_func.deriv(x_val, m_val);
            }
            else {
              Y.value(row) = x_val;
              w[row] = weight;
            }
          });
        }
        rand_pool.free_state(gen);
      }, "Genten::GCP_SGD::Uniform_Sample");
    }

    template <typename ExecSpace, typename LossFunction>
    void uniform_sample_tensor_hash(
      const SptensorT<ExecSpace>& X,
      const TensorHashMap<ExecSpace>& hash,
      const ttb_indx num_samples,
      const ttb_real weight,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams)
    {
      typedef Kokkos::TeamPolicy<ExecSpace> Policy;
      typedef typename Policy::member_type TeamMember;
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

      static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
      static const unsigned RowBlockSize = 1;
      static const unsigned FacBlockSize = 16; // FIXME
      static const unsigned VectorSize = is_gpu ? 16 : 1; //FIXME
      static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;
      static const unsigned RowsPerTeam = TeamSize * RowBlockSize;

      /*const*/ ttb_indx nnz = X.nnz();
      /*const*/ unsigned nd = u.ndims();
      /*const*/ ttb_indx ns = num_samples;
      const ttb_indx N = (ns+RowsPerTeam-1)/RowsPerTeam;

      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples;
      if (Y.nnz() < total_samples) {
        Y = SptensorT<ExecSpace>(X.size(), total_samples);
        w = ArrayT<ExecSpace>(total_samples);
      }

      // Generate samples of tensor
      Policy policy(N, TeamSize, VectorSize);
      Kokkos::parallel_for(
        policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx idx = offset + ii;
          if (idx >= ns)
            continue;

          // Sample tensor and get value
          ttb_real x_val = 0.0;
          Kokkos::single( Kokkos::PerThread( team ), [&] (ttb_real& xv)
          {
            for (ttb_indx m=0; m<nd; ++m)
              ind[m] = Rand::draw(gen,0,X.size(m));
            const auto hash_index = hash.find(ind);
            if (hash.valid_at(hash_index)) {
              const ttb_indx i = hash.value_at(hash_index);
              xv = X.value(i);
            }
            else
              xv = 0.0;
          }, x_val);

          // Compute Ktensor value
          ttb_real m_val;
          if (compute_gradient)
            m_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(u, ind);

          // Add new entry
          const ttb_indx row = idx;
          Kokkos::single( Kokkos::PerThread( team ), [&] ()
          {
            for (ttb_indx m=0; m<nd; ++m)
              Y.subscript(row,m) = ind[m];
            if (compute_gradient) {
              Y.value(row) =
                weight * loss_func.deriv(x_val, m_val);
            }
            else {
              Y.value(row) = x_val;
              w[row] = weight;
            }
          });
        }
        rand_pool.free_state(gen);
      }, "Genten::GCP_SGD::Uniform_Sample");
    }

    template <typename ExecSpace, typename LossFunction>
    void stratified_sample_tensor(
      const SptensorT<ExecSpace>& X,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams)
    {
      typedef Kokkos::TeamPolicy<ExecSpace> Policy;
      typedef typename Policy::member_type TeamMember;
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

      static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
      static const unsigned RowBlockSize = 1;
      static const unsigned FacBlockSize = 16; // FIXME
      static const unsigned VectorSize = is_gpu ? 16 : 1; //FIXME
      static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;
      static const unsigned RowsPerTeam = TeamSize * RowBlockSize;

      /*const*/ ttb_indx nnz = X.nnz();
      /*const*/ unsigned nd = X.ndims();
      /*const*/ ttb_indx ns_nz = num_samples_nonzeros;
      /*const*/ ttb_indx ns_z = num_samples_zeros;
      const ttb_indx N_nz = (ns_nz+RowsPerTeam-1)/RowsPerTeam;
      const ttb_indx N_z = (ns_z+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples_nonzeros + num_samples_zeros;
      if (Y.nnz() < total_samples) {
        Y = SptensorT<ExecSpace>(X.size(), total_samples);
        w = ArrayT<ExecSpace>(total_samples);
      }

      // Generate samples of nonzeros
      Policy policy_nz(N_nz, TeamSize, VectorSize);
      Kokkos::parallel_for(
        policy_nz.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          /*const*/ ttb_indx idx = offset + ii;
          if (idx >= ns_nz)
            continue;

          // Generate random tensor index
          ttb_real x_val = 0.0;
          Kokkos::single( Kokkos::PerThread( team ), [&] (ttb_real& xv)
          {
            ttb_indx i;
            if (ns_nz == nnz)
              i = idx;        // Don't sample if all nonzeros were requested
            else
              i = Rand::draw(gen,0,nnz);
            for (ttb_indx m=0; m<nd; ++m)
              ind[m] = X.subscript(i,m);
            xv = X.value(i);
          }, x_val);

          // Compute Ktensor value
          ttb_real m_val = 0.0;
          if (compute_gradient)
            m_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(u, ind);

          // Add new nonzero
          Kokkos::single( Kokkos::PerThread( team ), [&] ()
          {
            for (ttb_indx m=0; m<nd; ++m)
              Y.subscript(idx,m) = ind[m];
            if (compute_gradient) {
              Y.value(idx) =
                weight_nonzeros * loss_func.deriv(x_val, m_val);
            }
            else {
              Y.value(idx) = x_val;
              w[idx] = weight_nonzeros;
            }
          });
        }
        rand_pool.free_state(gen);
      }, "Genten::GCP_SGD::Stratified_Sample_Nonzeros");


      // Generate samples of zeros
      Policy policy_z(N_z, TeamSize, VectorSize);
      Kokkos::parallel_for(
        policy_z.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx idx = offset + ii;
          if (idx >= ns_z)
            continue;

          // Keep generating samples until we get one not in the tensor
          int found = 1;
          while (found) {
            Kokkos::single( Kokkos::PerThread( team ), [&] (int& f)
            {
              // Generate index
              for (ttb_indx m=0; m<nd; ++m)
                ind[m] = Rand::draw(gen,0,X.size(m));

              // Search for index
              f = (X.index(ind) < nnz);
            }, found);
          }

          // Compute Ktensor value
          ttb_real m_val;
          if (compute_gradient)
            m_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(u, ind);

          // Add new nonzero
          const ttb_indx row = num_samples_nonzeros + idx;
          Kokkos::single( Kokkos::PerThread( team ), [&] ()
          {
            for (ttb_indx m=0; m<nd; ++m)
              Y.subscript(row,m) = ind[m];
            if (compute_gradient) {
              Y.value(row) =
                weight_zeros * loss_func.deriv(ttb_real(0.0), m_val);
            }
            else {
              Y.value(row) = 0.0;
              w[row] = weight_zeros;
            }
          });
        }
        rand_pool.free_state(gen);
      }, "Genten::GCP_SGD::Stratified_Sample_Zeros");
    }

    template <typename ExecSpace, typename LossFunction>
    void stratified_sample_tensor_hash(
      const SptensorT<ExecSpace>& X,
      const TensorHashMap<ExecSpace>& hash,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams)
    {
      typedef Kokkos::TeamPolicy<ExecSpace> Policy;
      typedef typename Policy::member_type TeamMember;
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

      static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
      static const unsigned RowBlockSize = 1;
      static const unsigned FacBlockSize = 16; // FIXME
      static const unsigned VectorSize = is_gpu ? 16 : 1; //FIXME
      static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;
      static const unsigned RowsPerTeam = TeamSize * RowBlockSize;

      /*const*/ ttb_indx nnz = X.nnz();
      /*const*/ unsigned nd = u.ndims();
      /*const*/ ttb_indx ns_nz = num_samples_nonzeros;
      /*const*/ ttb_indx ns_z = num_samples_zeros;
      const ttb_indx N_nz = (ns_nz+RowsPerTeam-1)/RowsPerTeam;
      const ttb_indx N_z = (ns_z+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples_nonzeros + num_samples_zeros;
      if (Y.nnz() < total_samples) {
        Y = SptensorT<ExecSpace>(X.size(), total_samples);
        w = ArrayT<ExecSpace>(total_samples);
      }

      // Generate samples of nonzeros
      Policy policy_nz(N_nz, TeamSize, VectorSize);
      Kokkos::parallel_for(
        policy_nz.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx idx = offset + ii;
          if (idx >= ns_nz)
            continue;

          // Generate random tensor index
          ttb_real x_val = 0.0;
          Kokkos::single( Kokkos::PerThread( team ), [&] (ttb_real& xv)
          {
            const ttb_indx i = Rand::draw(gen,0,nnz);
            for (ttb_indx m=0; m<nd; ++m)
              ind[m] = X.subscript(i,m);
            xv = X.value(i);
          }, x_val);

          // Compute Ktensor value
          ttb_real m_val = 0.0;
          if (compute_gradient)
            m_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(u, ind);

          // Add new nonzero
          Kokkos::single( Kokkos::PerThread( team ), [&] ()
          {
            for (ttb_indx m=0; m<nd; ++m)
              Y.subscript(idx,m) = ind[m];
            if (compute_gradient) {
              Y.value(idx) =
                weight_nonzeros * loss_func.deriv(x_val, m_val);
            }
            else {
              Y.value(idx) = x_val;
              w[idx] = weight_nonzeros;
            }
          });
        }
        rand_pool.free_state(gen);
      }, "Genten::GCP_SGD::Stratified_Sample_Nonzeros");

      // Generate samples of zeros
      Policy policy_z(N_z, TeamSize, VectorSize);
      Kokkos::parallel_for(
        policy_z.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx idx = offset + ii;
          if (idx >= ns_z)
            continue;

          // Keep generating samples until we get one not in the tensor
          int found = 1;
          while (found) {
            Kokkos::single( Kokkos::PerThread( team ), [&] (int& f)
            {
              // Generate index
              for (ttb_indx m=0; m<nd; ++m)
                ind[m] = Rand::draw(gen,0,X.size(m));

              // Search for index
              f = hash.exists(ind);
            }, found);
          }

          // Compute Ktensor value
          ttb_real m_val;
          if (compute_gradient)
            m_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(u, ind);

          // Add new nonzero
          const ttb_indx row = num_samples_nonzeros + idx;
          Kokkos::single( Kokkos::PerThread( team ), [&] ()
          {
            for (ttb_indx m=0; m<nd; ++m)
              Y.subscript(row,m) = ind[m];
            if (compute_gradient) {
              Y.value(row) =
                weight_zeros * loss_func.deriv(ttb_real(0.0), m_val);
            }
            else {
              Y.value(row) = 0.0;
              w[row] = weight_zeros;
            }
          });
        }
        rand_pool.free_state(gen);
      }, "Genten::GCP_SGD::Stratified_Sample_Zeros");
    }

    template <typename ExecSpace, typename LossFunction>
    void semi_stratified_sample_tensor(
      const SptensorT<ExecSpace>& X,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& w,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams)
    {
      typedef Kokkos::TeamPolicy<ExecSpace> Policy;
      typedef typename Policy::member_type TeamMember;
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

      static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
      static const unsigned RowBlockSize = 1;
      static const unsigned FacBlockSize = 16; // FIXME
      static const unsigned VectorSize = is_gpu ? 16 : 1; //FIXME
      static const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;
      static const unsigned RowsPerTeam = TeamSize * RowBlockSize;

      /*const*/ ttb_indx nnz = X.nnz();
      /*const*/ unsigned nd = u.ndims();
      /*const*/ ttb_indx ns_nz = num_samples_nonzeros;
      /*const*/ ttb_indx ns_z = num_samples_zeros;
      const ttb_indx N_nz = (ns_nz+RowsPerTeam-1)/RowsPerTeam;
      const ttb_indx N_z = (ns_z+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples_nonzeros + num_samples_zeros;
      if (Y.nnz() < total_samples) {
        Y = SptensorT<ExecSpace>(X.size(), total_samples);
        w = ArrayT<ExecSpace>(total_samples);
      }

      // Generate samples of nonzeros
      Policy policy_nz(N_nz, TeamSize, VectorSize);
      Kokkos::parallel_for(
        policy_nz.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx idx = offset + ii;
          if (idx >= ns_nz)
            continue;

          // Generate random tensor index
          ttb_real x_val = 0.0;
          Kokkos::single( Kokkos::PerThread( team ), [&] (ttb_real& xv)
          {
            const ttb_indx i = Rand::draw(gen,0,nnz);
            for (ttb_indx m=0; m<nd; ++m)
              ind[m] = X.subscript(i,m);
            xv = X.value(i);
          }, x_val);

          // Compute Ktensor value
          ttb_real m_val = 0.0;
          if (compute_gradient)
            m_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(u, ind);

          // Add new nonzero
          Kokkos::single( Kokkos::PerThread( team ), [&] ()
          {
            for (ttb_indx m=0; m<nd; ++m)
              Y.subscript(idx,m) = ind[m];
            if (compute_gradient) {
              Y.value(idx) =
                weight_nonzeros * ( loss_func.deriv(x_val, m_val) -
                                    loss_func.deriv(ttb_real(0.0), m_val) );
            }
            else {
              Y.value(idx) = x_val;
              w[idx] = weight_nonzeros;
            }
          });
        }
        rand_pool.free_state(gen);
      }, "Genten::GCP_SGD::SemiStratified_Sample_Nonzeros");

      // Generate samples of zeros
      Policy policy_z(N_z, TeamSize, VectorSize);
      Kokkos::parallel_for(
        policy_z.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx idx = offset + ii;
          if (idx >= ns_z)
            continue;

          // Generate index -- use broadcast form to force warp sync
          // so that ind is updated before used by other threads
          int sync = 0;
          Kokkos::single( Kokkos::PerThread( team ), [&] (int& s)
          {
            for (ttb_indx m=0; m<nd; ++m)
              ind[m] = Rand::draw(gen,0,X.size(m));
            s = 1;
          }, sync);

          // Compute Ktensor value
          ttb_real m_val = 0.0;
          if (compute_gradient)
            m_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(u, ind);

          // Add new nonzero
          const ttb_indx row = num_samples_nonzeros + idx;
          Kokkos::single( Kokkos::PerThread( team ), [&] ()
          {
            for (ttb_indx m=0; m<nd; ++m)
              Y.subscript(row,m) = ind[m];
            if (compute_gradient) {
              Y.value(row) =
                weight_zeros * loss_func.deriv(ttb_real(0.0), m_val);
            }
            else {
              Y.value(row) = 0.0;
              w[row] = weight_zeros;
            }
          });
        }
        rand_pool.free_state(gen);
      }, "Genten::GCP_SGD::SemiStratified_Sample_Zeros");
    }

    template <typename ExecSpace, typename LossFunction>
    void sample_tensor_nonzeros(
      const SptensorT<ExecSpace>& X,
      const ttb_indx offset,
      const ttb_indx num_samples,
      const ttb_real weight,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Y,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams)
    {
      const ttb_indx nnz = X.nnz();
      const ttb_indx nd = X.ndims();

      // Resize Y if necessary
      if (Y.nnz() < offset+num_samples) {
        Y = SptensorT<ExecSpace>(X.size(), offset+num_samples);
      }

      // Parallel sampling on the device
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      const ttb_indx nloops = algParams.rng_iters;
      const ttb_indx N_nonzeros = (num_samples+nloops-1)/nloops;
      Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,N_nonzeros),
                           KOKKOS_LAMBDA(const ttb_indx k)
      {
        generator_type gen = rand_pool.get_state();
        for (ttb_indx l=0; l<nloops; ++l) {
          const ttb_indx i = k*nloops+l;
          if (i<num_samples) {
            const ttb_indx idx = Rand::draw(gen,0,nnz);
            const auto& ind = X.getSubscripts(idx);
            if (compute_gradient) {
              const ttb_real m_val = compute_Ktensor_value(u, ind);
              Y.value(offset+i) =
                weight * loss_func.deriv(X.value(idx), m_val);
            }
            else {
              Y.value(offset+i) = X.value(idx);
            }
            for (ttb_indx j=0; j<nd; ++j)
              Y.subscript(offset+i,j) = ind[j];
          }
        }
        rand_pool.free_state(gen);
      }, "Genten::GCP_SGD::sample_tensor_nonzeros");

    }

    template <typename ExecSpace, typename LossFunction>
    void sample_tensor_nonzeros(
      const SptensorT<ExecSpace>& X,
      const ArrayT<ExecSpace>& w,
      const ttb_indx num_samples,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      SptensorT<ExecSpace>& Y,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams)
    {
      const ttb_indx nnz = X.nnz();
      const ttb_indx nd = X.ndims();

      // Resize Y if necessary
      if (Y.nnz() < num_samples) {
        Y = SptensorT<ExecSpace>(X.size(), num_samples);
      }

      // Parallel sampling on the device
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      const ttb_indx nloops = algParams.rng_iters;
      const ttb_indx N_nonzeros = (num_samples+nloops-1)/nloops;
      Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,N_nonzeros),
                           KOKKOS_LAMBDA(const ttb_indx k)
      {
        generator_type gen = rand_pool.get_state();
        for (ttb_indx l=0; l<nloops; ++l) {
          const ttb_indx i = k*nloops+l;
          if (i<num_samples) {
            const ttb_indx idx = Rand::draw(gen,0,nnz);
            const auto& ind = X.getSubscripts(idx);
            const ttb_real m_val = compute_Ktensor_value(u, ind);
            Y.value(i) = w[idx] * loss_func.deriv(X.value(idx), m_val);
            for (ttb_indx j=0; j<nd; ++j)
              Y.subscript(i,j) = ind[j];
          }
        }
        rand_pool.free_state(gen);
      }, "Genten::GCP_SGD::sample_tensor_nonzeros");

    }

    template <typename ExecSpace>
    void sample_tensor_nonzeros(
      const SptensorT<ExecSpace>& X,
      const ArrayT<ExecSpace>& w,
      const ttb_indx num_samples,
      SptensorT<ExecSpace>& Y,
      ArrayT<ExecSpace>& z,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams)
    {
      const ttb_indx nnz = X.nnz();
      const ttb_indx nd = X.ndims();

      // Resize Y if necessary
      if (Y.nnz() < num_samples) {
        Y = SptensorT<ExecSpace>(X.size(), num_samples);
        z = ArrayT<ExecSpace>(num_samples);
      }

      // Parallel sampling on the device
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      const ttb_indx nloops = algParams.rng_iters;
      const ttb_indx N_nonzeros = (num_samples+nloops-1)/nloops;
      Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,N_nonzeros),
                           KOKKOS_LAMBDA(const ttb_indx k)
      {
        generator_type gen = rand_pool.get_state();
        for (ttb_indx l=0; l<nloops; ++l) {
          const ttb_indx i = k*nloops+l;
          if (i<num_samples) {
            const ttb_indx idx = Rand::draw(gen,0,nnz);
            Y.value(i) = X.value(idx);
            for (ttb_indx j=0; j<nd; ++j)
              Y.subscript(i,j) = X.subscript(idx,j);
            z[i] = w[idx];
          }
        }
        rand_pool.free_state(gen);
      }, "Genten::GCP_SGD::sample_tensor_nonzeros");

    }

    template <typename ExecSpace>
    void sample_tensor_zeros(
      const SptensorT<ExecSpace>& X,
      const ttb_indx offset,
      const ttb_indx num_samples,
      SptensorT<ExecSpace>& Y,
      SptensorT<ExecSpace>& Z,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams)
    {
      const ttb_indx nnz = X.nnz();
      const ttb_indx nd = X.ndims();
      const ttb_real ne = X.numel_float();
      const ttb_indx ns = std::min(num_samples, ttb_indx(ne-nnz));

      // Resize Y if necessary
      if (Y.nnz() < offset+ns) {
        Y = SptensorT<ExecSpace>(X.size(), offset+ns);
      }

      // Parallel sampling on the device
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      const ttb_indx nloops = algParams.rng_iters;

      // First generate oversample_factor*ns samples of potential zeros
      const ttb_real oversample = algParams.oversample_factor;
      const ttb_indx total_samples =
        static_cast<ttb_indx>(oversample*num_samples);
      if (Z.nnz() < total_samples) {
        Z = SptensorT<ExecSpace>(X.size(), total_samples);
      }
      const ttb_indx N_zeros_gen = (total_samples+nloops-1)/nloops;
      Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,N_zeros_gen),
                           KOKKOS_LAMBDA(const ttb_indx k)
      {
        generator_type gen = rand_pool.get_state();
        for (ttb_indx l=0; l<nloops; ++l) {
          const ttb_indx i = k*nloops+l;
          if (i<total_samples) {
            for (ttb_indx j=0; j<nd; ++j)
              Z.subscript(i,j) = Rand::draw(gen,0,X.size(j));
            Z.value(i) = 0.0;
          }
        }
        rand_pool.free_state(gen);
      }, "Genten::GCP_SGD::sample_tensor_zeros_bulk");

      // Sort Z
      Z.setIsSorted(false);
      Z.sort();

      // Copy Z into Y skipping entries that are in X and duplicates

#if 0
      // Serial version
      ttb_indx n = 0;
      ttb_indx i = 0;
      ttb_indx ind = 0;
      while (n<total_samples && i<ns) {
        const auto sub = Z.getSubscripts(n);
        ind = X.sorted_lower_bound(sub,ind);
        const bool found = X.isSubscriptEqual(ind,sub);
        if (!found) {
          bool in_Y = i > 0 ? true : false;
          if (i>0) {
            for (ttb_indx j=0; j<nd; ++j) {
              if (Y.subscript(offset+i-1,j) != Z.subscript(n,j)) {
                in_Y = false;
                break;
              }
            }
          }
          if (!in_Y) {
            for (ttb_indx j=0; j<nd; ++j)
              Y.subscript(offset+i,j) = Z.subscript(n,j);
            Y.value(offset+i) = 0.0;
            ++i;
          }
        }
        ++n;
      }
      if (n == total_samples && i < ns)
        Kokkos::abort("Ran out of zeros before completion!");
#elif 0
      // Parallel version using a hash map
      // Determine unique entries in Z
      Kokkos::UnorderedMap<ttb_indx,void,ExecSpace> unique_map(total_samples);
      Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,total_samples),
                           KOKKOS_LAMBDA(const ttb_indx n)
      {
        bool duplicate = false;
        if (n > 0) {
          duplicate = true;
          for (ttb_indx j=0; j<nd; ++j) {
            if (Z.subscript(n-1,j) != Z.subscript(n,j)) {
              duplicate = false;
              break;
            }
          }
        }
        if (!duplicate) {
          if (unique_map.insert(n).failed())
            Kokkos::abort("Unordered map insert failed!");
        }
      }, "Genten::GCP_SGD::sample_tensor_zeros_duplicates");

      // Determine which entries in Z are not in X
      Kokkos::UnorderedMap<ttb_indx,void,ExecSpace> map(total_samples);
      Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,N_zeros_gen),
                           KOKKOS_LAMBDA(const ttb_indx k)
      {
        ttb_indx ind = 0;
        for (ttb_indx l=0; l<nloops; ++l) {
          const ttb_indx n = k*nloops+l;
          if (n < total_samples) {
            const auto sub = Z.getSubscripts(n);
            ind = X.sorted_lower_bound(sub,ind);
            const bool found = X.isSubscriptEqual(ind,sub);
            if (!found) {
              if (unique_map.exists(n)) {
                if (map.insert(n).failed())
                  Kokkos::abort("Unordered map insert failed!");
              }
            }
          }
        }
      }, "Genten::GCP_SGD::sample_tensor_zeros_search");

      // Add entries in Z not in X into Y, eliminating duplicates
      const ttb_indx sz = map.capacity();
      Kokkos::View<ttb_indx,ExecSpace> idx("idx");
      // for (ttb_indx m=0; m<sz; ++m)
      Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,sz),
                           KOKKOS_LAMBDA(const ttb_indx m)
      {
        if (map.valid_at(m)) {
          const ttb_indx n = map.key_at(m);
          const ttb_indx i = Kokkos::atomic_fetch_add(&idx(), ttb_indx(1));
          if (i < ns) {
            for (ttb_indx j=0; j<nd; ++j)
              Y.subscript(offset+i,j) = Z.subscript(n,j);
            Y.value(offset+i) = 0.0;
          }
        }
      }, "Genten::GCP_SGD::sample_tensor_zeros_search");
      typename Kokkos::View<ttb_indx,ExecSpace>::HostMirror idx_h =
        create_mirror_view(idx);
      deep_copy(idx_h, idx);
      if (idx_h() < ns)
        Kokkos::abort("Ran out of zeros before completion!");
#else
      // Direct parallel version
      Kokkos::View<ttb_indx,ExecSpace> idx("idx");
      Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,N_zeros_gen),
                           KOKKOS_LAMBDA(const ttb_indx k)
      {
        ttb_indx ind = 0;
        for (ttb_indx l=0; l<nloops; ++l) {
          const ttb_indx n = k*nloops+l;
          if (n < total_samples) {
            const auto sub = Z.getSubscripts(n);
            bool duplicate = false;
            if (n > 0) {
              duplicate = true;
              for (ttb_indx j=0; j<nd; ++j) {
                if (Z.subscript(n-1,j) != sub(j)) {
                  duplicate = false;
                  break;
                }
              }
            }
            if (!duplicate) {
              ind = X.sorted_lower_bound(sub,ind);
              const bool found = X.isSubscriptEqual(ind,sub);
              if (!found) {
                const ttb_indx i =
                  Kokkos::atomic_fetch_add(&idx(), ttb_indx(1));
                if (i < ns) {
                  for (ttb_indx j=0; j<nd; ++j)
                    Y.subscript(offset+i,j) = sub(j);
                  Y.value(offset+i) = 0.0;
                }
              }
            }
          }
        }
      }, "Genten::GCP_SGD::sample_tensor_zeros_search");
      typename Kokkos::View<ttb_indx,ExecSpace>::HostMirror idx_h =
        create_mirror_view(idx);
      deep_copy(idx_h, idx);
      if (idx_h() < ns)
        Kokkos::abort("Ran out of zeros before completion!");
#endif

    }

    template <typename ExecSpace>
    void merge_sampled_tensors(const SptensorT<ExecSpace>& X_nz,
                               const SptensorT<ExecSpace>& X_z,
                               SptensorT<ExecSpace>& X,
                               const AlgParams& algParams)
    {
      const ttb_indx xnz = X_nz.nnz();
      const ttb_indx xz = X_z.nnz();
      const ttb_indx nz = xnz+xz;
      const ttb_indx nd = X_nz.ndims();

      // Resize X if necessary
      if (X.nnz() < nz) {
        X = SptensorT<ExecSpace>(X_nz.size(), nz);
      }

      // Copy X_nz
      Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,xnz),
                           KOKKOS_LAMBDA(const ttb_indx i)
      {
        for (ttb_indx j=0; j<nd; ++j)
          X.subscript(i,j) = X_nz.subscript(i,j);
        X.value(i) = X_nz.value(i);
      }, "Genten::GCP_SGD::merge_sampled_tensors_nz");

      // Copy X_z
      Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,xz),
                           KOKKOS_LAMBDA(const ttb_indx i)
      {
        for (ttb_indx j=0; j<nd; ++j)
          X.subscript(i+xnz,j) = X_z.subscript(i,j);
        X.value(i+xnz) = X_z.value(i);
      }, "Genten::GCP_SGD::merge_sampled_tensors_z");
    }

    // In this function, X is a stratified sampled tensor from some other
    // tensor, and the history term uses the same set of samples from the
    // samples tensor for each slice in the history term
    template <typename ExecSpace, typename LossFunction>
    void stratified_ktensor_grad(
      const SptensorT<ExecSpace>& X,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& u,
      const KtensorT<ExecSpace>& up,
      const ArrayT<ExecSpace>& window,
      const ttb_real window_penalty,
      const LossFunction& loss_func,
      SptensorT<ExecSpace>& Y,
      const AlgParams& algParams)
    {
      typedef Kokkos::TeamPolicy<ExecSpace> Policy;
      typedef typename Policy::member_type TeamMember;
      typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

      static const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;
      static const unsigned RowBlockSize = 1;
      static const unsigned FacBlockSize = 16; // FIXME
      static const unsigned VectorSize = is_cuda ? 16 : 1; //FIXME
      static const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;
      static const unsigned RowsPerTeam = TeamSize * RowBlockSize;

      /*const*/ ttb_indx nnz = X.nnz();
      /*const*/ unsigned nd = X.ndims();
      /*const*/ ttb_indx ns_nz = num_samples_nonzeros;
      /*const*/ ttb_indx total_samples =
        num_samples_nonzeros + num_samples_zeros;
      const ttb_indx N = (total_samples+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);
      /*const*/ ttb_indx nh = window.size();

      if (u[nd-1].nRows() != nh)
        Genten::error("stratified_ktensor_grad():  temporal mode size of ktensor u does not match given history window!");
      if (up[nd-1].nRows() != nh)
        Genten::error("stratified_ktensor_grad():  temporal mode size of ktensor up does not match given history window!");

      // Resize Y if necessary
      if (Y.nnz() < total_samples*nh) {
        IndxArrayT<ExecSpace> sz = X.size();
        auto sz_host = create_mirror_view(sz);
        deep_copy(sz_host, sz);
        sz_host[nd-1] = nh;
        deep_copy(sz, sz_host);
        Y = SptensorT<ExecSpace>(sz, total_samples*nh);
      }

      // Generate terms from given tensor X
      Policy policy(N, TeamSize, VectorSize);
      Kokkos::parallel_for(
        policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          /*const*/ ttb_indx idx = offset + ii;
          if (idx >= total_samples)
            continue;

          // Weight is based on whether sample is for a zero or nonzero
          ttb_real w;
          if (idx < ns_nz)
            w = weight_nonzeros;
          else
            w = weight_zeros;

          for (ttb_indx h=0; h<nh; ++h) {

            // Get and set subscripts
            Kokkos::single( Kokkos::PerThread( team ), [&] ()
            {
              for (ttb_indx m=0; m<nd; ++m) {
                ind[m] = m == nd-1 ? h : X.subscript(idx,m);
                Y.subscript(idx+total_samples*h,m) = ind[m];
              }
            });

            // Compute Ktensor values
            ttb_real m_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(u, ind);
            ttb_real mp_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(up, ind);

            // Compute tensor value
            Kokkos::single( Kokkos::PerThread( team ), [&] ()
            {
              Y.value(idx+total_samples*h) =
                  window_penalty * window[h] * w *
                  loss_func.deriv(mp_val, m_val);
            });
          }
        }
      }, "Genten::GCP_SGD::stratified_ktensor_grad");
    }

    // This function uniformly samples a Ktensor for computing the history
    // window term gradient.  It currently samples each slice in the history
    // term independently.
    template <typename ExecSpace, typename LossFunction>
    void uniform_ktensor_grad(
      const ttb_indx num_samples,
      const ttb_real weight,
      const KtensorT<ExecSpace>& u,
      const KtensorT<ExecSpace>& up,
      const ArrayT<ExecSpace>& window,
      const ttb_real window_penalty,
      const LossFunction& loss_func,
      SptensorT<ExecSpace>& Y,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams)
    {
      typedef Kokkos::TeamPolicy<ExecSpace> Policy;
      typedef typename Policy::member_type TeamMember;
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;

      static const bool is_cuda = Genten::is_cuda_space<ExecSpace>::value;
      static const unsigned RowBlockSize = 1;
      static const unsigned FacBlockSize = 16; // FIXME
      static const unsigned VectorSize = is_cuda ? 16 : 1; //FIXME
      static const unsigned TeamSize = is_cuda ? 128/VectorSize : 1;
      static const unsigned RowsPerTeam = TeamSize * RowBlockSize;

      /*const*/ unsigned nd = u.ndims();
      /*const*/ ttb_indx ns = num_samples;
      const ttb_indx N = (num_samples+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);
      /*const*/ ttb_indx nh = window.size();

      if (u[nd-1].nRows() != nh)
        Genten::error("uniform_ktensor_grad():  temporal mode size of ktensor u does not match given history window!");
      if (up[nd-1].nRows() != nh)
        Genten::error("uniform_ktensor_grad():  temporal mode size of ktensor up does not match given history window!");

      // Resize Y if necessary
      if (Y.nnz() < num_samples*nh) {
        IndxArrayT<ExecSpace> sz(nd);
        auto sz_host = create_mirror_view(sz);
        for (unsigned i=0; i<nd; ++i)
          sz_host[i] = u[i].nRows();
        deep_copy(sz, sz_host);
        Y = SptensorT<ExecSpace>(sz, num_samples*nh);
      }

      // Generate terms from given tensor X
      Policy policy(N, TeamSize, VectorSize);
      Kokkos::parallel_for(
        policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();
        TmpScratchSpace team_ind(team.team_scratch(0), TeamSize, nd);
        ttb_indx *ind = &(team_ind(team.team_rank(),0));

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          /*const*/ ttb_indx idx = offset + ii;
          if (idx >= ns)
            continue;

          for (ttb_indx h=0; h<nh; ++h) {

            // Sample and set subscripts
            Kokkos::single( Kokkos::PerThread( team ), [&] ()
            {
              for (ttb_indx m=0; m<nd; ++m) {
                ind[m] = m == nd-1 ? h : Rand::draw(gen,0,u[m].nRows());
                Y.subscript(idx+ns*h,m) = ind[m];
              }
            });

            // Compute Ktensor values
            ttb_real m_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(u, ind);
            ttb_real mp_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(up, ind);

            // Compute tensor value
            Kokkos::single( Kokkos::PerThread( team ), [&] ()
            {
              Y.value(idx+num_samples*h) = window_penalty * window[h] * weight *
                loss_func.deriv(mp_val, m_val);
            });

          }
        }
      }, "Genten::GCP_SGD::uniform_ktensor_grad");
    }

  }

}

#include "Genten_GCP_LossFunctions.hpp"

#define LOSS_INST_MACRO(SPACE,LOSS)                                     \
  template void Impl::uniform_sample_tensor(                            \
    const SptensorT<SPACE>& X,                                          \
    const ttb_indx num_samples,                                         \
    const ttb_real weight,                                              \
    const KtensorT<SPACE>& u,                                           \
    const LOSS& loss_func,                                              \
    const bool compute_gradient,                                        \
    SptensorT<SPACE>& Y,                                                \
    ArrayT<SPACE>& w,                                                   \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams);                                        \
                                                                        \
  template void Impl::uniform_sample_tensor_hash(                       \
    const SptensorT<SPACE>& X,                                          \
    const TensorHashMap<SPACE>& hash,                                   \
    const ttb_indx num_samples,                                         \
    const ttb_real weight,                                              \
    const KtensorT<SPACE>& u,                                           \
    const LOSS& loss_func,                                              \
    const bool compute_gradient,                                        \
    SptensorT<SPACE>& Y,                                                \
    ArrayT<SPACE>& w,                                                   \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams);                                        \
                                                                        \
  template void Impl::stratified_sample_tensor(                         \
    const SptensorT<SPACE>& X,                                          \
    const ttb_indx num_samples_nonzeros,                                \
    const ttb_indx num_samples_zeros,                                   \
    const ttb_real weight_nonzeros,                                     \
    const ttb_real weight_zeros,                                        \
    const KtensorT<SPACE>& u,                                           \
    const LOSS& loss_func,                                              \
    const bool compute_gradient,                                        \
    SptensorT<SPACE>& Y,                                                \
    ArrayT<SPACE>& w,                                                   \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams);                                        \
                                                                        \
  template void Impl::stratified_sample_tensor_hash(                    \
    const SptensorT<SPACE>& X,                                          \
    const TensorHashMap<SPACE>& hash, \
    const ttb_indx num_samples_nonzeros,                                \
    const ttb_indx num_samples_zeros,                                   \
    const ttb_real weight_nonzeros,                                     \
    const ttb_real weight_zeros,                                        \
    const KtensorT<SPACE>& u,                                           \
    const LOSS& loss_func,                                              \
    const bool compute_gradient,                                        \
    SptensorT<SPACE>& Y,                                                \
    ArrayT<SPACE>& w,                                                   \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams);                                        \
                                                                        \
  template void Impl::semi_stratified_sample_tensor(                    \
    const SptensorT<SPACE>& X,                                          \
    const ttb_indx num_samples_nonzeros,                                \
    const ttb_indx num_samples_zeros,                                   \
    const ttb_real weight_nonzeros,                                     \
    const ttb_real weight_zeros,                                        \
    const KtensorT<SPACE>& u,                                           \
    const LOSS& loss_func,                                              \
    const bool compute_gradient,                                        \
    SptensorT<SPACE>& Y,                                                \
    ArrayT<SPACE>& w,                                                   \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams);                                        \
                                                                        \
  template void Impl::sample_tensor_nonzeros(                           \
    const SptensorT<SPACE>& X,                                          \
    const ttb_indx offset,                                              \
    const ttb_indx num_samples,                                         \
    const ttb_real weight,                                              \
    const KtensorT<SPACE>& u,                                           \
    const LOSS& loss_func,                                              \
    const bool compute_gradient,                                        \
    SptensorT<SPACE>& Y,                                                \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams);                                        \
                                                                        \
  template void Impl::sample_tensor_nonzeros(                           \
    const SptensorT<SPACE>& X,                                          \
    const ArrayT<SPACE>& w,                                             \
    const ttb_indx num_samples,                                         \
    const KtensorT<SPACE>& u,                                           \
    const LOSS& loss_func,                                              \
    SptensorT<SPACE>& Y,                                                \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams);                                        \
                                                                        \
  template void Impl::stratified_ktensor_grad(                          \
    const SptensorT<SPACE>& X,                                          \
    const ttb_indx num_samples_nonzeros,                                \
    const ttb_indx num_samples_zeros,                                   \
    const ttb_real weight_nonzeros,                                     \
    const ttb_real weight_zeros,                                        \
    const KtensorT<SPACE>& u,                                           \
    const KtensorT<SPACE>& up,                                          \
    const ArrayT<SPACE>& window,                                        \
    const ttb_real window_penalty,                                      \
    const LOSS& loss_func,                                              \
    SptensorT<SPACE>& Y,                                                \
    const AlgParams& algParams);                                        \
                                                                        \
  template void Impl::uniform_ktensor_grad(                             \
    const ttb_indx num_samples,                                         \
    const ttb_real weight,                                              \
    const KtensorT<SPACE>& u,                                           \
    const KtensorT<SPACE>& up,                                          \
    const ArrayT<SPACE>& window,                                        \
    const ttb_real window_penalty,                                      \
    const LOSS& loss_func,                                              \
    SptensorT<SPACE>& Y,                                                \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams);

#define INST_MACRO(SPACE)                                               \
  LOSS_INST_MACRO(SPACE,GaussianLossFunction)                           \
  LOSS_INST_MACRO(SPACE,RayleighLossFunction)                           \
  LOSS_INST_MACRO(SPACE,GammaLossFunction)                              \
  LOSS_INST_MACRO(SPACE,BernoulliLossFunction)                          \
  LOSS_INST_MACRO(SPACE,PoissonLossFunction)                            \
                                                                        \
  template void Impl::sample_tensor_nonzeros(                           \
    const SptensorT<SPACE>& X,                                          \
    const ArrayT<SPACE>& w,                                             \
    const ttb_indx num_samples,                                         \
    SptensorT<SPACE>& Y,                                                \
    ArrayT<SPACE>& z,                                                   \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams);                                        \
                                                                        \
  template void Impl::sample_tensor_zeros(                              \
    const SptensorT<SPACE>& X,                                          \
    const ttb_indx offset,                                              \
    const ttb_indx num_samples,                                         \
    SptensorT<SPACE>& Y,                                                \
    SptensorT<SPACE>& Z,                                                \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams);                                        \
                                                                        \
  template void Impl::merge_sampled_tensors(                            \
    const SptensorT<SPACE>& X_nz,                                       \
    const SptensorT<SPACE>& X_z,                                        \
    SptensorT<SPACE>& X,                                                \
    const AlgParams& algParams);

GENTEN_INST(INST_MACRO)
