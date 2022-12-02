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
#include "Genten_Tpetra.hpp"
#include "Genten_DistFacMatrix.hpp"

#include "Kokkos_Random.hpp"
#include "Kokkos_UnorderedMap.hpp"

namespace Genten {

  namespace Impl {

    template <typename ExecSpace, typename LossFunction>
    void uniform_sample_tensor(
      const SptensorT<ExecSpace>& Xd,
      const ttb_indx num_samples,
      const ttb_real weight,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Yd,
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

      const auto X = Xd.impl();

      /*const*/ ttb_indx nnz = X.nnz();
      /*const*/ unsigned nd = u.ndims();
      /*const*/ ttb_indx ns = num_samples;
      const ttb_indx N = (ns+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples;
      if (Yd.nnz() < total_samples) {
        Yd = SptensorT<ExecSpace>(X.size(), total_samples);
        w = ArrayT<ExecSpace>(total_samples);
      }
      const auto Y = Yd.impl();

      // Generate samples of tensor
      Policy policy(N, TeamSize, VectorSize);
      Kokkos::parallel_for(
        "Genten::GCP_SGD::Uniform_Sample",
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
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, u, ind);

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
      });
    }

    template <typename ExecSpace, typename LossFunction>
    void uniform_sample_tensor_hash(
      const SptensorT<ExecSpace>& Xd,
      const TensorHashMap<ExecSpace>& hash,
      const ttb_indx num_samples,
      const ttb_real weight,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Yd,
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

      const auto X = Xd.impl();

      /*const*/ ttb_indx nnz = X.nnz();
      /*const*/ unsigned nd = u.ndims();
      /*const*/ ttb_indx ns = num_samples;
      const ttb_indx N = (ns+RowsPerTeam-1)/RowsPerTeam;

      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples;
      if (Yd.nnz() < total_samples) {
        Yd = SptensorT<ExecSpace>(X.size(), total_samples);
        w = ArrayT<ExecSpace>(total_samples);
      }
      const auto Y = Yd.impl();

      // Generate samples of tensor
      Policy policy(N, TeamSize, VectorSize);
      Kokkos::parallel_for(
        "Genten::GCP_SGD::Uniform_Sample",
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
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, u, ind);

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
      });
    }

    template <typename ExecSpace, typename LossFunction>
    void stratified_sample_tensor(
      const SptensorT<ExecSpace>& Xd,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Yd,
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

      const auto X = Xd.impl();

      /*const*/ ttb_indx nnz = X.nnz();
      /*const*/ unsigned nd = u.ndims();
      /*const*/ ttb_indx ns_nz = num_samples_nonzeros;
      /*const*/ ttb_indx ns_z = num_samples_zeros;
      const ttb_indx N_nz = (ns_nz+RowsPerTeam-1)/RowsPerTeam;
      const ttb_indx N_z = (ns_z+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples_nonzeros + num_samples_zeros;
      if (Yd.nnz() < total_samples) {
        Yd = SptensorT<ExecSpace>(X.size(), total_samples);
        w = ArrayT<ExecSpace>(total_samples);
      }
      const auto Y = Yd.impl();

      // Generate samples of nonzeros
      Policy policy_nz(N_nz, TeamSize, VectorSize);
      Kokkos::parallel_for(
        "Genten::GCP_SGD::Stratified_Sample_Nonzeros",
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
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, u, ind);

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
      });


      // Generate samples of zeros
      Policy policy_z(N_z, TeamSize, VectorSize);
      Kokkos::parallel_for(
        "Genten::GCP_SGD::Stratified_Sample_Zeros",
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
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, u, ind);

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
      });
    }

    template <typename ExecSpace, typename LossFunction>
    void stratified_sample_tensor_hash(
      const SptensorT<ExecSpace>& Xd,
      const TensorHashMap<ExecSpace>& hash,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Yd,
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

      const auto X = Xd.impl();

      /*const*/ ttb_indx nnz = X.nnz();
      /*const*/ unsigned nd = u.ndims();
      /*const*/ ttb_indx ns_nz = num_samples_nonzeros;
      /*const*/ ttb_indx ns_z = num_samples_zeros;
      const ttb_indx N_nz = (ns_nz+RowsPerTeam-1)/RowsPerTeam;
      const ttb_indx N_z = (ns_z+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples_nonzeros + num_samples_zeros;
      if (Yd.nnz() < total_samples) {
        Yd = SptensorT<ExecSpace>(X.size(), total_samples);
        w = ArrayT<ExecSpace>(total_samples);
      }
      const auto Y = Yd.impl();

      // Generate samples of nonzeros
      Policy policy_nz(N_nz, TeamSize, VectorSize);
      Kokkos::parallel_for(
        "Genten::GCP_SGD::Stratified_Sample_Nonzeros",
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
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, u, ind);

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
      });

      // Generate samples of zeros
      Policy policy_z(N_z, TeamSize, VectorSize);
      Kokkos::parallel_for(
        "Genten::GCP_SGD::Stratified_Sample_Zeros",
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
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, u, ind);

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
      });
    }

    template <typename ExecSpace, typename Searcher, typename Gradient>
    void stratified_sample_tensor_tpetra(
      const SptensorT<ExecSpace>& Xd,
      const Searcher& searcher,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& u,
      const Gradient& gradient,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Yd,
      ArrayT<ExecSpace>& w,
      KtensorT<ExecSpace>& u_overlap,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams)
    {
#ifdef HAVE_TPETRA
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

      const auto X = Xd.impl();

      /*const*/ ttb_indx nnz = X.nnz();
      /*const*/ unsigned nd = u.ndims();
      /*const*/ ttb_indx ns_nz = num_samples_nonzeros;
      /*const*/ ttb_indx ns_z = num_samples_zeros;
      /*const*/ ttb_indx ns_t = num_samples_nonzeros+num_samples_zeros;
      const ttb_indx N_nz = (ns_nz+RowsPerTeam-1)/RowsPerTeam;
      const ttb_indx N_z = (ns_z+RowsPerTeam-1)/RowsPerTeam;
      const ttb_indx N_t = (ns_t+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples_nonzeros + num_samples_zeros;
      if (Yd.nnz() < total_samples) {
        Yd = SptensorT<ExecSpace>(X.size(), total_samples); // Correct size is set later
        Yd.allocGlobalSubscripts();
        Yd.setProcessorMap(Xd.getProcessorMap());
        w = ArrayT<ExecSpace>(total_samples);
      }
      const auto Y = Yd.impl();

      // Generate samples of nonzeros
      Policy policy_nz(N_nz, TeamSize, VectorSize);
      Kokkos::parallel_for(
        "Genten::GCP_SGD::Stratified_Sample_Nonzeros",
        policy_nz,
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();

        const ttb_indx offset =
          (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
        for (unsigned ii=0; ii<RowBlockSize; ++ii) {
          const ttb_indx idx = offset + ii;
          if (idx >= ns_nz)
            continue;

          // Generate random tensor index
          Kokkos::single( Kokkos::PerThread( team ), [&] ()
          {
            const ttb_indx i = Rand::draw(gen,0,nnz);
            for (ttb_indx m=0; m<nd; ++m)
              Y.globalSubscript(idx,m) = X.globalSubscript(i,m);
            Y.value(idx) = X.value(i); // We need x_val for both value and grad
            if (!compute_gradient)
              w[idx] = weight_nonzeros;
          });
        }
        rand_pool.free_state(gen);
      });

      // Generate samples of zeros
      Policy policy_z(N_z, TeamSize, VectorSize);
      Kokkos::parallel_for(
        "Genten::GCP_SGD::Stratified_Sample_Zeros",
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
                ind[m] = Rand::draw(gen,X.lowerBound(m),X.upperBound(m));

              // Search for index
              f = searcher.search(ind);
            }, found);
          }

          // Add new nonzero
          const ttb_indx row = num_samples_nonzeros + idx;
          Kokkos::single( Kokkos::PerThread( team ), [&] ()
          {
            for (ttb_indx m=0; m<nd; ++m)
              Y.globalSubscript(row,m) = ind[m];
            if (!compute_gradient) {
              Y.value(row) = 0.0; // We don't need the value for grad
              w[row] = weight_zeros;
            }
          });
        }
        rand_pool.free_state(gen);
      });

      // Build new communication maps for sampled tensor.
      // ToDo:  run this on the device
      using unordered_map_type = Kokkos::UnorderedMap<tpetra_go_type,tpetra_lo_type,DefaultHostExecutionSpace>;
      std::vector<unordered_map_type> map(nd);
      std::vector<tpetra_lo_type> cnt(nd, 0);
      auto subs_lids = Y.getSubscripts();
      auto subs_gids = Y.getGlobalSubscripts();
      auto subs_lids_host = create_mirror_view(subs_lids);
      auto subs_gids_host = create_mirror_view(subs_gids);
      deep_copy(subs_gids_host, subs_gids);
      for (unsigned n=0; n<nd; ++n)
        map[n].rehash(total_samples); // min(total_samples, X.upperBound(n)-X.lowerBound(n)) might be a more accurate bound, but that requires a device-host transfer
      for (ttb_indx i=0; i<total_samples; ++i) {
        for (unsigned n=0; n<nd; ++n) {
          const tpetra_go_type gid = subs_gids_host(i,n);
          auto idx = map[n].find(gid);
          tpetra_lo_type lid;
          if (map[n].valid_at(idx))
            lid = map[n].value_at(idx);
          else {
            lid = cnt[n]++;
            if (map[n].insert(gid,lid).failed())
              Genten::error("Insertion of GID failed, something is wrong!");
          }
          subs_lids_host(i,n) = lid;
        }
      }
      for (unsigned n=0; n<nd; ++n)
        assert(cnt[n] == map[n].size());
      deep_copy(subs_lids, subs_lids_host);

      // Construct sampled tpetra maps
      const tpetra_go_type indexBase = tpetra_go_type(0);
      const Tpetra::global_size_t invalid =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
      for (unsigned n=0; n<nd; ++n) {
        Kokkos::View<tpetra_go_type*,ExecSpace> gids("gids", cnt[n]);
        const unordered_map_type map_n = map[n];
        const ttb_indx sz = map_n.capacity();
        // Kokkos::parallel_for("Genten::GCP_SGD::Stratified_Build_Maps",
        //                      Kokkos::RangePolicy<ExecSpace>(0,sz),
        //                      KOKKOS_LAMBDA(const ttb_indx idx)
        auto gids_host = create_mirror_view(gids);
        for (ttb_indx idx=0; idx<sz; ++idx)
        {
          if (map_n.valid_at(idx)) {
            const tpetra_go_type gid = map_n.key_at(idx);
            const tpetra_lo_type lid = map_n.value_at(idx);
            gids_host[lid] = gid;
          }
        }//);
        deep_copy(gids, gids_host);
        Yd.factorMap(n) = Xd.factorMap(n);
        Yd.tensorMap(n) = Teuchos::rcp(new tpetra_map_type<ExecSpace>(
          invalid, gids, indexBase, Xd.tensorMap(n)->getComm()));
        Yd.importer(n) = Teuchos::rcp(new tpetra_import_type<ExecSpace>(
          Yd.factorMap(n), Yd.tensorMap(n)));
      }

      // Set correct size in tensor
      auto sz_host = Y.size_host();
      for (unsigned n=0; n<nd; ++n)
        sz_host[n] = Yd.tensorMap(n)->getLocalNumElements();
      deep_copy(Y.size(), sz_host);

      // Import u to overlapped tensor map
      const unsigned nc = u.ncomponents();
      u_overlap = KtensorT<ExecSpace>(nc, nd);
      for (unsigned n=0; n<nd; ++n) {
        FacMatrixT<ExecSpace> mat(Yd.tensorMap(n)->getLocalNumElements(), nc);
        u_overlap.set_factor(n, mat);
      }
      for (unsigned n=0; n<nd; ++n) {
        DistFacMatrix<ExecSpace> src(u[n], Yd.factorMap(n));
        DistFacMatrix<ExecSpace> dst(u_overlap[n], Yd.tensorMap(n));
        dst.doImport(src, *(Yd.importer(n)), Tpetra::INSERT);
      }
      u_overlap.setProcessorMap(u.getProcessorMap());

      // Set gradient values in sampled tensor
      if (compute_gradient) {
        Policy policy_t(N_t, TeamSize, VectorSize);
        Kokkos::parallel_for(
          "Genten::GCP_SGD::Stratified_Gradient",
          policy_t,
          KOKKOS_LAMBDA(const TeamMember& team)
        {
          const ttb_indx offset =
            (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
          for (unsigned ii=0; ii<RowBlockSize; ++ii) {
            const ttb_indx idx = offset + ii;
            if (idx >= ns_t)
              continue;

            // Compute Ktensor value
            const ttb_real m_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, u_overlap, Y.getSubscripts(idx));

            // Set value in tensor
            Kokkos::single( Kokkos::PerThread( team ), [&] ()
            {
              if (idx < ns_nz) { // This is a nonzero
                const ttb_real x_val = Y.value(idx);
                Y.value(idx) = gradient.evalNonZero(x_val, m_val, weight_nonzeros);
              }
              else { // This is a zero
                Y.value(idx) = gradient.evalZero(m_val, weight_zeros);
              }
            });
          }
        });
      }
#else
      Genten::error("Stratified sampling with dist-update-method == tpetra requires tpetra!");
#endif
    }

    template <typename ExecSpace, typename LossFunction>
    void semi_stratified_sample_tensor(
      const SptensorT<ExecSpace>& Xd,
      const ttb_indx num_samples_nonzeros,
      const ttb_indx num_samples_zeros,
      const ttb_real weight_nonzeros,
      const ttb_real weight_zeros,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Yd,
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

      const auto X = Xd.impl();

      /*const*/ ttb_indx nnz = X.nnz();
      /*const*/ unsigned nd = u.ndims();
      /*const*/ ttb_indx ns_nz = num_samples_nonzeros;
      /*const*/ ttb_indx ns_z = num_samples_zeros;
      const ttb_indx N_nz = (ns_nz+RowsPerTeam-1)/RowsPerTeam;
      const ttb_indx N_z = (ns_z+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples_nonzeros + num_samples_zeros;
      if (Yd.nnz() < total_samples) {
        Yd = SptensorT<ExecSpace>(X.size(), total_samples);
        w = ArrayT<ExecSpace>(total_samples);
      }
      const auto Y = Yd.impl();

      // Generate samples of nonzeros
      Policy policy_nz(N_nz, TeamSize, VectorSize);
      Kokkos::parallel_for(
        "Genten::GCP_SGD::SemiStratified_Sample_Nonzeros",
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
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, u, ind);

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
      });

      // Generate samples of zeros
      Policy policy_z(N_z, TeamSize, VectorSize);
      Kokkos::parallel_for(
        "Genten::GCP_SGD::SemiStratified_Sample_Zeros",
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
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(team, u, ind);

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
      });
    }

    template <typename ExecSpace, typename LossFunction>
    void sample_tensor_nonzeros(
      const SptensorT<ExecSpace>& Xd,
      const ttb_indx offset,
      const ttb_indx num_samples,
      const ttb_real weight,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      const bool compute_gradient,
      SptensorT<ExecSpace>& Yd,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams)
    {
      const auto X = Xd.impl();

      const ttb_indx nnz = X.nnz();
      const ttb_indx nd = X.ndims();

      // Resize Y if necessary
      if (Yd.nnz() < offset+num_samples) {
        Yd = SptensorT<ExecSpace>(X.size(), offset+num_samples);
      }
      const auto Y = Yd.impl();

      // Parallel sampling on the device
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      const ttb_indx nloops = algParams.rng_iters;
      const ttb_indx N_nonzeros = (num_samples+nloops-1)/nloops;
      Kokkos::parallel_for("Genten::GCP_SGD::sample_tensor_nonzeros",
                           Kokkos::RangePolicy<ExecSpace>(0,N_nonzeros),
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
      });

    }

    template <typename ExecSpace, typename LossFunction>
    void sample_tensor_nonzeros(
      const SptensorT<ExecSpace>& Xd,
      const ArrayT<ExecSpace>& w,
      const ttb_indx num_samples,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
      SptensorT<ExecSpace>& Yd,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams)
    {
      const auto X = Xd.impl();

      const ttb_indx nnz = X.nnz();
      const ttb_indx nd = X.ndims();

      // Resize Y if necessary
      if (Yd.nnz() < num_samples) {
        Yd = SptensorT<ExecSpace>(X.size(), num_samples);
      }
      const auto Y = Yd.impl();

      // Parallel sampling on the device
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      const ttb_indx nloops = algParams.rng_iters;
      const ttb_indx N_nonzeros = (num_samples+nloops-1)/nloops;
      Kokkos::parallel_for("Genten::GCP_SGD::sample_tensor_nonzeros",
                           Kokkos::RangePolicy<ExecSpace>(0,N_nonzeros),
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
      });

    }

    template <typename ExecSpace>
    void sample_tensor_nonzeros(
      const SptensorT<ExecSpace>& Xd,
      const ArrayT<ExecSpace>& w,
      const ttb_indx num_samples,
      SptensorT<ExecSpace>& Yd,
      ArrayT<ExecSpace>& z,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams)
    {
      const auto X = Xd.impl();

      const ttb_indx nnz = X.nnz();
      const ttb_indx nd = X.ndims();

      // Resize Y if necessary
      if (Yd.nnz() < num_samples) {
        Yd = SptensorT<ExecSpace>(X.size(), num_samples);
        z = ArrayT<ExecSpace>(num_samples);
      }

      const auto Y = Yd.impl();

      // Parallel sampling on the device
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      const ttb_indx nloops = algParams.rng_iters;
      const ttb_indx N_nonzeros = (num_samples+nloops-1)/nloops;
      Kokkos::parallel_for("Genten::GCP_SGD::sample_tensor_nonzeros",
                           Kokkos::RangePolicy<ExecSpace>(0,N_nonzeros),
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
      });

    }

    template <typename ExecSpace>
    void sample_tensor_zeros(
      const SptensorT<ExecSpace>& Xd,
      const ttb_indx offset,
      const ttb_indx num_samples,
      SptensorT<ExecSpace>& Yd,
      SptensorT<ExecSpace>& Zd,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const AlgParams& algParams)
    {
      const auto X = Xd.impl();

      const ttb_indx nnz = X.nnz();
      const ttb_indx nd = X.ndims();
      const ttb_real ne = X.numel_float();
      const ttb_indx ns = std::min(num_samples, ttb_indx(ne-nnz));

      // Resize Y if necessary
      if (Yd.nnz() < offset+ns) {
        Yd = SptensorT<ExecSpace>(X.size(), offset+ns);
      }

      const auto Y = Yd.impl();

      // Parallel sampling on the device
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      const ttb_indx nloops = algParams.rng_iters;

      // First generate oversample_factor*ns samples of potential zeros
      const ttb_real oversample = algParams.oversample_factor;
      const ttb_indx total_samples =
        static_cast<ttb_indx>(oversample*num_samples);
      if (Zd.nnz() < total_samples) {
        Zd = SptensorT<ExecSpace>(X.size(), total_samples);
      }
      auto& Z = Zd.impl();
      const ttb_indx N_zeros_gen = (total_samples+nloops-1)/nloops;
      Kokkos::parallel_for("Genten::GCP_SGD::sample_tensor_zeros_bulk",
                           Kokkos::RangePolicy<ExecSpace>(0,N_zeros_gen),
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
      });

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
      Kokkos::parallel_for("Genten::GCP_SGD::sample_tensor_zeros_duplicates",
                           Kokkos::RangePolicy<ExecSpace>(0,total_samples),
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
      });

      // Determine which entries in Z are not in X
      Kokkos::UnorderedMap<ttb_indx,void,ExecSpace> map(total_samples);
      Kokkos::parallel_for("Genten::GCP_SGD::sample_tensor_zeros_search",
                           Kokkos::RangePolicy<ExecSpace>(0,N_zeros_gen),
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
      });

      // Add entries in Z not in X into Y, eliminating duplicates
      const ttb_indx sz = map.capacity();
      Kokkos::View<ttb_indx,ExecSpace> idx("idx");
      // for (ttb_indx m=0; m<sz; ++m)
      Kokkos::parallel_for("Genten::GCP_SGD::sample_tensor_zeros_search",
                           Kokkos::RangePolicy<ExecSpace>(0,sz),
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
      });
      typename Kokkos::View<ttb_indx,ExecSpace>::HostMirror idx_h =
        create_mirror_view(idx);
      deep_copy(idx_h, idx);
      if (idx_h() < ns)
        Kokkos::abort("Ran out of zeros before completion!");
#else
      // Direct parallel version
      Kokkos::View<ttb_indx,ExecSpace> idx("idx");
      Kokkos::parallel_for("Genten::GCP_SGD::sample_tensor_zeros_search",
                           Kokkos::RangePolicy<ExecSpace>(0,N_zeros_gen),
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
      });
      typename Kokkos::View<ttb_indx,ExecSpace>::HostMirror idx_h =
        create_mirror_view(idx);
      deep_copy(idx_h, idx);
      if (idx_h() < ns)
        Kokkos::abort("Ran out of zeros before completion!");
#endif

    }

    template <typename ExecSpace>
    void merge_sampled_tensors(const SptensorT<ExecSpace>& Xd_nz,
                               const SptensorT<ExecSpace>& Xd_z,
                               SptensorT<ExecSpace>& Xd,
                               const AlgParams& algParams)
    {
      const auto X_nz = Xd_nz.impl();
      const auto X_z = Xd_z.impl();

      const ttb_indx xnz = X_nz.nnz();
      const ttb_indx xz = X_z.nnz();
      const ttb_indx nz = xnz+xz;
      const ttb_indx nd = X_nz.ndims();

      // Resize X if necessary
      if (Xd.nnz() < nz) {
        Xd = SptensorT<ExecSpace>(X_nz.size(), nz);
      }

      const auto X = Xd.impl();

      // Copy X_nz
      Kokkos::parallel_for("Genten::GCP_SGD::merge_sampled_tensors_nz",
                           Kokkos::RangePolicy<ExecSpace>(0,xnz),
                           KOKKOS_LAMBDA(const ttb_indx i)
      {
        for (ttb_indx j=0; j<nd; ++j)
          X.subscript(i,j) = X_nz.subscript(i,j);
        X.value(i) = X_nz.value(i);
      });

      // Copy X_z
      Kokkos::parallel_for("Genten::GCP_SGD::merge_sampled_tensors_z",
                           Kokkos::RangePolicy<ExecSpace>(0,xz),
                           KOKKOS_LAMBDA(const ttb_indx i)
      {
        for (ttb_indx j=0; j<nd; ++j)
          X.subscript(i+xnz,j) = X_z.subscript(i,j);
        X.value(i+xnz) = X_z.value(i);
      });
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
    const TensorHashMap<SPACE>& hash,                                   \
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
  template void Impl::stratified_sample_tensor_tpetra(                  \
    const SptensorT<SPACE>& X,                                          \
    const Impl::SortSearcher<SPACE>& searcher,                          \
    const ttb_indx num_samples_nonzeros,                                \
    const ttb_indx num_samples_zeros,                                   \
    const ttb_real weight_nonzeros,                                     \
    const ttb_real weight_zeros,                                        \
    const KtensorT<SPACE>& u,                                           \
    const Impl::StratifiedGradient<LOSS>& gradient,                     \
    const bool compute_gradient,                                        \
    SptensorT<SPACE>& Y,                                                \
    ArrayT<SPACE>& w,                                                   \
    KtensorT<SPACE>& u_overlap,                                         \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams);                                        \
                                                                        \
  template void Impl::stratified_sample_tensor_tpetra(                  \
    const SptensorT<SPACE>& X,                                          \
    const Impl::HashSearcher<SPACE>& searcher,                          \
    const ttb_indx num_samples_nonzeros,                                \
    const ttb_indx num_samples_zeros,                                   \
    const ttb_real weight_nonzeros,                                     \
    const ttb_real weight_zeros,                                        \
    const KtensorT<SPACE>& u,                                           \
    const Impl::StratifiedGradient<LOSS>& gradient,                     \
    const bool compute_gradient,                                        \
    SptensorT<SPACE>& Y,                                                \
    ArrayT<SPACE>& w,                                                   \
    KtensorT<SPACE>& u_overlap,                                         \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams);                                        \
                                                                        \
  template void Impl::stratified_sample_tensor_tpetra(                  \
    const SptensorT<SPACE>& X,                                          \
    const Impl::SemiStratifiedSearcher<SPACE>& searcher,                \
    const ttb_indx num_samples_nonzeros,                                \
    const ttb_indx num_samples_zeros,                                   \
    const ttb_real weight_nonzeros,                                     \
    const ttb_real weight_zeros,                                        \
    const KtensorT<SPACE>& u,                                           \
    const Impl::SemiStratifiedGradient<LOSS>& gradient,                 \
    const bool compute_gradient,                                        \
    SptensorT<SPACE>& Y,                                                \
    ArrayT<SPACE>& w,                                                   \
    KtensorT<SPACE>& u_overlap,                                         \
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
