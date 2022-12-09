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

#ifdef HAVE_TPETRA
namespace {
  template <typename ExecSpace, typename subs_type>
  void build_tensor_maps(
    const subs_type& subs_lids, const subs_type& subs_gids,
    const std::vector< Teuchos::RCP< const tpetra_map_type<ExecSpace> > >& factorMaps,
    std::vector< Teuchos::RCP< const tpetra_map_type<ExecSpace> > >& tensorMaps,
    std::vector< Teuchos::RCP< const tpetra_import_type<ExecSpace> > >& importers,
    const AlgParams& algParams)
  {
    GENTEN_TIME_MONITOR_DIFF("build tensor maps",build_tensor_maps);
    const ttb_indx total_samples = subs_lids.extent(0);
    const unsigned nd = subs_lids.extent(1);

    // Build new communication maps for sampled tensor.
    // ToDo:  run this on the device
    GENTEN_START_TIMER("compute LIDs");
    using unordered_map_type = Kokkos::UnorderedMap<tpetra_go_type,tpetra_lo_type,DefaultHostExecutionSpace>;
    std::vector<unordered_map_type> map(nd);
    std::vector<tpetra_lo_type> cnt(nd, 0);
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
    GENTEN_STOP_TIMER("compute LIDs");

    // Construct sampled tpetra maps
    GENTEN_START_TIMER("construct maps");
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
      GENTEN_START_TIMER("tensor map constructor");
      tensorMaps[n] = Teuchos::rcp(new tpetra_map_type<ExecSpace>(
        invalid, gids, indexBase, factorMaps[n]->getComm()));
      GENTEN_STOP_TIMER("tensor map constructor");
      if (algParams.optimize_maps) {
        GENTEN_START_TIMER("optimize tensor maps");
        bool err = false;
        auto p = Tpetra::Details::makeOptimizedColMapAndImport(
          std::cerr, err, *factorMaps[n], *tensorMaps[n]);
        if (err)
          Genten::error("Tpetra::Details::makeOptimizedColMap failed!");
        tensorMaps[n] = p.first;
        importers[n] = p.second;
        for (ttb_indx i=0; i<total_samples; ++i)
          subs_lids_host(i,n) =
            tensorMaps[n]->getLocalElement(subs_gids_host(i,n));
        GENTEN_STOP_TIMER("optimize tensor maps");
      }
      else {
        GENTEN_START_TIMER("import constructor");
        importers[n] = Teuchos::rcp(new tpetra_import_type<ExecSpace>(
          factorMaps[n], tensorMaps[n]));
        GENTEN_STOP_TIMER("import constructor");
      }
      deep_copy(subs_lids, subs_lids_host);
    }
    GENTEN_STOP_TIMER("construct maps");
  }

  template <typename ExecSpace>
  KtensorT<ExecSpace> import_ktensor_to_tensor_map(
    const KtensorT<ExecSpace>& u,
    const std::vector< Teuchos::RCP< const tpetra_map_type<ExecSpace> > >& factorMaps,
    const std::vector< Teuchos::RCP< const tpetra_map_type<ExecSpace> > >& tensorMaps,
    const std::vector< Teuchos::RCP< const tpetra_import_type<ExecSpace> > >& importers)
  {
    GENTEN_START_TIMER("create overlapped k-tensor");
    const unsigned nd = u.ndims();
    const unsigned nc = u.ncomponents();
    KtensorT<ExecSpace> u_overlap(nc, nd);
    for (unsigned n=0; n<nd; ++n) {
      FacMatrixT<ExecSpace> mat(tensorMaps[n]->getLocalNumElements(), nc);
      u_overlap.set_factor(n, mat);
    }
    u_overlap.setProcessorMap(u.getProcessorMap());
    GENTEN_STOP_TIMER("create overlapped k-tensor");

    GENTEN_START_TIMER("ktensor import");
    for (unsigned n=0; n<nd; ++n) {
      DistFacMatrix<ExecSpace> src(u[n], factorMaps[n]);
      DistFacMatrix<ExecSpace> dst(u_overlap[n], tensorMaps[n]);
      dst.doImport(src, *(importers[n]), Tpetra::INSERT);
    }
    GENTEN_STOP_TIMER("ktensor import");

    return u_overlap;
  }
}
#endif

  namespace Impl {

    template <typename ExecSpace, typename Searcher, typename LossFunction>
    void uniform_sample_tensor(
      const SptensorT<ExecSpace>& Xd,
      const Searcher& searcher,
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
            xv = searcher.value(ind);
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

    template <typename ExecSpace, typename Searcher, typename LossFunction>
    void uniform_sample_tensor_tpetra(
      const SptensorT<ExecSpace>& Xd,
      const Searcher& searcher,
      const ttb_indx num_samples,
      const ttb_real weight,
      const KtensorT<ExecSpace>& u,
      const LossFunction& loss_func,
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
      /*const*/ ttb_indx ns = num_samples;
      const ttb_indx N = (ns+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples;
      if (Yd.nnz() < total_samples) {
        Yd = SptensorT<ExecSpace>(X.size(), total_samples);// Correct size is set later
        Yd.allocGlobalSubscripts();
        deep_copy(Yd.getLowerBounds(), Xd.getLowerBounds());
        deep_copy(Yd.getUpperBounds(), Xd.getUpperBounds());
        Yd.setProcessorMap(Xd.getProcessorMap());
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
          Kokkos::single( Kokkos::PerThread( team ), [&] ()
          {
            for (ttb_indx m=0; m<nd; ++m)
              ind[m] = Rand::draw(gen,X.lowerBound(m),X.upperBound(m));
            Y.value(idx) = searcher.value(ind);
            for (ttb_indx m=0; m<nd; ++m)
              Y.globalSubscript(idx,m) = ind[m];
            if (!compute_gradient)
              w[idx] = weight;
          });
        }
        rand_pool.free_state(gen);
      });

      // Build tensor maps
      Yd.getFactorMaps() = Xd.getFactorMaps();
      build_tensor_maps(Yd.getSubscripts(), Yd.getGlobalSubscripts(),
                        Xd.getFactorMaps(), Yd.getTensorMaps(),
                        Yd.getImporters(), algParams);

      // Set correct size in tensor
      auto sz_host = Yd.size_host();
      for (unsigned n=0; n<nd; ++n)
        sz_host[n] = Yd.tensorMap(n)->getLocalNumElements();
      deep_copy(Y.size(), sz_host);

      // Import u to overlapped tensor map
      u_overlap = import_ktensor_to_tensor_map(
        u, Yd.getFactorMaps(), Yd.getTensorMaps(), Yd.getImporters());

      // Set gradient values in sampled tensor
      if (compute_gradient) {

        Kokkos::parallel_for(
          "Genten::GCP_SGD::Uniform_Gradient",
          policy,
          KOKKOS_LAMBDA(const TeamMember& team)
        {
          const ttb_indx offset =
            (team.league_rank()*TeamSize+team.team_rank())*RowBlockSize;
          for (unsigned ii=0; ii<RowBlockSize; ++ii) {
            const ttb_indx idx = offset + ii;
            if (idx >= ns)
              continue;

            // Compute Ktensor value
            ttb_real m_val =
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(
                team, u_overlap, Y.getSubscripts(idx));

            // Set value in tensor
            Kokkos::single( Kokkos::PerThread( team ), [&] ()
            {
              const ttb_real x_val = Y.value(idx);
              Y.value(idx) = weight * loss_func.deriv(x_val, m_val);
            });
          }
        });
      }
#else
      Genten::error("Uniform sampling with dist-update-method == tpetra requires tpetra!");
#endif
    }

    template <typename ExecSpace, typename Searcher, typename Gradient>
    void stratified_sample_tensor(
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
                gradient.evalNonZero(x_val, m_val, weight_nonzeros);
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
              f = searcher.search(ind);
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
                gradient.evalZero(m_val, weight_zeros);
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
      /*const*/ unsigned nd = X.ndims();
      /*const*/ ttb_indx ns_nz = num_samples_nonzeros;
      /*const*/ ttb_indx ns_z = num_samples_zeros;
      /*const*/ ttb_indx ns_t = ns_nz + ns_z;
      const ttb_indx N_nz = (ns_nz+RowsPerTeam-1)/RowsPerTeam;
      const ttb_indx N_z = (ns_z+RowsPerTeam-1)/RowsPerTeam;
      const ttb_indx N_t = (ns_t+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes = TmpScratchSpace::shmem_size(TeamSize,nd);

      // Resize Y if necessary
      const ttb_indx total_samples = num_samples_nonzeros + num_samples_zeros;
      if (Yd.nnz() < total_samples) {
        Yd = SptensorT<ExecSpace>(Xd.size(), total_samples); // Correct size is set later
        Yd.allocGlobalSubscripts();
        deep_copy(Yd.getLowerBounds(), Xd.getLowerBounds());
        deep_copy(Yd.getUpperBounds(), Xd.getUpperBounds());
        Yd.setProcessorMap(Xd.getProcessorMap());
        w = ArrayT<ExecSpace>(total_samples);
      }
      auto Y = Yd.impl();

      // Generate samples of nonzeros
      GENTEN_START_TIMER("sample nonzeros");
      Policy policy_nz(N_nz, TeamSize, VectorSize);
      Kokkos::parallel_for(
        "Genten::GCP_SGD::Sample_Nonzeros",
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
      GENTEN_STOP_TIMER("sample nonzeros");

      // Generate samples of zeros
      GENTEN_START_TIMER("sample zeros");
      Policy policy_z(N_z, TeamSize, VectorSize);
      Kokkos::parallel_for(
        "Genten::GCP_SGD::Sample_Zeros",
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
      GENTEN_STOP_TIMER("sample zeros");

      // Build tensor maps
      Yd.getFactorMaps() = Xd.getFactorMaps();
      build_tensor_maps(Yd.getSubscripts(), Yd.getGlobalSubscripts(),
                        Xd.getFactorMaps(), Yd.getTensorMaps(),
                        Yd.getImporters(), algParams);

      // Set correct size in tensor
      auto sz_host = Y.size_host();
      for (unsigned n=0; n<nd; ++n)
        sz_host[n] = Yd.tensorMap(n)->getLocalNumElements();
      deep_copy(Y.size(), sz_host);

      // Import u to overlapped tensor map
      u_overlap = import_ktensor_to_tensor_map(
        u, Yd.getFactorMaps(), Yd.getTensorMaps(), Yd.getImporters());

      // Set gradient values in sampled tensor
      if (compute_gradient) {
        GENTEN_TIME_MONITOR_DIFF("compute gradient tensor",compute_grad);
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
              compute_Ktensor_value<ExecSpace,FacBlockSize,VectorSize>(
                team, u_overlap, Y.getSubscripts(idx));

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

  }

}

#include "Genten_GCP_LossFunctions.hpp"

#define LOSS_INST_MACRO(SPACE,LOSS)                                     \
  template void Impl::uniform_sample_tensor(                            \
    const SptensorT<SPACE>& X,                                          \
    const Impl::SortSearcher<SPACE>& searcher,                          \
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
  template void Impl::uniform_sample_tensor(                            \
    const SptensorT<SPACE>& X,                                          \
    const Impl::HashSearcher<SPACE>& searcher,                          \
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
  template void Impl::uniform_sample_tensor_tpetra(                     \
    const SptensorT<SPACE>& X,                                          \
    const Impl::SortSearcher<SPACE>& searcher,                          \
    const ttb_indx num_samples,                                         \
    const ttb_real weight,                                              \
    const KtensorT<SPACE>& u,                                           \
    const LOSS& loss_func,                                              \
    const bool compute_gradient,                                        \
    SptensorT<SPACE>& Y,                                                \
    ArrayT<SPACE>& w,                                                   \
    KtensorT<SPACE>& u_overlap,                                         \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams);                                        \
                                                                        \
  template void Impl::uniform_sample_tensor_tpetra(                     \
    const SptensorT<SPACE>& X,                                          \
    const Impl::HashSearcher<SPACE>& searcher,                          \
    const ttb_indx num_samples,                                         \
    const ttb_real weight,                                              \
    const KtensorT<SPACE>& u,                                           \
    const LOSS& loss_func,                                              \
    const bool compute_gradient,                                        \
    SptensorT<SPACE>& Y,                                                \
    ArrayT<SPACE>& w,                                                   \
    KtensorT<SPACE>& u_overlap,                                         \
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams);                                        \
                                                                        \
  template void Impl::stratified_sample_tensor(                         \
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
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams);                                        \
                                                                        \
  template void Impl::stratified_sample_tensor(                         \
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
    Kokkos::Random_XorShift64_Pool<SPACE>& rand_pool,                   \
    const AlgParams& algParams);                                        \
                                                                        \
  template void Impl::stratified_sample_tensor(                         \
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
    const AlgParams& algParams);

#define INST_MACRO(SPACE)                                               \
  LOSS_INST_MACRO(SPACE,GaussianLossFunction)                           \
  LOSS_INST_MACRO(SPACE,RayleighLossFunction)                           \
  LOSS_INST_MACRO(SPACE,GammaLossFunction)                              \
  LOSS_INST_MACRO(SPACE,BernoulliLossFunction)                          \
  LOSS_INST_MACRO(SPACE,PoissonLossFunction)

GENTEN_INST(INST_MACRO)
