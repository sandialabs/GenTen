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

#pragma once

#include <cmath>
#include "Genten_GCP_SGD_Iter.hpp"

// To do:
//  * Test on Volta.  Do we need warp sync's?
//  * Add history

namespace Genten {

  namespace Impl {

    template <typename ExecSpace, typename LossFunction, typename Stepper>
    void gcp_sgd_iter_async_kernel(
      const SptensorImpl<ExecSpace>& X,
      const KtensorImpl<ExecSpace>& u,
      const LossFunction& f,
      const ttb_indx nsz,
      const ttb_indx nsnz,
      const ttb_real wz,
      const ttb_real wnz,
      Kokkos::Random_XorShift64_Pool<ExecSpace>& rand_pool,
      const Stepper& stepper,
      const ttb_indx mode_beg,
      const ttb_indx mode_end,
      const AlgParams& algParams,
      const ttb_indx total_iters)
    {
      using std::floor;
      using std::pow;
      using std::log2;
      using std::min;

      typedef Kokkos::TeamPolicy<ExecSpace> Policy;
      typedef typename Policy::member_type TeamMember;
      typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
      typedef typename RandomPool::generator_type generator_type;
      typedef Kokkos::rand<generator_type, ttb_indx> Rand;
      typedef Kokkos::View< ttb_indx**, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > IndScratchSpace;
      typedef Kokkos::View< ttb_real***, Kokkos::LayoutRight, typename ExecSpace::scratch_memory_space , Kokkos::MemoryUnmanaged > KtnScratchSpace;

      /*const*/ ttb_indx num_samples = (nsz+nsnz)*algParams.epoch_iters;
      /*const*/ ttb_indx nnz = X.nnz();
      /*const*/ unsigned nd = u.ndims();
      /*const*/ unsigned nc = u.ncomponents();
      /*const*/ unsigned mb = mode_beg;
      /*const*/ unsigned me = mode_end;

      static const bool is_gpu = Genten::is_gpu_space<ExecSpace>::value;
      const unsigned RowBlockSize = algParams.mttkrp_nnz_tile_size;
      const unsigned VectorSize =
        is_gpu ? min(unsigned(128), unsigned(pow(2.0, floor(log2(nc))))) : 1;
      const unsigned TeamSize = is_gpu ? 128/VectorSize : 1;
      const unsigned RowsPerTeam = TeamSize * RowBlockSize;
      const ttb_indx N = (num_samples+RowsPerTeam-1)/RowsPerTeam;
      const size_t bytes =
        IndScratchSpace::shmem_size(TeamSize,nd) +
        KtnScratchSpace::shmem_size(TeamSize,nd,nc);

      Policy policy(N, TeamSize, VectorSize);
      Kokkos::parallel_for(
        "gcp_sgd_iter_asyn_kernel",
        policy.set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const TeamMember& team)
      {
        generator_type gen = rand_pool.get_state();
        const unsigned team_rank = team.team_rank();
        const unsigned team_size = team.team_size();
        IndScratchSpace team_ind(team.team_scratch(0), team_size, nd);
        KtnScratchSpace team_ktn(team.team_scratch(0), team_size, nd, nc);

        for (unsigned ii=0; ii<RowBlockSize; ++ii) {

          // Randomly choose if this is a zero or nonzero sample based
          // on the fraction of requested zero/nonzero samples
          ttb_indx idx = 0;
          Kokkos::single( Kokkos::PerThread( team ), [&] (ttb_indx& i)
          {
            i = Rand::draw(gen,0,nsz+nsnz);
          }, idx);
          const bool nonzero_sample = idx < nsnz;

          ttb_real x_val = 0.0;
          if (nonzero_sample) {
            // Generate random tensor index
            Kokkos::single( Kokkos::PerThread( team ), [&] (ttb_real& xv)
            {
              const ttb_indx i = Rand::draw(gen,0,nnz);
              for (ttb_indx m=0; m<nd; ++m)
                team_ind(team_rank,m) = X.subscript(i,m);
              xv = X.value(i);
            }, x_val);
          }
          else {
            // Generate index -- use broadcast form to force warp sync
            // so that ind is updated before used by other threads
            int sync = 0;
            Kokkos::single( Kokkos::PerThread( team ), [&] (int& s)
            {
              for (ttb_indx m=0; m<nd; ++m)
                team_ind(team_rank,m) = Rand::draw(gen,0,X.size(m));
              s = 1;
            }, sync);
            x_val = 0.0;
          }

          // Read u
          for (unsigned m=0; m<nd; ++m) {
            const ttb_indx k = team_ind(team_rank,m);
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,nc),
                                 [&] (const unsigned& j)
            {
              team_ktn(team_rank, m, j) = u[m].entry(k,j);
            });
          }

          // Compute Ktensor value
          ttb_real m_val = 0.0;
          Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,nc),
                                  [&] (const unsigned& j, ttb_real& mv)
          {
            ttb_real tmp = 1.0;
            for (unsigned m=0; m<nd; ++m)
              tmp *= team_ktn(team_rank, m, j);
            mv += tmp;
          }, m_val);

          // Compute Y value
          ttb_real y_val;
          if (nonzero_sample)
            y_val = wnz * (f.deriv(x_val, m_val) -
                           f.deriv(ttb_real(0.0), m_val));
          else
             y_val = wz * f.deriv(ttb_real(0.0), m_val);

          // Compute gradient contribution
          for (unsigned n=mb; n<me; ++n) {
            const ttb_indx k = team_ind(team_rank,n);
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,nc),
                                 [&] (const unsigned& j)
            {
              ttb_real tmp = y_val;
              for (unsigned m=0; m<nd; ++m)
                if (m != n)
                  tmp *= team_ktn(team_rank, m, j);
              stepper.eval_async(n,k,j,tmp,u);
            });
          }
        }
        stepper.update_async(RowBlockSize, team);
        rand_pool.free_state(gen);
      });

      // Fence to make sure kernel is finished before updates to stepper
      Kokkos::fence();
    }

    template <typename ExecSpace, typename LossFunction>
    class GCP_SGD_Iter_Async :
      public GCP_SGD_Iter<SptensorT<ExecSpace>, LossFunction> {
    public:

      GCP_SGD_Iter_Async(const KtensorT<ExecSpace>& u0,
                         const StreamingHistory<ExecSpace>& hist,
                         const ttb_real penalty,
                         const ttb_indx mode_beg,
                         const ttb_indx mode_end,
                         const AlgParams& algParams) :
        GCP_SGD_Iter<SptensorT<ExecSpace>, LossFunction>(
          u0, hist, penalty, mode_beg, mode_end, algParams)
      {
        if (u0.getProcessorMap() != nullptr &&
            u0.getProcessorMap()->gridSize() > 0)
          Genten::error("Asynchronous GCP iterator does not work with > 1 MPI processor.");
      }

      virtual ~GCP_SGD_Iter_Async() {}

      virtual void run(SptensorT<ExecSpace>& X,
                       const LossFunction& loss_func,
                       Sampler<SptensorT<ExecSpace>,LossFunction>& sampler,
                       GCP_SGD_Step<ExecSpace,LossFunction>& stepper,
                       ttb_indx& total_iters)
      {
        // Cast sampler to SemiStratified and extract data
        SemiStratifiedSampler<ExecSpace,LossFunction>* semi_strat_sampler =
          dynamic_cast<SemiStratifiedSampler<ExecSpace,LossFunction>*>(&sampler);
        if (semi_strat_sampler == nullptr)
          Genten::error("Asynchronous iterator requires semi-stratified sampler!");
        const ttb_indx nsz = semi_strat_sampler->getNumSamplesZerosGrad();
        const ttb_indx nsnz = semi_strat_sampler->getNumSamplesNonzerosGrad();
        const ttb_real wz = semi_strat_sampler->getWeightZerosGrad();
        const ttb_real wnz = semi_strat_sampler->getWeightNonzerosGrad();
        auto& rand_pool = semi_strat_sampler->getRandPool();
        stepper.setNumSamples(nsz+nsnz);

        this->timer.start(this->timer_grad);

        // Run kernel
        // To do:  handle history terms, penalty
        SGDStep<ExecSpace,LossFunction>* sgd_step =
          dynamic_cast<SGDStep<ExecSpace,LossFunction>*>(&stepper);
        AdaGradStep<ExecSpace,LossFunction>* adagrad_step =
          dynamic_cast<AdaGradStep<ExecSpace,LossFunction>*>(&stepper);
        AdamStep<ExecSpace,LossFunction>* adam_step =
          dynamic_cast<AdamStep<ExecSpace,LossFunction>*>(&stepper);
        AMSGradStep<ExecSpace,LossFunction>* amsgrad_step =
          dynamic_cast<AMSGradStep<ExecSpace,LossFunction>*>(&stepper);
        if (sgd_step != nullptr)
          gcp_sgd_iter_async_kernel(
            X.impl(),this->ut.impl(),loss_func,nsz,nsnz,wz,wnz,rand_pool,*sgd_step,
            this->mode_beg, this->mode_end, this->algParams, total_iters);
        else if (adagrad_step != nullptr)
          gcp_sgd_iter_async_kernel(
            X.impl(),this->ut.impl(),loss_func,nsz,nsnz,wz,wnz,rand_pool,*adagrad_step,
            this->mode_beg, this->mode_end, this->algParams, total_iters);
        else if (adam_step != nullptr)
          gcp_sgd_iter_async_kernel(
            X.impl(),this->ut.impl(),loss_func,nsz,nsnz,wz,wnz,rand_pool,*adam_step,
            this->mode_beg, this->mode_end, this->algParams, total_iters);
        else if (amsgrad_step != nullptr)
          gcp_sgd_iter_async_kernel(
            X.impl(),this->ut.impl(),loss_func,nsz,nsnz,wz,wnz,rand_pool,*amsgrad_step,
            this->mode_beg, this->mode_end, this->algParams, total_iters);
        else
          Genten::error("Unsupported GCP-SGD stepper!");

        this->timer.stop(this->timer_grad);

        // Update number of iterations
        total_iters += this->algParams.epoch_iters;
      }

      virtual void printTimers(std::ostream& out) const
      {
        out << "\tgradient:  " << this->timer.getTotalTime(this->timer_grad)
            << " seconds\n";
      }
    };

  }

}
