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

#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>

#include "Genten_Kokkos.hpp"

#ifdef KOKKOS_ENABLE_OPENMP
#include "parallel_stable_sort.hpp"
#endif

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#endif

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

//#if defined(KOKKOS_ENABLE_SYCL)
//#include <execution>
//#endif

// Various utility algorithms using Kokkos

namespace Genten {

// Sort an array by computing permutation vector to sorted order.  The
// comparitor operator op encapsulates the array to be sorted, e.g.,
// to sort a 1-D view v:
//    auto op = KOKKOS_LAMBDA(const val_type& a, const val_type& b) {
//                 return v(a) < v(b);
//              };
template <typename PermType, typename Op>
void perm_sort_op(const PermType& perm, const Op& op)
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::perm_sort_op");
#endif

  typedef typename PermType::execution_space exec_space;
  typedef typename PermType::size_type size_type;

  // Initialize perm
  const size_type sz = perm.extent(0);
  Kokkos::parallel_for("Genten::perm_sort::perm_init",
                       Kokkos::RangePolicy<exec_space>(0,sz),
                       KOKKOS_LAMBDA(const size_type i)
  {
    perm(i) = i;
  } );

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  typedef typename PermType::non_const_value_type perm_val_type;
  if (is_gpu_space<exec_space>::value) {
    thrust::stable_sort(thrust::device_ptr<perm_val_type>(perm.data()),
                        thrust::device_ptr<perm_val_type>(perm.data()+sz),
                        op);
  }
  else
#endif

#if defined(KOKKOS_ENABLE_SYCL)
  if (is_sycl_space<exec_space>::value) {
    auto perm_mir = create_mirror_view(perm);
    deep_copy(perm_mir, perm);
    std::stable_sort(/*std::execution::par,*/ perm_mir.data(), perm_mir.data()+sz, op);
    deep_copy(perm, perm_mir);
  }
  else
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
  if (std::is_same<exec_space, Kokkos::OpenMP>::value) {
    pss::parallel_stable_sort(perm.data(), perm.data()+sz, op);
  }
  else
#endif

    std::stable_sort(perm.data(), perm.data()+sz, op);
}

// Sort an array by computing permutation vector to sorted order using
// standard comparitor on the given array v.
template <typename PermType, typename ViewType>
void perm_sort(const PermType& perm, const ViewType& v)
{
  typedef typename ViewType::size_type size_type;
  // We see a massive slowdown on the CPU if this lambda does capture-by-value,
  // which is what KOKKOS_LAMBDA always does. It seems that the view is
  // copied each time the op is executed!
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  perm_sort_op(perm, KOKKOS_LAMBDA(const size_type& a, const size_type& b)
#else
  perm_sort_op(perm, [&](const size_type& a, const size_type& b)
#endif
  {
    return v(a) < v(b);
  });
}

// Perform scan-by-key.  It is assumed keys is a 1-D view, vals is a 2-D
// view, and keys are sorted so entries with the same key are contiguous.
// Set check == true to check the result.
template <typename ValViewType, typename KeyViewType>
void key_scan(const ValViewType& vals, const KeyViewType& keys,
              const bool check = false)
{
  typedef typename ValViewType::non_const_value_type val_type;
  typedef typename KeyViewType::non_const_value_type key_type;
  typedef typename ValViewType::size_type size_type;
  typedef typename ValViewType::execution_space exec_space;
  typedef typename Kokkos::TeamPolicy<exec_space>::member_type TeamMember;
  typedef Kokkos::View<val_type*, Kokkos::LayoutRight, typename exec_space::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;
  typedef Genten::SpaceProperties<exec_space> Prop;

  // FIXME -- Tune these numbers
  const size_type n = vals.extent(0);
  const size_type r = vals.extent(1);
  size_type block_size, num_blocks, block_threshold, league_size, team_size,
    vector_size;
  if (Prop::is_gpu) {
    vector_size = r;
    team_size = 256 / vector_size;
    block_size = 32;
    if (n < block_size) {
      block_size = n;
      team_size = 1;
    }
    num_blocks = (n+block_size-1)/block_size;
    league_size = (n+team_size*block_size-1)/(team_size*block_size);
    block_threshold = 1;
  }
  else {
    const size_type num_threads = Prop::concurrency();
    vector_size = r;
    team_size = 1;
    if (n > num_threads) {
      block_size = (n+num_threads-1)/num_threads;
      num_blocks = (n+block_size-1)/block_size;
    }
    else {
      num_blocks = 1;
      block_size = n;
    }
    league_size = num_blocks;
    block_threshold = 8;
  }
  const size_t bytes = TmpScratchSpace::shmem_size(r);

  ValViewType vals_orig;
  if (check) {
    vals_orig = ValViewType("vals_orig",n,r);
    Kokkos::deep_copy(vals_orig, vals);
  }

  if (check)
    std::cout << "n = " << n << " block_size = " << block_size << " num_blocks = " << num_blocks << " league_size = " << league_size << " team_size = " << team_size << std::endl;

  if (num_blocks > block_threshold) {
    // Parallel scan
    ValViewType block_vals("block_vals", num_blocks, r);
    KeyViewType block_keys("block_keys", num_blocks);
    Kokkos::TeamPolicy<exec_space> policy(league_size,team_size,vector_size);
    Kokkos::parallel_for(
      "Genten::key_scan::parallel_scan",
      policy.set_scratch_size(0,Kokkos::PerThread(bytes)),
      KOKKOS_LAMBDA(const TeamMember& team)
      {
        const size_type block =
          team.league_rank()*team.team_size() + team.team_rank();
        if (block >= num_blocks) return;

        TmpScratchSpace s(team.thread_scratch(0), r);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          s[j] = val_type(0);
        });
        key_type key = 0;
        key_type key_prev = 0;
        for (size_type k=0; k<block_size; ++k) {
          size_type i = block*block_size + k;
          if (i >= n) continue;
          key_prev = key;
          key = keys(i);
          if (i == 0 || key != key_prev) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                                 [&] (const unsigned& j)
            {
              s[j] = vals(i,j);
            });
          }
          else
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                                 [&] (const unsigned& j)
            {
              s[j] += vals(i,j);
            });
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                               [&] (const unsigned& j)
          {
            vals(i,j) = s[j];
          });
        }
        block_keys(block) = key;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          block_vals(block,j) = s[j];
        });
      });

    // Scan the block results that are in the same segment
    key_scan(block_vals, block_keys, check);

    // Update scans for blocks [1,num_blocks) from inter-block scans
    Kokkos::parallel_for(
      "Genten::key_scan::block_scan",
      policy.set_scratch_size(0,Kokkos::PerThread(bytes)),
      KOKKOS_LAMBDA(const TeamMember& team)
      {
        const size_type block =
          team.league_rank()*team.team_size() + team.team_rank();
        size_type i = block*block_size;
        if (block >= num_blocks || i >= n) return;
        TmpScratchSpace s(team.thread_scratch(0), r);
        if (block == 0) return;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          s[j] = block_vals(block-1,j);
        });
        const key_type block_key = block_keys(block-1);
        while (i<n && i<(block+1)*block_size && keys(i)==block_key) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                               [&] (const unsigned& j)
          {
            vals(i,j) += s[j];
          });
          ++i;
        }
      });
  }
  else {
    // Serial scan
    Kokkos::TeamPolicy<exec_space> policy(1,1,vector_size);
    Kokkos::parallel_for(
      "Genten::key_scan::serial_scan",
      policy.set_scratch_size(0,Kokkos::PerThread(bytes)),
      KOKKOS_LAMBDA(const TeamMember& team)
      {
        TmpScratchSpace s(team.thread_scratch(0), r);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          s[j] = val_type(0);
        });
        key_type key = 0;
        key_type key_prev = 0;
        for (size_type i=0; i<n; ++i) {
          key_prev = key;
          key = keys(i);
          if (i == 0 || key != key_prev)
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                                 [&] (const unsigned& j)
            {
              s[j] = vals(i,j);
            });
          else
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                                 [&] (const unsigned& j)
            {
              s[j] += vals(i,j);
            });
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                               [&] (const unsigned& j)
          {
            vals(i,j) = s[j];
          });
        }
      });
  }

  if (check) {
    // Check scan is correct
    typename ValViewType::HostMirror vals_host =
      Kokkos::create_mirror_view(vals_orig);
    typename KeyViewType::HostMirror keys_host =
      Kokkos::create_mirror_view(keys);
    typename ValViewType::HostMirror scans_host =
      Kokkos::create_mirror_view(vals);
    Kokkos::deep_copy(vals_host, vals_orig);
    Kokkos::deep_copy(keys_host, keys);
    Kokkos::deep_copy(scans_host, vals);
    bool correct = true;
    std::vector<val_type> s(r);
    key_type key = 0;
    key_type key_prev = 0;
    for (size_type i=0; i<n; ++i) {
      key_prev = key;
      key = keys_host(i);
      if (i == 0 || key != key_prev)
        for (size_type j=0; j<r; ++j)
          s[j] = vals_host(i,j);
      else
        for (size_type j=0; j<r; ++j)
          s[j] += vals_host(i,j);
      for (size_type j=0; j<r; ++j) {
        if (scans_host(i,j) != s[j]) {
          correct = false;
          break;
        }
      }
    }

    // Print incorrect values
    if (!correct) {
      const size_type w1 = std::ceil(std::log10(n))+2;
      const val_type w2 = std::ceil(std::log10(100))+2;
      std::cout << std::setw(w1) << "i" << " "
                << std::setw(w2-1) << "k" << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2-1) << "v" << j << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2) << "s" << j << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2) << "t" << j << " ";
      std::cout << std::endl
                << std::setw(w1) << "==" << " "
                << std::setw(w2) << "==" << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2) << "==" << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2+1) << "==" << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2+1) << "==" << " ";
      std::cout << std::endl;
      key_type key = 0;
      key_type key_prev = 0;
      for (size_type i=0; i<n; ++i) {
        key_prev = key;
        key = keys_host(i);
        if (i == 0 || key != key_prev)
          for (size_type j=0; j<r; ++j)
            s[j] = vals_host(i,j);
        else
          for (size_type j=0; j<r; ++j)
            s[j] += vals_host(i,j);
        bool line_correct = true;
        for (size_type j=0; j<r; ++j) {
          if (scans_host(i,j) != s[j]) {
            line_correct = false;
            break;
          }
        }
        if (!line_correct) {
          std::cout << std::setw(w1) << i << " "
                    << std::setw(w2) << keys_host(i) << " ";
          for (size_type j=0; j<r; ++j)
            std::cout << std::setw(w2) << vals_host(i,j) << " ";
          for (size_type j=0; j<r; ++j)
            std::cout << std::setw(w2+1) << scans_host(i,j) << " ";
          for (size_type j=0; j<r; ++j)
            std::cout << std::setw(w2+1) << s[j] << " ";
          std::cout << "Wrong!" << std::endl;
        }
      }
      std::cout << "Scan is not correct!" << std::endl;
    }
    else
      std::cout << "Scan is correct!" << std::endl;
  }
}

// Perform scan-by-key with permutation.  It is assumed keys is a 1-D view,
// vals is a 2-D view, and the 1-D permutation array perm permutes keys to
// sorted order.  Set check == true to check the result.
template <typename ValViewType, typename KeyViewType, typename PermViewType>
void key_scan(const ValViewType& vals, const KeyViewType& keys,
              const PermViewType& perm, const bool check = false)
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::key_scan_perm");
#endif

  typedef typename ValViewType::non_const_value_type val_type;
  typedef typename KeyViewType::non_const_value_type key_type;
  typedef typename PermViewType::non_const_value_type perm_type;
  typedef typename ValViewType::size_type size_type;
  typedef typename ValViewType::execution_space exec_space;
  typedef typename Kokkos::TeamPolicy<exec_space>::member_type TeamMember;
  typedef Kokkos::View<val_type*, Kokkos::LayoutRight, typename exec_space::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;
  typedef Genten::SpaceProperties<exec_space> Prop;

  // FIXME -- Tune these numbers
  const size_type n = vals.extent(0);
  const size_type r = vals.extent(1);
  size_type block_size, num_blocks, block_threshold, league_size, team_size,
    vector_size;
  if (Prop::is_gpu) {
    vector_size = r;
    team_size = 256 / vector_size;
    block_size = 32;
    if (n < block_size) {
      block_size = n;
      team_size = 1;
    }
    num_blocks = (n+block_size-1)/block_size;
    league_size = (n+team_size*block_size-1)/(team_size*block_size);
    block_threshold = 1;
  }
  else {
    const size_type num_threads = Prop::concurrency();
    vector_size = r;
    team_size = 1;
    if (n > num_threads) {
      block_size = (n+num_threads-1)/num_threads;
      num_blocks = (n+block_size-1)/block_size;
    }
    else {
      num_blocks = 1;
      block_size = n;
    }
    league_size = num_blocks;
    block_threshold = 1;
  }
  const size_t bytes = TmpScratchSpace::shmem_size(r);

  ValViewType vals_orig;
  if (check) {
    vals_orig = ValViewType("vals_orig",n,r);
    Kokkos::deep_copy(vals_orig, vals);
  }

  if (check)
    std::cout << "n = " << n << " block_size = " << block_size << " num_blocks = " << num_blocks << " league_size = " << league_size << " team_size = " << team_size << std::endl;

  if (num_blocks > block_threshold) {
    // Parallel scan
    ValViewType block_vals("block_vals", num_blocks, r);
    KeyViewType block_keys("block_keys", num_blocks);
    Kokkos::TeamPolicy<exec_space> policy(league_size,team_size,vector_size);
    Kokkos::parallel_for(
      "Genten::key_scan_perm::parallel_scan",
      policy.set_scratch_size(0,Kokkos::PerThread(bytes)),
      KOKKOS_LAMBDA(const TeamMember& team)
      {
        const size_type block =
          team.league_rank()*team.team_size() + team.team_rank();
        if (block >= num_blocks) return;

        TmpScratchSpace s(team.thread_scratch(0), r);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          s[j] = val_type(0);
        });
        key_type key = 0;
        key_type key_prev = 0;
        perm_type p = 0;
        for (size_type k=0; k<block_size; ++k) {
          size_type i = block*block_size + k;
          if (i >= n) continue;
          p = perm(i);
          key_prev = key;
          key = keys(p);
          if (p == 0 || key != key_prev) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                                 [&] (const unsigned& j)
            {
              s[j] = vals(p,j);
            });
          }
          else
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                                 [&] (const unsigned& j)
            {
              s[j] += vals(p,j);
            });
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                               [&] (const unsigned& j)
          {
            vals(p,j) = s[j];
          });
        }
        block_keys(block) = key;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          block_vals(block,j) = s[j];
        });
      });

    // Scan the block results that are in the same segment (does not use
    // permutation)
    key_scan(block_vals, block_keys, check);

    // Update scans for blocks [1,num_blocks) from inter-block scans
    Kokkos::parallel_for(
      "Genten::key_scan_perm::block_scan",
      policy.set_scratch_size(0,Kokkos::PerThread(bytes)),
      KOKKOS_LAMBDA(const TeamMember& team)
      {
        const size_type block =
          team.league_rank()*team.team_size() + team.team_rank();
        size_type i = block*block_size;
        if (block >= num_blocks || i >= n) return;
        TmpScratchSpace s(team.thread_scratch(0), r);
        if (block == 0) return;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          s[j] = block_vals(block-1,j);
        });
        const key_type block_key = block_keys(block-1);
        perm_type p = perm(i);
        while (i<n && i<(block+1)*block_size && keys(p)==block_key) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                               [&] (const unsigned& j)
          {
            vals(p,j) += s[j];
          });
          ++i;
          p = perm(i);
        }
      });
  }
  else {
    // Serial scan
    Kokkos::TeamPolicy<exec_space> policy(1,1,vector_size);
    Kokkos::parallel_for(
      "Genten::key_scan_perm::serial_scan",
      policy.set_scratch_size(0,Kokkos::PerThread(bytes)),
      KOKKOS_LAMBDA(const TeamMember& team)
      {
        TmpScratchSpace s(team.thread_scratch(0), r);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          s[j] = val_type(0);
        });
        key_type key = 0;
        key_type key_prev = 0;
        perm_type p = 0;
        for (size_type i=0; i<n; ++i) {
          p = perm(i);
          key_prev = key;
          key = keys(p);
          if (p == 0 || key != key_prev)
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                                 [&] (const unsigned& j)
            {
              s[j] = vals(p,j);
            });
          else
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                                 [&] (const unsigned& j)
            {
              s[j] += vals(p,j);
            });
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                               [&] (const unsigned& j)
          {
            vals(p,j) = s[j];
          });
        }
      });
  }

  if (check) {
    // Check scan is correct
    typename ValViewType::HostMirror vals_host =
      Kokkos::create_mirror_view(vals_orig);
    typename KeyViewType::HostMirror keys_host =
      Kokkos::create_mirror_view(keys);
    typename PermViewType::HostMirror perm_host =
      Kokkos::create_mirror_view(perm);
    typename ValViewType::HostMirror scans_host =
      Kokkos::create_mirror_view(vals);
    Kokkos::deep_copy(vals_host, vals_orig);
    Kokkos::deep_copy(keys_host, keys);
    Kokkos::deep_copy(perm_host, perm);
    Kokkos::deep_copy(scans_host, vals);
    bool correct = true;
    std::vector<val_type> s(r);
    key_type key = 0;
    key_type key_prev = 0;
    for (size_type i=0; i<n; ++i) {
      perm_type p = perm_host(i);
      key_prev = key;
      key = keys_host(p);
      if (p == 0 || key != key_prev)
        for (size_type j=0; j<r; ++j)
          s[j] = vals_host(p,j);
      else
        for (size_type j=0; j<r; ++j)
          s[j] += vals_host(p,j);
      for (size_type j=0; j<r; ++j) {
        if (scans_host(p,j) != s[j]) {
          correct = false;
          break;
        }
      }
    }

    // Print incorrect values
    if (!correct) {
      const size_type w1 = std::ceil(std::log10(n))+2;
      const val_type w2 = std::ceil(std::log10(100))+2;
      std::cout << std::setw(w1) << "i" << " "
                << std::setw(w2-1) << "k" << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2-1) << "v" << j << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2) << "s" << j << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2) << "t" << j << " ";
      std::cout << std::endl
                << std::setw(w1) << "==" << " "
                << std::setw(w2) << "==" << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2) << "==" << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2+1) << "==" << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2+1) << "==" << " ";
      std::cout << std::endl;
      key_type key = 0;
      key_type key_prev = 0;
      for (size_type i=0; i<n; ++i) {
        perm_type p = perm_host(i);
        key_prev = key;
        key = keys_host(p);
        if (p == 0 || key != key_prev)
          for (size_type j=0; j<r; ++j)
            s[j] = vals_host(p,j);
        else
          for (size_type j=0; j<r; ++j)
            s[j] += vals_host(p,j);
        bool line_correct = true;
        for (size_type j=0; j<r; ++j) {
          if (scans_host(p,j) != s[j]) {
            line_correct = false;
            break;
          }
        }
        if (!line_correct) {
          std::cout << std::setw(w1) << p << " "
                    << std::setw(w2) << keys_host(p) << " ";
          for (size_type j=0; j<r; ++j)
            std::cout << std::setw(w2) << vals_host(p,j) << " ";
          for (size_type j=0; j<r; ++j)
            std::cout << std::setw(w2+1) << scans_host(p,j) << " ";
          for (size_type j=0; j<r; ++j)
            std::cout << std::setw(w2+1) << s[j] << " ";
          std::cout << "Wrong!" << std::endl;
        }
      }
      std::cout << "Scan is not correct!" << std::endl;
    }
    else
      std::cout << "Scan is correct!" << std::endl;
  }
}

// Perform segmented scan using head flags.  It is assumed flags is a 1-D view,
// vals is a 2-D view, and a new segment starts whenver flags(i) == 1.
// Set check == true to check the result.
template <typename ValViewType, typename FlagViewType>
void seg_scan(const ValViewType& vals, const FlagViewType& flags,
              const bool check = false)
{
  typedef typename ValViewType::non_const_value_type val_type;
  typedef typename FlagViewType::non_const_value_type flag_type;
  typedef typename ValViewType::size_type size_type;
  typedef typename ValViewType::execution_space exec_space;
  typedef typename Kokkos::TeamPolicy<exec_space>::member_type TeamMember;
  typedef Kokkos::View<val_type*, Kokkos::LayoutRight, typename exec_space::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;
  typedef Genten::SpaceProperties<exec_space> Prop;

  const size_type n = vals.extent(0);
  const size_type r = vals.extent(1);
  size_type block_size, num_blocks, block_threshold, league_size, team_size,
    vector_size;
  if (Prop::is_gpu) {
    vector_size = r;
    team_size = 256 / vector_size;
    block_size = 32;
    if (n < block_size) {
      block_size = n;
      team_size = 1;
    }
    num_blocks = (n+block_size-1)/block_size;
    league_size = (n+team_size*block_size-1)/(team_size*block_size);
    block_threshold = 1;
  }
  else {
    const size_type num_threads = Prop::concurrency();
    vector_size = r;
    team_size = 1;
    if (n > num_threads) {
      block_size = (n+num_threads-1)/num_threads;
      num_blocks = (n+block_size-1)/block_size;
    }
    else {
      num_blocks = 1;
      block_size = n;
    }
    league_size = num_blocks;
    block_threshold = 8;
  }
  const size_t bytes = TmpScratchSpace::shmem_size(r);

  ValViewType vals_orig;
  if (check) {
    vals_orig = ValViewType("vals_orig",n,r);
    Kokkos::deep_copy(vals_orig, vals);
  }

  if (check)
    std::cout << "n = " << n << " block_size = " << block_size << " num_blocks = " << num_blocks << " league_size = " << league_size << " team_size = " << team_size << std::endl;

  if (num_blocks > block_threshold) {
    // Parallel scan
    ValViewType block_vals("block_vals", num_blocks, r);
    FlagViewType block_flags("block_flags", num_blocks);
    Kokkos::TeamPolicy<exec_space> policy(league_size,team_size,vector_size);
    Kokkos::parallel_for(
      policy.set_scratch_size(0,Kokkos::PerThread(bytes)),
      KOKKOS_LAMBDA(const TeamMember& team)
      {
        const size_type block =
          team.league_rank()*team.team_size() + team.team_rank();
        if (block >= num_blocks) return;

        TmpScratchSpace s(team.thread_scratch(0), r);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          s[j] = val_type(0);
        });
        flag_type f = block == 0 ? 1 : 0;
        for (size_type k=0; k<block_size; ++k) {
          size_type i = block*block_size + k;
          if (i >= n) continue;
          if (flags(i) == 1) {
            f = flag_type(1);
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                                 [&] (const unsigned& j)
            {
              s[j] = vals(i,j);
            });
          }
          else
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                                 [&] (const unsigned& j)
            {
              s[j] += vals(i,j);
            });
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                               [&] (const unsigned& j)
          {
            vals(i,j) = s[j];
          });
        }
        block_flags(block) = f;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          block_vals(block,j) = s[j];
        });
      });

    // Scan the block results that are in the same segment
    seg_scan(block_vals, block_flags, check);

    // Update scans for blocks [1,num_blocks) from inter-block scans
    Kokkos::parallel_for(
      policy.set_scratch_size(0,Kokkos::PerThread(bytes)),
      KOKKOS_LAMBDA(const TeamMember& team)
      {
        const size_type block =
          team.league_rank()*team.team_size() + team.team_rank();
        size_type i = block*block_size;
        if (block >= num_blocks || i >= n) return;
        TmpScratchSpace s(team.thread_scratch(0), r);
        if (block == 0) return;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          s[j] = block_vals(block-1,j);
        });
        while (i<n && i<(block+1)*block_size && flags(i)==0) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                               [&] (const unsigned& j)
          {
            vals(i,j) += s[j];
          });
          ++i;
        }
      });
  }
  else {
    // Serial scan
    Kokkos::TeamPolicy<exec_space> policy(1,1,vector_size);
    Kokkos::parallel_for(
      policy.set_scratch_size(0,Kokkos::PerThread(bytes)),
      KOKKOS_LAMBDA(const TeamMember& team)
      {
        TmpScratchSpace s(team.thread_scratch(0), r);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          s[j] = val_type(0);
        });
        for (size_type i=0; i<n; ++i) {
          if (flags(i) == 1)
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                                 [&] (const unsigned& j)
            {
              s[j] = vals(i,j);
            });
          else
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                                 [&] (const unsigned& j)
            {
              s[j] += vals(i,j);
            });
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                               [&] (const unsigned& j)
          {
            vals(i,j) = s[j];
          });
        }
      });
  }

  if (check) {
    // Check scan is correct
    typename ValViewType::HostMirror vals_host =
      Kokkos::create_mirror_view(vals_orig);
    typename FlagViewType::HostMirror flags_host =
      Kokkos::create_mirror_view(flags);
    typename ValViewType::HostMirror scans_host =
      Kokkos::create_mirror_view(vals);
    Kokkos::deep_copy(vals_host, vals_orig);
    Kokkos::deep_copy(flags_host, flags);
    Kokkos::deep_copy(scans_host, vals);
    bool correct = true;
    std::vector<val_type> s(r);
    for (size_type i=0; i<n; ++i) {
      if (i == 0 || flags_host(i) == 1)
        for (size_type j=0; j<r; ++j)
          s[j] = vals_host(i,j);
      else
        for (size_type j=0; j<r; ++j)
          s[j] += vals_host(i,j);
      for (size_type j=0; j<r; ++j) {
        if (scans_host(i,j) != s[j]) {
          correct = false;
          break;
        }
      }
    }

    // Print incorrect values
    if (!correct) {
      const size_type w1 = std::ceil(std::log10(n))+2;
      const val_type w2 = std::ceil(std::log10(100))+2;
      std::cout << std::setw(w1) << "i" << " "
                << std::setw(2) << "k" << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2-1) << "v" << j << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2) << "s" << j << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2) << "t" << j << " ";
      std::cout << std::endl
                << std::setw(w1) << "==" << " "
                << std::setw(2) << "=" << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2) << "==" << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2+1) << "==" << " ";
      for (size_type j=0; j<r; ++j)
        std::cout << std::setw(w2+1) << "==" << " ";
      std::cout << std::endl;
      for (size_type i=0; i<n; ++i) {
        if (i == 0 || flags_host(i) == 1)
          for (size_type j=0; j<r; ++j)
            s[j] = vals_host(i,j);
        else
          for (size_type j=0; j<r; ++j)
            s[j] += vals_host(i,j);
        bool line_correct = true;
        for (size_type j=0; j<r; ++j) {
          if (scans_host(i,j) != s[j]) {
            line_correct = false;
            break;
          }
        }
        if (!line_correct) {
          std::cout << std::setw(w1) << i << " "
                    << std::setw(2) << flags_host(i) << " ";
          for (size_type j=0; j<r; ++j)
            std::cout << std::setw(w2) << vals_host(i,j) << " ";
          for (size_type j=0; j<r; ++j)
            std::cout << std::setw(w2+1) << scans_host(i,j) << " ";
          for (size_type j=0; j<r; ++j)
            std::cout << std::setw(w2+1) << s[j] << " ";
          std::cout << "Wrong!" << std::endl;
        }
      }
      std::cout << "Scan is not correct!" << std::endl;
    }
    else
      std::cout << "Scan is correct!" << std::endl;
  }
}

}
