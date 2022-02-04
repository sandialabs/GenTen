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
#include <ostream>

#include "Genten_Kokkos.hpp"
#if defined(KOKKOS_ENABLE_CUDA)
#include "Cuda/Kokkos_Cuda_Team.hpp"
#if (CUDA_VERSION < 10000)
#define KOKKOS_DEFAULTED_DEVICE_FUNCTION __device__ inline
#else
#define KOKKOS_DEFAULTED_DEVICE_FUNCTION inline
#endif
#endif

#if defined (KOKKOS_ENABLE_HIP)
#include "HIP/Kokkos_HIP_Team.hpp"
#define KOKKOS_DEFAULTED_DEVICE_FUNCTION __device__ inline
#endif

namespace Genten {

  namespace Impl {

#ifdef __CUDA_ARCH__
    // Reduce y across the warp and broadcast to all lanes
    template <typename T, typename Ordinal>
     __device__ inline T warpReduce(T y, const Ordinal warp_size) {
      Kokkos::Impl::CudaTeamMember::vector_reduce(Kokkos::Sum<T>(y));
      return y;
    }
#endif

#ifdef __HIP_DEVICE_COMPILE__
    // Reduce y across the warp and broadcast to all lanes
    template <typename T, typename Ordinal>
     __device__ inline T warpReduce(T y, const Ordinal warp_size) {
      Kokkos::Impl::HIPTeamMember::vector_reduce(Kokkos::Sum<T>(y));
      return y;
    }
#endif

  }

  // A compile-time sized polymorphic array used in, e.g., MTTKRP
  template <typename ExecSpace, typename Scalar, typename Ordinal,
            unsigned Length, unsigned Size, unsigned WarpDim,
            typename Enabled = void>
  class TinyVec {
  public:

    typedef ExecSpace exec_space;
    typedef Scalar scalar_type;
    typedef Ordinal ordinal_type;

    static const ordinal_type len = Length / WarpDim;
    Kokkos::Impl::integral_nonzero_constant<ordinal_type,Size/WarpDim> sz;
    alignas(64) scalar_type v[len];

    KOKKOS_DEFAULTED_FUNCTION
    TinyVec() = default;

    KOKKOS_INLINE_FUNCTION
    TinyVec(const ordinal_type size, const scalar_type x) :
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
#else
      sz(size)
#endif
    {
      broadcast(x);
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec(const ordinal_type size, const scalar_type* x) :
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
#else
      sz(size)
#endif
    {
      load(x);
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec(const TinyVec& x) : sz(x.sz.value) {
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] = x.v[i];
    }

    KOKKOS_DEFAULTED_FUNCTION
    ~TinyVec() = default;

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator=(const TinyVec& x) {
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] = x.v[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator=(const scalar_type x) {
      broadcast(x);
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator=(const scalar_type* x) {
      load(x);
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    constexpr ordinal_type size() const { return sz.value; }

    KOKKOS_INLINE_FUNCTION
    scalar_type operator[](ordinal_type i) const { return v[i]; }

    KOKKOS_INLINE_FUNCTION
    scalar_type& operator[](ordinal_type i) { return v[i]; }

    KOKKOS_INLINE_FUNCTION
    void broadcast(const scalar_type x) {
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] = x;
    }

    KOKKOS_INLINE_FUNCTION
    void load(const scalar_type* x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] = x[i*WarpDim+threadIdx.x];
#else
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] = x[i];
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void store(scalar_type* x) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      for (ordinal_type i=0; i<sz.value; ++i)
        x[i*WarpDim+threadIdx.x] = v[i];
#else
      for (ordinal_type i=0; i<sz.value; ++i)
        x[i] = v[i];
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void store_plus(scalar_type* x) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      for (ordinal_type i=0; i<sz.value; ++i)
        x[i*WarpDim+threadIdx.x] += v[i];
#else
      for (ordinal_type i=0; i<sz.value; ++i)
        x[i] += v[i];
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void atomic_store_plus(volatile scalar_type* x) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      for (ordinal_type i=0; i<sz.value; ++i)
        Kokkos::atomic_add(x+i*WarpDim+threadIdx.x, v[i]);
#else
      for (ordinal_type i=0; i<sz.value; ++i)
        Kokkos::atomic_add(x+i, v[i]);
#endif
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec atomic_exchange(volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      for (ordinal_type i=0; i<sz.value; ++i)
        c.v[i] = Kokkos::atomic_exchange(x+i*WarpDim+threadIdx.x, v[i]);
#else
      for (ordinal_type i=0; i<sz.value; ++i)
        c.v[i] = Kokkos::atomic_exchange(x+i, v[i]);
#endif
      return c;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec atomic_fetch_max(volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      for (ordinal_type i=0; i<sz.value; ++i)
        c.v[i] = Kokkos::atomic_fetch_max(x+i*WarpDim+threadIdx.x, v[i]);
#else
      for (ordinal_type i=0; i<sz.value; ++i)
        c.v[i] = Kokkos::atomic_fetch_max(x+i, v[i]);
#endif
      return c;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec atomic_fetch_min(volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      for (ordinal_type i=0; i<sz.value; ++i)
        c.v[i] = Kokkos::atomic_fetch_min(x+i*WarpDim+threadIdx.x, v[i]);
#else
      for (ordinal_type i=0; i<sz.value; ++i)
        c.v[i] = Kokkos::atomic_fetch_min(x+i, v[i]);
#endif
      return c;
    }

    template <typename Oper>
    KOKKOS_INLINE_FUNCTION
    TinyVec atomic_oper_fetch(const Oper& op, volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      for (ordinal_type i=0; i<sz.value; ++i)
        c.v[i] = Genten::atomic_oper_fetch(op, x+i*WarpDim+threadIdx.x, v[i]);
#else
      for (ordinal_type i=0; i<sz.value; ++i)
        c.v[i] = Genten::atomic_oper_fetch(op, x+i, v[i]);
#endif
      return c;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator+=(const TinyVec& x) {
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] += x.v[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator-=(const TinyVec& x) {
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] -= x.v[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator*=(const TinyVec& x) {
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] *= x.v[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator/=(const TinyVec& x) {
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] /= x.v[i];
      return *this;
    }

    template <typename Op>
    KOKKOS_INLINE_FUNCTION
    void apply_unary(const Op& op, const TinyVec& x) {
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] = op(x.v[i]);
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator+=(const scalar_type x) {
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] += x;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator-=(const scalar_type x) {
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] -= x;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator*=(const scalar_type x) {
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] *= x;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator/=(const scalar_type x) {
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] /= x;
      return *this;
    }

    template <unsigned S>
    KOKKOS_INLINE_FUNCTION
    TinyVec& operator+=(
      const TinyVec<ExecSpace,Scalar,Ordinal,Length,S,WarpDim,Enabled>& x) {
      for (ordinal_type i=0; i<x.sz.value; ++i)
        v[i] += x.v[i];
      return *this;
    }

    template <unsigned S>
    KOKKOS_INLINE_FUNCTION
    TinyVec& operator-=(
      const TinyVec<ExecSpace,Scalar,Ordinal,Length,S,WarpDim,Enabled>& x) {
      for (ordinal_type i=0; i<x.sz.value; ++i)
        v[i] -= x.v[i];
      return *this;
    }

    template <unsigned S>
    KOKKOS_INLINE_FUNCTION
    TinyVec& operator*=(
      const TinyVec<ExecSpace,Scalar,Ordinal,Length,S,WarpDim,Enabled>& x) {
      for (ordinal_type i=0; i<x.sz.value; ++i)
        v[i] *= x.v[i];
      return *this;
    }

    template <unsigned S>
    KOKKOS_INLINE_FUNCTION
    TinyVec& operator/=(
      const TinyVec<ExecSpace,Scalar,Ordinal,Length,S,WarpDim,Enabled>& x) {
      for (ordinal_type i=0; i<x.sz.value; ++i)
        v[i] /= x.v[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator+=(const scalar_type* x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] += x[i*WarpDim+threadIdx.x];
#else
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] += x[i];
#endif
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator-=(const scalar_type* x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] -= x[i*WarpDim+threadIdx.x];
#else
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] -= x[i];
#endif
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator*=(const scalar_type* x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] *= x[i*WarpDim+threadIdx.x];
#else
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] *= x[i];
#endif
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator/=(const scalar_type* x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] /= x[i*WarpDim+threadIdx.x];
#else
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] /= x[i];
#endif
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    scalar_type sum() const {
      scalar_type s = 0.0;
      for (ordinal_type i=0; i<sz.value; ++i)
        s += v[i];
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      s = Impl::warpReduce(s, WarpDim);
#endif
      return s;
    }

    void print(std::ostream& os) const {
      os << "[ ";
      for (ordinal_type i=0; i<sz.value; ++i)
        os << v[i] << " ";
      os << "]";
    }

  };

#if (defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)) || \
    (defined(KOKKOS_ENABLE_HIP) && defined(__HIP_DEVICE_COMPILE__))

  // Specialization for Cuda or HIP where Length / WarpDim == 1.  Store the vector
  // components in register space since Cuda or HIP may store them in global memory
  // (especially in the dynamically sized case).
  template <typename Scalar, typename Ordinal,
            unsigned Length, unsigned Size, unsigned WarpDim>
  class TinyVec< Kokkos_GPU_Space,Scalar,Ordinal,Length,Size,WarpDim,
                 typename std::enable_if<Length/WarpDim == 1>::type >
  {
  public:

    typedef Kokkos_GPU_Space exec_space;
    typedef Scalar scalar_type;
    typedef Ordinal ordinal_type;

    static const ordinal_type len = 1; // = Length/WarpDim
    Kokkos::Impl::integral_nonzero_constant<ordinal_type,Size/WarpDim> sz;
    scalar_type v0;

    KOKKOS_DEFAULTED_DEVICE_FUNCTION
    TinyVec() = default;

    __device__ inline
    TinyVec(const ordinal_type size, const scalar_type x)
      : sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
    {
      broadcast(x);
    }

    __device__ inline
    TinyVec(const ordinal_type size, const scalar_type* x)
      : sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
    {
      load(x);
    }

    __device__ inline
    TinyVec(const TinyVec& x) : sz(x.sz.value) {
      v0 = x.v0;
    }

    KOKKOS_DEFAULTED_DEVICE_FUNCTION
    ~TinyVec() = default;

    __device__ inline
    TinyVec& operator=(const TinyVec& x) {
      v0 = x.v0;
      return *this;
    }

    __device__ inline
    TinyVec& operator=(const scalar_type x) {
      broadcast(x);
      return *this;
    }

    __device__ inline
    TinyVec& operator=(const scalar_type* x) {
      load(x);
      return *this;
    }

    __device__ inline
    constexpr ordinal_type size() const { return sz.value; }

    __device__ inline
    void broadcast(const scalar_type x) {
      v0 = x;
    }

    __device__ inline
    void load(const scalar_type* x) {
      if (sz.value > 0) v0 = x[threadIdx.x];
    }

    __device__ inline
    void store(scalar_type* x) const {
      if (sz.value > 0) x[threadIdx.x] = v0;
    }

    __device__ inline
    void store_plus(scalar_type* x) const {
      if (sz.value > 0) x[threadIdx.x] += v0;
    }

    __device__ inline
    void atomic_store_plus(volatile scalar_type* x) const {
      if (sz.value > 0) Kokkos::atomic_add(x+threadIdx.x, v0);
    }

    __device__ inline
    TinyVec atomic_exchange(volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
      if (sz.value > 0) c.v0 = Kokkos::atomic_exchange(x+threadIdx.x, v0);
      return c;
    }

    __device__ inline
    TinyVec atomic_fetch_max(volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
      if (sz.value > 0) c.v0 = Kokkos::atomic_fetch_max(x+threadIdx.x, v0);
      return c;
    }

    __device__ inline
    TinyVec atomic_fetch_min(volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
      if (sz.value > 0) c.v0 = Kokkos::atomic_fetch_min(x+threadIdx.x, v0);
      return c;
    }

    template <typename Oper>
    __device__ inline
    TinyVec atomic_oper_fetch(const Oper& op, volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
      if (sz.value > 0) c.v0 = Genten::atomic_oper_fetch(op, x+threadIdx.x, v0);
      return c;
    }

    __device__ inline
    TinyVec& operator+=(const TinyVec& x) {
      v0 += x.v0;
      return *this;
    }

    __device__ inline
    TinyVec& operator-=(const TinyVec& x) {
      v0 -= x.v0;
      return *this;
    }

    __device__ inline
    TinyVec& operator*=(const TinyVec& x) {
      v0 *= x.v0;
      return *this;
    }

    __device__ inline
    TinyVec& operator/=(const TinyVec& x) {
      v0 /= x.v0;
      return *this;
    }

    template <typename Op>
    __device__ inline
    void apply_unary(const Op& op, const TinyVec& x) {
      v0 = op(x.v0);
    }

    __device__ inline
    TinyVec& operator+=(const scalar_type x) {
      v0 += x;
      return *this;
    }

    __device__ inline
    TinyVec& operator-=(const scalar_type x) {
      v0 -= x;
      return *this;
    }

    __device__ inline
    TinyVec& operator*=(const scalar_type x) {
      v0 *= x;
      return *this;
    }

    __device__ inline
    TinyVec& operator/=(const scalar_type x) {
      v0 /= x;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator+=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 += x.v0;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator-=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 -= x.v0;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator*=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 *= x.v0;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator/=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 /= x.v0;
      return *this;
    }

    __device__ inline
    TinyVec& operator+=(const scalar_type* x) {
      if (sz.value > 0) v0 += x[threadIdx.x];
      return *this;
    }

    __device__ inline
    TinyVec& operator-=(const scalar_type* x) {
      if (sz.value > 0) v0 -= x[threadIdx.x];
      return *this;
    }

    __device__ inline
    TinyVec& operator*=(const scalar_type* x) {
      if (sz.value > 0) v0 *= x[threadIdx.x];
      return *this;
    }

    __device__ inline
    TinyVec& operator/=(const scalar_type* x) {
      if (sz.value > 0) v0 /= x[threadIdx.x];
      return *this;
    }

    __device__ inline
    scalar_type sum() const {
      scalar_type s = 0.0;
      if (sz.value > 0) s += v0;
      s = Impl::warpReduce(s, WarpDim);
      return s;
    }

  };

  // Specialization for Cuda or HIP where Length / WarpDim == 2.  Store the vector
  // components in register space since Cuda or HIP may store them in global memory
  // (especially in the dynamically sized case).
  template <typename Scalar, typename Ordinal,
            unsigned Length, unsigned Size, unsigned WarpDim>
  class TinyVec< Kokkos_GPU_Space,Scalar,Ordinal,Length,Size,WarpDim,
                 typename std::enable_if<Length/WarpDim == 2>::type >
  {
  public:

    typedef Kokkos_GPU_Space exec_space;
    typedef Scalar scalar_type;
    typedef Ordinal ordinal_type;

    static const ordinal_type len = 2;  // = Length/WarpDim
    Kokkos::Impl::integral_nonzero_constant<ordinal_type,Size/WarpDim> sz;
    scalar_type v0, v1;

    KOKKOS_DEFAULTED_DEVICE_FUNCTION
    TinyVec() = default;

    __device__ inline
    TinyVec(const ordinal_type size, const scalar_type x)
      : sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
    {
      broadcast(x);
    }

    __device__ inline
    TinyVec(const ordinal_type size, const scalar_type* x)
      : sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
    {
      load(x);
    }

    __device__ inline
    TinyVec(const TinyVec& x) : sz(x.sz.value) {
      v0 = x.v0; v1 = x.v1;
    }

    KOKKOS_DEFAULTED_DEVICE_FUNCTION
    ~TinyVec() = default;

    __device__ inline
    TinyVec& operator=(const TinyVec& x) {
      v0 = x.v0; v1 = x.v1;
      return *this;
    }

    __device__ inline
    TinyVec& operator=(const scalar_type x) {
      broadcast(x);
      return *this;
    }

    __device__ inline
    TinyVec& operator=(const scalar_type* x) {
      load(x);
      return *this;
    }

    __device__ inline
    constexpr ordinal_type size() const { return sz.value; }

    __device__ inline
    void broadcast(const scalar_type x) {
      v0 = v1 = x;
    }

    __device__ inline
    void load(const scalar_type* x) {
      if (sz.value > 0) v0 = x[threadIdx.x];
      if (sz.value > 1) v1 = x[WarpDim + threadIdx.x];
    }

    __device__ inline
    void store(scalar_type* x) const {
      if (sz.value > 0) x[threadIdx.x] = v0;
      if (sz.value > 1) x[WarpDim + threadIdx.x] = v1;
    }

    __device__ inline
    void store_plus(scalar_type* x) const {
      if (sz.value > 0) x[threadIdx.x] += v0;
      if (sz.value > 1) x[WarpDim + threadIdx.x] += v1;
    }

    __device__ inline
    void atomic_store_plus(volatile scalar_type* x) const {
      if (sz.value > 0) Kokkos::atomic_add(x+threadIdx.x, v0);
      if (sz.value > 1) Kokkos::atomic_add(x+WarpDim+threadIdx.x, v1);
    }

    __device__ inline
    TinyVec atomic_exchange(volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
      if (sz.value > 0)
        c.v0 = Kokkos::atomic_exchange(x+threadIdx.x, v0);
      if (sz.value > 1)
        c.v1 = Kokkos::atomic_exchange(x+WarpDim+threadIdx.x, v1);
      return c;
    }

    __device__ inline
    TinyVec atomic_fetch_max(volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
      if (sz.value > 0)
        c.v0 = Kokkos::atomic_fetch_max(x+threadIdx.x, v0);
      if (sz.value > 1)
        c.v1 = Kokkos::atomic_fetch_max(x+WarpDim+threadIdx.x, v1);
      return c;
    }

    __device__ inline
    TinyVec atomic_fetch_min(volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
      if (sz.value > 0)
        c.v0 = Kokkos::atomic_fetch_min(x+threadIdx.x, v0);
      if (sz.value > 1)
        c.v1 = Kokkos::atomic_fetch_min(x+WarpDim+threadIdx.x, v1);
      return c;
    }

    template <typename Oper>
    __device__ inline
    TinyVec atomic_oper_fetch(const Oper& op, volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
      if (sz.value > 0)
        c.v0 = Genten::atomic_oper_fetch(op, x+threadIdx.x, v0);
      if (sz.value > 1)
        c.v1 = Genten::atomic_oper_fetch(op, x+WarpDim+threadIdx.x, v1);
      return c;
    }

    __device__ inline
    TinyVec& operator+=(const TinyVec& x) {
      v0 += x.v0;
      v1 += x.v1;
      return *this;
    }

    __device__ inline
    TinyVec& operator-=(const TinyVec& x) {
      v0 -= x.v0;
      v1 -= x.v1;
      return *this;
    }

    __device__ inline
    TinyVec& operator*=(const TinyVec& x) {
      v0 *= x.v0;
      v1 *= x.v1;
      return *this;
    }

    __device__ inline
    TinyVec& operator/=(const TinyVec& x) {
      v0 /= x.v0;
      v1 /= x.v1;
      return *this;
    }

    template <typename Op>
    __device__ inline
    void apply_unary(const Op& op, const TinyVec& x) {
      v0 = op(x.v0);
      v1 = op(x.v1);
    }

    __device__ inline
    TinyVec& operator+=(const scalar_type x) {
      v0 += x;
      v1 += x;
      return *this;
    }

    __device__ inline
    TinyVec& operator-=(const scalar_type x) {
      v0 -= x;
      v1 -= x;
      return *this;
    }

    __device__ inline
    TinyVec& operator*=(const scalar_type x) {
      v0 *= x;
      v1 *= x;
      return *this;
    }

    __device__ inline
    TinyVec& operator/=(const scalar_type x) {
      v0 /= x;
      v1 /= x;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator+=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 += x.v0;
      if (x.sz.value > 1) v1 += x.v1;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator-=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 -= x.v0;
      if (x.sz.value > 1) v1 -= x.v1;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator*=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 *= x.v0;
      if (x.sz.value > 1) v1 *= x.v1;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator/=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 /= x.v0;
      if (x.sz.value > 1) v1 /= x.v1;
      return *this;
    }

    __device__ inline
    TinyVec& operator+=(const scalar_type* x) {
      if (sz.value > 0) v0 += x[threadIdx.x];
      if (sz.value > 1) v1 += x[WarpDim + threadIdx.x];
      return *this;
    }

    __device__ inline
    TinyVec& operator-=(const scalar_type* x) {
      if (sz.value > 0) v0 -= x[threadIdx.x];
      if (sz.value > 1) v1 -= x[WarpDim + threadIdx.x];
      return *this;
    }

   __device__ inline
    TinyVec& operator*=(const scalar_type* x) {
      if (sz.value > 0) v0 *= x[threadIdx.x];
      if (sz.value > 1) v1 *= x[WarpDim + threadIdx.x];
      return *this;
    }

    __device__ inline
    TinyVec& operator/=(const scalar_type* x) {
      if (sz.value > 0) v0 /= x[threadIdx.x];
      if (sz.value > 1) v1 /= x[WarpDim + threadIdx.x];
      return *this;
    }

    __device__ inline
    scalar_type sum() const {
      scalar_type s = 0.0;
      if (sz.value > 0) s += v0;
      if (sz.value > 1) s += v1;
      s = Impl::warpReduce(s, WarpDim);
      return s;
    }

  };

  // Specialization for Cuda or HIP where Length / WarpDim == 3.  Store the vector
  // components in register space since Cuda or HIP may store them in global memory
  // (especially in the dynamically sized case).
  template <typename Scalar, typename Ordinal,
            unsigned Length, unsigned Size, unsigned WarpDim>
  class TinyVec< Kokkos_GPU_Space,Scalar,Ordinal,Length,Size,WarpDim,
                 typename std::enable_if<Length/WarpDim == 3>::type >
  {
  public:

    typedef Kokkos_GPU_Space exec_space;
    typedef Scalar scalar_type;
    typedef Ordinal ordinal_type;

    static const ordinal_type len = 3;  // = Length/WarpDim
    Kokkos::Impl::integral_nonzero_constant<ordinal_type,Size/WarpDim> sz;
    scalar_type v0, v1, v2;

    KOKKOS_DEFAULTED_DEVICE_FUNCTION
    TinyVec() = default;

    __device__ inline
    TinyVec(const ordinal_type size, const scalar_type x)
      : sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
    {
      broadcast(x);
    }

    __device__ inline
    TinyVec(const ordinal_type size, const scalar_type* x)
      : sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
    {
      load(x);
    }

    __device__ inline
    TinyVec(const TinyVec& x) : sz(x.sz.value) {
      v0 = x.v0; v1 = x.v1; v2 = x.v2;
    }

    KOKKOS_DEFAULTED_DEVICE_FUNCTION
    ~TinyVec() = default;

    __device__ inline
    TinyVec& operator=(const TinyVec& x) {
      v0 = x.v0; v1 = x.v1; v2 = x.v2;
      return *this;
    }

    __device__ inline
    TinyVec& operator=(const scalar_type x) {
      broadcast(x);
      return *this;
    }

    __device__ inline
    TinyVec& operator=(const scalar_type* x) {
      load(x);
      return *this;
    }

    __device__ inline
    constexpr ordinal_type size() const { return sz.value; }

    __device__ inline
    void broadcast(const scalar_type x) {
      v0 = v1 = v2 = x;
    }

    __device__ inline
    void load(const scalar_type* x) {
      if (sz.value > 0) v0 = x[threadIdx.x];
      if (sz.value > 1) v1 = x[WarpDim + threadIdx.x];
      if (sz.value > 2) v2 = x[2*WarpDim + threadIdx.x];
    }

    __device__ inline
    void store(scalar_type* x) const {
      if (sz.value > 0) x[threadIdx.x] = v0;
      if (sz.value > 1) x[WarpDim + threadIdx.x] = v1;
      if (sz.value > 2) x[2*WarpDim + threadIdx.x] = v2;
    }

    __device__ inline
    void store_plus(scalar_type* x) const {
      if (sz.value > 0) x[threadIdx.x] += v0;
      if (sz.value > 1) x[WarpDim + threadIdx.x] += v1;
      if (sz.value > 2) x[2*WarpDim + threadIdx.x] += v2;
    }

    __device__ inline
    void atomic_store_plus(volatile scalar_type* x) const {
      if (sz.value > 0) Kokkos::atomic_add(x+threadIdx.x, v0);
      if (sz.value > 1) Kokkos::atomic_add(x+WarpDim+threadIdx.x, v1);
      if (sz.value > 2) Kokkos::atomic_add(x+2*WarpDim+threadIdx.x, v2);
    }

    __device__ inline
    TinyVec atomic_exchange(volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
      if (sz.value > 0)
        c.v0 = Kokkos::atomic_exchange(x+threadIdx.x, v0);
      if (sz.value > 1)
        c.v1 = Kokkos::atomic_exchange(x+WarpDim+threadIdx.x, v1);
      if (sz.value > 2)
        c.v2 = Kokkos::atomic_exchange(x+2*WarpDim+threadIdx.x, v2);
      return c;
    }

    __device__ inline
    TinyVec atomic_fetch_max(volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
      if (sz.value > 0)
        c.v0 = Kokkos::atomic_fetch_max(x+threadIdx.x, v0);
      if (sz.value > 1)
        c.v1 = Kokkos::atomic_fetch_max(x+WarpDim+threadIdx.x, v1);
      if (sz.value > 2)
        c.v2 = Kokkos::atomic_fetch_max(x+2*WarpDim+threadIdx.x, v2);
      return c;
    }

    __device__ inline
    TinyVec atomic_fetch_min(volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
      if (sz.value > 0)
        c.v0 = Kokkos::atomic_fetch_min(x+threadIdx.x, v0);
      if (sz.value > 1)
        c.v1 = Kokkos::atomic_fetch_min(x+WarpDim+threadIdx.x, v1);
      if (sz.value > 2)
        c.v2 = Kokkos::atomic_fetch_min(x+2*WarpDim+threadIdx.x, v2);
      return c;
    }

    template <typename Oper>
    __device__ inline
    TinyVec atomic_oper_fetch(const Oper& op, volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
      if (sz.value > 0)
        c.v0 = Genten::atomic_oper_fetch(op, x+threadIdx.x, v0);
      if (sz.value > 1)
        c.v1 = Genten::atomic_oper_fetch(op, x+WarpDim+threadIdx.x, v1);
      if (sz.value > 2)
        c.v2 = Genten::atomic_oper_fetch(op, x+2*WarpDim+threadIdx.x, v2);
      return c;
    }

    __device__ inline
    TinyVec& operator+=(const TinyVec& x) {
      v0 += x.v0;
      v1 += x.v1;
      v2 += x.v2;
      return *this;
    }

    __device__ inline
    TinyVec& operator-=(const TinyVec& x) {
      v0 -= x.v0;
      v1 -= x.v1;
      v2 -= x.v2;
      return *this;
    }

    __device__ inline
    TinyVec& operator*=(const TinyVec& x) {
      v0 *= x.v0;
      v1 *= x.v1;
      v2 *= x.v2;
      return *this;
    }

    __device__ inline
    TinyVec& operator/=(const TinyVec& x) {
      v0 /= x.v0;
      v1 /= x.v1;
      v2 /= x.v2;
      return *this;
    }

    template <typename Op>
    __device__ inline
    void apply_unary(const Op& op, const TinyVec& x) {
      v0 = op(x.v0);
      v1 = op(x.v1);
      v2 = op(x.v2);
    }

    __device__ inline
    TinyVec& operator+=(const scalar_type x) {
      v0 += x;
      v1 += x;
      v2 += x;
      return *this;
    }

    __device__ inline
    TinyVec& operator-=(const scalar_type x) {
      v0 -= x;
      v1 -= x;
      v2 -= x;
      return *this;
    }

    __device__ inline
    TinyVec& operator*=(const scalar_type x) {
      v0 *= x;
      v1 *= x;
      v2 *= x;
      return *this;
    }

    __device__ inline
    TinyVec& operator/=(const scalar_type x) {
      v0 /= x;
      v1 /= x;
      v2 /= x;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator+=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 += x.v0;
      if (x.sz.value > 1) v1 += x.v1;
      if (x.sz.value > 2) v2 += x.v2;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator-=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 -= x.v0;
      if (x.sz.value > 1) v1 -= x.v1;
      if (x.sz.value > 2) v2 -= x.v2;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator*=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 *= x.v0;
      if (x.sz.value > 1) v1 *= x.v1;
      if (x.sz.value > 2) v2 *= x.v2;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator/=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 /= x.v0;
      if (x.sz.value > 1) v1 /= x.v1;
      if (x.sz.value > 2) v2 /= x.v2;
      return *this;
    }

    __device__ inline
    TinyVec& operator+=(const scalar_type* x) {
      if (sz.value > 0) v0 += x[threadIdx.x];
      if (sz.value > 1) v1 += x[WarpDim + threadIdx.x];
      if (sz.value > 2) v2 += x[2*WarpDim + threadIdx.x];
      return *this;
    }

    __device__ inline
    TinyVec& operator-=(const scalar_type* x) {
      if (sz.value > 0) v0 -= x[threadIdx.x];
      if (sz.value > 1) v1 -= x[WarpDim + threadIdx.x];
      if (sz.value > 2) v2 -= x[2*WarpDim + threadIdx.x];
      return *this;
    }

    __device__ inline
    TinyVec& operator*=(const scalar_type* x) {
      if (sz.value > 0) v0 *= x[threadIdx.x];
      if (sz.value > 1) v1 *= x[WarpDim + threadIdx.x];
      if (sz.value > 2) v2 *= x[2*WarpDim + threadIdx.x];
      return *this;
    }

    __device__ inline
    TinyVec& operator/=(const scalar_type* x) {
      if (sz.value > 0) v0 /= x[threadIdx.x];
      if (sz.value > 1) v1 /= x[WarpDim + threadIdx.x];
      if (sz.value > 2) v2 /= x[2*WarpDim + threadIdx.x];
      return *this;
    }

    __device__ inline
    scalar_type sum() const {
      scalar_type s = 0.0;
      if (sz.value > 0) s += v0;
      if (sz.value > 1) s += v1;
      if (sz.value > 2) s += v2;
      s = Impl::warpReduce(s, WarpDim);
      return s;
    }

  };

  // Specialization for Cuda or HIP where Length / WarpDim == 4.  Store the vector
  // components in register space since Cuda or HIP may store them in global memory
  // (especially in the dynamically sized case).
  template <typename Scalar, typename Ordinal,
            unsigned Length, unsigned Size, unsigned WarpDim>
  class TinyVec< Kokkos_GPU_Space,Scalar,Ordinal,Length,Size,WarpDim,
                 typename std::enable_if<Length/WarpDim == 4>::type >
  {
  public:

    typedef Kokkos_GPU_Space exec_space;
    typedef Scalar scalar_type;
    typedef Ordinal ordinal_type;

    static const ordinal_type len = 4;  // = Length/WarpDim
    Kokkos::Impl::integral_nonzero_constant<ordinal_type,Size/WarpDim> sz;
    scalar_type v0, v1, v2, v3;

    KOKKOS_DEFAULTED_DEVICE_FUNCTION
    TinyVec() = default;

    __device__ inline
    TinyVec(const ordinal_type size, const scalar_type x)
      : sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
    {
      broadcast(x);
    }

    __device__ inline
    TinyVec(const ordinal_type size, const scalar_type* x)
      : sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
    {
      load(x);
    }

    __device__ inline
    TinyVec(const TinyVec& x) : sz(x.sz.value) {
      v0 = x.v0; v1 = x.v1; v2 = x.v2; v3 = x.v3;
    }

    KOKKOS_DEFAULTED_DEVICE_FUNCTION
    ~TinyVec() = default;

    __device__ inline
    TinyVec& operator=(const TinyVec& x) {
      v0 = x.v0; v1 = x.v1; v2 = x.v2; v3 = x.v3;
      return *this;
    }

    __device__ inline
    TinyVec& operator=(const scalar_type x) {
      broadcast(x);
      return *this;
    }

    __device__ inline
    TinyVec& operator=(const scalar_type* x) {
      load(x);
      return *this;
    }

    __device__ inline
    constexpr ordinal_type size() const { return sz.value; }

    __device__ inline
    void broadcast(const scalar_type x) {
      v0 = v1 = v2 = v3 = x;
    }

    __device__ inline
    void load(const scalar_type* x) {
      if (sz.value > 0) v0 = x[threadIdx.x];
      if (sz.value > 1) v1 = x[WarpDim + threadIdx.x];
      if (sz.value > 2) v2 = x[2*WarpDim + threadIdx.x];
      if (sz.value > 3) v3 = x[3*WarpDim + threadIdx.x];
    }

    __device__ inline
    void store(scalar_type* x) const {
      if (sz.value > 0) x[threadIdx.x] = v0;
      if (sz.value > 1) x[WarpDim + threadIdx.x] = v1;
      if (sz.value > 2) x[2*WarpDim + threadIdx.x] = v2;
      if (sz.value > 3) x[3*WarpDim + threadIdx.x] = v3;
    }

    __device__ inline
    void store_plus(scalar_type* x) const {
      if (sz.value > 0) x[threadIdx.x] += v0;
      if (sz.value > 1) x[WarpDim + threadIdx.x] += v1;
      if (sz.value > 2) x[2*WarpDim + threadIdx.x] += v2;
      if (sz.value > 3) x[3*WarpDim + threadIdx.x] += v3;
    }

    __device__ inline
    void atomic_store_plus(volatile scalar_type* x) const {
      if (sz.value > 0) Kokkos::atomic_add(x+threadIdx.x, v0);
      if (sz.value > 1) Kokkos::atomic_add(x+WarpDim+threadIdx.x, v1);
      if (sz.value > 2) Kokkos::atomic_add(x+2*WarpDim+threadIdx.x, v2);
      if (sz.value > 3) Kokkos::atomic_add(x+3*WarpDim+threadIdx.x, v3);
    }

    __device__ inline
    TinyVec atomic_exchange(volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
      if (sz.value > 0)
        c.v0 = Kokkos::atomic_exchange(x+threadIdx.x, v0);
      if (sz.value > 1)
        c.v1 = Kokkos::atomic_exchange(x+WarpDim+threadIdx.x, v1);
      if (sz.value > 2)
        c.v2 = Kokkos::atomic_exchange(x+2*WarpDim+threadIdx.x, v2);
      if (sz.value > 3)
        c.v3 = Kokkos::atomic_exchange(x+3*WarpDim+threadIdx.x, v3);
      return c;
    }

    __device__ inline
    TinyVec atomic_fetch_max(volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
      if (sz.value > 0)
        c.v0 = Kokkos::atomic_fetch_max(x+threadIdx.x, v0);
      if (sz.value > 1)
        c.v1 = Kokkos::atomic_fetch_max(x+WarpDim+threadIdx.x, v1);
      if (sz.value > 2)
        c.v2 = Kokkos::atomic_fetch_max(x+2*WarpDim+threadIdx.x, v2);
      if (sz.value > 3)
        c.v3 = Kokkos::atomic_fetch_max(x+3*WarpDim+threadIdx.x, v3);
      return c;
    }

    __device__ inline
    TinyVec atomic_fetch_min(volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
      if (sz.value > 0)
        c.v0 = Kokkos::atomic_fetch_min(x+threadIdx.x, v0);
      if (sz.value > 1)
        c.v1 = Kokkos::atomic_fetch_min(x+WarpDim+threadIdx.x, v1);
      if (sz.value > 2)
        c.v2 = Kokkos::atomic_fetch_min(x+2*WarpDim+threadIdx.x, v2);
      if (sz.value > 3)
        c.v3 = Kokkos::atomic_fetch_min(x+3*WarpDim+threadIdx.x, v3);
      return c;
    }

    template <typename Oper>
    __device__ inline
    TinyVec atomic_oper_fetch(const Oper& op, volatile scalar_type* x) const {
      TinyVec c(sz.value, 0.0);
      if (sz.value > 0)
        c.v0 = Genten::atomic_oper_fetch(op, x+threadIdx.x, v0);
      if (sz.value > 1)
        c.v1 = Genten::atomic_oper_fetch(op, x+WarpDim+threadIdx.x, v1);
      if (sz.value > 2)
        c.v2 = Genten::atomic_oper_fetch(op, x+2*WarpDim+threadIdx.x, v2);
      if (sz.value > 3)
        c.v3 = Genten::atomic_oper_fetch(op, x+3*WarpDim+threadIdx.x, v3);
      return c;
    }

    __device__ inline
    TinyVec& operator+=(const TinyVec& x) {
      v0 += x.v0;
      v1 += x.v1;
      v2 += x.v2;
      v3 += x.v3;
      return *this;
    }

    __device__ inline
    TinyVec& operator-=(const TinyVec& x) {
      v0 -= x.v0;
      v1 -= x.v1;
      v2 -= x.v2;
      v3 -= x.v3;
      return *this;
    }

    __device__ inline
    TinyVec& operator*=(const TinyVec& x) {
      v0 *= x.v0;
      v1 *= x.v1;
      v2 *= x.v2;
      v3 *= x.v3;
      return *this;
    }

    __device__ inline
    TinyVec& operator/=(const TinyVec& x) {
      v0 /= x.v0;
      v1 /= x.v1;
      v2 /= x.v2;
      v3 /= x.v3;
      return *this;
    }

    template <typename Op>
    __device__ inline
    void apply_unary(const Op& op, const TinyVec& x) {
      v0 = op(x.v0);
      v1 = op(x.v1);
      v2 = op(x.v2);
      v3 = op(x.v3);
    }

    __device__ inline
    TinyVec& operator+=(const scalar_type x) {
      v0 += x;
      v1 += x;
      v2 += x;
      v3 += x;
      return *this;
    }

    __device__ inline
    TinyVec& operator-=(const scalar_type x) {
      v0 -= x;
      v1 -= x;
      v2 -= x;
      v3 -= x;
      return *this;
    }

    __device__ inline
    TinyVec& operator*=(const scalar_type x) {
      v0 *= x;
      v1 *= x;
      v2 *= x;
      v3 *= x;
      return *this;
    }

    __device__ inline
    TinyVec& operator/=(const scalar_type x) {
      v0 /= x;
      v1 /= x;
      v2 /= x;
      v3 /= x;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator+=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 += x.v0;
      if (x.sz.value > 1) v1 += x.v1;
      if (x.sz.value > 2) v2 += x.v2;
      if (x.sz.value > 3) v3 += x.v3;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator-=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 -= x.v0;
      if (x.sz.value > 1) v1 -= x.v1;
      if (x.sz.value > 2) v2 -= x.v2;
      if (x.sz.value > 3) v3 -= x.v3;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator*=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 *= x.v0;
      if (x.sz.value > 1) v1 *= x.v1;
      if (x.sz.value > 2) v2 *= x.v2;
      if (x.sz.value > 3) v3 *= x.v3;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator/=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 /= x.v0;
      if (x.sz.value > 1) v1 /= x.v1;
      if (x.sz.value > 2) v2 /= x.v2;
      if (x.sz.value > 3) v3 /= x.v3;
      return *this;
    }

    __device__ inline
    TinyVec& operator+=(const scalar_type* x) {
      if (sz.value > 0) v0 += x[threadIdx.x];
      if (sz.value > 1) v1 += x[WarpDim + threadIdx.x];
      if (sz.value > 2) v2 += x[2*WarpDim + threadIdx.x];
      if (sz.value > 3) v3 += x[3*WarpDim + threadIdx.x];
      return *this;
    }

    __device__ inline
    TinyVec& operator-=(const scalar_type* x) {
      if (sz.value > 0) v0 -= x[threadIdx.x];
      if (sz.value > 1) v1 -= x[WarpDim + threadIdx.x];
      if (sz.value > 2) v2 -= x[2*WarpDim + threadIdx.x];
      if (sz.value > 3) v3 -= x[3*WarpDim + threadIdx.x];
      return *this;
    }

    __device__ inline
    TinyVec& operator*=(const scalar_type* x) {
      if (sz.value > 0) v0 *= x[threadIdx.x];
      if (sz.value > 1) v1 *= x[WarpDim + threadIdx.x];
      if (sz.value > 2) v2 *= x[2*WarpDim + threadIdx.x];
      if (sz.value > 3) v3 *= x[3*WarpDim + threadIdx.x];
      return *this;
    }

    __device__ inline
    TinyVec& operator/=(const scalar_type* x) {
      if (sz.value > 0) v0 /= x[threadIdx.x];
      if (sz.value > 1) v1 /= x[WarpDim + threadIdx.x];
      if (sz.value > 2) v2 /= x[2*WarpDim + threadIdx.x];
      if (sz.value > 3) v3 /= x[3*WarpDim + threadIdx.x];
      return *this;
    }

    __device__ inline
    scalar_type sum() const {
      scalar_type s = 0.0;
      if (sz.value > 0) s += v0;
      if (sz.value > 1) s += v1;
      if (sz.value > 2) s += v2;
      if (sz.value > 3) s += v3;
      s = Impl::warpReduce(s, WarpDim);
      return s;
    }

  };

#endif

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  TinyVec<E,Sc,O,L,S,W>
  operator+(const TinyVec<E,Sc,O,L,S,W>& a, const TinyVec<E,Sc,O,L,S,W>& b)
  {
    TinyVec<E,Sc,O,L,S,W> c(a);
    c += b;
    return c;
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  TinyVec<E,Sc,O,L,S,W>
  operator+(const Sc& a, const TinyVec<E,Sc,O,L,S,W>& b)
  {
    TinyVec<E,Sc,O,L,S,W> c(b.size(), a);
    c += b;
    return c;
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  TinyVec<E,Sc,O,L,S,W>
  operator+(const TinyVec<E,Sc,O,L,S,W>& a, const Sc& b)
  {
    TinyVec<E,Sc,O,L,S,W> c(a);
    c += b;
    return c;
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  TinyVec<E,Sc,O,L,S,W>
  operator-(const TinyVec<E,Sc,O,L,S,W>& a, const TinyVec<E,Sc,O,L,S,W>& b)
  {
    TinyVec<E,Sc,O,L,S,W> c(a);
    c -= b;
    return c;
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  TinyVec<E,Sc,O,L,S,W>
  operator-(const Sc& a, const TinyVec<E,Sc,O,L,S,W>& b)
  {
    TinyVec<E,Sc,O,L,S,W> c(b.size(), a);
    c -= b;
    return c;
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  TinyVec<E,Sc,O,L,S,W>
  operator-(const TinyVec<E,Sc,O,L,S,W>& a, const Sc& b)
  {
    TinyVec<E,Sc,O,L,S,W> c(a);
    c -= b;
    return c;
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  TinyVec<E,Sc,O,L,S,W>
  operator*(const TinyVec<E,Sc,O,L,S,W>& a, const TinyVec<E,Sc,O,L,S,W>& b)
  {
    TinyVec<E,Sc,O,L,S,W> c(a);
    c *= b;
    return c;
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  TinyVec<E,Sc,O,L,S,W>
  operator*(const Sc& a, const TinyVec<E,Sc,O,L,S,W>& b)
  {
    TinyVec<E,Sc,O,L,S,W> c(b.size(), a);
    c *= b;
    return c;
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  TinyVec<E,Sc,O,L,S,W>
  operator*(const TinyVec<E,Sc,O,L,S,W>& a, const Sc& b)
  {
    TinyVec<E,Sc,O,L,S,W> c(a);
    c *= b;
    return c;
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  TinyVec<E,Sc,O,L,S,W>
  operator/(const TinyVec<E,Sc,O,L,S,W>& a, const TinyVec<E,Sc,O,L,S,W>& b)
  {
    TinyVec<E,Sc,O,L,S,W> c(a);
    c /= b;
    return c;
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  TinyVec<E,Sc,O,L,S,W>
  operator/(const Sc& a, const TinyVec<E,Sc,O,L,S,W>& b)
  {
    TinyVec<E,Sc,O,L,S,W> c(b.size(), a);
    c /= b;
    return c;
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  TinyVec<E,Sc,O,L,S,W>
  operator/(const TinyVec<E,Sc,O,L,S,W>& a, const Sc& b)
  {
    TinyVec<E,Sc,O,L,S,W> c(a);
    c /= b;
    return c;
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  TinyVec<E,Sc,O,L,S,W>
  sqrt(const TinyVec<E,Sc,O,L,S,W>& a)
  {
    using std::sqrt;
    TinyVec<E,Sc,O,L,S,W> c(a);
    c.apply_unary([&](const Sc& x) { return sqrt(x); }, a);
    return c;
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  std::ostream& operator << (std::ostream& os, const TinyVec<E,Sc,O,L,S,W>& a)
  {
    a.print(os);
    return os;
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W,
            typename Oper>
  KOKKOS_INLINE_FUNCTION
  Genten::TinyVec<E,Sc,O,L,S,W>
  atomic_oper_fetch(const Oper& op, volatile Sc* x,
                    const Genten::TinyVec<E,Sc,O,L,S,W>& tv)
  {
    return tv.atomic_oper_fetch(op, x);
  }

}

namespace Kokkos {

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  void atomic_add(volatile Sc* x, const Genten::TinyVec<E,Sc,O,L,S,W>& tv)
  {
    tv.atomic_store_plus(x);
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  Genten::TinyVec<E,Sc,O,L,S,W>
  atomic_exchange(volatile Sc* x, const Genten::TinyVec<E,Sc,O,L,S,W>& tv)
  {
    return tv.atomic_exchange(x);
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  Genten::TinyVec<E,Sc,O,L,S,W>
  atomic_fetch_max(volatile Sc* x, const Genten::TinyVec<E,Sc,O,L,S,W>& tv)
  {
    return tv.atomic_fetch_max(x);
  }

  template <typename E, typename Sc, typename O,
            unsigned L, unsigned S, unsigned W>
  KOKKOS_INLINE_FUNCTION
  Genten::TinyVec<E,Sc,O,L,S,W>
  atomic_fetch_min(volatile Sc* x, const Genten::TinyVec<E,Sc,O,L,S,W>& tv)
  {
    return tv.atomic_fetch_min(x);
  }

}
