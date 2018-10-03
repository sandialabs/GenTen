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

#include "Genten_Kokkos.hpp"
#if defined(KOKKOS_ENABLE_CUDA)
#include "Cuda/Kokkos_Cuda_Vectorization.hpp"
#endif

namespace Genten {

  namespace Impl {

#ifdef __CUDA_ARCH__
    // Reduce y across the warp and broadcast to all lanes
    template <typename T, typename Ordinal>
     __device__ inline T warpReduce(T y, const Ordinal warp_size) {
      for (Ordinal i=1; i<warp_size; i*=2) {
        y += Kokkos::shfl_down(y, i, warp_size);
      }
      y = Kokkos::shfl(y, 0, warp_size);
      return y;
    }
#endif

  }

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

    KOKKOS_INLINE_FUNCTION
    TinyVec() = default;

    KOKKOS_INLINE_FUNCTION
    TinyVec(const ordinal_type size, const scalar_type x) :
#ifdef __CUDA_ARCH__
      sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
#else
      sz(size)
#endif
    {
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] = x;
    }

    KOKKOS_INLINE_FUNCTION
    ~TinyVec() = default;

    KOKKOS_INLINE_FUNCTION
    TinyVec(const TinyVec&) = default;

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator=(const TinyVec&) = default;

    KOKKOS_INLINE_FUNCTION
    void broadcast(const scalar_type x) {
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] = x;
    }

    KOKKOS_INLINE_FUNCTION
    void load(const scalar_type* x) {
#ifdef __CUDA_ARCH__
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] = x[i*WarpDim+threadIdx.x];
#else
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] = x[i];
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void store_plus(scalar_type* x) const {
#ifdef __CUDA_ARCH__
      for (ordinal_type i=0; i<sz.value; ++i)
        x[i*WarpDim+threadIdx.x] += v[i];
#else
      for (ordinal_type i=0; i<sz.value; ++i)
        x[i] += v[i];
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void atomic_store_plus(volatile scalar_type* x) const {
#ifdef __CUDA_ARCH__
      for (ordinal_type i=0; i<sz.value; ++i)
        Kokkos::atomic_add(x+i*WarpDim+threadIdx.x, v[i]);
#else
      for (ordinal_type i=0; i<sz.value; ++i)
        Kokkos::atomic_add(x+i, v[i]);
#endif
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator+=(const TinyVec& x) {
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] += x.v[i];
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

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator*=(const scalar_type x) {
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] *= x;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator*=(const scalar_type* x) {
#ifdef __CUDA_ARCH__
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] *= x[i*WarpDim+threadIdx.x];
#else
      for (ordinal_type i=0; i<sz.value; ++i)
        v[i] *= x[i];
#endif
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    scalar_type sum() const {
      scalar_type s = 0.0;
      for (ordinal_type i=0; i<sz.value; ++i)
        s += v[i];
#ifdef __CUDA_ARCH__
      s = Impl::warpReduce(s, WarpDim);
#endif
      return s;
    }

  };

#if defined(KOKKOS_HAVE_CUDA) && defined(__CUDA_ARCH__)

  // Specialization for Cuda where Length / WarpDim == 1.  Store the vector
  // components in register space since Cuda may store them in global memory
  // (especially in the dynamically sized case).
  template <typename Scalar, typename Ordinal,
            unsigned Length, unsigned Size, unsigned WarpDim>
  class TinyVec< Kokkos::Cuda,Scalar,Ordinal,Length,Size,WarpDim,
                 typename std::enable_if<Length/WarpDim == 1>::type >
  {
  public:

    typedef Kokkos::Cuda exec_space;
    typedef Scalar scalar_type;
    typedef Ordinal ordinal_type;

    static const ordinal_type len = 1; // = Length/WarpDim
    Kokkos::Impl::integral_nonzero_constant<ordinal_type,Size/WarpDim> sz;
    scalar_type v0;

    __device__ inline
    TinyVec() = default;

    __device__ inline
    TinyVec(const ordinal_type size, const scalar_type x)
      : sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
    {
      v0 = x;
    }

    __device__ inline
    ~TinyVec() = default;

    __device__ inline
    TinyVec(const TinyVec&) = default;

    __device__ inline
    TinyVec& operator=(const TinyVec&) = default;

    __device__ inline
    void broadcast(const scalar_type x) {
      v0 = x;
    }

    __device__ inline
    void load(const scalar_type* x) {
      if (sz.value > 0) v0 = x[threadIdx.x];
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
    TinyVec& operator+=(const TinyVec& x) {
      v0 += x.v0;
      return *this;
    }

    template <unsigned S>
    __device__ inline
    TinyVec& operator+=(
      const TinyVec<exec_space,Scalar,Ordinal,Length,S,WarpDim,void>& x) {
      if (x.sz.value > 0) v0 += x.v0;
      return *this;
    }

    __device__ inline
    TinyVec& operator*=(const scalar_type x) {
      v0 *= x;
      return *this;
    }

    __device__ inline
    TinyVec& operator*=(const scalar_type* x) {
      if (sz.value > 0) v0 *= x[threadIdx.x];
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

  // Specialization for Cuda where Length / WarpDim == 2.  Store the vector
  // components in register space since Cuda may store them in global memory
  // (especially in the dynamically sized case).
  template <typename Scalar, typename Ordinal,
            unsigned Length, unsigned Size, unsigned WarpDim>
  class TinyVec< Kokkos::Cuda,Scalar,Ordinal,Length,Size,WarpDim,
                 typename std::enable_if<Length/WarpDim == 2>::type >
  {
  public:

    typedef Kokkos::Cuda exec_space;
    typedef Scalar scalar_type;
    typedef Ordinal ordinal_type;

    static const ordinal_type len = 2;  // = Length/WarpDim
    Kokkos::Impl::integral_nonzero_constant<ordinal_type,Size/WarpDim> sz;
    scalar_type v0, v1;

    __device__ inline
    TinyVec() = default;

    __device__ inline
    TinyVec(const ordinal_type size, const scalar_type x)

      : sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
    {
      v0 = v1 = x;
    }

    __device__ inline
    ~TinyVec() = default;

    __device__ inline
    TinyVec(const TinyVec&) = default;

    __device__ inline
    TinyVec& operator=(const TinyVec&) = default;

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
    TinyVec& operator+=(const TinyVec& x) {
      v0 += x.v0;
      v1 += x.v1;
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

    __device__ inline
    TinyVec& operator*=(const scalar_type x) {
      v0 *= x;
      v1 *= x;
      return *this;
    }

    __device__ inline
    TinyVec& operator*=(const scalar_type* x) {
      if (sz.value > 0) v0 *= x[threadIdx.x];
      if (sz.value > 1) v1 *= x[WarpDim + threadIdx.x];
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

  // Specialization for Cuda where Length / WarpDim == 3.  Store the vector
  // components in register space since Cuda may store them in global memory
  // (especially in the dynamically sized case).
  template <typename Scalar, typename Ordinal,
            unsigned Length, unsigned Size, unsigned WarpDim>
  class TinyVec< Kokkos::Cuda,Scalar,Ordinal,Length,Size,WarpDim,
                 typename std::enable_if<Length/WarpDim == 3>::type >
  {
  public:

    typedef Kokkos::Cuda exec_space;
    typedef Scalar scalar_type;
    typedef Ordinal ordinal_type;

    static const ordinal_type len = 3;  // = Length/WarpDim
    Kokkos::Impl::integral_nonzero_constant<ordinal_type,Size/WarpDim> sz;
    scalar_type v0, v1, v2;

    __device__ inline
    TinyVec() = default;

    __device__ inline
    TinyVec(const ordinal_type size, const scalar_type x)

      : sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
    {
      v0 = v1 = v2 = x;
    }

    __device__ inline
    ~TinyVec() = default;

    __device__ inline
    TinyVec(const TinyVec&) = default;

    __device__ inline
    TinyVec& operator=(const TinyVec&) = default;

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
    TinyVec& operator+=(const TinyVec& x) {
      v0 += x.v0;
      v1 += x.v1;
      v2 += x.v2;
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

    __device__ inline
    TinyVec& operator*=(const scalar_type x) {
      v0 *= x;
      v1 *= x;
      v2 *= x;
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
    scalar_type sum() const {
      scalar_type s = 0.0;
      if (sz.value > 0) s += v0;
      if (sz.value > 1) s += v1;
      if (sz.value > 2) s += v2;
      s = Impl::warpReduce(s, WarpDim);
      return s;
    }

  };

  // Specialization for Cuda where Length / WarpDim == 4.  Store the vector
  // components in register space since Cuda may store them in global memory
  // (especially in the dynamically sized case).
  template <typename Scalar, typename Ordinal,
            unsigned Length, unsigned Size, unsigned WarpDim>
  class TinyVec< Kokkos::Cuda,Scalar,Ordinal,Length,Size,WarpDim,
                 typename std::enable_if<Length/WarpDim == 4>::type >
  {
  public:

    typedef Kokkos::Cuda exec_space;
    typedef Scalar scalar_type;
    typedef Ordinal ordinal_type;

    static const ordinal_type len = 4;  // = Length/WarpDim
    Kokkos::Impl::integral_nonzero_constant<ordinal_type,Size/WarpDim> sz;
    scalar_type v0, v1, v2, v3;

    __device__ inline
    TinyVec() = default;

    __device__ inline
    TinyVec(const ordinal_type size, const scalar_type x)

      : sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
    {
      v0 = v1 = v2 = v3 = x;
    }

    __device__ inline
    ~TinyVec() = default;

    __device__ inline
    TinyVec(const TinyVec&) = default;

    __device__ inline
    TinyVec& operator=(const TinyVec&) = default;

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
    TinyVec& operator+=(const TinyVec& x) {
      v0 += x.v0;
      v1 += x.v1;
      v2 += x.v2;
      v3 += x.v3;
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

    __device__ inline
    TinyVec& operator*=(const scalar_type x) {
      v0 *= x;
      v1 *= x;
      v2 *= x;
      v3 *= x;
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

}

namespace Kokkos {

  template <typename ExecSpace, typename Scalar, typename Ordinal,
            unsigned Length, unsigned Size, unsigned WarpDim,
            typename Enabled>
  KOKKOS_INLINE_FUNCTION
  void atomic_add(
    volatile Scalar* x,
    const Genten::TinyVec<ExecSpace,Scalar,Ordinal,Length,Size,WarpDim,Enabled>& tv)
  {
    tv.atomic_store_plus(x);
  }

}
