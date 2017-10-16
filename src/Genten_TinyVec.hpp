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
#include <cmath>

// extern "C" {
// #include <immintrin.h>
// }

namespace Genten {

  template <typename Scalar, typename Ordinal,
            unsigned Length, unsigned Size, unsigned WarpDim,
            typename Tag = void,
            bool Nonzero = ( Size != Ordinal(0) )>
  class TinyVec {
  public:
    
    typedef Scalar scalar_type;
    typedef Ordinal ordinal_type;
    typedef Tag tag_type;
    
    static const ordinal_type len = Length / WarpDim;
    static const ordinal_type sz = Size / WarpDim;
    alignas(64) scalar_type v[len];

    KOKKOS_INLINE_FUNCTION
    TinyVec() = default;

    KOKKOS_INLINE_FUNCTION
    TinyVec(const ordinal_type size) {}

    KOKKOS_INLINE_FUNCTION
    TinyVec(const ordinal_type size, const scalar_type x) {
      for (ordinal_type i=0; i<sz; ++i)
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
      for (ordinal_type i=0; i<sz; ++i)
        v[i] = x;
    }

    KOKKOS_INLINE_FUNCTION
    void load(const scalar_type* x) {
#ifdef __CUDA_ARCH__
      for (ordinal_type i=0; i<sz; ++i)
        v[i] = x[i*WarpDim+threadIdx.x];
#else
      for (ordinal_type i=0; i<sz; ++i)
        v[i] = x[i];
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void store(scalar_type* x) const {
#ifdef __CUDA_ARCH__
      for (ordinal_type i=0; i<sz; ++i)
        x[i*WarpDim+threadIdx.x] = v[i];
#else
      for (ordinal_type i=0; i<sz; ++i)
        x[i] = v[i];
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void store_plus(scalar_type* x) const {
#ifdef __CUDA_ARCH__
      for (ordinal_type i=0; i<sz; ++i)
        x[i*WarpDim+threadIdx.x] += v[i];
#else
      for (ordinal_type i=0; i<sz; ++i)
        x[i] += v[i];
#endif
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator+=(const scalar_type x) {
      for (ordinal_type i=0; i<sz; ++i)
        v[i] += x;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator-=(const scalar_type x) {
      for (ordinal_type i=0; i<sz; ++i)
        v[i] -= x;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator*=(const scalar_type x) {
      for (ordinal_type i=0; i<sz; ++i)
        v[i] *= x;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator/=(const scalar_type x) {
      for (ordinal_type i=0; i<sz; ++i)
        v[i] /= x;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator+=(const TinyVec& x) {
      for (ordinal_type i=0; i<sz; ++i)
        v[i] += x.v[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator-=(const TinyVec& x) {
      for (ordinal_type i=0; i<sz; ++i)
        v[i] -= x.v[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator*=(const TinyVec& x) {
      for (ordinal_type i=0; i<sz; ++i)
        v[i] *= x.v[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator/=(const TinyVec& x) {
      for (ordinal_type i=0; i<sz; ++i)
        v[i] /= x.v[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Scalar sum() const {
      Scalar s(0.0);
      for (ordinal_type i=0; i<sz; ++i)
        s += v[i];
#ifdef __CUDA_ARCH__
      for (ordinal_type i=1; i<WarpDim; i*=2) {
        s += Kokkos::shfl_down(s, i, WarpDim);
      }
      s = Kokkos::shfl(s, 0, WarpDim);
#endif
      return s;
    }

  };

  // Specialization for dynamically sized array where Size == 0
  template <typename Scalar, typename Ordinal,
            unsigned Length, unsigned Size, unsigned WarpDim,
            typename Tag>
  class TinyVec<Scalar,Ordinal,Length,Size,WarpDim,Tag,false> {
  public:
    
    typedef Scalar scalar_type;
    typedef Ordinal ordinal_type;
    typedef Tag tag_type;
    
    static const ordinal_type len = Length / WarpDim;
    ordinal_type sz;
    alignas(64) scalar_type v[len];

    KOKKOS_INLINE_FUNCTION
    TinyVec() = default;

    KOKKOS_INLINE_FUNCTION
    TinyVec(const ordinal_type size) :
#ifdef __CUDA_ARCH__
      sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
#else
      sz(size)
#endif
    {}

    KOKKOS_INLINE_FUNCTION
    TinyVec(const ordinal_type size, const scalar_type x) :
#ifdef __CUDA_ARCH__
      sz( (size+WarpDim-1-threadIdx.x) / WarpDim )
#else
      sz(size)
#endif
    {
      for (ordinal_type i=0; i<sz; ++i)
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
      for (ordinal_type i=0; i<sz; ++i)
        v[i] = x;
    }

    KOKKOS_INLINE_FUNCTION
    void load(const scalar_type* x) {
#ifdef __CUDA_ARCH__
      for (ordinal_type i=0; i<sz; ++i)
        v[i] = x[i*WarpDim+threadIdx.x];
#else
      for (ordinal_type i=0; i<sz; ++i)
        v[i] = x[i];
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void store(scalar_type* x) const {
#ifdef __CUDA_ARCH__
      for (ordinal_type i=0; i<sz; ++i)
        x[i*WarpDim+threadIdx.x] = v[i];
#else
      for (ordinal_type i=0; i<sz; ++i)
        x[i] = v[i];
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void store_plus(scalar_type* x) const {
#ifdef __CUDA_ARCH__
      for (ordinal_type i=0; i<sz; ++i)
        x[i*WarpDim+threadIdx.x] += v[i];
#else
      for (ordinal_type i=0; i<sz; ++i)
        x[i] += v[i];
#endif
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator+=(const scalar_type x) {
      for (ordinal_type i=0; i<sz; ++i)
        v[i] += x;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator-=(const scalar_type x) {
      for (ordinal_type i=0; i<sz; ++i)
        v[i] -= x;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator*=(const scalar_type x) {
      for (ordinal_type i=0; i<sz; ++i)
        v[i] *= x;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator/=(const scalar_type x) {
      for (ordinal_type i=0; i<sz; ++i)
        v[i] /= x;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator+=(const TinyVec& x) {
      for (ordinal_type i=0; i<sz; ++i)
        v[i] += x.v[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator-=(const TinyVec& x) {
      for (ordinal_type i=0; i<sz; ++i)
        v[i] -= x.v[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator*=(const TinyVec& x) {
      for (ordinal_type i=0; i<sz; ++i)
        v[i] *= x.v[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator/=(const TinyVec& x) {
      for (ordinal_type i=0; i<sz; ++i)
        v[i] /= x.v[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Scalar sum() const {
      Scalar s(0.0);
      for (ordinal_type i=0; i<sz; ++i)
        s += v[i];
#ifdef __CUDA_ARCH__
      for (ordinal_type i=1; i<WarpDim; i*=2) {
        s += Kokkos::shfl_down(s, i, WarpDim);
      }
      s = Kokkos::shfl(s, 0, WarpDim);
#endif
      return s;
    }

  };

  template <typename Scalar, typename Ordinal,
            unsigned Length, unsigned Size, unsigned WarpDim,
            typename Tag>
  KOKKOS_INLINE_FUNCTION
  Genten::TinyVec<Scalar,Ordinal,Length,Size,WarpDim,Tag>
  abs(const Genten::TinyVec<Scalar,Ordinal,Length,Size,WarpDim,Tag>& x)
  {
    using std::abs;
    Genten::TinyVec<Scalar,Ordinal,Length,Size,WarpDim,Tag> y(x.sz);
    for (Ordinal i=0; i<x.sz; ++i)
      y.v[i] = abs(x.v[i]);
    return y;
  }
}

namespace Kokkos {

  template <typename Scalar, typename Ordinal,
            unsigned Length, unsigned Size, unsigned WarpDim,
            typename Tag>
  KOKKOS_INLINE_FUNCTION
  void atomic_add(
    volatile Scalar* x,
    const Genten::TinyVec<Scalar,Ordinal,Length,Size,WarpDim,Tag>& tv)
  {
#ifdef __CUDA_ARCH__
    for (Ordinal i=0; i<tv.sz; ++i)
      atomic_add(x+i*WarpDim+threadIdx.x, tv.v[i]);
#else
    for (Ordinal i=0; i<tv.sz; ++i)
      atomic_add(x+i, tv.v[i]);
#endif
  }

//   template <typename ExecSpace, typename Scalar, typename Ordinal,
//             unsigned Length>
//   KOKKOS_INLINE_FUNCTION
//   void atomic_add(
//     volatile Scalar* x,
//     const Genten::TinyVec<ExecSpace,Scalar,Ordinal,Length,Length,Length>& tv)
//   {
// #ifdef __CUDA_ARCH__
//     atomic_add(x+threadIdx.x, tv.v);
// #else
//     atomic_add(x, tv.v);
// #endif
//   }

  template <typename Scalar, typename Ordinal,
            unsigned Length, unsigned Size, unsigned WarpDim,
            typename Tag>
  struct reduction_identity< Genten::TinyVec<Scalar,Ordinal,Length,Size,WarpDim,Tag> >
  {
    typedef Genten::TinyVec<Scalar,Ordinal,Length,Size,WarpDim,Tag> scalar;
    typedef reduction_identity<Scalar> ris;
    KOKKOS_FORCEINLINE_FUNCTION static scalar sum()  {
      scalar x(Length);
      x.broadcast(0.0);
      return x;
    }
    KOKKOS_FORCEINLINE_FUNCTION static scalar prod() {
      scalar x(Length);
      x.broadcast(1.0);
      return x;
    }
    KOKKOS_FORCEINLINE_FUNCTION static scalar max()  {
      scalar x(Length);
      x.broadcast(ris::max());
      return x;
    }
    KOKKOS_FORCEINLINE_FUNCTION static scalar min()  {
      scalar x(Length);
      x.broadcast(ris::min());
      return x;
    }
  };

}

#if 0 && defined(__AVX__)

#if 1

namespace Genten {
  template <typename ExecSpace, typename Ordinal>
  class TinyVec<ExecSpace,double,Ordinal,16,16,true> {
  public:
    
    typedef Ordinal ordinal_type;
    typedef double scalar_type;
    typedef ExecSpace execution_space;
    
    static const ordinal_type len = 16;
    static const ordinal_type sz = 16;
    static const ordinal_type vec_len = 4;
    static const ordinal_type N = len / vec_len;
    __m256d v[N];

    KOKKOS_INLINE_FUNCTION
    TinyVec() = default;

    KOKKOS_INLINE_FUNCTION
    TinyVec(const ordinal_type size) {}

    KOKKOS_INLINE_FUNCTION
    TinyVec(const ordinal_type size, const scalar_type x) {
      for (ordinal_type i=0; i<N; ++i)
        v[i] = _mm256_set1_pd(x);
    }

    KOKKOS_INLINE_FUNCTION
    ~TinyVec() = default;

    KOKKOS_INLINE_FUNCTION
    TinyVec(const TinyVec&) = default;

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator=(const TinyVec&) = default;

    KOKKOS_INLINE_FUNCTION
    void broadcast(const scalar_type x) {
      for (ordinal_type i=0; i<N; ++i)
        v[i] = _mm256_set1_pd(x);
    }

    KOKKOS_INLINE_FUNCTION
    void load(const scalar_type* x) {
      for (ordinal_type i=0; i<N; ++i)
        v[i] = _mm256_load_pd(x+i*vec_len);
    }

    KOKKOS_INLINE_FUNCTION
    void store(scalar_type* x) const {
      for (ordinal_type i=0; i<N; ++i)
        _mm256_store_pd(x+i*vec_len, v[i]);
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator+=(const scalar_type x) {
      __m256d xv = _mm256_set1_pd(x);
      for (ordinal_type i=0; i<N; ++i)
        v[i] = _mm256_add_pd(v[i], xv);
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator-=(const scalar_type x) {
      __m256d xv = _mm256_set1_pd(x);
      for (ordinal_type i=0; i<N; ++i)
        v[i] = _mm256_sub_pd(v[i], xv);
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator*=(const scalar_type x) {
      __m256d xv = _mm256_set1_pd(x);
      for (ordinal_type i=0; i<N; ++i)
        v[i] = _mm256_mul_pd(v[i], xv);
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator/=(const scalar_type x) {
      __m256d xv = _mm256_set1_pd(x);
      for (ordinal_type i=0; i<N; ++i)
        v[i] = _mm256_div_pd(v[i], xv);
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator+=(const TinyVec& x) {
      for (ordinal_type i=0; i<N; ++i)
        v[i] = _mm256_add_pd(v[i], x.v[i]);
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator-=(const TinyVec& x) {
      for (ordinal_type i=0; i<N; ++i)
        v[i] = _mm256_sub_pd(v[i], x.v[i]);
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator*=(const TinyVec& x) {
      for (ordinal_type i=0; i<N; ++i)
        v[i] = _mm256_mul_pd(v[i], x.v[i]);
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator/=(const TinyVec& x) {
      for (ordinal_type i=0; i<N; ++i)
        v[i] = _mm256_div_pd(v[i], x.v[i]);
      return *this;
    }

  };
}

namespace Kokkos {
  template <typename ExecSpace, typename Ordinal>
  KOKKOS_INLINE_FUNCTION
  void atomic_add(
    volatile double* x,
    const Genten::TinyVec<ExecSpace,double,Ordinal,16,16>& tv)
  {
    for (Ordinal i=0; i<tv.N; ++i)
      for (Ordinal j=0; j<tv.vec_len; ++j)
        atomic_add(x+i*tv.vec_len+j, tv.v[i][j]);
  }
}

#else

namespace Genten {
  template <typename ExecSpace, typename Ordinal>
  class TinyVec<ExecSpace,double,Ordinal,16,16,true> {
  public:
    
    typedef Ordinal ordinal_type;
    typedef double scalar_type;
    typedef ExecSpace execution_space;
    
    static const ordinal_type len = 16;
    static const ordinal_type sz = 16;
    __m256d v1, v2, v3, v4;

    KOKKOS_INLINE_FUNCTION
    TinyVec() = default;

    KOKKOS_INLINE_FUNCTION
    TinyVec(const ordinal_type size) {}

    KOKKOS_INLINE_FUNCTION
    TinyVec(const ordinal_type size, const scalar_type x) {
      v1 = _mm256_set1_pd(x);
      v2 = _mm256_set1_pd(x);
      v3 = _mm256_set1_pd(x);
      v4 = _mm256_set1_pd(x);
    }

    KOKKOS_INLINE_FUNCTION
    ~TinyVec() = default;

    KOKKOS_INLINE_FUNCTION
    TinyVec(const TinyVec&) = default;

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator=(const TinyVec&) = default;

    KOKKOS_INLINE_FUNCTION
    void broadcast(const scalar_type x) {
      v1 = _mm256_set1_pd(x);
      v2 = _mm256_set1_pd(x);
      v3 = _mm256_set1_pd(x);
      v4 = _mm256_set1_pd(x);
    }

    KOKKOS_INLINE_FUNCTION
    void load(const scalar_type* x) {
      v1 = _mm256_load_pd(x);
      v2 = _mm256_load_pd(x+4);
      v3 = _mm256_load_pd(x+8);
      v4 = _mm256_load_pd(x+12);
    }

    KOKKOS_INLINE_FUNCTION
    void store(scalar_type* x) const {
      _mm256_store_pd(x, v1);
      _mm256_store_pd(x+4, v2);
      _mm256_store_pd(x+8, v3);
      _mm256_store_pd(x+12, v4);
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator+=(const scalar_type x) {
      __m256d xv = _mm256_set1_pd(x);
      v1 = _mm256_add_pd(v1, xv);
      v2 = _mm256_add_pd(v2, xv);
      v3 = _mm256_add_pd(v3, xv);
      v4 = _mm256_add_pd(v4, xv);
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator-=(const scalar_type x) {
      __m256d xv = _mm256_set1_pd(x);
      v1 = _mm256_sub_pd(v1, xv);
      v2 = _mm256_sub_pd(v2, xv);
      v3 = _mm256_sub_pd(v3, xv);
      v4 = _mm256_sub_pd(v4, xv);
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator*=(const scalar_type x) {
      __m256d xv = _mm256_set1_pd(x);
      v1 = _mm256_mul_pd(v1, xv);
      v2 = _mm256_mul_pd(v2, xv);
      v3 = _mm256_mul_pd(v3, xv);
      v4 = _mm256_mul_pd(v4, xv);
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator/=(const scalar_type x) {
      __m256d xv = _mm256_set1_pd(x);
      v1 = _mm256_div_pd(v1, xv);
      v2 = _mm256_div_pd(v2, xv);
      v3 = _mm256_div_pd(v3, xv);
      v4 = _mm256_div_pd(v4, xv);
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator+=(const TinyVec& x) {
      v1 = _mm256_add_pd(v1, x.v1);
      v2 = _mm256_add_pd(v2, x.v2);
      v3 = _mm256_add_pd(v3, x.v3);
      v4 = _mm256_add_pd(v4, x.v4);
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator-=(const TinyVec& x) {
      v1 = _mm256_sub_pd(v1, x.v1);
      v2 = _mm256_sub_pd(v2, x.v2);
      v3 = _mm256_sub_pd(v3, x.v3);
      v4 = _mm256_sub_pd(v4, x.v4);
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator*=(const TinyVec& x) {
      v1 = _mm256_mul_pd(v1, x.v1);
      v2 = _mm256_mul_pd(v2, x.v2);
      v3 = _mm256_mul_pd(v3, x.v3);
      v4 = _mm256_mul_pd(v4, x.v4);
    }

    KOKKOS_INLINE_FUNCTION
    TinyVec& operator/=(const TinyVec& x) {
      v1 = _mm256_div_pd(v1, x.v1);
      v2 = _mm256_div_pd(v2, x.v2);
      v3 = _mm256_div_pd(v3, x.v3);
      v4 = _mm256_div_pd(v4, x.v4);
    }

  };
}

namespace Kokkos {
  template <typename ExecSpace, typename Ordinal>
  void atomic_add(
    volatile double* x,
    const Genten::TinyVec<ExecSpace,double,Ordinal,16,16>& tv)
  {
    for (Ordinal i=0; i<4; ++i)
      atomic_add(x+i, tv.v1[i]);
    for (Ordinal i=0; i<4; ++i)
      atomic_add(x+i+4, tv.v2[i]);
    for (Ordinal i=0; i<4; ++i)
      atomic_add(x+i+8, tv.v3[i]);
    for (Ordinal i=0; i<4; ++i)
      atomic_add(x+i+12, tv.v4[i]);
  }
}
#endif

#endif
