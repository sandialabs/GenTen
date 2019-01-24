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

#include "Kokkos_UnorderedMap.hpp"

namespace Genten {
  namespace Impl {

    // Statically sized array for storing tensor indices as keys in UnorderedMap
    template <typename T, unsigned N>
    class Array {
    public:
      typedef T &                                 reference ;
      typedef typename std::add_const<T>::type &  const_reference ;
      typedef size_t                              size_type ;
      typedef ptrdiff_t                           difference_type ;
      typedef T                                   value_type ;
      typedef T *                                 pointer ;
      typedef typename std::add_const<T>::type *  const_pointer ;

      KOKKOS_INLINE_FUNCTION static constexpr size_type size() { return N ; }

      template< typename iType >
      KOKKOS_INLINE_FUNCTION
      reference operator[]( const iType & i ) { return x[i]; }

      template< typename iType >
      KOKKOS_INLINE_FUNCTION
      const_reference operator[]( const iType & i ) const { return x[i]; }

    private:
      T x[N];

    };

    // 128-bit version of MurmurHash3 -- see http://github.com/aappleby/smhasher
    // MurmurHash3 was written by Austin Appleby, and is placed in the public
    // domain. The author hereby disclaims copyright to this source code.
    void MurmurHash3_x86_128 ( const void * key, int len, uint32_t seed, void * out );

  }
}

/*
namespace Kokkos {

  // Specialization of pod_hash for Array<T,N> using 128-bit hash.  Doesn't
  // really work because __uint128_t doesn't work with Kokkos.
  // Note the default implementation uses 32-bit MurmurHash3
  template <typename TT, unsigned N>
  struct pod_hash< Genten::Impl::Array<TT,N> >
  {
    typedef Genten::Impl::Array<TT,N> T;
    typedef T argument_type;
    typedef T first_argument_type;
    typedef __uint128_t second_argument_type;
    typedef __uint128_t result_type;

    KOKKOS_FORCEINLINE_FUNCTION
    __uint128_t operator()(T const & t) const
    {
      __uint128_t out;
      Genten::Impl::MurmurHash3_x86_128( &t, sizeof(T), 0, &out);
      return out;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    __uint128_t operator()(T const & t, uint32_t seed) const
    {
      __uint128_t out;
      return Genten::Impl::MurmurHash3_x86_128( &t, sizeof(T), seed);
      return out;
    }
  };

}
*/
