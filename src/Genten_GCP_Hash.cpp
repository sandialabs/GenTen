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

#include "Genten_GCP_Hash.hpp"

namespace Genten {

  namespace Impl {

    // MurmurHash3 was written by Austin Appleby, and is placed in the public
    // domain. The author hereby disclaims copyright to this source code.
    KOKKOS_FORCEINLINE_FUNCTION
    uint32_t getblock32 ( const uint32_t * p, int i )
    {
      return p[i];
    }

    KOKKOS_FORCEINLINE_FUNCTION
    uint32_t rotl32 ( uint32_t x, int8_t r )
    { return (x << r) | (x >> (32 - r)); }

    KOKKOS_FORCEINLINE_FUNCTION
    uint32_t fmix32 ( uint32_t h )
    {
      h ^= h >> 16;
      h *= 0x85ebca6b;
      h ^= h >> 13;
      h *= 0xc2b2ae35;
      h ^= h >> 16;
      return h;
    }

    // 128-bit version of MurmurHash3 -- see http://github.com/aappleby/smhasher
    // MurmurHash3 was written by Austin Appleby, and is placed in the public
    // domain. The author hereby disclaims copyright to this source code.
    void MurmurHash3_x86_128 ( const void * key, const int len,
                               uint32_t seed, void * out )
    {
      const uint8_t * data = (const uint8_t*)key;
      const int nblocks = len / 16;

      uint32_t h1 = seed;
      uint32_t h2 = seed;
      uint32_t h3 = seed;
      uint32_t h4 = seed;

      const uint32_t c1 = 0x239b961b;
      const uint32_t c2 = 0xab0e9789;
      const uint32_t c3 = 0x38b34ae5;
      const uint32_t c4 = 0xa1e38b93;

      //----------
      // body

      const uint32_t * blocks = (const uint32_t *)(data + nblocks*16);

      for(int i = -nblocks; i; i++)
      {
        uint32_t k1 = getblock32(blocks,i*4+0);
        uint32_t k2 = getblock32(blocks,i*4+1);
        uint32_t k3 = getblock32(blocks,i*4+2);
        uint32_t k4 = getblock32(blocks,i*4+3);

        k1 *= c1; k1  = rotl32(k1,15); k1 *= c2; h1 ^= k1;

        h1 = rotl32(h1,19); h1 += h2; h1 = h1*5+0x561ccd1b;

        k2 *= c2; k2  = rotl32(k2,16); k2 *= c3; h2 ^= k2;

        h2 = rotl32(h2,17); h2 += h3; h2 = h2*5+0x0bcaa747;

        k3 *= c3; k3  = rotl32(k3,17); k3 *= c4; h3 ^= k3;

        h3 = rotl32(h3,15); h3 += h4; h3 = h3*5+0x96cd1c35;

        k4 *= c4; k4  = rotl32(k4,18); k4 *= c1; h4 ^= k4;

        h4 = rotl32(h4,13); h4 += h1; h4 = h4*5+0x32ac3b17;
      }

      //----------
      // tail

      const uint8_t * tail = (const uint8_t*)(data + nblocks*16);

      uint32_t k1 = 0;
      uint32_t k2 = 0;
      uint32_t k3 = 0;
      uint32_t k4 = 0;

      switch(len & 15)
      {
      case 15: k4 ^= tail[14] << 16;
      case 14: k4 ^= tail[13] << 8;
      case 13: k4 ^= tail[12] << 0;
        k4 *= c4; k4  = rotl32(k4,18); k4 *= c1; h4 ^= k4;

      case 12: k3 ^= tail[11] << 24;
      case 11: k3 ^= tail[10] << 16;
      case 10: k3 ^= tail[ 9] << 8;
      case  9: k3 ^= tail[ 8] << 0;
        k3 *= c3; k3  = rotl32(k3,17); k3 *= c4; h3 ^= k3;

      case  8: k2 ^= tail[ 7] << 24;
      case  7: k2 ^= tail[ 6] << 16;
      case  6: k2 ^= tail[ 5] << 8;
      case  5: k2 ^= tail[ 4] << 0;
        k2 *= c2; k2  = rotl32(k2,16); k2 *= c3; h2 ^= k2;

      case  4: k1 ^= tail[ 3] << 24;
      case  3: k1 ^= tail[ 2] << 16;
      case  2: k1 ^= tail[ 1] << 8;
      case  1: k1 ^= tail[ 0] << 0;
           k1 *= c1; k1  = rotl32(k1,15); k1 *= c2; h1 ^= k1;
      };

      //----------
      // finalization

      h1 ^= len; h2 ^= len; h3 ^= len; h4 ^= len;

      h1 += h2; h1 += h3; h1 += h4;
      h2 += h1; h3 += h1; h4 += h1;

      h1 = fmix32(h1);
      h2 = fmix32(h2);
      h3 = fmix32(h3);
      h4 = fmix32(h4);

      h1 += h2; h1 += h3; h1 += h4;
      h2 += h1; h3 += h1; h4 += h1;

      ((uint32_t*)out)[0] = h1;
      ((uint32_t*)out)[1] = h2;
      ((uint32_t*)out)[2] = h3;
      ((uint32_t*)out)[3] = h4;
    }

  }

}
