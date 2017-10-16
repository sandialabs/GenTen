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

#include  "Genten_RandomMT.hpp"


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  File-global Data
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

static const int  N = 624;
static const int  M = 397;

#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */

static unsigned long  _mt[N];     //-- State vector
static int            _mti;


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++  Class Definition:  RandomMT
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//-----------------------------------------------------------------------------
//  Constructor
//-----------------------------------------------------------------------------
Genten::RandomMT::RandomMT (const unsigned long  nnSeed)
{
  _mti = N + 1;

  _mt[0]= nnSeed & 0xffffffffUL;
  for (_mti = 1; _mti < N; _mti++)
  {
    _mt[_mti] =
      (1812433253UL * (_mt[_mti - 1] ^ (_mt[_mti - 1] >> 30)) + _mti);
    _mt[_mti] &= 0xffffffffUL;
  }

  return;
}


//-----------------------------------------------------------------------------
//  Destructor
//-----------------------------------------------------------------------------
Genten::RandomMT::~RandomMT (void)
{
  return;
}


//-----------------------------------------------------------------------------
//  Method:  genrnd_int32
//-----------------------------------------------------------------------------
/** The core random number generator, called by the other methods.
 */
unsigned long  Genten::RandomMT::genrnd_int32 (void)
{
  static unsigned long  mag01[2]={0x0UL, MATRIX_A};
  unsigned long  y;

  if (_mti >= N) { /* generate N words at one time */
    int  kk;

    for (kk = 0; kk < N-M; kk++)
    {
      y = (_mt[kk] & UPPER_MASK) | (_mt[kk+1] & LOWER_MASK);
      _mt[kk] = _mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    for ( ; kk < N-1; kk++)
    {
      y = (_mt[kk] & UPPER_MASK) |(_mt[kk+1] & LOWER_MASK);
      _mt[kk] = _mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    y = (_mt[N-1] & UPPER_MASK) |(_mt[0] & LOWER_MASK);
    _mt[N-1] = _mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

    _mti = 0;
  }

  y = _mt[_mti++];

  /* Tempering */
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);

  return( y );
}


//-----------------------------------------------------------------------------
//  Method:  genrnd_double
//-----------------------------------------------------------------------------
double  Genten::RandomMT::genrnd_double (void)
{
  //---- DIVIDE BY 2^32 TO CONFORM WITH [0,1).
  return( genrnd_int32() * (1.0 / 4294967296.0) );
}


//-----------------------------------------------------------------------------
//  Method:  genrnd_doubleInclusive
//-----------------------------------------------------------------------------
double  Genten::RandomMT::genrnd_doubleInclusive (void)
{
  //---- DIVIDE BY 2^32 - 1 TO CONFORM WITH [0,1].
  return( genrnd_int32() * (1.0 / 4294967295.0) );
}


//-----------------------------------------------------------------------------
//  Method:  genMatlabMT
//-----------------------------------------------------------------------------
double  Genten::RandomMT::genMatlabMT (void)
{
  long  nn1 = genrnd_int32() >> 5;
  long  nn2 = genrnd_int32() >> 6;

  //---- USE 2^26 = 67,108,864 AND 2^53 = 9,007,199,254,740,992.
  double  dResult = (((double) nn1) * 67108864.0 + ((double) nn2))
    * (1.0 / 9007199254740992.0);

  return( dResult );
}
