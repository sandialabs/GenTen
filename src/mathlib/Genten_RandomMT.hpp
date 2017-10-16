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

/*!
  @file Genten_RandomMT.h
  @brief Class declaration for a Mersenne Twister random number generator.
*/


#ifndef GENTEN_RANDOMMT_H
#define GENTEN_RANDOMMT_H

namespace Genten
{


  //! Mersenne Twister random number generator.
  class RandomMT
  {
  public:

    //! Constructor.
    RandomMT (const unsigned long  nnSeed);

    //! Destructor.
    ~RandomMT (void);


    //! Return a uniform random number on the interval [0,0xffffffff].
    unsigned long  genrnd_int32 (void);

    //! Return a uniform random number on the interval [0,1).
    double genrnd_double (void);

    //! Return a uniform random number on the interval [0,1].
    double  genrnd_doubleInclusive (void);

    //! Return a uniform random number on the interval [0,1).
    /*!
     *  Calling this method generates the same stream of random samples as
     *  the following Matlab code:
     *    > rstrm = RandStream('mt19937ar', 'Seed', 1);
     *    > rand (rstrm, 1, 10);
     *
     *  Note that the C++ instance must be constructed with the same seed as
     *  Matlab, and the seed must be positive.
     *
     *  Matlab RandStream behavior was checked in versions 7.10 (2010a)
     *  and 8.1 (R2013a), and found to be in agreement to 16 or more digits.
     *  Unofficial documentation indicates Matlab uses the genrand_res53()
     *  method of Matsumoto and Nishimura.
     */
    double  genMatlabMT (void);


  private:

    //! By design, there is no copy constructor.
    RandomMT (const RandomMT &);
    //! By design, there is no assignment operator.
    RandomMT &  operator=(const RandomMT &);
  };


}          //-- namespace Genten

#endif     //-- GENTEN_RANDOMMT_H
