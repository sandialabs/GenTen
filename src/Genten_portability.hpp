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


#ifndef GENTEN_portability_h
#define GENTEN_portability_h

#include "Genten_Util.hpp"

using namespace std;

// Forward declarations.
void Genten_Test_TTM(int infolevel);
void Genten_Test_Array(int infolevel);
void Genten_Test_CpAls(int infolevel);
#ifdef HAVE_LBFGSB
void Genten_Test_CpOptLbfgsb(int infolevel);
#endif
#ifdef HAVE_ROL
void Genten_Test_CpOptRol(int infolevel);
#endif
void Genten_Test_FacMatrix(int infolevel, const string & dirname);
void Genten_Test_IndxArray(int infolevel);
void Genten_Test_IO(int infolevel, const string & dirname);
void Genten_Test_Ktensor(int infolevel);
void Genten_Test_MixedFormats(int infolevel);
void Genten_Test_Sptensor(int infolevel);
void Genten_Test_Tensor(int infolevel);
void Genten_MomentTensor(int infolevel);
#ifdef HAVE_GCP
// #ifdef HAVE_ROL
// void Genten_Test_GCP_Opt(int infolevel);
// #endif
void Genten_Test_GCP_SGD(int infolevel);
#endif

#ifdef __PGI
// for pgi RESTRICT requires --restrict on CXXFLAGS, CFLAGS
#define RESTRICT restrict
#endif

#ifndef RESTRICT
#ifdef __GNUG__
#define RESTRICT __restrict__
#else
// unknown compiler case. don't assume restrict support
#define RESTRICT
#endif // gnug
#endif // not previously seen
  // Genten_MomentTensor(infolevel);

  // std::cout << "Unit tests complete for " << Genten::getGentenVersion() << endl;


namespace Genten {
  // Return true if the number is valid (finite).
  /* Invalid numbers (infinity or NaN) are the result of invalid
   * arithmetic operations. */
  bool  isRealValid (const ttb_real  d);
}


#endif //  GENTEN_portability_h
