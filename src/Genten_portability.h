//@HEADER
// ************************************************************************
//     Genten Tensor Toolbox
//     Software package for tensor math by Sandia National Laboratories
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
// ************************************************************************
//@HEADER


#ifndef GENTEN_portability_h
#define GENTEN_portability_h

#include "Genten_Util.h"

#ifdef __INTEL_COMPILER
// for intel RESTRICT requires -restrict on CXXFLAGS, CFLAGS
#define RESTRICT restrict
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


namespace Genten {
  // Return true if the number is valid (finite).
  /* Invalid numbers (infinity or NaN) are the result of invalid
   * arithmetic operations. */
  bool  isRealValid (const ttb_real  d);
}


#endif //  GENTEN_portability_h
