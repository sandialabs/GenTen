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


#include "Genten_portability.hpp"

#if defined(_WIN32)
  #include "float.h"
#elif defined(__APPLE__)
//  #include "math.h"          //-- FOR isinf AND isnan
  #include <cmath>
#elif defined (__SUNPRO_CC)
  #include "ieeefp.h"        //-- FOR finite
#else
  #include "math.h"          //-- FOR isinf AND isnan
#endif


bool Genten::isRealValid (const ttb_real  d)
{
  //---- CMAKE IS NOT HELPFUL IN FINDING THESE FUNCTIONS, SO THEY
  //---- ARE SPECIFICALLY CODED FOR EACH PLATFORM.
  //---- IF SUITABLE PRIMITIVES CANNOT BE FOUND FOR A COMPILER,
  //---- THEN IT IS ACCEPTABLE TO ALWAYS RETURN true.

#if defined(_WIN32)
  if (_finite (d) == 0)
    return( false );
  if (d != d)
    return( false );

#elif defined(__APPLE__)
  if (std::isinf (d) != 0)
    return( false );
  if (std::isnan (d) != 0)
    return( false );

#elif defined (__SUNPRO_CC)
  if (finite (d) == 0)
    return( false );
  if (d != d)
    return( false );

#else
  if (isinf (d) != 0)
    return( false );
  if (isnan (d) != 0)
    return( false );

#endif

  return( true );
}
