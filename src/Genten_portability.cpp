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


#include "Genten_portability.h"

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
