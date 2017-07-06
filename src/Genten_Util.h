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

/* ----- Standard Include Statements ----- */
#include <iostream>
#include <string>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstddef>

#include "Genten_Kokkos.h"

/* ----- Typedefs ----- */
/* We use typedefs to make the code portable, especially for
   varying sizes of integers, etc.
   Note that ttb_indx should always be an unsigned integer type.
*/
typedef double ttb_real;
//typedef float ttb_real;
typedef size_t ttb_indx;
//typedef unsigned ttb_indx;
typedef bool ttb_bool;


//---- Define machine epsilon for double precision numbers.
//---- Most compilers define DBL_EPSILON, but not all.
#if defined(DBL_EPSILON)
  #define MACHINE_EPSILON  DBL_EPSILON
#else
  #define MACHINE_EPSILON  2.220446049250313e-16
#endif

//---- Most compilers define DBL_MAX = 1.7976931348623158e+308.
//---- Most compilers define DBL_MIN = 2.2250738585072014e-308.
#if defined(DBL_MAX)
  #define DOUBLE_MAX  DBL_MAX
#else
  #define DOUBLE_MAX  1.0e+300
#endif
#if defined(DBL_MIN)
  #define DOUBLE_MIN  DBL_MIN
#else
  #define DOUBLE_MIN  1.0e-300
#endif


/* ----- Enums ----- */
namespace Genten {

  enum NormType {NormOne, NormTwo, NormInf};
}

/* ----- Utility Functions ----- */
namespace Genten {

  // Throw an error.
  void error(std::string s);

  class IndxArray; // Forward declaration

  // Subscipt to linear index conversion.
  /* Converts an n-way subscript for a tensor element into a linear
     index that can be used to index that data array. */
  //TBD should move this into Tensor file
  void sub2ind(const Genten::IndxArray & siz, const Genten::IndxArray & sub, ttb_indx & ind);

  // Linear index to subscript conversion.
  /* Converts an linear index that can be used to index the data array
     into an n-way subscript. */
  //TBD should move this into Tensor file
  void ind2sub(const Genten::IndxArray & siz, const ttb_indx ind, Genten::IndxArray & sub);

  //! Return true if two real numbers are equal to a specified tolerance.
  /*!
   *  Numbers are equal if
   *
   *  <pre>
   *         fabs(d1 - d2)
   *    --------------------------  < dTol .
   *    max(1, fabs(d1), fabs(d2))
   *  </pre>
   *
   *  @param d1    1st real number
   *  @param d2    2nd real number
   *  @param dTol  Relative tolerance, must be >= 0
   *  @return      true if d1 and d2 are within dTol
   */
  bool  isEqualToTol(ttb_real  d1,
                     ttb_real  d2,
                     ttb_real  dTol);

  // Return the offcial version for this release of the Tensor Toolbox.
  char *  getGentenVersion(void);

  // Connect executable to vtune for profiling
  void connect_vtune(const int p_rank = 0);

}
