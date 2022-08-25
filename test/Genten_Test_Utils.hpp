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


#include <string>
#include <vector>
#include <cmath>
#include "Genten_Util.hpp"


namespace Genten
{
  namespace Test
  {
    void initialize(const::std::string & msg, int infolevel = 0);
    void message(const std::string & msg, int line, const char * fname);
    bool assertcheck(bool test, const std::string & desc,
                     int line, const char * fname);
    void finalize();
    /* timing routines which only work on Linux
       void gettime(timeval * timer);
       double timediff(timeval start, timeval end);
    */

    int num_failed();

    inline ttb_real max_abs(const ttb_real a, const ttb_real b) {
      return std::fabs(a) > std::fabs(b) ? std::fabs(a) : std::fabs(b);
    }
    inline ttb_real rel_diff(const ttb_real a, const ttb_real b) {
      return std::fabs(a-b)/max_abs(ttb_real(1.0),max_abs(a,b));
    }
    inline bool float_eq(const ttb_real a, const ttb_real b) {
      return rel_diff(a, b) < ttb_real(10.0)*MACHINE_EPSILON;
    }
    inline bool float_eq(const ttb_real a, const ttb_real b, const ttb_real tol)
    {
      return rel_diff(a, b) < tol;
    }
  }
}

#define ASSERT(cond,msg) Genten::Test::assertcheck(cond,msg,__LINE__,__FILE__)
#define MESSAGE(msg) Genten::Test::message(msg,__LINE__,__FILE__)


/* It's useful to disable cerr when doing error testing.
   Amazingly, the /dev/null seems to work even in windows. */
#include <fstream>
#define SETUP_DISABLE_CERR std::ofstream file("/dev/null"); std::streambuf * strm_buffer;
#define DISABLE_CERR strm_buffer = std::cerr.rdbuf(); std::cerr.rdbuf(file.rdbuf());
#define REENABLE_CERR std::cerr.rdbuf (strm_buffer);

// For comparing real numbers
#define MAXABS(a,b) (Genten::Test::max_abs(a,b))
#define RELDIFF(a,b) (Genten::Test::rel_diff(a,b))
#define EQ(a,b) (Genten::Test::float_eq(a,b))
#define FLOAT_EQ(a,b,tol) (Genten::Test::float_eq(a,b,tol))
