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


#include <iostream>
#include <sstream>
#include "Genten_Test_Utils.hpp"

using namespace std;

namespace Genten
{
  namespace Test
  {
    int infolevel;
    std::vector<std::string> errors;
  }
}

void Genten::Test::initialize(const string & msg, int infolevel)
{
  cout << "----------------------------------------------------" << endl;
  cout << msg << endl;
  cout << "----------------------------------------------------" << endl;
  Genten::Test::errors.clear();
  Genten::Test::infolevel = infolevel;
}

bool Genten::Test::assertcheck(bool test,
                            const string & desc,
                            int line,
                            const char * fname)
{
  std::ostringstream oss;
  oss << desc << " (File: " << fname << ", Line: " << line << ")";

  if (Genten::Test::infolevel > 0)
  {
    if (test)
    {
      cout << "PASS: ";
    }
    else
    {
      cout << "FAIL: ";
    }
    cout << oss.str() << endl;
  }
  else
  {
    if (test)
    {
      cout << '.';
    }
    else
    {
      cout << 'X';
      Genten::Test::errors.push_back(oss.str());
    }

  }

  return(test);
}

void Genten::Test::message(const string & msg, int line, const char * fname)
{
  if (Genten::Test::infolevel > 0)
  {
    cout << "INFO: " << msg << " (File: " << fname << ", Line: " << line << ")" << endl;
  }
}



void Genten::Test::finalize()
{
  if (Genten::Test::infolevel == 0)
  {
    cout << endl;
    cout << endl;
    size_t n = Genten::Test::errors.size();
    for (size_t i = 0; i < n; i ++)
    {
      cout << "Error " << i+1 << " of " << n << ": " ;
      cout << Genten::Test::errors[i] << endl;
    }
    cout << endl;
  }
}

int Genten::Test::num_failed()
{
  return Genten::Test::errors.size();
}

/* timing routines which work on Linux
   void Genten::Test::gettime(timeval * time)
   {
   gettimeofday(time,NULL);
   }

   double Genten::Test::timediff(timeval start, timeval end)
   {
   double secs;
   struct timeval temp;

   if ( (end.tv_usec - start.tv_usec) < 0 )
   {
   temp.tv_sec  = end.tv_sec - start.tv_sec - 1.0;
   temp.tv_usec = 1.0e6 + end.tv_usec - start.tv_usec;
   }
   else
   {
   temp.tv_sec  = end.tv_sec - start.tv_sec;
   temp.tv_usec = end.tv_usec - start.tv_usec;
   }

   secs = (double) temp.tv_sec;
   secs = secs + ((double) temp.tv_usec)*1.0e-6;

   return secs;
   }
*/
