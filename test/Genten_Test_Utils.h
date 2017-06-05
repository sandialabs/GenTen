//@HEADER
// ************************************************************************
//     Genten Tensor Toolbox
//     Software package for tensor math by Sandia National Laboratories
//
// Sandia National Laboratories is a multimission laboratory managed
// and operated by National Technology and Engineering Solutions of Sandia,
// LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
// U.S. Department of Energy’s National Nuclear Security Administration under
// contract DE-NA0003525.
//
// Copyright 20XX, Sandia Corporation.
// ************************************************************************
//@HEADER


#include<string>
#include<vector>
#include "Genten_Util.h"


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
#define MAXABS(a,b) ((fabs(a) > fabs(b)) ? fabs(a) : fabs(b))
#define RELDIFF(a,b) (fabs(a-b)/MAXABS((ttb_real)1,MAXABS(a,b)))
#define EQ(a,b) (RELDIFF((ttb_real)a, (ttb_real)b) < MACHINE_EPSILON)
