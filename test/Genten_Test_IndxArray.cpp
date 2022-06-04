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


#include "Genten_IndxArray.hpp"
#include "Genten_Test_Utils.hpp"

using namespace Genten::Test;

// These tests are just on the host, so not templated on the space
void Genten_Test_IndxArray(int infolevel)
{
  //SETUP_DISABLE_CERR;

  bool tf;

  initialize("Tests on Genten::IndxArray", infolevel);

  // Empty constructor
  // a = []
  MESSAGE("Testing empty constructor");
  Genten::IndxArray a;
  ASSERT(a.empty(), "Empty constructor works as expected");

  // Constructor of length n
  // b = []
  MESSAGE("Testing Constructor of length n");
  Genten::IndxArray b(0);
  ASSERT(b.empty(), "Constructor of length n works for zero size");

  // Constructor of length n
  // c = [ ? ? ? ]
  MESSAGE("Testing Constructor of length n");
  Genten::IndxArray c(3);
  ASSERT(c.size() == 3, "Constructor of length n for positive size");

  // Copy constructor from array
  // d = [ 1 2 3 ]
  MESSAGE("Testing Copy constructor from array");
  ttb_indx ddata[] = {1, 2, 3};
  Genten::IndxArray d(3, ddata);
  ASSERT(d.size() == 3, "Size correct for Copy constructor from array");
  tf = true;
  for (int i = 0; i < 3; i ++)
  {
    if (d[i] != (i+1))
    {
      tf = false;
      break;
    }
  }
  ASSERT(tf, "Data correct for Copy constructor from array");

  // Copy constructor from double array
  // de= [ 0 1 2 ]
  MESSAGE("Testing Copy constructor from double array");
  ttb_indx edata[] = {0, 1, 2};
  Genten::IndxArray e(3, edata);
  ASSERT(e.size() == 3, "Size correct for Copy constructor from double array");
  tf = true;
  for (int i = 0; i < 3; i ++)
  {
    if (e[i] != i)
    {
      tf = false;
      break;
    }
  }
  ASSERT(tf, "Data correct for Copy constructor from double array");

  // Copy Constructor
  // f = [ 1 2 3 ]
  MESSAGE("Testing copy constructor");
  Genten::IndxArray f(d.size());
  deep_copy(f,d);
  ASSERT(f == d, "Copy constructor works");

  // Leave-one-out copy constructor
  // g = [1 3]
  MESSAGE("Testing leave-one-out copy constructor");
  Genten::IndxArray g(f,1);
  ASSERT(g.size() == (f.size() - 1), "Array size correct for leave-one-out copy constructor");
  ASSERT((g[0] == 1) && (g[1] == 3), "Entries correct for leave-one-out copy constructor");

  // Destructor (but there is no real test here)
  MESSAGE("Testing deletion of IndxArray");
  Genten::IndxArray * h = new Genten::IndxArray(d);
  delete h;

  // OPERATOR==
  ASSERT(d == f, "Operator== works as expected");
  ASSERT(!(a == c), "Operator== works as expected");
  ASSERT(!(e == d), "Operator== works as expected");

  // OPERATOR<=
  ASSERT(d <= f, "Operator<= works as expected");
  f[1] = d[1]+1;
  ASSERT(d <= f, "Operator<= works as expected");

  // EMPTY
  ASSERT(a.empty(), "Empty works as expected");
  ASSERT(b.empty(), "Empty works as expected");

  // SIZE
  ASSERT(a.size() == 0, "Size works as expected");
  ASSERT(b.size() == 0, "Size works as expected");
  ASSERT(c.size() == 3, "Size works as expected");
  ASSERT(d.size() == 3, "Size works as expected");
  ASSERT(e.size() == 3, "Size works as expected");
  ASSERT(f.size() == 3, "Size works as expected");
  ASSERT(g.size() == 2, "Size works as expected");

  // RESIZE
  b = Genten::IndxArray(5);
  ASSERT(b.size() == 5, "Resize works as expected");
  f = Genten::IndxArray(0);
  ASSERT(f.empty(), "Resize works as expected");

  // OPERATOR[] CONST
  const Genten::IndxArray i(e);
  tf = true;
  for (ttb_indx ii = 0; ii < i.size(); ii ++)
  {
    if (i[ii] != ii)
    {
      tf = false;
      break;
    }
  }
  ASSERT(tf, "Operator[] for const works as expected");

  // OPERATOR[] Non-Const
  for (ttb_indx i = 0; i < b.size(); i ++)
  {
    b[i] = i+5;
  }
  tf = true;
  for (ttb_indx i = 0; i < b.size(); i ++)
  {
    if (b[i] != i+5)
    {
      tf = false;
      break;
    }
  }
  ASSERT(tf, "Operator[] for non-const works as expected");

  // PROD
  ASSERT(b.prod() == 15120, "Prod works as expected");
  ASSERT(i.prod() == 0, "Prod works as expected");
  ASSERT(d.prod() == 6, "Prod works as expected");
  ASSERT(a.prod() == 0, "Prod works as expected");

  // PROD w/ default
  ASSERT(b.prod(1) == 15120, "Prod works as expected");
  ASSERT(i.prod(1) == 0, "Prod works as expected");
  ASSERT(d.prod(1) == 6, "Prod works as expected");
  ASSERT(a.prod(1) == 1, "Prod works as expected");

  // PROD w / start & end
  ASSERT(b.prod(1,5) == 3024, "Prod works as expected");
  ASSERT(b.prod(2,2) == 0, "Prod works as expected");

  // PROD w / start & end & default
  ASSERT(b.prod(2,2,1) == 1, "Prod works as expected");

  // CUMPROD
  c = Genten::IndxArray(b.size()+1);
  c.cumprod(b);
  ASSERT(c.size() == (b.size() + 1), "Size correct for cumprod");
  ttb_real tmp = 1;
  tf = true;
  for (ttb_indx i = 0; i < c.size(); i ++)
  {
    if (c[i] != tmp)
    {
      tf = false;
      break;
    }
    if (i < b.size())
    {
      tmp = tmp * b[i];
    }
  }
  ASSERT(tf, "Cumprod works as expected");

  // OPERATOR=
  a = c;
  ASSERT(a == c, "Operator= works as expected");

  // ZERO
  e.zero();
  tf = true;
  for (ttb_indx i = 0; i < e.size(); i ++)
  {
    if (e[i] != 0)
    {
      tf = false;
      break;
    }
  }
  ASSERT(tf, "zero works as expected");

  // ISPERMUTATION
  f = Genten::IndxArray(5);
  for (int i = 0; i < 5; i ++)
  {
    f[i] = 4-i;
  }
  ASSERT(f.isPermutation(), "ispermuation works as expected");
  ASSERT(!g.isPermutation(), "ispermuation works as expected");

  MESSAGE("Testing increment function");
  Genten::IndxArray dims(4), check(4), ii(4);
  dims[0] = 2; dims[1] = 3; dims[2] = 2; dims[3] = 4;
  ii[0] = 0; ii[1] = 0; ii[2] = 0; ii[3] = 0;
  ii.increment(dims);
  check[0] = 0; check[1] = 0; check[2] = 0; check[3] = 1;
  ASSERT(ii == check, "increment works as expected (a)");
  ii[0] = 0; ii[1] = 2; ii[2] = 1; ii[3] = 3;
  ii.increment(dims);
  check[0] = 1; check[1] = 0; check[2] = 0; check[3] = 0;
  ASSERT(ii == check, "increment works as expected (b)");
  ii[0] = 1; ii[1] = 2; ii[2] = 1; ii[3] = 3;
  ii.increment(dims);
  check[0] = 2; check[1] = 0; check[2] = 0; check[3] = 0;
  ASSERT(ii == check, "increment works as expected (c)");

  finalize();
}
