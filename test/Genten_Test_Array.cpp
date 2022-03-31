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
#include <cmath>
#include "Genten_Array.hpp"
#include "Genten_Test_Utils.hpp"
#include "Genten_Util.hpp"

using namespace Genten::Test;

template <typename ExecSpace>
void Genten_Test_Array_Space(int infolevel)
{
  typedef ExecSpace exec_space;
  typedef Genten::DefaultHostExecutionSpace host_exec_space;

  //SETUP_DISABLE_CERR;

  bool tf;

  std::string space_name = Genten::SpaceProperties<exec_space>::name();
  initialize("Tests on Genten::Array (" + space_name + ")", infolevel);

  // EMPTY CONSTRUCTOR
  // a = []
  MESSAGE("Creating empty array");
  Genten::Array a;
  ASSERT(a.empty(), "Array is emtpy");

  // LENGTH CONSTRUCTOR
  // b = []
  MESSAGE("Creating Array with a specified length of 0");
  Genten::Array b(0);
  ASSERT(b.empty(), "Array is empty");

  // c = [? ? ? ? ?]
  MESSAGE("Creating Array with a specified length");
  Genten::Array c(5);
  ASSERT(c.size() == 5, "Array has correct length");

  // DATA CONSTRUCTOR W/ SHADOWING
  // d = [ 0 1 2 3 4 ] SHADOW
  MESSAGE("Creating an array that *shadows* existing ttb_real* data");
  ttb_real ddata[] = {0,1,2,3,4};
  Genten::Array d(5, ddata);
  ASSERT(d.size() == 5, "Array has correct length");
  tf = true;
  for (ttb_indx i = 0; i < 5; i ++)
  {
    if (d[i] != i)
    {
      tf = false;
      break;
    }
  }
  ASSERT(tf, "Data is correct");

  // d = [ -5 1 2 3 4 ] SHADOW
  ddata[0] = -5;
  ASSERT(d[0] == -5, "Entry is changed when shadowed data is changed");

  // d = [ 0 1 2 3 4 ] SHADOW
  ddata[0] = 0;

  // DATA CONSTRUCTOR W/O SHADOWING
  // e = [ 0 1 2 3 4 ]
  MESSAGE("Creating an array named that *copies* existing ttb_real* data");
  ttb_real * edata = (ttb_real *) malloc(5 * sizeof(ttb_real));
  for (int i = 0; i < 5; i ++)
  {
    edata[i] = i;
  }
  Genten::Array e(5, edata, false);
  free(edata);
  ASSERT(e == d, "Arrays are equal");

  // COPY CONSTRUCTOR
  // f = [ 0 1 2 3 4 ]
  MESSAGE("Creating an array that copies an existing array");
  Genten::Array f(d.size());
  deep_copy(f,d);
  ASSERT(f == d, "Arrays are equal");

  // f = [ -1 1 2 3 4 ]
  MESSAGE("Checking for deep copy, even against a shadow'd array");
  f[0] = -1;
  ASSERT(f[0] != d[0], "Deep copy successful");

  // f = [ 0 1 2 3 4 ]
  f[0] = 0;

  // DESTRUCTOR
  // g (created and deleted)
  MESSAGE("Creating and freeing an Array that *shadows* existing data");
  ttb_real * gdata = (ttb_real *) malloc(5 * sizeof(ttb_real));
  for (int i = 0; i < 5; i ++)
  {
    gdata[i] = i;
  }
  Genten::Array * g = new Genten::Array(5, gdata);
  delete g;
  ASSERT(gdata != 0, "Shadowed gdata is still non-null");

  // d = [ 1 2 3 ] *
  MESSAGE("Resizing shadow'd Array to less than its current length");
  d = Genten::Array(3);
  ASSERT(d.size() == 3, "Resized array has correct length");

  // OPERATOR= (Array input)
  // a = [ 1 2 3 ]
  MESSAGE("Testing operator= with Array");
  a = d;
  ASSERT(a == d, "Arrays are equal");

  // OPERATOR= (scalar input)
  // a = [0.5 0.5 0.5]
  MESSAGE("Testing operator= with scalar");
  a = 0.5;
  ASSERT(a.size() == 3, "Size of Array is correct");
  tf = true;
  for (int i = 0; i < 3; i ++)
  {
    if (a[i] != 0.5)
    {
      tf = false;
      break;
    }
  }
  ASSERT(tf, "All entries of Array are equal correct");

  // RESET
  MESSAGE("Testing reset");
  b = Genten::Array(3, ttb_real(0.5));
  ASSERT(a==b, "Reset worked as expected");
  e = Genten::Array(0, ttb_real(0.0));
  ASSERT(e.empty(), "Reset to empty worked as expected");

  // SIZE
  ASSERT(a.size() == 3, "Correct size reported, case 1");
  ASSERT(b.size() == 3, "Correct size reported, case 2");
  ASSERT(c.size() == 5, "Correct size reported, case 3");
  ASSERT(d.size() == 3, "Correct size reported, case 4");
  ASSERT(e.size() == 0, "Correct size reported, case 5");
  ASSERT(f.size() == 5, "Correct size reported, case 6");

  // EMPTY
  ASSERT(e.empty(), "Empty worked as expected");
  ASSERT(!f.empty(), "Empty worked as exptected");

  // OPERATOR[] for CONST
  // h = [ 1 2 3 4 5 ? ? ? ? ? ] CONST
  const Genten::Array h(f);
  ASSERT(h == f, "Const copy constructor worked");
  tf = true;
  for (int i = 0; i < 5; i ++)
  {
    if (h[i] != i)
    {
      tf = false;
      break;
    }
  }
  ASSERT(tf, "Operator[] works for const arrays");

  // OPERATOR[] for non-const
  for (ttb_indx i = 0; i < f.size(); i ++)
  {
    f[i] = (ttb_real) i;
  }
  tf = true;
  for (ttb_indx i = 0; i < f.size(); i ++)
  {
    if (f[i] != i)
    {
      tf = false;
      break;
    }
  }
  ASSERT(tf, "Operator[] works for non-const arrays");

  // NORM_TWO
  MESSAGE("Testing norm_two");
  a = Genten::Array(5);
  ttb_real ans = 0;
  for (ttb_indx i = 0; i < 5; i ++)
  {
    a[i] = i/11.0;
    ans += (i/11.0)*(i/11.0);
  }
  ans = sqrt(ans);
  Genten::ArrayT<exec_space> a_dev = create_mirror_view( exec_space(), a );
  deep_copy( a_dev, a );
  ASSERT( EQ(a_dev.norm(Genten::NormTwo), ans), "norm_two works as expected");


  // NORM_ONE
  // a = [0 -1/11 2/11 -3/11 4/11]
  MESSAGE("Testing norm_one");
  a = Genten::Array(5);
  ans = 0;
  for (ttb_indx i = 0; i < 5; i ++)
  {
    a[i] = pow((ttb_real)-1,(int)i)*i/11.0;
    ans += i/11.0;
  }
  a_dev = create_mirror_view( exec_space(), a );
  deep_copy( a_dev, a );
  ASSERT( EQ(a_dev.norm(Genten::NormOne), ans), "norm_one works as expected");

  // NORM_INF -- Not on device
  MESSAGE("Testing norm_inf");
  ans = 4.0/11.0;
  ASSERT( EQ(a.norm(Genten::NormInf), ans), "norm_inf works as expected");

  // NNZ
  ASSERT(a.nnz() == 4, "nnz works as expected");

  // DOT
  // b = [2 2 2 2 2]
  b = Genten::Array(5,ttb_real(2.0));
  ans = 0;
  for (ttb_indx i = 0; i < 5; i ++)
  {
    ans += 2 * a[i];
  }
  ASSERT( EQ(a.dot(b), ans), "dot works as expected");

  // EQUAL
  ASSERT(!b.isEqual(a,MACHINE_EPSILON), "equal in the false case");
  c = a;
  ASSERT(c.isEqual(a,MACHINE_EPSILON), "equal in the true case");

  // TIMES
  Genten::Array answ;
  a = Genten::Array(5, ttb_real(2.5));
  a.times(3);
  answ = Genten::Array(a.size(), ttb_real(7.5));
  ASSERT(a.isEqual(answ,MACHINE_EPSILON), "times with scalar argument");

  // INVERT
  a.invert(9.375);
  answ = Genten::Array(a.size(), ttb_real(1.25));
  ASSERT(a.isEqual(answ,MACHINE_EPSILON), "invert with scalar argument");

  // SHIFT
  a.shift(3);
  answ = Genten::Array(5, ttb_real(4.25));
  ASSERT(a.isEqual(answ,MACHINE_EPSILON), "shift with scalar argument");

  // POWER
  a.power(2);
  answ = Genten::Array(5, ttb_real(18.0625));
  ASSERT(a.isEqual(answ,MACHINE_EPSILON), "power with scalar argument");

  // TIMES
  b = Genten::Array(5, ttb_real(2.5));
  a.times(3, b);
  answ = Genten::Array(5, ttb_real(7.5));
  ASSERT(a.isEqual(answ,MACHINE_EPSILON), "times with two arguments");

  // INVERT
  b = a;
  a.invert(9.375, b);
  answ = Genten::Array(5, ttb_real(1.25));
  ASSERT(a.isEqual(answ,MACHINE_EPSILON), "a = 9.375 ./ b");

  // SHIFT
  b = a;
  a.shift(3, b);
  answ = Genten::Array(5, ttb_real(4.25));
  ASSERT(a.isEqual(answ,MACHINE_EPSILON), "shift with two arguments");

  // POWER
  b = a;
  a.power(2, b);
  answ = Genten::Array(5, ttb_real(18.0625));
  ASSERT(a.isEqual(answ,MACHINE_EPSILON), "power with two arguments");

  // PLUS
  a = Genten::Array(5, ttb_real(2.3));
  b = Genten::Array(5, ttb_real(2.5));
  a.plus(b);
  answ = Genten::Array(5, ttb_real(4.8));
  ASSERT(a.isEqual(answ, MACHINE_EPSILON), "plus with one input");

  // MINUS
  a = Genten::Array(5, ttb_real(2.3));
  a.minus(b);
  answ = Genten::Array(5, ttb_real(-0.2));
  ASSERT(a.isEqual(answ, MACHINE_EPSILON), "minus with one input");

  // TIMES
  a = Genten::Array(5, ttb_real(2.3));
  a_dev = create_mirror_view( exec_space(), a );
  Genten::ArrayT<exec_space> b_dev = create_mirror_view( exec_space(), b );
  deep_copy( a_dev, a );
  deep_copy( b_dev, b );
  a_dev.times(b_dev);
  deep_copy( a, a_dev );
  answ = Genten::Array(5, ttb_real(5.75));
  ASSERT(a.isEqual(answ, MACHINE_EPSILON), "a = a.* b");

  // DIVIDE
  a = Genten::Array(5, ttb_real(2.3));
  a.divide(b);
  answ = Genten::Array(5, ttb_real(.92));
  ASSERT(a.isEqual(answ, MACHINE_EPSILON), "a = a ./ b");

  // PLUS
  a = Genten::Array(5, ttb_real(2.3));
  b = Genten::Array(5, ttb_real(2.5));
  c.plus(a,b);
  answ = Genten::Array(5, ttb_real(4.8));
  ASSERT(c.isEqual(answ, MACHINE_EPSILON), "c = a + b");

  // MINUS
  c.minus(a,b);
  answ = Genten::Array(5, ttb_real(-0.2));
  ASSERT(c.isEqual(answ, MACHINE_EPSILON), "c = a - b");

  // TIMES
  c.times(a,b);
  answ = Genten::Array(5, ttb_real(5.75));
  ASSERT(c.isEqual(answ, MACHINE_EPSILON), "c = a .* b");

  // DIVIDE
  c.divide(a,b);
  answ = Genten::Array(5, ttb_real(.92));
  ASSERT(c.isEqual(answ, MACHINE_EPSILON), "c = a ./ b");

  finalize();
}

void Genten_Test_Array(int infolevel) {
#ifdef KOKKOS_ENABLE_CUDA
  Genten_Test_Array_Space<Kokkos::Cuda>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_HIP
  Genten_Test_Array_Space<Kokkos::Experimental::HIP>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  Genten_Test_Array_Space<Kokkos::OpenMP>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_THREADS
  Genten_Test_Array_Space<Kokkos::Threads>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_SERIAL
  Genten_Test_Array_Space<Kokkos::Serial>(infolevel);
#endif
}
