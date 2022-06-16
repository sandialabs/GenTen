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
#include <time.h>
#include "Genten_Sptensor.hpp"
#include "Genten_Test_Utils.hpp"
#include "Genten_IOtext.hpp"

using namespace Genten::Test;

// These tests are just on the host, so not templated on the space
void Genten_Test_Sptensor(int infolevel)
{
  bool tf;

  initialize("Tests on Genten::Sptensor", infolevel);

  // EMPTY CONSTRUCTOR
  // a = []
  MESSAGE("Creating empty sparse tensor");
  Genten::Sptensor a;
  ASSERT(a.nnz() == 0, "Sparse tensor is empty");

  // LENGTH CONSTRUCTOR
  // b = []
  MESSAGE("Creating sparse tensor with a specified nnz of 0 and empty IndxArray");
  Genten::IndxArray dims;
  Genten::Sptensor b(dims,0);
  ASSERT(b.nnz() == 0, "Sparse tensor is empty");

  // c = [? ? ? ? ?]
  MESSAGE("Creating sparse tensor with a specified nnz and size");
  dims = Genten::IndxArray(3); dims[0] = 3; dims[1] = 4; dims[2] = 5;
  Genten::Sptensor c(dims,5);
  ASSERT(c.nnz() == 5, "Array has correct length");

  // DATA CONSTRUCTOR FOR MATLAB
  // nnz = 5, size = [5 2 4]
  // vls = [ 0 1 2 3 4 ]
  // sbs = [[1 1 1], [2 2 2], [3 1 3], [4 2 4], [5 1 1]]
  MESSAGE("Creating a sparse tensor from MATLAB format");
  ttb_indx nd = 3, nz = 5;
  ttb_real sz[3] = {5,2,4};
  ttb_real * vls = (ttb_real *) malloc(5 * sizeof(ttb_real));
  ttb_real * sbs = (ttb_real *) malloc(15 * sizeof(ttb_real));
  for (int i = 0; i < 5; i ++)
  {
    vls[i] = i;
    sbs[i] = i % 5 + 1;
    sbs[5+i] = i % 2 + 1;
    sbs[10+i] = i % 4 + 1;
  }
  Genten::Sptensor d(nd, sz, nz, vls, sbs);
  tf = true;
  if (d.nnz() != 5 || d.ndims() != 3)
  {
    tf = false;
  }
  for (ttb_indx i = 0; i < d.nnz(); i ++)
  {
    if (d.value(i) != i)
    {
      tf = false;
    }
    for (ttb_indx j = 0; j < d.ndims(); j ++)
    {
      if (d.subscript(i,j) != sbs[i+j*nz]-1)
      {
        tf = false;
      }
    }
  }
  ASSERT(tf, "Sparse tensor constructed as expected");
  free(vls);
  free(sbs);

  // DESTRUCTOR (but there is no real test here)
  // g (created and deleted)
  MESSAGE("Creating and freeing a sparse tensor");
  dims = Genten::IndxArray(3);
  Genten::Sptensor * g = new Genten::Sptensor(dims,10);
  delete g;

  /*
  // SORT
  MESSAGE("Creating a tensor with unordered entries and one duplicate and calling sort");
  dims = Genten::IndxArray(3); dims[0] = 5; dims[1] = 2; dims[2] = 4;
  Genten::Sptensor h(dims,4);
  h.subscript(0,0) = 2; h.subscript(0,1) = 0; h.subscript(0,2) = 1; h.value(0) = 1;
  h.subscript(1,0) = 1; h.subscript(1,1) = 1; h.subscript(1,2) = 2; h.value(1) = 1;
  h.subscript(2,0) = 4; h.subscript(2,1) = 0; h.subscript(2,2) = 3; h.value(2) = 3;
  h.subscript(3,0) = 2; h.subscript(3,1) = 0; h.subscript(3,2) = 1; h.value(3) = 1;
  h.cleanup();
  tf = true;
  if (h.nnz() != 3)
  {
    tf = false;
  }
  for (ttb_indx i = 0; i < h.nnz(); i ++)
  {
    if (h.value(i) != i+1)
    {
      tf = false;
    }
  }
  ASSERT(tf, "Sorting removes duplicates and orders as expected");

  // VERIFY COPYING FROM ASSIGNMENT OPERATOR
  MESSAGE("Making a copy of a tensor");
  Genten::Sptensor hcopy;
  hcopy = h;
  tf = true;
  if (h.value(2) != hcopy.value(2))
    tf = false;
  if (h.subscript(2,2) != hcopy.subscript(2,2))
    tf = false;
  h.value(2) = 333.3;
  if (h.value(2) == hcopy.value(2))
    tf = false;
  h.subscript(2,2) = 333;
  if (h.subscript(2,2) == hcopy.subscript(2,2))
    tf = false;
  ASSERT(tf, "Deep copy includes values and subscripts");
  */

  // SEARCH
  dims = Genten::IndxArray(3);
  dims[0] = 2;  dims[1] = 3;  dims[2] = 4;
  Genten::Sptensor X(dims,10);
  X.subscript(0,0) = 1;  X.subscript(0,1) = 0;  X.subscript(0,2) = 0;
  X.value(0) = 1.0;
  X.subscript(1,0) = 0;  X.subscript(1,1) = 1;  X.subscript(1,2) = 0;
  X.value(1) = 1.0;
  X.subscript(2,0) = 1;  X.subscript(2,1) = 1;  X.subscript(2,2) = 0;
  X.value(2) = 1.0;
  X.subscript(3,0) = 0;  X.subscript(3,1) = 2;  X.subscript(3,2) = 0;
  X.value(3) = 1.0;
  X.subscript(4,0) = 0;  X.subscript(4,1) = 0;  X.subscript(4,2) = 1;
  X.value(4) = 1.0;
  X.subscript(5,0) = 0;  X.subscript(5,1) = 2;  X.subscript(5,2) = 1;
  X.value(5) = 1.0;
  X.subscript(6,0) = 0;  X.subscript(6,1) = 0;  X.subscript(6,2) = 3;
  X.value(6) = 1.0;
  X.subscript(7,0) = 1;  X.subscript(7,1) = 0;  X.subscript(7,2) = 3;
  X.value(7) = 1.0;
  X.subscript(8,0) = 0;  X.subscript(8,1) = 1;  X.subscript(8,2) = 3;
  X.value(8) = 1.0;
  X.subscript(9,0) = 1;  X.subscript(9,1) = 1;  X.subscript(9,2) = 3;
  X.value(9) = 1.0;
  ASSERT(X.nnz() == 10, "Data tensor has 10 nonzeroes");

  MESSAGE("Testing unsorted tensor search");
  ASSERT(X.index(1, 0, 0) == 0, "Index found");
  ASSERT(X.index(1, 0, 0) == 0, "Index found");
  ASSERT(X.index(0, 1, 0) == 1, "Index found");
  ASSERT(X.index(1, 1, 0) == 2, "Index found");
  ASSERT(X.index(0, 2, 0) == 3, "Index found");
  ASSERT(X.index(0, 0, 1) == 4, "Index found");
  ASSERT(X.index(0, 2, 1) == 5, "Index found");
  ASSERT(X.index(0, 0, 3) == 6, "Index found");
  ASSERT(X.index(1, 0, 3) == 7, "Index found");
  ASSERT(X.index(0, 1, 3) == 8, "Index found");
  ASSERT(X.index(1, 1, 3) == 9, "Index found");
  ASSERT(X.index(0, 0, 0) == 10, "Index not found");
  ASSERT(X.index(0, 1, 2) == 10, "Index not found");
  ASSERT(X.index(1, 2, 3) == 10, "Index not found");
  ASSERT(X.index(3, 0, 0) == 10, "Index not found");

  X.sort();
  MESSAGE("Testing sorted tensor search");
  //std::cout << std::endl;
  //Genten::print_sptensor(X, std::cout, "Sorted X");
  ASSERT(X.index(0, 0, 1) == 0, "Index found");
  ASSERT(X.index(0, 0, 3) == 1, "Index found");
  ASSERT(X.index(0, 1, 0) == 2, "Index found");
  ASSERT(X.index(0, 1, 3) == 3, "Index found");
  ASSERT(X.index(0, 2, 0) == 4, "Index found");
  ASSERT(X.index(0, 2, 1) == 5, "Index found");
  ASSERT(X.index(1, 0, 0) == 6, "Index found");
  ASSERT(X.index(1, 0, 3) == 7, "Index found");
  ASSERT(X.index(1, 1, 0) == 8, "Index found");
  ASSERT(X.index(1, 1, 3) == 9, "Index found");
  ASSERT(X.index(0, 0, 0) == 10, "Index not found");
  ASSERT(X.index(0, 1, 2) == 10, "Index not found");
  ASSERT(X.index(1, 2, 3) == 10, "Index not found");
  ASSERT(X.index(3, 0, 0) == 10, "Index not found");

  finalize();
}
