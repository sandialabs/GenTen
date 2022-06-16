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
#include "Genten_IOtext.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_Test_Utils.hpp"

using namespace Genten::Test;

// These tests are just on the host, so not templated on the space
void Genten_Test_Ktensor(int infolevel)
{
  bool tf;

  initialize("Tests on Genten::Ktensor", infolevel);

  // Test empty constructor.
  MESSAGE("Creating an empty Ktensor");
  Genten::Ktensor a;
  ASSERT((a.ncomponents() == 0) && (a.ndims() == 0),
         "Ktensor is empty");

  // Test constructor with arguments.
  MESSAGE("Creating a new Ktensor");
  ttb_indx  nc = 2;
  Genten::Ktensor b(nc, 3);
  ASSERT((b.ncomponents() == nc) && (b.ndims() == 3),
         "Ktensor has correct size");

  MESSAGE("Setting factor matrices with correct size");
  Genten::IndxArray dims = {1, 2, 3};
  Genten::Ktensor c(nc, 3, dims);
  ASSERT(c.isConsistent(), "Ktensor is consistent");
  ASSERT((c[0].nRows() == 1) && (c[0].nCols() == nc),
         "Ktensor factor 0 has correct size");
  ASSERT((c[1].nRows() == 2) && (c[1].nCols() == nc),
         "Ktensor factor 0 has correct size");
  ASSERT((c[2].nRows() == 3) && (c[2].nCols() == nc),
         "Ktensor factor 0 has correct size");

  // Test element access of weights and factors.
  MESSAGE("Setting factors and weights");
  c.weights(0) = 1.0;
  c.weights(1) = 2.0;
  c[0].entry(0,0) = 1.0;
  c[0].entry(0,1) = 2.0;
  c[2].entry(2,1) = 3.0;
  c[1].entry(1,1) = 4.0;
  ASSERT(c[2].entry(2,1) == 3.0, "Ktensor element set");
  ASSERT( EQ(c.normFsq(), 48*48), "Ktensor normFsq correct");

  // Test copy constructor.
  Genten::Ktensor cCopy1 (c);
  tf = true;
  if (cCopy1.ncomponents() != c.ncomponents())  tf = false;
  if (cCopy1.ndims() != c.ndims())  tf = false;
  if (cCopy1.weights(0) != 1.0)  tf = false;
  if (cCopy1[0].entry(0,0) != 1.0)  tf = false;
  ASSERT(tf, "Copy constructor works");

  // Test assignment operator.
  Genten::Ktensor cCopy2;
  cCopy2 = c;
  tf = true;
  if (cCopy2.ncomponents() != c.ncomponents())  tf = false;
  if (cCopy2.ndims() != c.ndims())  tf = false;
  if (cCopy2.weights(0) != 1.0)  tf = false;
  if (cCopy2[0].entry(0,0) != 1.0)  tf = false;
  ASSERT(tf, "Assignment operator works");

  // Test arrange.
  // Weights {1,2,3} should reorder to {3,2,1}.
  Genten::Ktensor c2(3, 3, dims);
  c2.weights(0) = 1.0;
  c2.weights(1) = 2.0;
  c2.weights(2) = 3.0;
  c2[0].entry(0,0) = 1.0;
  c2[0].entry(0,1) = 2.0;
  c2[0].entry(0,2) = 3.0;
  c2[1].entry(0,0) = 4.0;
  c2[1].entry(1,0) = 5.0;
  c2[1].entry(0,1) = 6.0;
  c2[1].entry(1,1) = 7.0;
  c2[1].entry(0,2) = 8.0;
  c2[1].entry(1,2) = 9.0;
  c2[2].entry(0,0) = 1.0;
  c2[2].entry(1,0) = 2.0;
  c2[2].entry(2,0) = 3.0;
  c2[2].entry(0,1) = 4.0;
  c2[2].entry(1,1) = 5.0;
  c2[2].entry(2,1) = 6.0;
  c2[2].entry(0,2) = 7.0;
  c2[2].entry(1,2) = 8.0;
  c2[2].entry(2,2) = 9.0;
  ASSERT( EQ(c2.normFsq(), 3443252.0), "Ktensor normFsq correct");
  c2.arrange();
  ASSERT(   (c2.weights(0) == 3.0)
            && (c2.weights(1) == 2.0)
            && (c2.weights(2) == 1.0),
            "Arrange reordered weights correctly");
  ASSERT(   (c2[0].entry(0,0) == 3.0)
            && (c2[0].entry(0,1) == 2.0)
            && (c2[0].entry(0,2) == 1.0),
            "Arrange reordered factor[0] correctly");
  ASSERT( EQ(c2.normFsq(), 3443252.0), "Ktensor normFsq correct after arrange");

  finalize();
  return;
}
