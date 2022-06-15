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

/*!
  @file Genten_Test_IOtext.cpp
  @brief Unit tests for methods in Genten_IOtext.
*/

#include "Genten_IndxArray.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Test_Utils.hpp"
#include "Genten_Util.hpp"

using namespace std;
using namespace Genten::Test;

// These tests are just on the host, so not templated on the space
void Genten_Test_IO(int            infolevel,
                    const string & dirname)
{
  initialize("Tests of tensor I/O methods", infolevel);

  string fname;
  bool  bIsOK;
  double  dExpectedValue;

  // Test I/O methods for Sptensor.

  MESSAGE("Reading sparse tensor from file");
  Genten::Sptensor oSpt;
  fname = "C_sptensor.txt";
  Genten::import_sptensor (dirname + fname, oSpt);
  ASSERT((oSpt.ndims() == 3) && (oSpt.nnz() == 5) && (oSpt.numel() == 24),
         "Sptensor size verified");
  ASSERT(   (oSpt.subscript(4,0) == 3) && (oSpt.subscript(4,1) == 0)
            && (oSpt.subscript(4,2) == 1) && EQ(oSpt.value(4), 5.0),
            "Verified an Sptensor element read from file");

  if (infolevel == 1)
  {
    MESSAGE("Printing sparse tensor");
    Genten::print_sptensor (oSpt, cout, "From " + fname);
    MESSAGE("Done");
  }

  MESSAGE("Writing Sptensor to temporary file");
  fname = "tmp_Test_IOtext.txt";
  Genten::export_sptensor (fname, oSpt);
  Genten::Sptensor oSpt2;
  Genten::import_sptensor (fname, oSpt2);
  ASSERT(oSpt.isEqual(oSpt2, MACHINE_EPSILON),
         "Sptensor unchanged after write and read");
  ASSERT(remove (fname.c_str()) == 0, "Temp file for export_sptensor deleted");


  // Test I/O methods for FacMatrix.

  MESSAGE("Reading matrix from file");
  Genten::FacMatrix oFM;
  fname = "B_matrix.txt";
  Genten::import_matrix (dirname + fname, oFM);
  ASSERT((oFM.nRows() == 3) && (oFM.nCols() == 2), "Matrix size verified");
  MESSAGE("Verifying contents of matrix read from file");
  bIsOK = true;
  dExpectedValue = 0.1;
  for (ttb_indx j = 0; j < oFM.nCols(); j++)
  {
    for (ttb_indx i = 0; i < oFM.nRows(); i++)
    {
      if (EQ(oFM.entry(i,j),dExpectedValue) == false)
      {
        bIsOK = false;
        break;
      }
      dExpectedValue += 0.1;
    }
    if (bIsOK == false)
    {
      break;
    }
  }
  ASSERT(bIsOK, "All matrix elements verified");

  if (infolevel == 1)
  {
    MESSAGE("Printing matrix");
    Genten::print_matrix (oFM, cout, "Contents of " + dirname + fname);
    MESSAGE("Done");
  }

  MESSAGE("Writing matrix to temporary file");
  fname = "tmp_Test_IOtext.txt";
  Genten::export_matrix (fname, oFM);
  Genten::FacMatrix oFM2;
  Genten::import_matrix (fname, oFM2);
  ASSERT(oFM.isEqual(oFM2, MACHINE_EPSILON),
         "Matrix unchanged after write and read");
  ASSERT(remove (fname.c_str()) == 0, "Temp file for export_matrix deleted");


  // Test I/O methods for Ktensor.

  MESSAGE("Reading Ktensor from file");
  Genten::Ktensor oK;
  fname = "E_ktensor.txt";
  Genten::import_ktensor (dirname + fname, oK);
  ASSERT((oK.ndims() == 3) && (oK.ncomponents() == 2),
         "Ktensor size verified");
  ASSERT(EQ(oK.weights(0),1.0) && EQ(oK.weights(1),2.0),
         "Ktensor weights verified");
  bIsOK = true;
  bIsOK =    (EQ(oK[0].entry(1,1), 0.9))
    && (EQ(oK[1].entry(0,1), 2.0))
    && (EQ(oK[2].entry(2,0), 0.03));
  ASSERT(bIsOK, "Verified selected Ktensor elements read from file");

  MESSAGE("Writing Ktensor to temporary file");
  fname = "tmp_Test_IOtext.txt";
  Genten::export_ktensor (fname, oK);
  Genten::Ktensor oK2;
  Genten::import_ktensor (fname, oK2);
  ASSERT(oK.isEqual(oK2, MACHINE_EPSILON),
         "Ktensor unchanged after write and read");
  ASSERT(remove (fname.c_str()) == 0, "Temp file for export_matrix deleted");


  finalize();
  return;
}
