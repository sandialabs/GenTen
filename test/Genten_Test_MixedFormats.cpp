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


#include "Genten_Array.h"
#include "Genten_FacMatrix.h"
#include "Genten_IOtext.h"             // In case debug lines are uncommented
#include "Genten_Ktensor.h"
#include "Genten_MixedFormatOps.h"
#include "Genten_Sptensor.h"
#include "Genten_Test_Utils.h"
#include "Genten_Util.h"

using namespace Genten::Test;


/* This file contains unit tests for operations involving mixed tensor formats.
 */
void Genten_Test_MixedFormats(int infolevel)
{
  initialize("Tests involving mixed format tensors", infolevel);

  Genten::IndxArray dims;


  //----------------------------------------------------------------------
  // Test innerprod() between Sptensor/Tensor and Ktensor.
  //----------------------------------------------------------------------

  MESSAGE("Creating an Sptensor for innerprod test");
  dims = Genten::IndxArray(3); dims[0] = 4; dims[1] = 2; dims[2] = 3;
  Genten::Sptensor a(dims,4);
  a.subscript(0,0) = 2;  a.subscript(0,1) = 0;  a.subscript(0,2) = 0;
  a.value(0) = 1.0;
  a.subscript(1,0) = 1;  a.subscript(1,1) = 1;  a.subscript(1,2) = 1;
  a.value(1) = 2.0;
  a.subscript(2,0) = 3;  a.subscript(2,1) = 0;  a.subscript(2,2) = 2;
  a.value(2) = 3.0;
  a.subscript(3,0) = 0;  a.subscript(3,1) = 1;  a.subscript(3,2) = 2;
  a.value(3) = 4.0;

  MESSAGE("Creating a Ktensor of matching shape");
  dims = Genten::IndxArray(3); dims[0] = 4; dims[1] = 2; dims[2] = 3;
  ttb_indx  nc = 2;
  Genten::Ktensor  oKtens(nc, 3, dims);

  // The sparse tensor has 4 nonzeroes.  Populate the Ktensor so that two
  // of its nonzeroes match, testing different weights.
  // Answers were obtained using Matlab tensor toolbox.

  oKtens.weights(0) = 1.0;
  oKtens.weights(1) = 2.0;

  oKtens[0].entry(2,0) = 1.0;
  oKtens[1].entry(0,0) = 1.0;
  oKtens[2].entry(0,0) = 1.0;

  oKtens[0].entry(1,0) = -1.0;

  oKtens[0].entry(3,1) = 0.3;
  oKtens[1].entry(0,1) = 0.3;
  oKtens[2].entry(2,1) = 0.3;

  ttb_real d;
  d = innerprod (a, oKtens);
  ASSERT( EQ(d, 1.162), "Inner product between sptensor and ktensor");

  Genten::Array  altLambda(2);
  altLambda[0] = 3.0;
  altLambda[1] = 1.0;
  d = innerprod(a, oKtens, altLambda);
  ASSERT( EQ(d, 3.081), "Inner product with alternate lambda is correct");


  //----------------------------------------------------------------------
  // Test times() and divide() between Sptensor and Ktensor.
  //----------------------------------------------------------------------

  MESSAGE("Resizing Sptensor for times/divide test");
  dims = Genten::IndxArray(3); dims[0] = 3; dims[1] = 4; dims[2] = 2;
  a = Genten::Sptensor(dims,3);
  a.subscript(0,0) = 1;  a.subscript(0,1) = 0;  a.subscript(0,2) = 0;
  a.value(0) = 1.0;
  a.subscript(1,0) = 1;  a.subscript(1,1) = 0;  a.subscript(1,2) = 1;
  a.value(1) = 1.0;
  a.subscript(2,0) = 1;  a.subscript(2,1) = 1;  a.subscript(2,2) = 0;
  a.value(2) = 1.0;
  ASSERT(a.nnz() == 3, "Sptensor resized to correct nnz");

  MESSAGE("Resizing Ktensor for times test");
  dims = Genten::IndxArray(3); dims[0] = 3; dims[1] = 4; dims[2] = 2;
  nc = 2;
  oKtens = Genten::Ktensor(nc, 3, dims);
  oKtens.setWeights (1.0);
  ASSERT(oKtens.isConsistent(), "Ktensor resized consistently");

  // Set elements in the factors to unique values, counting up by 1.
  // The Ktensor has a nonzero value for every entry in the reconstructed
  // tensor; element-wise multiplication will result in modifications to
  // just the nonzeroes of the Sptensor.
  d = 1.0;
  for (ttb_indx i = 0; i < oKtens.ndims(); i++)
  {
    for (ttb_indx j = 0; j < oKtens[i].nRows(); j++)
    {
      for (ttb_indx r = 0; r < oKtens.ncomponents(); r++)
      {
        oKtens[i].entry(j,r) = d;
        d = d + 1.0;
      }
    }
  }
  /* Uncomment to manually check what the answer should be.
     Genten::print_sptensor(a, std::cout, "Sparse tensor for times/divide test");
     Genten::print_ktensor(oKtens, std::cout, "Ktensor for times/divide test");
  */

  // Test times().
  Genten::Sptensor  oTest(a.size(), a.nnz());
  oTest.times (oKtens, a);
  ASSERT( EQ(oTest.value(0), (3*7*15 + 4*8*16)), "times() element 0 OK");
  ASSERT( EQ(oTest.value(1), (3*7*17 + 4*8*18)), "times() element 1 OK");
  ASSERT( EQ(oTest.value(2), (3*9*15 + 4*10*16)), "times() element 2 OK");

  // Test that divide() undoes the multiplication.
  oTest.divide (oKtens, oTest, 1.0e-10);
  ASSERT( EQ(oTest.value(0), 1.0), "divide() element 0 OK");
  ASSERT( EQ(oTest.value(1), 1.0), "divide() element 1 OK");
  ASSERT( EQ(oTest.value(2), 1.0), "divide() element 2 OK");


  //----------------------------------------------------------------------
  // Test mttkrp() between Sptensor and Ktensor.
  //
  // Corresponding Matlab code for the tests:
  //   Asparse = sptensor([], [], [2 3 4]);
  //   Asparse(1,1,1) = 1.0
  //   Afac = [10 ; 11];  Bfac = [12 ; 13 ; 14];  Cfac = [15 ; 16 ; 17 ; 18];
  //   K = ktensor({Afac,Bfac,Cfac})
  //   mttkrp(Asparse,K.U,1)     % Matricizes on 1st index
  //   mttkrp(Asparse,K.U,2)     % Matricizes on 2nd index
  //   mttkrp(Asparse,K.U,3)     % Matricizes on 3rd index
  //   Asparse(2,3,4) = 1.0
  //   mttkrp(Asparse,K.U,1)     % Matricizes on 1st index
  //   mttkrp(Asparse,K.U,2)     % Matricizes on 2nd index
  //   mttkrp(Asparse,K.U,3)     % Matricizes on 3rd index
  //----------------------------------------------------------------------

  MESSAGE("Resizing Sptensor for mttkrp test");
  dims = Genten::IndxArray(3); dims[0] = 2; dims[1] = 3; dims[2] = 4;
  a = Genten::Sptensor(dims,2);
  a.subscript(0,0) = 0;  a.subscript(0,1) = 0;  a.subscript(0,2) = 0;
  a.value(0) = 1.0;
  a.subscript(1,0) = 1;  a.subscript(1,1) = 2;  a.subscript(1,2) = 3;
  a.value(1) = 0.0;

  MESSAGE("Resizing Ktensor of matching shape");
  nc = 1;
  oKtens = Genten::Ktensor(nc, 3, dims);
  oKtens.setWeights (1.0);
  ASSERT(oKtens.isConsistent(), "Ktensor resized consistently");

  // Set elements in the factors to unique values.
  // The sparse tensor has only one nonzero value; hence, mttkrp will involve
  // just one element from two of the factor matrices.
  oKtens[0].entry(0,0) = 10.0;
  oKtens[0].entry(1,0) = 11.0;
  oKtens[1].entry(0,0) = 12.0;
  oKtens[1].entry(1,0) = 13.0;
  oKtens[1].entry(2,0) = 14.0;
  oKtens[2].entry(0,0) = 15.0;
  oKtens[2].entry(1,0) = 16.0;
  oKtens[2].entry(2,0) = 17.0;
  oKtens[2].entry(3,0) = 18.0;

  Genten::FacMatrix  oFM;

  // Matricizing on index 0 has result 12*15 = 180.
  mttkrp (a, oKtens, 0, oFM);
  ASSERT((oFM.nRows() == 2) && (oFM.nCols() == 1),
         "mttkrp result shape correct for index [0]");
  ASSERT( EQ(oFM.entry(0,0), 180.0) && EQ(oFM.entry(1,0), 0.0),
          "mttkrp result values correct for index [0]");

  /* Uncomment to manually check what the answer should be.
     Genten::print_sptensor(a, std::cout, "Sparse tensor for mttkrp test");
     Genten::print_ktensor(oKtens, std::cout, "Ktensor for mttkrp test");
     Genten::print_matrix(oFM, std::cout, "Matrix result from mttkrp");
  */

  // Matricizing on index 1 has result 10*15 = 150.
  mttkrp (a, oKtens, 1, oFM);
  ASSERT((oFM.nRows() == 3) && (oFM.nCols() == 1),
         "mttkrp result shape correct for index [1]");
  ASSERT( EQ(oFM.entry(0,0), 150.0) && EQ(oFM.entry(1,0), 0.0),
          "mttkrp result values correct for index [1]");

  // Matricizing on index 2 has result 10*12 = 120.
  mttkrp (a, oKtens, 2, oFM);
  ASSERT((oFM.nRows() == 4) && (oFM.nCols() == 1),
         "mttkrp result shape correct for index [2]");
  ASSERT( EQ(oFM.entry(0,0), 120.0) && EQ(oFM.entry(1,0), 0.0),
          "mttkrp result values correct for index [2]");

  // Add another nonzero and repeat the three tests.
  a.subscript(1,0) = 1;  a.subscript(1,1) = 2;  a.subscript(1,2) = 3;
  a.value(1) = 1.0;
  mttkrp (a, oKtens, 0, oFM);
  ASSERT( EQ(oFM.entry(0,0), 180.0) && EQ(oFM.entry(1,0), 252.0),
          "mttkrp result values correct for index [0], 2 sparse nnz");
  mttkrp (a, oKtens, 1, oFM);
  ASSERT( EQ(oFM.entry(0,0), 150.0) && EQ(oFM.entry(2,0), 198.0),
          "mttkrp result values correct for index [0], 2 sparse nnz");
  mttkrp (a, oKtens, 2, oFM);
  ASSERT( EQ(oFM.entry(0,0), 120.0) && EQ(oFM.entry(3,0), 154.0),
          "mttkrp result values correct for index [0], 2 sparse nnz");

  finalize();
  return;
}
