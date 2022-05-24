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


#include "Genten_Array.hpp"
#include "Genten_FacMatrix.hpp"
#include "Genten_IOtext.hpp"             // In case debug lines are uncommented
#include "Genten_Ktensor.hpp"
#include "Genten_MixedFormatOps.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_Test_Utils.hpp"
#include "Genten_Util.hpp"

using namespace Genten::Test;


/* This file contains unit tests for operations involving mixed tensor formats.
 */

template <typename ExecSpace>
void Genten_Test_MixedFormats_Impl(int infolevel)
{
  typedef ExecSpace exec_space;
  typedef Genten::DefaultHostExecutionSpace host_exec_space;
  typedef Genten::SptensorT<exec_space> Sptensor_type;
  typedef Genten::SptensorT<host_exec_space> Sptensor_host_type;
  typedef Genten::KtensorT<exec_space> Ktensor_type;
  typedef Genten::KtensorT<host_exec_space> Ktensor_host_type;
  typedef Genten::TensorT<exec_space> Tensor_type;
  typedef Genten::TensorT<host_exec_space> Tensor_host_type;

  std::string space_name = Genten::SpaceProperties<exec_space>::name();
  initialize("Tests involving mixed format tensors (" + space_name + ")",
             infolevel);

  Genten::IndxArray dims;

  //----------------------------------------------------------------------
  // Test innerprod() between Sptensor/Tensor and Ktensor.
  //----------------------------------------------------------------------

  MESSAGE("Creating an Sptensor for innerprod test");
  dims = Genten::IndxArray(3); dims[0] = 4; dims[1] = 2; dims[2] = 3;
  Sptensor_host_type a(dims,4);
  a.subscript(0,0) = 2;  a.subscript(0,1) = 0;  a.subscript(0,2) = 0;
  a.value(0) = 1.0;
  a.subscript(1,0) = 1;  a.subscript(1,1) = 1;  a.subscript(1,2) = 1;
  a.value(1) = 2.0;
  a.subscript(2,0) = 3;  a.subscript(2,1) = 0;  a.subscript(2,2) = 2;
  a.value(2) = 3.0;
  a.subscript(3,0) = 0;  a.subscript(3,1) = 1;  a.subscript(3,2) = 2;
  a.value(3) = 4.0;
  Sptensor_type a_dev = create_mirror_view( exec_space(), a );
  deep_copy( a_dev, a );

  MESSAGE("Creating a (dense) Tensor for innerprod test");
  Tensor_type t_dev(a_dev);

  MESSAGE("Creating a Ktensor of matching shape");
  dims = Genten::IndxArray(3); dims[0] = 4; dims[1] = 2; dims[2] = 3;
  ttb_indx  nc = 2;
  Ktensor_host_type  oKtens(nc, 3, dims);

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

  Ktensor_type oKtens_dev = create_mirror_view( exec_space(), oKtens );
  deep_copy( oKtens_dev, oKtens );

  ttb_real d;
  d = innerprod (a_dev, oKtens_dev);
  ASSERT( EQ(d, 1.162), "Inner product between sptensor and ktensor");
  d = innerprod (t_dev, oKtens_dev);
  ASSERT( EQ(d, 1.162), "Inner product between tensor and ktensor");

  Genten::Array  altLambda(2);
  altLambda[0] = 3.0;
  altLambda[1] = 1.0;
  Genten::ArrayT<exec_space> altLambda_dev =
    create_mirror_view( exec_space(), altLambda );
  deep_copy( altLambda_dev, altLambda );
  d = innerprod(a_dev, oKtens_dev, altLambda_dev);
  ASSERT( EQ(d, 3.081), "Inner product with alternate lambda is correct");
  d = innerprod(t_dev, oKtens_dev, altLambda_dev);
  ASSERT( EQ(d, 3.081), "Inner product with alternate lambda is correct");


  //----------------------------------------------------------------------
  // Test times() and divide() between Sptensor and Ktensor.
  // (note these only run on the host currently)
  //----------------------------------------------------------------------

  MESSAGE("Resizing Sptensor for times/divide test");
  dims = Genten::IndxArray(3); dims[0] = 3; dims[1] = 4; dims[2] = 2;
  a = Sptensor_host_type(dims,3);
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
  Sptensor_host_type  oTest(a.size(), a.nnz());
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
  // Test mttkrp() between Tensor and Ktensor.
  //
  // Corresponding Matlab code for the tests:
  //   Adense = tenzeros([2 3 4]);
  //   Adense(1,1,1) = 1.0
  //   Afac = [10 ; 11];  Bfac = [12 ; 13 ; 14];  Cfac = [15 ; 16 ; 17 ; 18];
  //   K = ktensor({Afac,Bfac,Cfac})
  //   mttkrp(Adense,K.U,1)     % Matricizes on 1st index
  //   mttkrp(Adense,K.U,2)     % Matricizes on 2nd index
  //   mttkrp(Adense,K.U,3)     % Matricizes on 3rd index
  //   Adense(2,3,4) = 1.0
  //   mttkrp(Adense,K.U,1)     % Matricizes on 1st index
  //   mttkrp(Adense,K.U,2)     % Matricizes on 2nd index
  //   mttkrp(Adense,K.U,3)     % Matricizes on 3rd index
  //----------------------------------------------------------------------

  MESSAGE("Resizing Tensor for mttkrp test");
  dims = Genten::IndxArray(3); dims[0] = 2; dims[1] = 3; dims[2] = 4;
  Tensor_host_type t2(dims, 0.0);
  t2(0,0,0) = 1.0;
  Tensor_type t2_dev = create_mirror_view( exec_space(), t2);
  deep_copy(t2_dev, t2);

  MESSAGE("Resizing Ktensor of matching shape");
  nc = 1;
  oKtens = Ktensor_host_type(nc, 3, dims);
  oKtens.setWeights (1.0);
  ASSERT(oKtens.isConsistent(), "Ktensor resized consistently");

  // Set elements in the factors to unique values.
  oKtens[0](0,0) = 10.0;
  oKtens[0](1,0) = 11.0;
  oKtens[1](0,0) = 12.0;
  oKtens[1](1,0) = 13.0;
  oKtens[1](2,0) = 14.0;
  oKtens[2](0,0) = 15.0;
  oKtens[2](1,0) = 16.0;
  oKtens[2](2,0) = 17.0;
  oKtens[2](3,0) = 18.0;
  oKtens_dev = create_mirror_view( exec_space(), oKtens );
  deep_copy( oKtens_dev, oKtens );

  Genten::FacMatrix oFM;
  Genten::FacMatrixT<exec_space> oFM_dev;

  // Matricizing on index 0 has result 12*15 = 180.
  oFM = Genten::FacMatrix(t2.size(0), oKtens.ncomponents());
  oFM_dev = create_mirror_view( exec_space(), oFM );
  deep_copy( oFM_dev, oFM );
  mttkrp (t2_dev, oKtens_dev, 0, oFM_dev);
  deep_copy( oFM, oFM_dev );
  ASSERT((oFM.nRows() == 2) && (oFM.nCols() == 1),
         "mttkrp result shape correct for index [0]");
  ASSERT( EQ(oFM.entry(0,0), 180.0) && EQ(oFM.entry(1,0), 0.0),
          "mttkrp result values correct for index [0]");

  /* Uncomment to manually check what the answer should be.
     Genten::print_tensor(t2, std::cout, "Tensor for mttkrp test");
     Genten::print_ktensor(oKtens, std::cout, "Ktensor for mttkrp test");
     Genten::print_matrix(oFM, std::cout, "Matrix result from mttkrp");
  */

  // Matricizing on index 1 has result 10*15 = 150.
  oFM = Genten::FacMatrix(t2.size(1), oKtens.ncomponents());
  oFM_dev = create_mirror_view( exec_space(), oFM );
  deep_copy( oFM_dev, oFM );
  mttkrp (t2_dev, oKtens_dev, 1, oFM_dev);
  deep_copy( oFM, oFM_dev );
  ASSERT((oFM.nRows() == 3) && (oFM.nCols() == 1),
         "mttkrp result shape correct for index [1]");
  ASSERT( EQ(oFM.entry(0,0), 150.0) && EQ(oFM.entry(1,0), 0.0),
          "mttkrp result values correct for index [1]");

  // Matricizing on index 2 has result 10*12 = 120.
  oFM = Genten::FacMatrix(t2.size(2), oKtens.ncomponents());
  oFM_dev = create_mirror_view( exec_space(), oFM );
  deep_copy( oFM_dev, oFM );
  mttkrp (t2_dev, oKtens_dev, 2, oFM_dev);
  deep_copy( oFM, oFM_dev );
  ASSERT((oFM.nRows() == 4) && (oFM.nCols() == 1),
         "mttkrp result shape correct for index [2]");
  ASSERT( EQ(oFM.entry(0,0), 120.0) && EQ(oFM.entry(1,0), 0.0),
          "mttkrp result values correct for index [2]");

  // Add another nonzero and repeat the three tests.
  t2(1,2,3) = 1.0;
  deep_copy(t2_dev, t2);
  oFM = Genten::FacMatrix(t2.size(0), oKtens.ncomponents());
  oFM_dev = create_mirror_view( exec_space(), oFM );
  deep_copy( oFM_dev, oFM );
  mttkrp (t2_dev, oKtens_dev, 0, oFM_dev);
  deep_copy( oFM, oFM_dev );
  ASSERT( EQ(oFM.entry(0,0), 180.0) && EQ(oFM.entry(1,0), 252.0),
          "mttkrp result values correct for index [0], 2 sparse nnz");
  oFM = Genten::FacMatrix(t2.size(1), oKtens.ncomponents());
  oFM_dev = create_mirror_view( exec_space(), oFM );
  deep_copy( oFM_dev, oFM );
  mttkrp (t2_dev, oKtens_dev, 1, oFM_dev);
  deep_copy( oFM, oFM_dev );
  ASSERT( EQ(oFM.entry(0,0), 150.0) && EQ(oFM.entry(2,0), 198.0),
          "mttkrp result values correct for index [0], 2 sparse nnz");
  oFM = Genten::FacMatrix(t2.size(2), oKtens.ncomponents());
  oFM_dev = create_mirror_view( exec_space(), oFM );
  deep_copy( oFM_dev, oFM );
  mttkrp (t2_dev, oKtens_dev, 2, oFM_dev);
  deep_copy( oFM, oFM_dev );
  ASSERT( EQ(oFM.entry(0,0), 120.0) && EQ(oFM.entry(3,0), 154.0),
          "mttkrp result values correct for index [0], 2 sparse nnz");

  finalize();

}

template <typename ExecSpace>
void Genten_Test_MTTKRP_Type(Genten::MTTKRP_Method::type mttkrp_method,
                             int infolevel, const std::string& label)
{
  typedef ExecSpace exec_space;
  typedef Genten::DefaultHostExecutionSpace host_exec_space;
  typedef Genten::SptensorT<exec_space> Sptensor_type;
  typedef Genten::SptensorT<host_exec_space> Sptensor_host_type;
  typedef Genten::KtensorT<exec_space> Ktensor_type;
  typedef Genten::KtensorT<host_exec_space> Ktensor_host_type;

  std::string space_name = Genten::SpaceProperties<exec_space>::name();
  initialize("MTTKRP tests ("+label+", "+space_name+")", infolevel);

  Genten::IndxArray dims;

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
  Sptensor_host_type a(dims,2);
  a.subscript(0,0) = 0;  a.subscript(0,1) = 0;  a.subscript(0,2) = 0;
  a.value(0) = 1.0;
  a.subscript(1,0) = 1;  a.subscript(1,1) = 2;  a.subscript(1,2) = 3;
  a.value(1) = 0.0;

  MESSAGE("Resizing Ktensor of matching shape");
  ttb_indx nc = 1;
  Genten::Ktensor oKtens(nc, 3, dims);
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

  // Copy a and oKtens to device
  Sptensor_type a_dev = create_mirror_view( exec_space(), a );
  deep_copy( a_dev, a );
  if (mttkrp_method == Genten::MTTKRP_Method::Perm)
    a_dev.createPermutation();
  Ktensor_type oKtens_dev = create_mirror_view( exec_space(), oKtens );
  deep_copy( oKtens_dev, oKtens );

  Genten::FacMatrix oFM;
  Genten::FacMatrixT<exec_space> oFM_dev;

  Genten::AlgParams algParams;
  algParams.mttkrp_method = mttkrp_method;
  algParams.mttkrp_duplicated_threshold = 1.0e6; // Ensure duplicated is used

  // Matricizing on index 0 has result 12*15 = 180.
  oFM = Genten::FacMatrix(a.size(0), oKtens.ncomponents());
  oFM_dev = create_mirror_view( exec_space(), oFM );
  deep_copy( oFM_dev, oFM );
  mttkrp (a_dev, oKtens_dev, 0, oFM_dev, algParams);
  deep_copy( oFM, oFM_dev );
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
  oFM = Genten::FacMatrix(a.size(1), oKtens.ncomponents());
  oFM_dev = create_mirror_view( exec_space(), oFM );
  deep_copy( oFM_dev, oFM );
  mttkrp (a_dev, oKtens_dev, 1, oFM_dev, algParams);
  deep_copy( oFM, oFM_dev );
  ASSERT((oFM.nRows() == 3) && (oFM.nCols() == 1),
         "mttkrp result shape correct for index [1]");
  ASSERT( EQ(oFM.entry(0,0), 150.0) && EQ(oFM.entry(1,0), 0.0),
          "mttkrp result values correct for index [1]");

  // Matricizing on index 2 has result 10*12 = 120.
  oFM = Genten::FacMatrix(a.size(2), oKtens.ncomponents());
  oFM_dev = create_mirror_view( exec_space(), oFM );
  deep_copy( oFM_dev, oFM );
  mttkrp (a_dev, oKtens_dev, 2, oFM_dev, algParams);
  deep_copy( oFM, oFM_dev );
  ASSERT((oFM.nRows() == 4) && (oFM.nCols() == 1),
         "mttkrp result shape correct for index [2]");
  ASSERT( EQ(oFM.entry(0,0), 120.0) && EQ(oFM.entry(1,0), 0.0),
          "mttkrp result values correct for index [2]");

  // Add another nonzero and repeat the three tests.
  a.subscript(1,0) = 1;  a.subscript(1,1) = 2;  a.subscript(1,2) = 3;
  a.value(1) = 1.0;
  a_dev = create_mirror_view( exec_space(), a );
  deep_copy( a_dev, a );
  if (mttkrp_method == Genten::MTTKRP_Method::Perm)
    a_dev.createPermutation();
  oFM = Genten::FacMatrix(a.size(0), oKtens.ncomponents());
  oFM_dev = create_mirror_view( exec_space(), oFM );
  deep_copy( oFM_dev, oFM );
  mttkrp (a_dev, oKtens_dev, 0, oFM_dev, algParams);
  deep_copy( oFM, oFM_dev );
  ASSERT( EQ(oFM.entry(0,0), 180.0) && EQ(oFM.entry(1,0), 252.0),
          "mttkrp result values correct for index [0], 2 sparse nnz");
  oFM = Genten::FacMatrix(a.size(1), oKtens.ncomponents());
  oFM_dev = create_mirror_view( exec_space(), oFM );
  deep_copy( oFM_dev, oFM );
  mttkrp (a_dev, oKtens_dev, 1, oFM_dev, algParams);
  deep_copy( oFM, oFM_dev );
  ASSERT( EQ(oFM.entry(0,0), 150.0) && EQ(oFM.entry(2,0), 198.0),
          "mttkrp result values correct for index [0], 2 sparse nnz");
  oFM = Genten::FacMatrix(a.size(2), oKtens.ncomponents());
  oFM_dev = create_mirror_view( exec_space(), oFM );
  deep_copy( oFM_dev, oFM );
  mttkrp (a_dev, oKtens_dev, 2, oFM_dev, algParams);
  deep_copy( oFM, oFM_dev );
  ASSERT( EQ(oFM.entry(0,0), 120.0) && EQ(oFM.entry(3,0), 154.0),
          "mttkrp result values correct for index [0], 2 sparse nnz");

  finalize();
  return;
}

template <typename ExecSpace>
void Genten_Test_MTTKRP_All_Type(Genten::MTTKRP_All_Method::type mttkrp_method,
                                 int infolevel, const std::string& label)
{
  typedef ExecSpace exec_space;
  typedef Genten::DefaultHostExecutionSpace host_exec_space;
  typedef Genten::SptensorT<exec_space> Sptensor_type;
  typedef Genten::SptensorT<host_exec_space> Sptensor_host_type;
  typedef Genten::KtensorT<exec_space> Ktensor_type;
  typedef Genten::KtensorT<host_exec_space> Ktensor_host_type;

  std::string space_name = Genten::SpaceProperties<exec_space>::name();
  initialize("MTTKRP-All tests ("+label+", "+space_name+")", infolevel);

  Genten::IndxArray dims;

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
  Sptensor_host_type a(dims,2);
  a.subscript(0,0) = 0;  a.subscript(0,1) = 0;  a.subscript(0,2) = 0;
  a.value(0) = 1.0;
  a.subscript(1,0) = 1;  a.subscript(1,1) = 2;  a.subscript(1,2) = 3;
  a.value(1) = 0.0;

  MESSAGE("Resizing Ktensor of matching shape");
  ttb_indx nc = 1;
  Genten::Ktensor oKtens(nc, 3, dims);
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

  // Copy a and oKtens to device
  Sptensor_type a_dev = create_mirror_view( exec_space(), a );
  deep_copy( a_dev, a );
  Ktensor_type oKtens_dev = create_mirror_view( exec_space(), oKtens );
  deep_copy( oKtens_dev, oKtens );

  Genten::Ktensor v(nc, 3, dims);
  v.setWeights(1.0);
  Ktensor_type v_dev = create_mirror_view( exec_space(), v );
  deep_copy( v_dev, v );

  Genten::AlgParams algParams;
  algParams.mttkrp_all_method = mttkrp_method;
  algParams.mttkrp_method = Genten::MTTKRP_Method::Atomic;

  // Matricizing on index 0 has result 12*15 = 180.
  // Matricizing on index 1 has result 10*15 = 150.
  // Matricizing on index 2 has result 10*12 = 120.
  mttkrp_all(a_dev, oKtens_dev, v_dev, algParams);
  deep_copy( v, v_dev );
  ASSERT((v[0].nRows() == 2) && (v[0].nCols() == 1),
         "mttkrp result shape correct for index [0]");
  ASSERT(EQ(v[0].entry(0,0), 180.0) && EQ(v[0].entry(1,0), 0.0),
          "mttkrp result values correct for index [0]");
  ASSERT((v[1].nRows() == 3) && (v[1].nCols() == 1),
         "mttkrp result shape correct for index [1]");
  ASSERT(EQ(v[1].entry(0,0), 150.0) && EQ(v[1].entry(1,0), 0.0),
          "mttkrp result values correct for index [1]");
  ASSERT((v[2].nRows() == 4) && (v[2].nCols() == 1),
         "mttkrp result shape correct for index [2]");
  ASSERT(EQ(v[2].entry(0,0), 120.0) && EQ(v[2].entry(1,0), 0.0),
          "mttkrp result values correct for index [2]");

  /* Uncomment to manually check what the answer should be.
     Genten::print_sptensor(a, std::cout, "Sparse tensor for mttkrp test");
     Genten::print_ktensor(oKtens, std::cout, "Ktensor for mttkrp test");
     Genten::print_ktensor(v, std::cout, "Ktensor result from mttkrp_all");
  */

  // Add another nonzero and repeat the three tests.
  a.subscript(1,0) = 1;  a.subscript(1,1) = 2;  a.subscript(1,2) = 3;
  a.value(1) = 1.0;
  a_dev = create_mirror_view( exec_space(), a );
  deep_copy( a_dev, a );
  mttkrp_all(a_dev, oKtens_dev, v_dev, algParams);
  deep_copy( v, v_dev );
  ASSERT(EQ(v[0].entry(0,0), 180.0) && EQ(v[0].entry(1,0), 252.0),
          "mttkrp result values correct for index [0], 2 sparse nnz");
  ASSERT(EQ(v[1].entry(0,0), 150.0) && EQ(v[1].entry(2,0), 198.0),
          "mttkrp result values correct for index [0], 2 sparse nnz");
  ASSERT(EQ(v[2].entry(0,0), 120.0) && EQ(v[2].entry(3,0), 154.0),
          "mttkrp result values correct for index [0], 2 sparse nnz");

  finalize();
  return;
}

template <typename ExecSpace>
void Genten_Test_MixedFormats_Space(int infolevel)
{
  typedef Genten::SpaceProperties<ExecSpace> space_prop;

  Genten_Test_MixedFormats_Impl<ExecSpace>(infolevel);

  Genten_Test_MTTKRP_Type<ExecSpace>(
    Genten::MTTKRP_Method::Atomic, infolevel, "Atomic");
  if (!space_prop::is_gpu)
    Genten_Test_MTTKRP_Type<ExecSpace>(
      Genten::MTTKRP_Method::Duplicated, infolevel, "Duplicated");
  Genten_Test_MTTKRP_Type<ExecSpace>(
    Genten::MTTKRP_Method::Perm, infolevel, "Perm");

  Genten_Test_MTTKRP_All_Type<ExecSpace>(
    Genten::MTTKRP_All_Method::Iterated, infolevel, "Iterated");
  Genten_Test_MTTKRP_All_Type<ExecSpace>(
    Genten::MTTKRP_All_Method::Atomic, infolevel, "Atomic");
  if (!space_prop::is_gpu)
    Genten_Test_MTTKRP_All_Type<ExecSpace>(
      Genten::MTTKRP_All_Method::Duplicated, infolevel, "Duplicated");
}

void Genten_Test_MixedFormats(int infolevel) {
#ifdef KOKKOS_ENABLE_CUDA
  Genten_Test_MixedFormats_Space<Kokkos::Cuda>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_HIP
  Genten_Test_MixedFormats_Space<Kokkos::Experimental::HIP>(infolevel);
#endif
#ifdef ENABLE_SYCL_FOR_CUDA
  Genten_Test_MixedFormats_Space<Kokkos::Experimental::SYCL>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  Genten_Test_MixedFormats_Space<Kokkos::OpenMP>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_THREADS
  Genten_Test_MixedFormats_Space<Kokkos::Threads>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_SERIAL
  Genten_Test_MixedFormats_Space<Kokkos::Serial>(infolevel);
#endif
}
