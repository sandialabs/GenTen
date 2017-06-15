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
// ************************************************************************
//@HEADER


#include "Genten_FacMatrix.h"
#include "Genten_IOtext.h"
#include "Genten_Test_Utils.h"
#include "Genten_Util.h"

using namespace Genten::Test;

void Genten_Test_FacMatrix(int infolevel, const std::string & datadir)
{
  SETUP_DISABLE_CERR;

  bool tf;

  initialize("Tests on Genten::FacMatrix", infolevel);

  // Empty Constructor
  MESSAGE("Testing empty constructor");
  Genten::FacMatrix a;
  ASSERT((a.nRows() == 0) & (a.nCols() == 0), "Empty constructor works as expected");

  // Size Constructor
  MESSAGE("Testing size constructor");
  Genten::FacMatrix a2(3,2);
  ASSERT((a2.nRows() == 3) && (a2.nCols() == 2), "Size constructor works as expected");

  // Data Constructor
  MESSAGE("Testing data constructor");
  ttb_real cdata[] = {1,2,3,4,5,6,7,8,9};
  Genten::FacMatrix c(3, 3, cdata);
  tf = true;
  double val = 1;
  for (ttb_indx j = 0; j < c.nRows(); j ++)
  {
    for (ttb_indx i = 0; i < c.nCols(); i ++)
    {
      if (c.entry(i,j) != val)
      {
        tf = false;
        break;
      }
      val += 1;
    }
  }
  ASSERT(tf, "Data constructor works as expected");

  // Copy Constructor
  MESSAGE("Copy constructor not tested explicitly");

  // Destructor
  MESSAGE("Destructor is not tested explicitly");


  // entry const
  const Genten::FacMatrix cconst(c);
  tf = cconst.isEqual(c, MACHINE_EPSILON);
  ASSERT(tf, "Entry for const works as expected");


  // entry non-const
  MESSAGE("Entry for non-const not tested explicitly");

  // resize
  a = Genten::FacMatrix(2,2);
  ASSERT(a.nRows() == 2, "Resize works as expected");
  ASSERT(a.nCols() == 2, "Resize works as expected");

  // operator=
  a = 5;
  ASSERT(EQ(a.entry(0,0), 5), "Operator= for scalar works as expected");
  ASSERT(EQ(a.entry(0,1), 5), "Operator= for scalar works as expected");
  ASSERT(EQ(a.entry(1,0), 5), "Operator= for scalar works as expected");
  ASSERT(EQ(a.entry(1,1), 5), "Operator= for scalar works as expected");

  // reset
  a = Genten::FacMatrix(1,1);
  a = 3;
  ASSERT(a.nRows() == 1, "Reset works as expected");
  ASSERT(a.nCols() == 1, "Reset works as expected");
  ASSERT(EQ(a.entry(0,0), 3), "Reset works as expected");

  // gramian
  Genten::FacMatrix b;
  Genten::import_matrix(datadir + "B_matrix.txt", b);
  Genten::FacMatrix d;
  Genten::import_matrix(datadir + "D_matrix.txt", d);
  Genten::FacMatrix e(b.nCols(), b.nCols());
  e.gramian(b);
  ASSERT(e.nCols() == d.nCols(), "Gramian works");
  ASSERT(e.nRows() == d.nRows(), "Gramian works");
  tf = true;
  for (ttb_indx j = 0; j < e.nCols(); j ++)
  {
    for (ttb_indx i = 0; i < e.nRows(); i ++)
    {
      if (!EQ(e.entry(i,j), d.entry(i,j)))
      {
        tf = false;
        break;
      }

    }

    if (!tf)
    {
      break;
    }
  }
  ASSERT(tf, "Gramian yields expected answer");

  // hadamard product
  MESSAGE("Checking error on wrong sized matrices");
  tf = false;
  DISABLE_CERR;
  try
  {
    b.times(d);
  }
  catch(...)
  {
    tf = true;
  }
  REENABLE_CERR;
  ASSERT(tf, "Expected exception is caught");

  MESSAGE("Now checking actual correctness");
  Genten::FacMatrix f(3,2);
  f = 2;
  f.times(b);
  ASSERT(f.nRows() == 3, "Hadamard works");
  ASSERT(f.nCols() == 2, "Hadamard works");
  tf = true;
  val = 0.1;
  for (ttb_indx j = 0; j < f.nCols(); j ++)
  {
    for (ttb_indx i = 0; i < f.nRows(); i ++)
    {
      if (!EQ(f.entry(i,j), 2*val))
      {
        tf = false;
        break;
      }
      val += 0.1;
    }

    if (tf == false)
    {
      break;
    }
  }
  ASSERT(tf, "Hadamard works as expected");

  // transpose
  Genten::FacMatrix g(b.nCols(), b.nRows());
  g.transpose(b);
  ASSERT(g.nCols() == b.nRows(), "Transpose # columns ok");
  ASSERT(g.nRows() == b.nCols(), "Transpose # rows ok");
  tf = true;
  for (ttb_indx j = 0; j < g.nCols(); j ++)
  {
    for (ttb_indx i = 0; i < g.nRows(); i ++)
    {
      if (!EQ(g.entry(i,j), b.entry(j,i)))
      {
        tf = false;
        break;
      }
    }

    if (tf == false)
    {
      break;
    }
  }
  ASSERT(tf, "Transpose works as expected");

  // oprod
  ttb_real hdata[] = {.1, .2, .3};
  const Genten::Array h(3, hdata);
  a = Genten::FacMatrix(h.size(), h.size());
  a.oprod(h);
  ASSERT(a.nRows() == 3, "Oprod # rows ok");
  ASSERT(a.nCols() == 3, "Oprod # cols ok");
  tf = true;
  for (ttb_indx j = 0; j < g.nCols(); j ++)
  {
    for (ttb_indx i = 0; i < g.nRows(); i ++)
    {
      val = h[i]*h[j];
      if (!EQ(a.entry(i,j),val))
      {
        tf = false;
        break;
      }
    }

    if (tf == false)
    {
      break;
    }
  }
  ASSERT(tf, "Oprod works as expected");

  // Linear solver, first for a diagonal matrix.
  a = Genten::FacMatrix(2,2);
  a.entry(0,0) = 1.0;
  a.entry(1,0) = 0.0;
  a.entry(0,1) = 0.0;
  a.entry(1,1) = 2.0;
  b = Genten::FacMatrix(1,2);
  b.entry(0,0) = 3.0;
  b.entry(0,1) = 4.0;
  b.solveTransposeRHS (a);
  c = Genten::FacMatrix(1,2);
  c.entry(0,0) = 3.0;
  c.entry(0,1) = 2.0;
  ASSERT(c.isEqual(b,MACHINE_EPSILON), "Solve works for diagonal matrix");

  // Linear solver, for an indefinite matrix and 3 right-hand sides.
  a = Genten::FacMatrix(2,2);
  a.entry(0,0) = 1.0;
  a.entry(1,0) = 2.0;
  a.entry(0,1) = 2.0;
  a.entry(1,1) = 1.0;
  b = Genten::FacMatrix(3,2);
  b.entry(0,0) = 1.0;
  b.entry(0,1) = 0.0;
  b.entry(1,0) = 0.0;
  b.entry(1,1) = 1.0;
  b.entry(2,0) = -1.0;
  b.entry(2,1) =  2.0;
  b.solveTransposeRHS (a);
  c = Genten::FacMatrix(3,2);
  c.entry(0,0) = -1.0 / 3.0;
  c.entry(0,1) =  2.0 / 3.0;
  c.entry(1,0) =  2.0 / 3.0;
  c.entry(1,1) = -1.0 / 3.0;
  c.entry(2,0) =  5.0 / 3.0;
  c.entry(2,1) = -4.0 / 3.0;
  ASSERT(c.isEqual(b,MACHINE_EPSILON), "Solve works for indefinite matrix");

  // colNorms
  Genten::Array nrms(3), nrms_chk(3);
  // set a = [3 0 0; 4 1 0; 0 0 0]
  a = Genten::FacMatrix(3,3);
  a.entry(0,0) = 3;
  a.entry(1,0) = 4;
  a.entry(1,1) = 1;
  a.colNorms(Genten::NormInf,nrms,0.0);
  nrms_chk[0] = 4;
  nrms_chk[1] = 1;
  nrms_chk[2] = 0;
  ASSERT(nrms.isEqual(nrms_chk,MACHINE_EPSILON),
         "ColNorms (max norm) works as expected");
  a.colNorms(Genten::NormOne,nrms,0.0);
  nrms_chk[0] = 7;
  ASSERT(nrms.isEqual(nrms_chk,MACHINE_EPSILON),
         "ColNorms (1-norm) works as expected");
  a.colNorms(Genten::NormTwo,nrms,0.0);
  nrms_chk[0] = 5;
  ASSERT(nrms.isEqual(nrms_chk,MACHINE_EPSILON),
         "ColNorms (2-norm) works as expected");

  // colScale
  Genten::Array weights(3);
  weights[0] = 3;
  weights[1] = 2;
  weights[2] = 1;
  b = Genten::FacMatrix(a.nRows(), a.nCols());
  b.deep_copy(a);
  a.colScale(weights, false);
  tf = false;
  for (ttb_indx i = 0; i < 3; i ++)
  {
    for (ttb_indx j = 0; j < 3; j ++)
    {
      tf = (a.entry(i,j) == (b.entry(i,j)*weights[j]));
      if (tf == false)
      {
        break;
      }
    }
    if (tf == false)
    {
      break;
    }
  }
  ASSERT(tf,"ColScale works as expected");
  a.colScale(weights,true);
  ASSERT(a.isEqual(b,MACHINE_EPSILON),
         "ColScale (inverse) works as expected");

  // permute
  ttb_real pdata[] = {1,2,3,4,5,6,7,8,9};
  ttb_real pdata_new[] = {7,8,9,4,5,6,1,2,3};
  ttb_indx idata[] = {2,1,0};

  Genten::IndxArray ind(3, idata);
  Genten::FacMatrix p(3, 3, pdata);
  Genten::FacMatrix p_new(3, 3, pdata_new);

  p.permute(ind);

  tf = false;
  for (ttb_indx i = 0; i < 3; i ++)
  {
    for (ttb_indx j = 0; j < 3; j ++)
    {
      tf = (p.entry(i,j) == p_new.entry(i,j));
      if (tf == false)
      {
        break;
      }
    }
    if (tf == false)
    {
      break;
    }
  }
  ASSERT(tf,"Permute works as expected");

  // sum TODO

  finalize();
}
