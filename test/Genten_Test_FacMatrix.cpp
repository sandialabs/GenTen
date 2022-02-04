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


#include "Genten_FacMatrix.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Test_Utils.hpp"
#include "Genten_Util.hpp"

using namespace Genten::Test;

template <typename ExecSpace>
void Genten_Test_FacMatrix_Space(int infolevel, const std::string & datadir)
{
  typedef ExecSpace exec_space;
  typedef Genten::DefaultHostExecutionSpace host_exec_space;

  SETUP_DISABLE_CERR;

  bool tf;
  std::string space_name = Genten::SpaceProperties<exec_space>::name();
  initialize("Tests on Genten::FacMatrix (" + space_name + ")", infolevel);

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
  Genten::FacMatrixT<exec_space> b_dev = create_mirror_view( exec_space(), b );
  Genten::FacMatrixT<exec_space> e_dev = create_mirror_view( exec_space(), e );
  deep_copy( b_dev, b );
  e_dev.gramian(b_dev, true);
  deep_copy( e, e_dev );
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
  Genten::FacMatrixT<exec_space> f_dev = create_mirror_view( exec_space(), f );
  deep_copy( f_dev, f );
  f_dev.times(b_dev);
  deep_copy( f, f_dev );
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
  Genten::FacMatrixT<exec_space> a_dev = create_mirror_view( exec_space(), a );
  Genten::ArrayT<exec_space> h_dev = create_mirror_view( exec_space(), h );
  deep_copy( a_dev, a );
  deep_copy( h_dev, h );
  a_dev.oprod(h_dev);
  deep_copy( a, a_dev );
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
  a_dev = create_mirror_view( exec_space(), a );
  b_dev = create_mirror_view( exec_space(), b );
  deep_copy( a_dev, a );
  deep_copy( b_dev, b );
  b_dev.solveTransposeRHS (a_dev);
  deep_copy( b, b_dev );
  c = Genten::FacMatrix(1,2);
  c.entry(0,0) = 3.0;
  c.entry(0,1) = 2.0;
  // Very slightly loosening of tolerance for GPU
  ASSERT(c.isEqual(b,MACHINE_EPSILON*10.), "Solve works for diagonal matrix");

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
  c = Genten::FacMatrix(3,2);
  c.entry(0,0) = -1.0 / 3.0;
  c.entry(0,1) =  2.0 / 3.0;
  c.entry(1,0) =  2.0 / 3.0;
  c.entry(1,1) = -1.0 / 3.0;
  c.entry(2,0) =  5.0 / 3.0;
  c.entry(2,1) = -4.0 / 3.0;
  a_dev = create_mirror_view( exec_space(), a );
  deep_copy( a_dev, a );

  Genten::FacMatrixT<exec_space> b1_dev( b.nRows(), b.nCols() );
  deep_copy( b1_dev, b );
  b1_dev.solveTransposeRHS (a_dev, true);
  auto b1 = create_mirror_view( host_exec_space(), b1_dev );
  deep_copy( b1, b1_dev );
  ASSERT(c.isEqual(b1,ttb_real(10.0)*MACHINE_EPSILON),
         "Solve (full) works for indefinite matrix");

  // Symmetric, indefinite solver currently doesn't work on GPU
  // (solver not fully implemented in cuSOLVER)
  if (!Genten::is_gpu_space<exec_space>::value) {
    Genten::FacMatrixT<exec_space> b2_dev( b.nRows(), b.nCols() );
    deep_copy( b2_dev, b );
    b2_dev.solveTransposeRHS (a_dev, false, Genten::Upper, true);
    auto b2 = create_mirror_view( host_exec_space(), b2_dev );
    deep_copy( b2, b2_dev );
    ASSERT(c.isEqual(b2,MACHINE_EPSILON),
           "Solve (sym, assume spd) works for indefinite matrix");

    Genten::FacMatrixT<exec_space> b3_dev( b.nRows(), b.nCols() );
    deep_copy( b3_dev, b );
    b3_dev.solveTransposeRHS (a_dev, false, Genten::Upper, false);
    auto b3 = create_mirror_view( host_exec_space(), b3_dev );
    deep_copy( b3, b3_dev );
    ASSERT(c.isEqual(b3,MACHINE_EPSILON),
           "Solve (sym, assume indefinite) works for indefinite matrix");
  }

  // colNorms
  Genten::Array nrms(3), nrms_chk(3);
  // set a = [3 0 0; 4 1 0; 0 0 0]
  a = Genten::FacMatrix(3,3);
  a.entry(0,0) = 3;
  a.entry(1,0) = 4;
  a.entry(1,1) = 1;
  a_dev = create_mirror_view( exec_space(), a );
  Genten::ArrayT<exec_space> nrms_dev = create_mirror_view( exec_space(), nrms );
  deep_copy( a_dev, a );
  a_dev.colNorms(Genten::NormInf,nrms_dev,0.0);
  deep_copy( nrms, nrms_dev );
  nrms_chk[0] = 4;
  nrms_chk[1] = 1;
  nrms_chk[2] = 0;
  ASSERT(nrms.isEqual(nrms_chk,MACHINE_EPSILON),
         "ColNorms (max norm) works as expected");
  a_dev.colNorms(Genten::NormOne,nrms_dev,0.0);
  deep_copy( nrms, nrms_dev );
  nrms_chk[0] = 7;
  ASSERT(nrms.isEqual(nrms_chk,MACHINE_EPSILON),
         "ColNorms (1-norm) works as expected");
  a_dev.colNorms(Genten::NormTwo,nrms_dev,0.0);
  deep_copy( nrms, nrms_dev );
  nrms_chk[0] = 5;
  ASSERT(nrms.isEqual(nrms_chk,MACHINE_EPSILON),
         "ColNorms (2-norm) works as expected");

  // colScale
  Genten::Array weights(3);
  weights[0] = 3;
  weights[1] = 2;
  weights[2] = 1;
  b = Genten::FacMatrix(a.nRows(), a.nCols());
  deep_copy(b,a);
  Genten::ArrayT<exec_space> weights_dev =
    create_mirror_view( exec_space(), weights );
  deep_copy( weights_dev, weights );
  a_dev.colScale(weights_dev, false);
  deep_copy( a, a_dev );
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
  a_dev.colScale(weights_dev,true);
  deep_copy( a, a_dev );
  ASSERT(a.isEqual(b,MACHINE_EPSILON),
         "ColScale (inverse) works as expected");

  // permute
  ttb_real pdata[] = {1,2,3,4,5,6,7,8,9};
  ttb_real pdata_new[] = {7,8,9,4,5,6,1,2,3};
  ttb_indx idata[] = {2,1,0};

  Genten::IndxArray ind(3, idata);
  Genten::FacMatrix p(3, 3, pdata);
  Genten::FacMatrix p_new(3, 3, pdata_new);

  Genten::FacMatrixT<exec_space> p_dev = create_mirror_view( exec_space(), p );
  deep_copy( p_dev, p );
  p_dev.permute(ind);
  deep_copy( p, p_dev );

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

  // innerprod
  const ttb_indx m = 50;
  const ttb_indx n = 20;
  a = Genten::FacMatrix(m,n);
  b = Genten::FacMatrix(m,n);
  Genten::Array w(n);
  ttb_real ip_true = 0.0;
  for (ttb_indx i=0; i<m; ++i) {
    for (ttb_indx j=0; j<n; ++j) {
      a.entry(i,j) = i+j;
      b.entry(i,j) = 10*(i+j);
      w[j] = j+1;
      ip_true += w[j]*a.entry(i,j) * b.entry(i,j);
    }
  }
  a_dev = create_mirror_view( exec_space(), a );
  b_dev = create_mirror_view( exec_space(), b );
  Genten::ArrayT<exec_space> w_dev = create_mirror_view( exec_space(), w );
  deep_copy( a_dev, a );
  deep_copy( b_dev, b );
  deep_copy( w_dev, w );
  ttb_real ip = a_dev.innerprod(b_dev, w_dev);
  ASSERT( EQ(ip, ip_true), "innerprod works as expected");

  // sum TODO

  finalize();
}

void Genten_Test_FacMatrix(int infolevel, const std::string & datadir) {
#ifdef KOKKOS_ENABLE_CUDA
  Genten_Test_FacMatrix_Space<Kokkos::Cuda>(infolevel, datadir);
#endif
#ifdef KOKKOS_ENABLE_HIP
  Genten_Test_FacMatrix_Space<Kokkos::Experimental::HIP>(infolevel, datadir);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  Genten_Test_FacMatrix_Space<Kokkos::OpenMP>(infolevel, datadir);
#endif
#ifdef KOKKOS_ENABLE_THREADS
  Genten_Test_FacMatrix_Space<Kokkos::Threads>(infolevel, datadir);
#endif
#ifdef KOKKOS_ENABLE_SERIAL
  Genten_Test_FacMatrix_Space<Kokkos::Serial>(infolevel, datadir);
#endif
}
