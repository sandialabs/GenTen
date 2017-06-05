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


#include <iostream>

#include "Genten_Util.h"

#include "Kokkos_Core.hpp"

using namespace std;

// Forward declarations.
void Genten_Test_Array(int infolevel);
void Genten_Test_CpAls(int infolevel);
void Genten_Test_FacMatrix(int infolevel, const string & dirname);
void Genten_Test_IndxArray(int infolevel);
void Genten_Test_IO(int infolevel, const string & dirname);
void Genten_Test_Ktensor(int infolevel);
void Genten_Test_MixedFormats(int infolevel);
void Genten_Test_Sptensor(int infolevel);

int main(int argc, char * argv[])
{

  Kokkos::initialize(argc, argv);

  // Level 0 is minimal output, 1 is more verbose.
  int infolevel = 0;
  if (argc == 2)
  {
    infolevel = atoi(argv[1]);
    if (infolevel < 0)
      infolevel = 0;
  }

  Genten_Test_Array(infolevel);
  Genten_Test_IndxArray(infolevel);
  Genten_Test_FacMatrix(infolevel, "./data/");
  Genten_Test_Sptensor(infolevel);
  Genten_Test_Ktensor(infolevel);
  Genten_Test_MixedFormats(infolevel);
  Genten_Test_IO(infolevel, "./data/");
  Genten_Test_CpAls(infolevel);

  cout << "Unit tests complete for " << Genten::getGentenVersion() << endl;

  Kokkos::finalize();

  return( 0 );
}
