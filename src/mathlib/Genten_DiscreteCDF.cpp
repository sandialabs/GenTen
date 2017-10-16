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


#include <stdio.h>

#include "Genten_Array.hpp"
#include "Genten_DiscreteCDF.hpp"
#include "Genten_FacMatrix.hpp"

using namespace std;


//-----------------------------------------------------------------------------
//  Constructor
//-----------------------------------------------------------------------------
Genten::DiscreteCDF::DiscreteCDF (void)
{
  return;
}


//-----------------------------------------------------------------------------
//  Destructor
//-----------------------------------------------------------------------------
Genten::DiscreteCDF::~DiscreteCDF (void)
{
  return;
}


//-----------------------------------------------------------------------------
//  Public method
//-----------------------------------------------------------------------------
bool Genten::DiscreteCDF::load (const Array &  cPDF)
{
  _cCDF = Array (cPDF.size());

  if (cPDF.size() == 1)
  {
    _cCDF[0] = 1.0;
    return( true );
  }

  for (ttb_indx  i = 0; i < cPDF.size(); i++)
  {
    ttb_real  dNext = cPDF[i];
    if ((dNext < 0.0) || (dNext >= 1.0))
    {
      cout << "*** Bad input to DiscreteCDF.load:  ("
           << i << ") = " << dNext << "\n";
      return( false );
    }
    if (i == 0)
      _cCDF[i] = dNext;
    else
      _cCDF[i] = dNext + _cCDF[i-1];
  }

  ttb_real  dTotal = _cCDF[_cCDF.size() - 1];
  if ((sizeof(ttb_real) == 8 && fabs (dTotal - 1.0) > 1.0e-14) ||
      (sizeof(ttb_real) == 4 && fabs (dTotal - 1.0) > 1.0e-6))
  {
    printf ("*** Bad input to DiscreteCDF.load: "
            " sums to %24.16f instead of 1 (error %e).\n",
            dTotal, fabs (dTotal - 1.0));
    return( false );
  }

  return( true );
}


//-----------------------------------------------------------------------------
//  Public method
//-----------------------------------------------------------------------------
bool Genten::DiscreteCDF::load (const FacMatrix &  cPDF,
                                const ttb_indx     nColumn)
{
  _cCDF = Array (cPDF.nRows());

  for (ttb_indx  r = 0; r < cPDF.nRows(); r++)
  {
    ttb_real  dNext = cPDF.entry (r, nColumn);
    if ((dNext < 0.0) || (dNext >= 1.0))
    {
      cout << "*** Bad input to DiscreteCDF.load:  ("
           << r << "," << nColumn << ") = " << dNext << "\n";
      return( false );
    }
    if (r == 0)
      _cCDF[r] = dNext;
    else
      _cCDF[r] = dNext + _cCDF[r-1];
  }

  ttb_real  dTotal = _cCDF[_cCDF.size() - 1];
  if ((sizeof(ttb_real) == 8 && fabs (dTotal - 1.0) > 1.0e-12) ||
      (sizeof(ttb_real) == 4 && fabs (dTotal - 1.0) > 1.0e-4))
  {
    printf ("*** Bad input to DiscreteCDF.load: "
            " sums to %18.16f instead of 1 (error %e).\n",
            dTotal, fabs (dTotal - 1.0));
    return( false );
  }

  return( true );
}


//-----------------------------------------------------------------------------
//  Public method
//-----------------------------------------------------------------------------
ttb_indx  Genten::DiscreteCDF::getRandomSample (ttb_real  dRandomNumber)
{
  const ttb_indx  nMAXLEN_FULLSEARCH = 16;

  // If the histogram is short, then just walk thru it to find the index.
  if (_cCDF.size() < nMAXLEN_FULLSEARCH)
  {
    for (ttb_indx  i = 0; i < _cCDF.size(); i++)
    {
      if (dRandomNumber < _cCDF[i])
        return(i);
    }
    return( _cCDF.size() - 1 );
  }

  // For a longer histogram, use a binary search.
  ttb_indx  nStart = 0;
  ttb_indx  nEnd = _cCDF.size() - 1;
  while (true)
  {
    if ((nEnd - nStart) <= 1)
    {
      if (dRandomNumber < _cCDF[nStart])
        return( nStart );
      else
        return( nEnd);
    }
    ttb_indx  nMiddle = (nStart + nEnd) / 2;
    if (dRandomNumber < _cCDF[nMiddle])
      nEnd = nMiddle;
    else
      nStart = nMiddle;
  }
}
