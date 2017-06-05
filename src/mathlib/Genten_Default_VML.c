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

#include "stddef.h"

void vdmul(const ptrdiff_t n, double * a, const double * b)
{
  // Quick return if possible
  if (n == 0)
    return;

  switch (n)
  {
  case 1:
    a[0] *= b[0];
    return;
  case 2:
    a[0] *= b[0];
    a[1] *= b[1];
    return;
  case 3:
    a[0] *= b[0];
    a[1] *= b[1];
    a[2] *= b[2];
    return;
  default:
  {
    ptrdiff_t i;
    // unroll loop 4 times
    for (i = 0; i <= n-4; i = i+4)
    {
      a[i]   *= b[i];
      a[i+1] *= b[i+1];
      a[i+2] *= b[i+2];
      a[i+3] *= b[i+3];
    }
    // take care of last few with recursive call
    vdmul(n-i, a+i, b+i);
  }
  }

  return;

}

void vsmul(const ptrdiff_t n, float * a, const float * b)
{
  // Quick return if possible
  if (n == 0)
    return;

  switch (n)
  {
  case 1:
    a[0] *= b[0];
    return;
  case 2:
    a[0] *= b[0];
    a[1] *= b[1];
    return;
  case 3:
    a[0] *= b[0];
    a[1] *= b[1];
    a[2] *= b[2];
    return;
  default:
  {
    ptrdiff_t i;
    // unroll loop 4 times
    for (i = 0; i <= n-4; i = i+4)
    {
      a[i]   *= b[i];
      a[i+1] *= b[i+1];
      a[i+2] *= b[i+2];
      a[i+3] *= b[i+3];
    }
    // take care of last few with recursive call
    vsmul(n-i, a+i, b+i);
  }
  }

  return;

}
