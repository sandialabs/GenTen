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

/*!
  @file Genten_Array.cpp
  @brief Data class for "flat" versions of vectors, matrices and tensors.
*/

#include <assert.h>
#include <cstring>
#include <limits>

#include "Genten_Array.h"
#include "Genten_MathLibs_Wpr.h"
#include "Genten_portability.h"
#include "Genten_RandomMT.h"
#include "Genten_Util.h"

Genten::Array::
Array(ttb_indx n, bool parallel):
  data("Genten::Array::data", n)
{
}

Genten::Array::Array(ttb_indx n, ttb_real val):
  data("Genten::Array::data", n)
{
  Kokkos::deep_copy(data, val);
}


Genten::Array::Array(ttb_indx n, ttb_real * d, ttb_bool shdw):
  data()
{
  if (!shdw)
  {
    data = view_type("Genten::Array::data", n);
    unmanaged_const_view_type d_view(d,n);
    Kokkos::deep_copy(data, d_view);
  }
  else
  {
    data = view_type(d, n);
  }
}

void Genten::Array::copyFrom(ttb_indx n, const ttb_real * src)
{
  if (n != data.dimension_0())
  {
    error("Genten::Array::copy - Destination array is not the correct size");
  }
  unmanaged_const_view_type src_view(src,n);
  Kokkos::deep_copy(data, src_view);
}

void Genten::Array::copyTo(ttb_indx n, ttb_real * dest) const
{
  if (n != data.dimension_0())
  {
    error("Genten::Array::copy - Destination array is not the correct size");
  }
  unmanaged_view_type dest_view(dest,n);
  Kokkos::deep_copy(dest_view, data);
}

void Genten::Array::operator=(ttb_real val)
{
  Kokkos::deep_copy(data, val);
}

void Genten::Array::rand()
{
  RandomMT  cRNG(0);
  const ttb_indx sz = data.dimension_0();
  for (ttb_indx  i = 0; i < sz; i++)
  {
    data[i] = cRNG.genrnd_double();
  }
  return;
}

void Genten::Array::scatter (const bool        bUseMatlabRNG,
                                 RandomMT &  cRMT)
{
  const ttb_indx sz = data.dimension_0();
  for (ttb_indx  i = 0; i < sz; i++)
  {
    ttb_real  dNextRan;
    if (bUseMatlabRNG)
      dNextRan = cRMT.genMatlabMT();
    else
      dNextRan = cRMT.genrnd_double();
    data[i] = dNextRan;
    /*TBD...legacy randomization
      data[i]= 0;
      while (data[i] == 0.0) {
      data[i] = drand48();
      }
    */
  }
  return;
}

bool Genten::Array::operator==(const Genten::Array & a) const
{
  const ttb_indx sz = data.dimension_0();
  if (sz != a.data.dimension_0())
  {
    return(false);
  }

  for (ttb_indx i = 0; i < sz; i ++)
  {
    if (data[i] != a.data[i])
    {
      return(false);
    }
  }

  return(true);
}

bool Genten::Array::hasNonFinite(ttb_indx &where) const
{
  const ttb_indx sz = data.dimension_0();
  for  (ttb_indx i = 0; i < sz; i ++)
  {
    if (isRealValid(data[i]) == false) {
      where = i;
      return true;
    }
  }
  return false;
}

ttb_real & Genten::Array::at(ttb_indx i) const
{
  const ttb_indx sz = data.dimension_0();
  if ( i < sz) {
    return(data[i]);
  }
  Genten::error("Genten::Array::at - out-of-bounds error");
  return data[0]; // not reached
}

ttb_real Genten::Array::norm(Genten::NormType ntype) const
{
  const ttb_indx sz = data.dimension_0();
  ttb_real nrm;
  switch(ntype)
  {
  case NormOne:
  {
    nrm = Genten::nrm1(sz, data.data(), 1);
    break;
  }
  case NormTwo:
  {
    nrm = Genten::nrm2(sz, data.data(), 1);
    break;
  }
  case NormInf:
  {
    ttb_indx idx = Genten::imax(sz, data.data(), 1);
    nrm = fabs(data[idx]);
    break;
  }
  default:
  {
    error("Genten::Array::norm - unimplemented norm type");
  }
  }
  return(nrm);
}


ttb_indx Genten::Array::nnz() const
{
  const ttb_indx sz = data.dimension_0();
  ttb_indx cnt = 0;
  for (ttb_indx i = 0; i < sz; i ++)
  {
    if (data[i] != 0)
    {
      cnt ++;
    }
  }
  return(cnt);
}

ttb_real Genten::Array::dot(const Genten::Array & y) const
{
  const ttb_indx sz = data.dimension_0();
  if (sz != y.data.dimension_0())
  {
    Genten::error("Genten::Array::dot - Size mismatch");
  }

  return(Genten::dot(sz, data.data(), 1, y.data.data(), 1));
}


bool Genten::Array::isEqual(const Genten::Array & y, ttb_real tol) const
{
  // Check for equal sizes.
  const ttb_indx sz = data.dimension_0();
  if (sz != y.data.dimension_0())
  {
    return( false );
  }

  // Check that elements are equal.
  for (ttb_indx  i = 0; i < sz; i++)
  {
    if (Genten::isEqualToTol(data[i], y.data[i], tol) == false)
      return( false );
  }

  return(true);
}


void Genten::Array::times(ttb_real a)
{
  const ttb_indx sz = data.dimension_0();
  Genten::scal(sz, a, data.data(), 1);
}

void Genten::Array::times(ttb_real a, const Genten::Array & y)
{
  const ttb_indx sz = data.dimension_0();
  if (sz != y.data.dimension_0())
  {
    Genten::error("Genten::Array::dot - Size mismatch");
  }
  Genten::copy(sz, y.data.data(), 1, data.data(), 1);
  Genten::scal(sz, a, data.data(), 1);
}

void Genten::Array::invert(ttb_real a)
{
  const ttb_indx sz = data.dimension_0();
  for (ttb_indx i = 0; i < sz; i ++)
  {
    data[i] = a / data[i];
  }
}

void Genten::Array::invert(ttb_real a, const Genten::Array & y)
{
  const ttb_indx sz = data.dimension_0();
  if (sz != y.data.dimension_0())
  {
    Genten::error("Genten::Array::dot - Size mismatch");
  }
  for (ttb_indx i = 0; i < data.dimension_0(); i ++)
  {
    data[i] = a / y.data[i];
  }
}
void Genten::Array::power(ttb_real a)
{
  const ttb_indx sz = data.dimension_0();
  for (ttb_indx i = 0; i < sz; i ++)
  {
    data[i] = pow(data[i], a);
  }
}

void Genten::Array::power(ttb_real a, const Genten::Array & y)
{
  const ttb_indx sz = data.dimension_0();
  if (sz != y.data.dimension_0())
  {
    Genten::error("Genten::Array::dot - Size mismatch");
  }
  for (ttb_indx i = 0; i < sz; i ++)
  {
    data[i] = pow(y.data[i], a);
  }
}


void Genten::Array::shift(ttb_real a)
{
  const ttb_indx sz = data.dimension_0();
  for (ttb_indx i = 0; i < sz; i ++)
  {
    data[i] = data[i] + a;
  }
}

void Genten::Array::shift(ttb_real a, const Genten::Array & y)
{
  const ttb_indx sz = data.dimension_0();
  if (sz != y.data.dimension_0())
  {
    Genten::error("Genten::Array::dot - Size mismatch");
  }
  for (ttb_indx i = 0; i < sz; i ++)
  {
    data[i] = y.data[i] + a;
  }
}

void Genten::Array::plus(const Genten::Array & y)
{
  const ttb_indx sz = data.dimension_0();
  if (sz != y.data.dimension_0())
  {
    Genten::error("Genten::Array::plus (one input) - size mismatch");
  }

  Genten::axpy(sz, 1.0, y.data.data(), 1, data.data(), 1);
}

// x = x + sum(y[i] )
void Genten::Array::plusVec(std::vector< const Array * > y)
{
  const ttb_indx sz = data.dimension_0();
  std::vector< const Array * >::iterator it;
  for (it = y.begin(); it < y.end(); it++) {
    if (sz != (*it)->data.dimension_0())
    {
      Genten::error("Genten::Array::plusVec - size mismatch");
    }
  }
  size_t nptrs = y.size();
  const ttb_real ** dptr = new const ttb_real *[nptrs];
  ttb_real * RESTRICT atmp = data.data();
  for (size_t i=0; i< nptrs; i++) {
    dptr[i] = y[i]->data.data();
  }
  for (ttb_indx i=0;i< sz; i++)
  {
    ttb_real stmp = 0.0;
    for (size_t j=0; j< nptrs; j++) {
      stmp += dptr[j][i];
    }
    atmp[i] += stmp;
  }
  delete dptr;
}


void Genten::Array::plus(const Genten::Array & y, const Genten::Array & z)
{
  const ttb_indx sz = data.dimension_0();
  if (y.data.dimension_0() != z.data.dimension_0())
  {
    Genten::error("Genten::Array::plus (two inputs) - size mismatch");
  }

  deep_copy(y);
  Genten::axpy(sz, 1.0, z.data.data(), 1, data.data(), 1);
}

void Genten::Array::minus(const Genten::Array & y)
{
  const ttb_indx sz = data.dimension_0();
  if (sz != y.data.dimension_0())
  {
    Genten::error("Genten::Array::minus (one input) - size mismatch");
  }

  Genten::axpy(sz, -1.0, y.data.data(), 1, data.data(), 1);
}

void Genten::Array::minus(const Genten::Array & y, const Genten::Array & z)
{
  const ttb_indx sz = data.dimension_0();
  if (y.data.dimension_0() != z.data.dimension_0())
  {
    Genten::error("Genten::Array::minus (two inputs) - size mismatch");
  }

  deep_copy(y);
  Genten::axpy(sz, -1.0, z.data.data(), 1, data.data(), 1);
}

void Genten::Array::times(const Genten::Array & y)
{
  const ttb_indx sz = data.dimension_0();
  if (sz != y.data.dimension_0())
  {
    Genten::error("Genten::Array::times (one input) - size mismatch");
  }

  vmul(sz, data.data(), y.data.data());

}

void Genten::Array::times(const Genten::Array & y, const Genten::Array & z)
{
  const ttb_indx sz = data.dimension_0();
  if (y.data.dimension_0() != z.data.dimension_0())
  {
    Genten::error("Genten::Array::times (two inputs) - size mismatch");
  }

  for (ttb_indx i = 0; i < sz; i ++)
  {
    data[i] = y.data[i] * z.data[i];
  }
}

void Genten::Array::divide(const Genten::Array & y)
{
  const ttb_indx sz = data.dimension_0();
  if (sz != y.data.dimension_0())
  {
    Genten::error("Genten::Array::divide (one input) - size mismatch");
  }

  for (ttb_indx i = 0; i < sz; i ++)
  {
    data[i] /= y.data[i];
  }
}

void Genten::Array::divide(const Genten::Array & y, const Genten::Array & z)
{
  const ttb_indx sz = data.dimension_0();
  if (y.data.dimension_0() != z.data.dimension_0())
  {
    Genten::error("Genten::Array::divide (two inputs) - size mismatch");
  }

  for (ttb_indx i = 0; i < sz; i ++)
  {
    data[i] = y.data[i] / z.data[i];
  }
}

ttb_real Genten::Array::sum() const
{
  const ttb_indx sz = data.dimension_0();
  ttb_real value = 0;
  for (ttb_indx i = 0; i < sz; i ++)
  {
    value += data[i];
  }
  return value;
}
