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

#include "Genten_IndxArray.h"
#include <cstring>
#include <cstdlib>

Genten::IndxArray::
IndxArray(ttb_indx n) :
  data("Genten::IndxArray::data", n)
{
}

Genten::IndxArray::
IndxArray(ttb_indx n, ttb_indx val) :
  data("Genten::IndxArray::data", n)
{
  Kokkos::deep_copy( data, val );
}

Genten::IndxArray::
IndxArray(ttb_indx n, ttb_indx * v) :
  data("Genten::IndxArray::data", n)
{
  unmanaged_const_view_type v_view(v,n);
  Kokkos::deep_copy(data, v_view);
}

Genten::IndxArray::
IndxArray(ttb_indx n, const ttb_real * v) :
  data("Genten::IndxArray::data", n)
{
  // This is a horribly slow loop. We know that the doubles are really storing positive integers, so it would be mightly nice if we could just pull out the mantissa or something elegant like that.
  for (ttb_indx i = 0; i < n; i ++)
  {
    data[i] = (ttb_indx) v[i];
  }
}

Genten::IndxArray::
IndxArray(const IndxArray & src, ttb_indx n):
  data("Genten::IndxArray::data", src.data.dimension_0()-1)
{
  const ttb_indx sz = data.dimension_0();
  Kokkos::deep_copy( Kokkos::subview( data,     std::make_pair(ttb_indx(0),n) ),
                     Kokkos::subview( src.data, std::make_pair(ttb_indx(0),n) ) );
  Kokkos::deep_copy( Kokkos::subview( data,     std::make_pair(n,sz) ),
                     Kokkos::subview( src.data, std::make_pair(n+1,sz+1) ) );
}

ttb_bool Genten::IndxArray::
operator==(const Genten::IndxArray & a) const
{
  const ttb_indx sz = data.dimension_0();
  if (sz != a.data.dimension_0())
  {
    return false;
  }

  for (ttb_indx i = 0; i < sz; i ++)
  {
    if (data[i] != a.data[i])
    {
      return false;
    }
  }

  return true;
}

ttb_bool Genten::IndxArray::
operator<=(const Genten::IndxArray & a) const
{
  const ttb_indx sz = data.dimension_0();
  if (sz != a.data.dimension_0())
    Genten::error("Genten::IndxArray::operator<= not comparable (different sizes).");

  for (ttb_indx i = 0; i < sz; i ++)
  {
    if (data[i] < a.data[i])
      return true;
    if (data[i] > a.data[i])
      return false;
  }

  // They're exactly equal
  return true;
}

ttb_indx & Genten::IndxArray::
at(ttb_indx i) const
{
  if (i < data.dimension_0()) {
    return(data[i]);
  }
  Genten::error("Genten::IndxArray::at ref - input i >= array size.");
  return data[0]; // notreached
}

ttb_indx Genten::IndxArray::
prod(ttb_indx i, ttb_indx j, ttb_indx dflt) const
{
  if (j <= i)
  {
    return(dflt);
  }

  ttb_indx p = 1;
  for (ttb_indx k = i; k < j; k ++)
  {
    p = p * data[k];
  }
  return(p);
}

void Genten::IndxArray::
cumprod(const Genten::IndxArray & src)
{
  const ttb_indx sz = data.dimension_0();
  if (sz != src.data.dimension_0()+1)
    Genten::error("Genten::IndxArray::cumprod not comparable (different sizes).");
  data[0] = 1;
  for (ttb_indx i = 0; i < sz-1; i++)
  {
    data[i+1] = data[i] * src.data[i];
  }
}

void Genten::IndxArray::
zero()
{
  Kokkos::deep_copy( data, ttb_indx(0) );
}

ttb_bool Genten::IndxArray::
isPermutation() const
{
  const ttb_indx sz = data.dimension_0();
  const ttb_indx invalid = ttb_indx(-1);
  Genten::IndxArray chk(sz);
  chk.zero();
  for (ttb_indx i = 0; i < sz; i ++)
  {
    if ((data[i] == invalid) || data[i] > (sz-1))
    {
      return(false);
    }
    chk.data[data[i]] += 1;
  }
  for (ttb_indx i = 0; i < sz; i ++)
  {
    if (chk.data[i] != 1)
    {
      return(false);
    }
  }
  return(true);
}

void Genten::IndxArray::
increment(const Genten::IndxArray & dims)
{
  const ttb_indx sz = data.dimension_0();
  if (sz != dims.data.dimension_0())
    Genten::error("Genten::IndxArray::increment different sizes");

  ttb_indx nd = dims.size();

  // increment least significant index in lex order (rightmost)
  data[nd-1]++;

  // handle carries if necessary
  for (ttb_indx i = nd-1; i > 0 && data[i] == dims.data[i]; i--)
  {
    data[i] = 0;
    data[i-1]++;
  }
  // most significant index will not be limited by dims[0]
}
