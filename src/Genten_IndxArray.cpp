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

#include "Genten_IndxArray.hpp"
#include <cstring>
#include <cstdlib>

template <typename ExecSpace>
Genten::IndxArrayT<ExecSpace>::
IndxArrayT(ttb_indx n) :
  data("Genten::IndxArray::data", n)
{
  if (Kokkos::Impl::MemorySpaceAccess< typename DefaultHostExecutionSpace::memory_space, typename ExecSpace::memory_space >::accessible)
    host_data = host_view_type(data.data(), n);
  else
    host_data = host_view_type("Genten::IndxArray::hsot_data",n);
}

template <typename ExecSpace>
Genten::IndxArrayT<ExecSpace>::
IndxArrayT(ttb_indx n, ttb_indx val) :
  IndxArrayT(n)
{
  deep_copy( data, val );
  if (!Kokkos::Impl::MemorySpaceAccess< typename DefaultHostExecutionSpace::memory_space, typename ExecSpace::memory_space >::accessible)
    deep_copy( host_data, val );
}

template <typename ExecSpace>
Genten::IndxArrayT<ExecSpace>::
IndxArrayT(ttb_indx n, ttb_indx * v) :
  IndxArrayT(n)
{
  unmanaged_const_view_type v_view(v,n);
  deep_copy(data, v_view);
  if (!Kokkos::Impl::MemorySpaceAccess< typename DefaultHostExecutionSpace::memory_space, typename ExecSpace::memory_space >::accessible)
    deep_copy( host_data, v_view );
}

template <typename ExecSpace>
Genten::IndxArrayT<ExecSpace>::
IndxArrayT(ttb_indx n, const ttb_real * v) :
  IndxArrayT(n)
{
  // This is a horribly slow loop. We know that the doubles are really storing positive integers, so it would be mightly nice if we could just pull out the mantissa or something elegant like that.
  for (ttb_indx i = 0; i < n; i ++)
  {
    data[i] = (ttb_indx) v[i];
  }
  if (!Kokkos::Impl::MemorySpaceAccess< typename DefaultHostExecutionSpace::memory_space, typename ExecSpace::memory_space >::accessible)
    deep_copy( host_data, data );
}

template <typename ExecSpace>
Genten::IndxArrayT<ExecSpace>::
IndxArrayT(const IndxArrayT<ExecSpace> & src, ttb_indx n):
  IndxArrayT(src.data.dimension_0()-1)
{
  const ttb_indx sz = data.dimension_0();
  deep_copy( Kokkos::subview( data,     std::make_pair(ttb_indx(0),n) ),
             Kokkos::subview( src.data, std::make_pair(ttb_indx(0),n) ) );
  deep_copy( Kokkos::subview( data,     std::make_pair(n,sz) ),
             Kokkos::subview( src.data, std::make_pair(n+1,sz+1) ) );
  if (!Kokkos::Impl::MemorySpaceAccess< typename DefaultHostExecutionSpace::memory_space, typename ExecSpace::memory_space >::accessible)
    deep_copy( host_data, data );
}

template <typename ExecSpace>
ttb_bool Genten::IndxArrayT<ExecSpace>::
operator==(const Genten::IndxArrayT<ExecSpace> & a) const
{
  const ttb_indx sz = host_data.dimension_0();
  if (sz != a.host_data.dimension_0())
  {
    return false;
  }

  for (ttb_indx i = 0; i < sz; i ++)
  {
    if (host_data[i] != a.host_data[i])
    {
      return false;
    }
  }

  return true;
}

template <typename ExecSpace>
ttb_bool Genten::IndxArrayT<ExecSpace>::
operator<=(const Genten::IndxArrayT<ExecSpace> & a) const
{
  const ttb_indx sz = host_data.dimension_0();
  if (sz != a.host_data.dimension_0())
    Genten::error("Genten::IndxArray::operator<= not comparable (different sizes).");

  for (ttb_indx i = 0; i < sz; i ++)
  {
    if (host_data[i] < a.host_data[i])
      return true;
    if (host_data[i] > a.host_data[i])
      return false;
  }

  // They're exactly equal
  return true;
}

template <typename ExecSpace>
ttb_indx & Genten::IndxArrayT<ExecSpace>::
at(ttb_indx i) const
{
  if (i < host_data.dimension_0()) {
    return(host_data[i]);
  }
  Genten::error("Genten::IndxArray::at ref - input i >= array size.");
  return host_data[0]; // notreached
}

template <typename ExecSpace>
ttb_indx Genten::IndxArrayT<ExecSpace>::
prod(ttb_indx i, ttb_indx j, ttb_indx dflt) const
{
  if (j <= i)
  {
    return(dflt);
  }

  ttb_indx p = 1;
  for (ttb_indx k = i; k < j; k ++)
  {
    p = p * host_data[k];
  }
  return(p);
}

template <typename ExecSpace>
void Genten::IndxArrayT<ExecSpace>::
cumprod(const Genten::IndxArrayT<ExecSpace> & src)
{
  const ttb_indx sz = host_data.dimension_0();
  if (sz != src.host_data.dimension_0()+1)
    Genten::error("Genten::IndxArray::cumprod not comparable (different sizes).");
  host_data[0] = 1;
  for (ttb_indx i = 0; i < sz-1; i++)
  {
    host_data[i+1] = host_data[i] * src.host_data[i];
  }
  deep_copy( data, host_data );
}

template <typename ExecSpace>
void Genten::IndxArrayT<ExecSpace>::
zero()
{
  deep_copy( data, ttb_indx(0) );
}

template <typename ExecSpace>
ttb_bool Genten::IndxArrayT<ExecSpace>::
isPermutation() const
{
  const ttb_indx sz = host_data.dimension_0();
  const ttb_indx invalid = ttb_indx(-1);
  Genten::IndxArrayT<ExecSpace> chk(sz);
  chk.zero();
  for (ttb_indx i = 0; i < sz; i ++)
  {
    if ((host_data[i] == invalid) || host_data[i] > (sz-1))
    {
      return(false);
    }
    chk.host_data[host_data[i]] += 1;
  }
  for (ttb_indx i = 0; i < sz; i ++)
  {
    if (chk.host_data[i] != 1)
    {
      return(false);
    }
  }
  return(true);
}

template <typename ExecSpace>
void Genten::IndxArrayT<ExecSpace>::
increment(const Genten::IndxArrayT<ExecSpace> & dims)
{
  const ttb_indx sz = host_data.dimension_0();
  if (sz != dims.host_data.dimension_0())
    Genten::error("Genten::IndxArray::increment different sizes");

  ttb_indx nd = dims.size();

  // increment least significant index in lex order (rightmost)
  host_data[nd-1]++;

  // handle carries if necessary
  for (ttb_indx i = nd-1; i > 0 && data[i] == dims.data[i]; i--)
  {
    host_data[i] = 0;
    host_data[i-1]++;
  }
  // most significant index will not be limited by dims[0]

  deep_copy( data, host_data );
}

#define INST_MACRO(SPACE) template class Genten::IndxArrayT<SPACE>;
GENTEN_INST(INST_MACRO)
