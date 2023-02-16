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

/*!
  @file Genten_Array.cpp
  @brief Data class for "flat" versions of vectors, matrices and tensors.
*/

#include <cstring>
#include <limits>

#include "Genten_Array.hpp"
#include "Genten_MathLibs_Wpr.hpp"
#include "Genten_portability.hpp"
#include "Genten_RandomMT.hpp"
#include "Genten_Util.hpp"
#include "Genten_Kokkos.hpp"
#include "Kokkos_Random.hpp"

#ifdef HAVE_CALIPER
#include <caliper/cali.h>
#endif

template <typename ExecSpace>
Genten::ArrayT<ExecSpace>::
ArrayT(ttb_indx n, bool parallel):
  data("Genten::ArrayT::data", n)
{
}

template <typename ExecSpace>
Genten::ArrayT<ExecSpace>::
ArrayT(ttb_indx n, ttb_real val):
  data("Genten::ArrayT::data", n)
{
  deep_copy(data, val);
}

template <typename ExecSpace>
Genten::ArrayT<ExecSpace>::
ArrayT(ttb_indx n, ttb_real * d, ttb_bool shdw):
  data()
{
  if (!shdw)
  {
    data = view_type(Kokkos::view_alloc("Genten::ArrayT::data",
                                        Kokkos::WithoutInitializing), n);
    unmanaged_const_view_type d_view(d,n);
    deep_copy(data, d_view);
  }
  else
  {
    data = view_type(d, n);
  }
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
copyFrom(ttb_indx n, const ttb_real * src) const
{
  if (n != data.extent(0))
  {
    error("Genten::ArrayT::copy - Destination array is not the correct size");
  }
  //unmanaged_const_view_type src_view(src,n);
  //deep_copy(data, src_view);
  view_type my_data = data;
  Kokkos::parallel_for("Genten::Array::copyFrom",
                       Kokkos::RangePolicy<ExecSpace>(0,n),
                       KOKKOS_LAMBDA(const ttb_indx i)
  {
    my_data(i) = src[i];
  });
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
copyTo(ttb_indx n, ttb_real * dest) const
{
  if (n != data.extent(0))
  {
    error("Genten::ArrayT::copy - Destination array is not the correct size");
  }
  unmanaged_view_type dest_view(dest,n);
  deep_copy(dest_view, data);
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
operator=(ttb_real val) const
{
  deep_copy(data, val);
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
operator=(ttb_real val)
{
  deep_copy(data, val);
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
rand() const
{
  RandomMT  cRNG(0);
  const ttb_indx sz = data.extent(0);
  for (ttb_indx  i = 0; i < sz; i++)
  {
    data[i] = cRNG.genrnd_double();
  }
  return;
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
scatter (const bool bUseMatlabRNG,
         const bool bUseParallelRNG,
         RandomMT &  cRMT) const
{
  const ttb_indx sz = data.extent(0);
  if (bUseParallelRNG)
  {
    const ttb_indx seed = cRMT.genrnd_int32();
    Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(seed);
    ttb_real min_val = 0.0;
    ttb_real max_val = 1.0;
    Kokkos::fill_random(data, rand_pool, min_val, max_val);
  }
  else
  {
    auto d = create_mirror_view(data);
    for (ttb_indx  i = 0; i < sz; i++)
    {
      ttb_real  dNextRan;
      if (bUseMatlabRNG)
        dNextRan = cRMT.genMatlabMT();
      else
        dNextRan = cRMT.genrnd_double();
      d[i] = dNextRan;
      /*TBD...legacy randomization
        data[i]= 0;
        while (data[i] == 0.0) {
        data[i] = drand48();
        }
      */
    }
    deep_copy(data, d);
  }
  return;
}

template <typename ExecSpace>
bool Genten::ArrayT<ExecSpace>::
operator==(const Genten::ArrayT<ExecSpace> & a) const
{
  const ttb_indx sz = data.extent(0);
  if (sz != a.data.extent(0))
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

template <typename ExecSpace>
bool Genten::ArrayT<ExecSpace>::
hasNonFinite(ttb_indx &where) const
{
  const ttb_indx sz = data.extent(0);
  for  (ttb_indx i = 0; i < sz; i ++)
  {
    if (isRealValid(data[i]) == false) {
      where = i;
      return true;
    }
  }
  return false;
}

template <typename ExecSpace>
ttb_real & Genten::ArrayT<ExecSpace>::
at(ttb_indx i) const
{
  const ttb_indx sz = data.extent(0);
  if ( i < sz) {
    return(data[i]);
  }
  Genten::error("Genten::ArrayT::at - out-of-bounds error");
  return data[0]; // not reached
}

template <typename ExecSpace>
ttb_real Genten::ArrayT<ExecSpace>::
norm(Genten::NormType ntype) const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::Array::norm");
#endif

  const ttb_indx sz = data.extent(0);
  Kokkos::RangePolicy<ExecSpace> policy(0,sz);
  auto my_data = data; // can't capture *this by value
  ttb_real nrm;
  switch(ntype)
  {
  case NormOne:
  {
    //nrm = Genten::nrm1(sz, data.data(), 1);
    Kokkos::parallel_reduce("Genten::Array::norm_1_kernel",
                            policy,
                            KOKKOS_LAMBDA(const ttb_indx i, ttb_real& t)
    {
      ttb_real v = my_data(i);
      t += (v >= 0 ? v : -v);
    }, nrm);
    Kokkos::fence();
    break;
  }
  case NormTwo:
  {
    //nrm = Genten::nrm2(sz, data.data(), 1);
    Kokkos::parallel_reduce("Genten::Array::norm_2_kernel",
                            policy,
                            KOKKOS_LAMBDA(const ttb_indx i, ttb_real& t)
    {
      t += my_data(i)*my_data(i);
    }, nrm);
    Kokkos::fence();
    nrm = std::sqrt(nrm);
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
    error("Genten::ArrayT::norm - unimplemented norm type");
  }
  }
  return(nrm);
}

template <typename ExecSpace>
ttb_indx Genten::ArrayT<ExecSpace>::
nnz() const
{
  const ttb_indx sz = data.extent(0);
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

template <typename ExecSpace>
ttb_real Genten::ArrayT<ExecSpace>::
dot(const Genten::ArrayT<ExecSpace> & y) const
{
  const ttb_indx sz = data.extent(0);
  if (sz != y.data.extent(0))
  {
    Genten::error("Genten::ArrayT::dot - Size mismatch");
  }

  //return(Genten::dot(sz, data.data(), 1, y.data.data(), 1));

  auto my_data = data; // can't capture *this by value
  ttb_real d = 0.0;
  Kokkos::parallel_reduce("Genten::Array::dot_kernel",
                          Kokkos::RangePolicy<ExecSpace>(0,sz),
                          KOKKOS_LAMBDA(const ttb_indx i, ttb_real& t)
  {
    t += my_data(i)*y.data(i);
  }, d);
  Kokkos::fence();

  return d;
}

template <typename ExecSpace>
bool Genten::ArrayT<ExecSpace>::
isEqual(const Genten::ArrayT<ExecSpace> & y, ttb_real tol) const
{
  // Check for equal sizes.
  const ttb_indx sz = data.extent(0);
  if (sz != y.data.extent(0))
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

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
times(ttb_real a) const
{
  const ttb_indx sz = data.extent(0);
  //Genten::scal(sz, a, data.data(), 1);
  view_type d = data;
  Kokkos::parallel_for("Genten::Array::scalar_times_kernel",
                       Kokkos::RangePolicy<ExecSpace>(0,sz),
                       KOKKOS_LAMBDA(const ttb_indx i)
  {
    d[i] *= a;
  });
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
times(ttb_real a, const Genten::ArrayT<ExecSpace> & y) const
{
  const ttb_indx sz = data.extent(0);
  if (sz != y.data.extent(0))
  {
    Genten::error("Genten::ArrayT::dot - Size mismatch");
  }
  Genten::copy(sz, y.data.data(), 1, data.data(), 1);
  Genten::scal(sz, a, data.data(), 1);
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
invert(ttb_real a) const
{
  const ttb_indx sz = data.extent(0);
  for (ttb_indx i = 0; i < sz; i ++)
  {
    data[i] = a / data[i];
  }
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
invert(ttb_real a, const Genten::ArrayT<ExecSpace> & y) const
{
  const ttb_indx sz = data.extent(0);
  if (sz != y.data.extent(0))
  {
    Genten::error("Genten::ArrayT::dot - Size mismatch");
  }
  for (ttb_indx i = 0; i < data.extent(0); i ++)
  {
    data[i] = a / y.data[i];
  }
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
power(ttb_real a) const
{
  view_type d = data;
  const ttb_indx sz = data.extent(0);
  Kokkos::parallel_for("Genten::Array::power_kernel",
                       Kokkos::RangePolicy<ExecSpace>(0,sz),
                       KOKKOS_LAMBDA(const ttb_indx i)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    using sycl::pow;
#else
    using std::pow;
#endif

    d[i] = pow(d[i], a);
  });
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
power(ttb_real a, const Genten::ArrayT<ExecSpace> & y) const
{
  const ttb_indx sz = data.extent(0);
  if (sz != y.data.extent(0))
  {
    Genten::error("Genten::ArrayT::power - Size mismatch");
  }
  for (ttb_indx i = 0; i < sz; i ++)
  {
    data[i] = pow(y.data[i], a);
  }
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
shift(ttb_real a) const
{
  const ttb_indx sz = data.extent(0);
  view_type d = data;
  Kokkos::parallel_for("Genten::Array::scalar_shift_kernel",
                       Kokkos::RangePolicy<ExecSpace>(0,sz),
                       KOKKOS_LAMBDA(const ttb_indx i)
  {
    d[i] += a;
  });
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
shift(ttb_real a, const Genten::ArrayT<ExecSpace> & y) const
{
  const ttb_indx sz = data.extent(0);
  if (sz != y.data.extent(0))
  {
    Genten::error("Genten::ArrayT::dot - Size mismatch");
  }
  for (ttb_indx i = 0; i < sz; i ++)
  {
    data[i] = y.data[i] + a;
  }
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
update(const ttb_real a, const Genten::ArrayT<ExecSpace> & y,
       const ttb_real b) const
{
  const ttb_indx sz = data.extent(0);
  if (sz != y.data.extent(0))
  {
    Genten::error("Genten::ArrayT::update - size mismatch");
  }

  view_type d = data;
  view_type yd = y.data;
  Kokkos::parallel_for("Genten::Array::update_kernel",
                       Kokkos::RangePolicy<ExecSpace>(0,sz),
                       KOKKOS_LAMBDA(const ttb_indx i)
 {
   d[i] = a*yd[i] + b*d[i];
 });
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
plus(const Genten::ArrayT<ExecSpace> & y, const ttb_real s) const
{
  const ttb_indx sz = data.extent(0);
  if (sz != y.data.extent(0))
  {
    Genten::error("Genten::ArrayT::plus (one input) - size mismatch");
  }

  //Genten::axpy(sz, s, y.data.data(), 1, data.data(), 1);
  view_type d = data;
  view_type yd = y.data;
  Kokkos::parallel_for("Genten::Array::plus_kernel",
                       Kokkos::RangePolicy<ExecSpace>(0,sz),
                       KOKKOS_LAMBDA(const ttb_indx i)
  {
    d[i] += s*yd[i];
  });
}

// x = x + sum(y[i] )
template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
plusVec(std::vector< const ArrayT<ExecSpace> * > y) const
{
  const ttb_indx sz = data.extent(0);
  typename std::vector< const ArrayT * >::iterator it;
  for (it = y.begin(); it < y.end(); it++) {
    if (sz != (*it)->data.extent(0))
    {
      Genten::error("Genten::ArrayT::plusVec - size mismatch");
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
  delete [] dptr;
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
plus(const Genten::ArrayT<ExecSpace> & y, const Genten::ArrayT<ExecSpace> & z) const
{
  const ttb_indx sz = data.extent(0);
  if (y.data.extent(0) != z.data.extent(0))
  {
    Genten::error("Genten::ArrayT::plus (two inputs) - size mismatch");
  }

  deep_copy(*this, y);
  Genten::axpy(sz, 1.0, z.data.data(), 1, data.data(), 1);
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
minus(const Genten::ArrayT<ExecSpace> & y) const
{
  const ttb_indx sz = data.extent(0);
  if (sz != y.data.extent(0))
  {
    Genten::error("Genten::ArrayT::minus (one input) - size mismatch");
  }

  Genten::axpy(sz, -1.0, y.data.data(), 1, data.data(), 1);
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
minus(const Genten::ArrayT<ExecSpace> & y, const Genten::ArrayT<ExecSpace> & z) const
{
  const ttb_indx sz = data.extent(0);
  if (y.data.extent(0) != z.data.extent(0))
  {
    Genten::error("Genten::ArrayT::minus (two inputs) - size mismatch");
  }

  deep_copy(*this, y);
  Genten::axpy(sz, -1.0, z.data.data(), 1, data.data(), 1);
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
times(const Genten::ArrayT<ExecSpace> & y) const
{
#ifdef HAVE_CALIPER
  cali::Function cali_func("Genten::Array::times");
#endif

  const ttb_indx sz = data.extent(0);
  if (sz != y.data.extent(0))
  {
    Genten::error("Genten::ArrayT::times (one input) - size mismatch");
  }

  //vmul(sz, data.data(), y.data.data());
  view_type d = data;
  view_type yd = y.data;
  Kokkos::parallel_for("Genten::Array::times_kernel",
                       Kokkos::RangePolicy<ExecSpace>(0,sz),
                       KOKKOS_LAMBDA(const ttb_indx i)
  {
    d[i] *= yd[i];
  });
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
times(const Genten::ArrayT<ExecSpace> & y, const Genten::ArrayT<ExecSpace> & z) const
{
  const ttb_indx sz = data.extent(0);
  if (y.data.extent(0) != z.data.extent(0))
  {
    Genten::error("Genten::ArrayT::times (two inputs) - size mismatch");
  }

  for (ttb_indx i = 0; i < sz; i ++)
  {
    data[i] = y.data[i] * z.data[i];
  }
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
divide(const Genten::ArrayT<ExecSpace> & y) const
{
  const ttb_indx sz = data.extent(0);
  if (sz != y.data.extent(0))
  {
    Genten::error("Genten::ArrayT::divide (one input) - size mismatch");
  }

  for (ttb_indx i = 0; i < sz; i ++)
  {
    data[i] /= y.data[i];
  }
}

template <typename ExecSpace>
void Genten::ArrayT<ExecSpace>::
divide(const Genten::ArrayT<ExecSpace> & y, const Genten::ArrayT<ExecSpace> & z) const
{
  const ttb_indx sz = data.extent(0);
  if (y.data.extent(0) != z.data.extent(0))
  {
    Genten::error("Genten::ArrayT::divide (two inputs) - size mismatch");
  }

  for (ttb_indx i = 0; i < sz; i ++)
  {
    data[i] = y.data[i] / z.data[i];
  }
}

template <typename ExecSpace>
ttb_real Genten::ArrayT<ExecSpace>::
sum() const
{
  const ttb_indx sz = data.extent(0);
  ttb_real value = 0;
  for (ttb_indx i = 0; i < sz; i ++)
  {
    value += data[i];
  }
  return value;
}

#define INST_MACRO(SPACE) template class Genten::ArrayT<SPACE>;
GENTEN_INST(INST_MACRO)
