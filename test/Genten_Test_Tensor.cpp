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


#include <iostream>
#include <cmath>
#include <time.h>

#include "Genten_Tensor.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Test_Utils.hpp"

using namespace Genten::Test;

template <typename ExecSpace>
void Genten_Test_Tensor_Space(int infolevel)
{
  typedef ExecSpace exec_space;
  typedef Genten::DefaultHostExecutionSpace host_exec_space;
  typedef Genten::SptensorT<exec_space> Sptensor_type;
  typedef Genten::SptensorT<host_exec_space> Sptensor_host_type;
  typedef Genten::TensorT<exec_space> Tensor_type;
  typedef Genten::TensorT<host_exec_space> Tensor_host_type;
  typedef Genten::KtensorT<exec_space> Ktensor_type;
  typedef Genten::KtensorT<host_exec_space> Ktensor_host_type;

  std::string space_name = Genten::SpaceProperties<exec_space>::name();
  initialize("Tests on Genten::Tensor (" + space_name + ")", infolevel);

  // Empty constructor, equivalent to Matlab:  a = [].
  MESSAGE("Creating empty tensor");
  Genten::Tensor a;
  ASSERT( (a.nnz() == 0) && (a.ndims() == 0), "Tensor is empty" );

  MESSAGE("Creating 0-th order tensors");
  ttb_indx nd = 0;
  Genten::Tensor b(nd);
  ASSERT( (b.nnz() == 0) && (b.ndims() == 0), "Tensor is empty");
  Genten::IndxArray dims;
  Genten::Tensor c(dims);
  ASSERT( (c.nnz() == 0) && (c.ndims() == 0), "Tensor is empty");

  MESSAGE("Creating tensor to test element access");
  dims = Genten::IndxArray(3); dims[0] = 4; dims[1] = 2; dims[2] = 3;
  Genten::Tensor d(dims, 0.0);
  d[0] = 1.0;
  d[23] = 2.0;
  ASSERT( d.nnz() == 24, "nnz() correct");
  Genten::IndxArray oSub(3);
  oSub[0] = 0;  oSub[1] = 0;  oSub[2] = 0;
  ASSERT( d[oSub] == 1.0, "First element [0,0,0] found");
  oSub[0] = 3;  oSub[1] = 1;  oSub[2] = 2;
  ASSERT( d[oSub] == 2.0, "Last element [3,1,2] found");
  d[oSub] = 3.0;
  ASSERT( d[23] == 3.0, "Last element modified and found");

  MESSAGE("Resizing and populating tensor to test norm");
  dims = Genten::IndxArray(2); dims[0] = 1; dims[1] = 2;
  d = Genten::Tensor(dims);
  d[0] = 1.0;
  d[1] = 3.0;
  ASSERT( EQ(d.norm(), sqrt(10.0)), "Frobenius norm correct");

  MESSAGE("Creating dense tensor from sparse tensor");
  // create test Sptensor
  dims = Genten::IndxArray(3); dims[0] = 3; dims[1] = 4; dims[2] = 5;
  Sptensor_host_type s(dims,5);
  for (ttb_indx i = 0; i < s.nnz(); i ++)
  {
    s.subscript(i,0) = i % 3;
    s.subscript(i,1) = (i+1) % 4;
    s.subscript(i,2) = (i+2) % 5;
    s.value(i) = i*1.5+1;
  }
  Sptensor_type s_dev = create_mirror_view(exec_space(), s);
  deep_copy(s_dev, s);
  Tensor_type e_dev(s_dev);
  ASSERT( EQ(e_dev.norm(), s_dev.norm()), "Constructor from Sptensor correct");

  MESSAGE("Creating dense tensor from Kruskal tensor");
  // create test Ktensor
  dims = Genten::IndxArray(2); dims[0] = 1; dims[1] = 2;
  Ktensor_host_type k(3, 2, dims);
  k.setWeightsRand();
  k.setMatricesRand();
  Ktensor_type k_dev = create_mirror_view(exec_space(), k);
  deep_copy(k_dev, k);
  Tensor_type f_dev(k_dev);
  Tensor_host_type f = create_mirror_view(host_exec_space(), f_dev);
  deep_copy(f,f_dev);
  // check entries
  Genten::IndxArray sub(2);
  sub[0] = 0; sub[1] = 0;
  ASSERT( EQ(f[sub], Genten::compute_Ktensor_value(k,sub)),
          "Constructor from Ktensor correct");
  sub[0] = 0; sub[1] = 1;
  ASSERT( EQ(f[sub], Genten::compute_Ktensor_value(k,sub)),
          "Constructor from Ktensor correct");

  finalize();
  return;
}

void Genten_Test_Tensor(int infolevel) {
#ifdef KOKKOS_ENABLE_CUDA
  Genten_Test_Tensor_Space<Kokkos::Cuda>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_HIP
  Genten_Test_Tensor_Space<Kokkos::Experimental::HIP>(infolevel);
#endif
#ifdef ENABLE_SYCL_FOR_CUDA
  Genten_Test_Tensor_Space<Kokkos::Experimental::SYCL>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  Genten_Test_Tensor_Space<Kokkos::OpenMP>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_THREADS
  Genten_Test_Tensor_Space<Kokkos::Threads>(infolevel);
#endif
#ifdef KOKKOS_ENABLE_SERIAL
  Genten_Test_Tensor_Space<Kokkos::Serial>(infolevel);
#endif
}
