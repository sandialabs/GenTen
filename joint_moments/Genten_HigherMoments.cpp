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

#include "Genten_Kokkos.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_HigherMoments.hpp"
#include "Genten_FormCokurtosisSlice.hpp"

//namespace Genten {

//namespace Impl{

template <typename ExecSpace>
void form_cokurtosis_tensor_naive(const Kokkos::View<ttb_real**, Kokkos::LayoutLeft, ExecSpace>& data_view,
                                  const ttb_indx nsamples, const ttb_indx nvars,
                                  Genten::TensorT<ExecSpace>& moment_tensor) 
{

}

//}// namespace Impl

double * FormRawMomentTensor(double *raw_data_ptr, int nsamples, int nvars, const int order=4) {

  typedef Genten::DefaultExecutionSpace Space;
  typedef Genten::TensorT<Space> Tensor_type;
  typedef Genten::DefaultHostExecutionSpace HostSpace;
  typedef Genten::TensorT<HostSpace> Tensor_host_type;

  //Create the size of moment tensor
  //On host first, then mirror copy to device
  //moment tensor is size nvars^d, where d is order of moment, i.e. nvars*nvars*.... (d times)
  Genten::IndxArrayT<HostSpace> moment_tensor_size_host(order, nvars);

  Genten::IndxArrayT<Space> moment_tensor_size = create_mirror_view( Space(), moment_tensor_size_host);
  deep_copy(moment_tensor_size, moment_tensor_size_host);

  //Now construct the tensor on the device
  Tensor_type X(moment_tensor_size, 0.0);

  //Create a Tensor_type of raw data
  //We will be basically casting the raw_data_ptr to a Kokkos Unmanaged View
  //Not as straightforward as it seems
  //Example: https://github.com/kokkos/kokkos-fortran-interop/blob/master/src/flcl-cxx.hpp/#L157
  //raw data is "viewed" as a 2D-array
  Kokkos::View<ttb_real**,Kokkos::LayoutLeft, HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged> > raw_data_host(raw_data_ptr, nsamples, nvars);

  //Create mirror of raw_data_host on device and copy over
  Kokkos::View<ttb_real**,Kokkos::LayoutLeft, Space> raw_data = Kokkos::create_mirror_view(Space(), raw_data_host);
  deep_copy(raw_data, raw_data_host);


  //---------Call the Kernel to Compute Moment Tensor----------------
  form_cokurtosis_tensor_naive(raw_data, nsamples, nvars, X);


  //Now Mirror the result back from device to host
  Tensor_host_type X_host = create_mirror_view(HostSpace(), X);
  deep_copy(X_host, X);

  return X_host.getValues().ptr();
}


//}// namespace Genten
