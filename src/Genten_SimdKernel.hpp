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

#pragma once

namespace Genten {
namespace Impl {

template <unsigned VS, typename Func>
void run_row_simd_kernel_impl(Func& f, const unsigned nc)
{
  static const unsigned VS4 = 4*VS;
  static const unsigned VS3 = 3*VS;
  static const unsigned VS2 = 2*VS;
  static const unsigned VS1 = 1*VS;

  if (nc > VS3)
    f.template run<VS4,VS>();
  else if (nc > VS2)
    f.template run<VS3,VS>();
  else if (nc > VS1)
    f.template run<VS2,VS>();
  else
    f.template run<VS1,VS>();
}

template <typename Func>
void run_row_simd_kernel(Func& f, const unsigned nc)
{
  if (nc >= 96)
    run_row_simd_kernel_impl<32>(f, nc);
  else if (nc >= 48)
    run_row_simd_kernel_impl<16>(f, nc);
  else if (nc >= 8)
    run_row_simd_kernel_impl<8>(f, nc);
  else if (nc >= 4)
    run_row_simd_kernel_impl<4>(f, nc);
  else if (nc >= 2)
    run_row_simd_kernel_impl<2>(f, nc);
  else
    run_row_simd_kernel_impl<1>(f, nc);
}

}
}
