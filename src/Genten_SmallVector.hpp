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

#include "CMakeInclude.h"
#ifdef HAVE_BOOST

#include <boost/container/small_vector.hpp>

namespace Genten {

namespace detail {
// Attempt to fill one 64 byte cacheline with the small_vector
template <typename T, int Guess = (64 / sizeof(T))>
constexpr auto default_small_vector_size() {
  namespace bc = boost::container;
  constexpr auto size = sizeof(bc::small_vector<T, Guess>);
  if constexpr (Guess > 1 && size > 64) {
    return default_small_vector_size<T, Guess - 1>();
  }
  static_assert(size <= 512,
                "Even with 1 element boost::container::small_vector was "
                "larger than 512 bytes, perhapse this type is not suitible "
                "for a small_vector, you may manually provide the size if "
                "you really want to put something this big on the stack.");
  return Guess;
}
} // namespace detail

template <typename T, int N = detail::default_small_vector_size<T>()>
using small_vector = boost::container::small_vector<T, N>;

} // namespace Genten

#else

#include <vector>

namespace Genten {

template <typename T, int N = 0>
using small_vector = std::vector<T>;

}

#endif

namespace Genten {

template <typename U, typename T>
small_vector<U>
convert(const small_vector<T>& in)
{
  small_vector<U> out(in.size());
  const int n = in.size();
  for (int i=0; i<n; ++i)
    out[i] = in[i];
  return out;
}

}
