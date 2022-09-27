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

#include <Genten_Util.hpp>

#include <gtest/gtest.h>

namespace Genten {
namespace UnitTests {

template <typename NewType, typename DummyType> struct TestingTypesPack {};

template <typename NewType, typename... Args>
struct TestingTypesPack<NewType, ::testing::Types<Args...>> {
  using type = ::testing::Types<Args..., NewType>;
};

using testing_types_pack_0 = ::testing::Types<>;

#if defined(HAVE_CUDA)
using testing_types_pack_1 =
    TestingTypesPack<Kokkos::Cuda, testing_types_pack_0>::type;
#else
using testing_types_pack_1 = testing_types_pack_0;
#endif

#if defined(HAVE_HIP)
using testing_types_pack_2 =
    TestingTypesPack<Kokkos::Experimental::HIP, testing_types_pack_1>::type;
#else
using testing_types_pack_2 = testing_types_pack_1;
#endif

#if defined(HAVE_SYCL)
using testing_types_pack_3 =
    TestingTypesPack<Kokkos::Experimental::SYCL, testing_types_pack_2>::type;
#else
using testing_types_pack_3 = testing_types_pack_2;
#endif

#if defined(HAVE_OPENMP)
using testing_types_pack_4 =
    TestingTypesPack<Kokkos::OpenMP, testing_types_pack_3>::type;
#else
using testing_types_pack_4 = testing_types_pack_3;
#endif

#if defined(HAVE_THREADS)
using testing_types_pack_5 =
    TestingTypesPack<Kokkos::Threads, testing_types_pack_4>::type;
#else
using testing_types_pack_5 = testing_types_pack_4;
#endif

#if defined(HAVE_SERIAL)
using testing_types_pack_6 =
    TestingTypesPack<Kokkos::Serial, testing_types_pack_5>::type;
#else
using testing_types_pack_6 = testing_types_pack_5;
#endif

using genten_test_types = testing_types_pack_6;

#define INFO_MSG(msg) Impl::info_message(msg, __LINE__, __FILE__)

#define TEST_MSG(msg) Impl::test_message(msg, __LINE__, __FILE__)

#define GENTEN_EQ(lhs, rhs, msg)                                               \
  Impl::test_message(msg, __LINE__, __FILE__);                                 \
  ASSERT_EQ(lhs, rhs)

#define GENTEN_NE(lhs, rhs, msg)                                               \
  Impl::test_message(msg, __LINE__, __FILE__);                                 \
  ASSERT_NE(lhs, rhs)

#define GENTEN_TRUE(cond, msg)                                                 \
  Impl::test_message(msg, __LINE__, __FILE__);                                 \
  ASSERT_TRUE(cond)

#define GENTEN_FALSE(cond, msg)                                                \
  Impl::test_message(msg, __LINE__, __FILE__);                                 \
  ASSERT_FALSE(cond)

#define GENTEN_FLOAT_EQ(lhs, rhs, msg)                                         \
  Impl::test_message(msg, __LINE__, __FILE__);                                 \
  ASSERT_FLOAT_EQ(lhs, rhs)

#define GENTEN_THROW(statement, expected_exception, msg)                       \
  Impl::test_message(msg, __LINE__, __FILE__);                                 \
  ASSERT_THROW(statement, expected_exception)

#define GENTEN_ANY_THROW(statement, msg)                                       \
  Impl::test_message(msg, __LINE__, __FILE__);                                 \
  ASSERT_ANY_THROW(statement)

#define GENTEN_LE(lhs, rhs, msg)                                               \
  Impl::test_message(msg, __LINE__, __FILE__);                                 \
  ASSERT_LE(lhs, rhs)

#define GENTEN_GE(lhs, rhs, msg)                                               \
  Impl::test_message(msg, __LINE__, __FILE__);                                 \
  ASSERT_GE(lhs, rhs)

#define EQ(a, b) Impl::float_eq(a, b)

#define FLOAT_EQ(a, b, tol) Impl::float_eq(a, b, tol)

namespace Impl {

enum class MsgColor { Green, Yellow };

const char *terminal_color(const MsgColor msg_color);
std::string make_verbose_message(const char *msg, const int line,
                                 const char *file_name);

void info_message(const char *msg, const int line, const char *file_name) {
  printf("%s[   INFO   ] \033[m", terminal_color(MsgColor::Green));
  printf("%s\n", make_verbose_message(msg, line, file_name).c_str());
}

void test_message(const char *msg, const int line, const char *file_name) {
  printf("%s[   TEST   ] \033[m", terminal_color(MsgColor::Yellow));
  printf("%s\n", make_verbose_message(msg, line, file_name).c_str());
}

const char *terminal_color(const MsgColor msg_color) {
  static const bool print_in_color = isatty((fileno(stdout)));
  if (not print_in_color) {
    return "\033[m";
  }

  switch (msg_color) {
  case MsgColor::Green: {
    return "\033[0;32m";
  }

  case MsgColor::Yellow: {
    return "\033[0;33m";
  }
  }

  return "";
}

std::string make_verbose_message(const char *msg, const int line,
                                 const char *file_name) {
  std::ostringstream oss;
  oss << msg << " (File: " << file_name << ", Line: " << line << ")";

  return oss.str();
}

KOKKOS_INLINE_FUNCTION
ttb_real max_abs(const ttb_real a, const ttb_real b) {
  return std::fabs(a) > std::fabs(b) ? std::fabs(a) : std::fabs(b);
}

KOKKOS_INLINE_FUNCTION
ttb_real rel_diff(const ttb_real a, const ttb_real b) {
  return std::fabs(a - b) / max_abs(ttb_real(1.0), max_abs(a, b));
}

KOKKOS_INLINE_FUNCTION
bool float_eq(const ttb_real a, const ttb_real b) {
  return rel_diff(a, b) < ttb_real(10.0) * MACHINE_EPSILON;
}

KOKKOS_INLINE_FUNCTION
bool float_eq(const ttb_real a, const ttb_real b, const ttb_real tol) {
  return rel_diff(a, b) < tol;
}

} // namespace Impl
} // namespace UnitTests
} // namespace Genten
