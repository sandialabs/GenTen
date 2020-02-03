// This file was GENERATED by command:
//     pump.py gmock-generated-function-mockers.h.pump
// DO NOT EDIT BY HAND!!!

// Copyright 2007, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


// Google Mock - a framework for writing C++ mock classes.
//
// This file implements function mockers of various arities.

// GOOGLETEST_CM0002 DO NOT DELETE

#ifndef GMOCK_INCLUDE_GMOCK_GMOCK_GENERATED_FUNCTION_MOCKERS_H_
#define GMOCK_INCLUDE_GMOCK_GMOCK_GENERATED_FUNCTION_MOCKERS_H_

#include <functional>
#include <utility>

#include "gmock/gmock-spec-builders.h"
#include "gmock/internal/gmock-internal-utils.h"

namespace testing {
namespace internal {
// Removes the given pointer; this is a helper for the expectation setter method
// for parameterless matchers.
//
// We want to make sure that the user cannot set a parameterless expectation on
// overloaded methods, including methods which are overloaded on const. Example:
//
//   class MockClass {
//     MOCK_METHOD0(GetName, string&());
//     MOCK_CONST_METHOD0(GetName, const string&());
//   };
//
//   TEST() {
//     // This should be an error, as it's not clear which overload is expected.
//     EXPECT_CALL(mock, GetName).WillOnce(ReturnRef(value));
//   }
//
// Here are the generated expectation-setter methods:
//
//   class MockClass {
//     // Overload 1
//     MockSpec<string&()> gmock_GetName() { ... }
//     // Overload 2. Declared const so that the compiler will generate an
//     // error when trying to resolve between this and overload 4 in
//     // 'gmock_GetName(WithoutMatchers(), nullptr)'.
//     MockSpec<string&()> gmock_GetName(
//         const WithoutMatchers&, const Function<string&()>*) const {
//       // Removes const from this, calls overload 1
//       return AdjustConstness_(this)->gmock_GetName();
//     }
//
//     // Overload 3
//     const string& gmock_GetName() const { ... }
//     // Overload 4
//     MockSpec<const string&()> gmock_GetName(
//         const WithoutMatchers&, const Function<const string&()>*) const {
//       // Does not remove const, calls overload 3
//       return AdjustConstness_const(this)->gmock_GetName();
//     }
//   }
//
template <typename MockType>
const MockType* AdjustConstness_const(const MockType* mock) {
  return mock;
}

// Removes const from and returns the given pointer; this is a helper for the
// expectation setter method for parameterless matchers.
template <typename MockType>
MockType* AdjustConstness_(const MockType* mock) {
  return const_cast<MockType*>(mock);
}

}  // namespace internal

// The style guide prohibits "using" statements in a namespace scope
// inside a header file.  However, the FunctionMocker class template
// is meant to be defined in the ::testing namespace.  The following
// line is just a trick for working around a bug in MSVC 8.0, which
// cannot handle it if we define FunctionMocker in ::testing.
using internal::FunctionMocker;

// GMOCK_RESULT_(tn, F) expands to the result type of function type F.
// We define this as a variadic macro in case F contains unprotected
// commas (the same reason that we use variadic macros in other places
// in this file).
// INTERNAL IMPLEMENTATION - DON'T USE IN USER CODE!!!
#define GMOCK_RESULT_(...) \
    typename ::testing::internal::Function<__VA_ARGS__>::Result

// The type of argument N of the given function type.
// INTERNAL IMPLEMENTATION - DON'T USE IN USER CODE!!!
#define GMOCK_ARG_(N, ...) \
    typename ::testing::internal::Function<__VA_ARGS__>::template Arg<N-1>::type

// The matcher type for argument N of the given function type.
// INTERNAL IMPLEMENTATION - DON'T USE IN USER CODE!!!
#define GMOCK_MATCHER_(N, ...) \
    const ::testing::Matcher<GMOCK_ARG_(N, __VA_ARGS__)>&

// The variable for mocking the given method.
// INTERNAL IMPLEMENTATION - DON'T USE IN USER CODE!!!
#define GMOCK_MOCKER_(arity, constness, Method) \
    GTEST_CONCAT_TOKEN_(gmock##constness##arity##_##Method##_, __LINE__)

// INTERNAL IMPLEMENTATION - DON'T USE IN USER CODE!!!
#define GMOCK_METHOD0_(constness, ct, Method, ...) \
  static_assert(0 == \
      ::testing::internal::Function<__VA_ARGS__>::ArgumentCount, \
      "MOCK_METHOD<N> must match argument count.");\
  GMOCK_RESULT_(__VA_ARGS__) ct Method( \
      ) constness { \
    GMOCK_MOCKER_(0, constness, Method).SetOwnerAndName(this, #Method); \
    return GMOCK_MOCKER_(0, constness, Method).Invoke(); \
  } \
  ::testing::MockSpec<__VA_ARGS__> \
      gmock_##Method() constness { \
    GMOCK_MOCKER_(0, constness, Method).RegisterOwner(this); \
    return GMOCK_MOCKER_(0, constness, Method).With(); \
  } \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method( \
      const ::testing::internal::WithoutMatchers&, \
      constness ::testing::internal::Function<__VA_ARGS__>* ) const { \
        return ::testing::internal::AdjustConstness_##constness(this)-> \
            gmock_##Method(); \
      } \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(0, constness, \
      Method)

// INTERNAL IMPLEMENTATION - DON'T USE IN USER CODE!!!
#define GMOCK_METHOD1_(constness, ct, Method, ...) \
  static_assert(1 == \
      ::testing::internal::Function<__VA_ARGS__>::ArgumentCount, \
      "MOCK_METHOD<N> must match argument count.");\
  GMOCK_RESULT_(__VA_ARGS__) ct Method( \
      GMOCK_ARG_(1, __VA_ARGS__) gmock_a1) constness { \
    GMOCK_MOCKER_(1, constness, Method).SetOwnerAndName(this, #Method); \
    return GMOCK_MOCKER_(1, constness, \
        Method).Invoke(::std::forward<GMOCK_ARG_(1, __VA_ARGS__)>(gmock_a1)); \
        \
  } \
  ::testing::MockSpec<__VA_ARGS__> \
      gmock_##Method(GMOCK_MATCHER_(1, __VA_ARGS__) gmock_a1) constness { \
    GMOCK_MOCKER_(1, constness, Method).RegisterOwner(this); \
    return GMOCK_MOCKER_(1, constness, Method).With(gmock_a1); \
  } \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method( \
      const ::testing::internal::WithoutMatchers&, \
      constness ::testing::internal::Function<__VA_ARGS__>* ) const { \
        return ::testing::internal::AdjustConstness_##constness(this)-> \
            gmock_##Method(::testing::A<GMOCK_ARG_(1, __VA_ARGS__)>()); \
      } \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(1, constness, \
      Method)

// INTERNAL IMPLEMENTATION - DON'T USE IN USER CODE!!!
#define GMOCK_METHOD2_(constness, ct, Method, ...) \
  static_assert(2 == \
      ::testing::internal::Function<__VA_ARGS__>::ArgumentCount, \
      "MOCK_METHOD<N> must match argument count.");\
  GMOCK_RESULT_(__VA_ARGS__) ct Method( \
      GMOCK_ARG_(1, __VA_ARGS__) gmock_a1, GMOCK_ARG_(2, \
          __VA_ARGS__) gmock_a2) constness { \
    GMOCK_MOCKER_(2, constness, Method).SetOwnerAndName(this, #Method); \
    return GMOCK_MOCKER_(2, constness, \
        Method).Invoke(::std::forward<GMOCK_ARG_(1, __VA_ARGS__)>(gmock_a1), \
        \
  ::std::forward<GMOCK_ARG_(2, __VA_ARGS__)>(gmock_a2)); \
  } \
  ::testing::MockSpec<__VA_ARGS__> \
      gmock_##Method(GMOCK_MATCHER_(1, __VA_ARGS__) gmock_a1, \
                     GMOCK_MATCHER_(2, __VA_ARGS__) gmock_a2) constness { \
    GMOCK_MOCKER_(2, constness, Method).RegisterOwner(this); \
    return GMOCK_MOCKER_(2, constness, Method).With(gmock_a1, gmock_a2); \
  } \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method( \
      const ::testing::internal::WithoutMatchers&, \
      constness ::testing::internal::Function<__VA_ARGS__>* ) const { \
        return ::testing::internal::AdjustConstness_##constness(this)-> \
            gmock_##Method(::testing::A<GMOCK_ARG_(1, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(2, __VA_ARGS__)>()); \
      } \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(2, constness, \
      Method)

// INTERNAL IMPLEMENTATION - DON'T USE IN USER CODE!!!
#define GMOCK_METHOD3_(constness, ct, Method, ...) \
  static_assert(3 == \
      ::testing::internal::Function<__VA_ARGS__>::ArgumentCount, \
      "MOCK_METHOD<N> must match argument count.");\
  GMOCK_RESULT_(__VA_ARGS__) ct Method( \
      GMOCK_ARG_(1, __VA_ARGS__) gmock_a1, GMOCK_ARG_(2, \
          __VA_ARGS__) gmock_a2, GMOCK_ARG_(3, \
          __VA_ARGS__) gmock_a3) constness { \
    GMOCK_MOCKER_(3, constness, Method).SetOwnerAndName(this, #Method); \
    return GMOCK_MOCKER_(3, constness, \
        Method).Invoke(::std::forward<GMOCK_ARG_(1, __VA_ARGS__)>(gmock_a1), \
        \
  ::std::forward<GMOCK_ARG_(2, __VA_ARGS__)>(gmock_a2), \
  ::std::forward<GMOCK_ARG_(3, __VA_ARGS__)>(gmock_a3)); \
  } \
  ::testing::MockSpec<__VA_ARGS__> \
      gmock_##Method(GMOCK_MATCHER_(1, __VA_ARGS__) gmock_a1, \
                     GMOCK_MATCHER_(2, __VA_ARGS__) gmock_a2, \
                     GMOCK_MATCHER_(3, __VA_ARGS__) gmock_a3) constness { \
    GMOCK_MOCKER_(3, constness, Method).RegisterOwner(this); \
    return GMOCK_MOCKER_(3, constness, Method).With(gmock_a1, gmock_a2, \
        gmock_a3); \
  } \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method( \
      const ::testing::internal::WithoutMatchers&, \
      constness ::testing::internal::Function<__VA_ARGS__>* ) const { \
        return ::testing::internal::AdjustConstness_##constness(this)-> \
            gmock_##Method(::testing::A<GMOCK_ARG_(1, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(2, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(3, __VA_ARGS__)>()); \
      } \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(3, constness, \
      Method)

// INTERNAL IMPLEMENTATION - DON'T USE IN USER CODE!!!
#define GMOCK_METHOD4_(constness, ct, Method, ...) \
  static_assert(4 == \
      ::testing::internal::Function<__VA_ARGS__>::ArgumentCount, \
      "MOCK_METHOD<N> must match argument count.");\
  GMOCK_RESULT_(__VA_ARGS__) ct Method( \
      GMOCK_ARG_(1, __VA_ARGS__) gmock_a1, GMOCK_ARG_(2, \
          __VA_ARGS__) gmock_a2, GMOCK_ARG_(3, __VA_ARGS__) gmock_a3, \
          GMOCK_ARG_(4, __VA_ARGS__) gmock_a4) constness { \
    GMOCK_MOCKER_(4, constness, Method).SetOwnerAndName(this, #Method); \
    return GMOCK_MOCKER_(4, constness, \
        Method).Invoke(::std::forward<GMOCK_ARG_(1, __VA_ARGS__)>(gmock_a1), \
        \
  ::std::forward<GMOCK_ARG_(2, __VA_ARGS__)>(gmock_a2), \
  ::std::forward<GMOCK_ARG_(3, __VA_ARGS__)>(gmock_a3), \
  ::std::forward<GMOCK_ARG_(4, __VA_ARGS__)>(gmock_a4)); \
  } \
  ::testing::MockSpec<__VA_ARGS__> \
      gmock_##Method(GMOCK_MATCHER_(1, __VA_ARGS__) gmock_a1, \
                     GMOCK_MATCHER_(2, __VA_ARGS__) gmock_a2, \
                     GMOCK_MATCHER_(3, __VA_ARGS__) gmock_a3, \
                     GMOCK_MATCHER_(4, __VA_ARGS__) gmock_a4) constness { \
    GMOCK_MOCKER_(4, constness, Method).RegisterOwner(this); \
    return GMOCK_MOCKER_(4, constness, Method).With(gmock_a1, gmock_a2, \
        gmock_a3, gmock_a4); \
  } \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method( \
      const ::testing::internal::WithoutMatchers&, \
      constness ::testing::internal::Function<__VA_ARGS__>* ) const { \
        return ::testing::internal::AdjustConstness_##constness(this)-> \
            gmock_##Method(::testing::A<GMOCK_ARG_(1, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(2, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(3, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(4, __VA_ARGS__)>()); \
      } \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(4, constness, \
      Method)

// INTERNAL IMPLEMENTATION - DON'T USE IN USER CODE!!!
#define GMOCK_METHOD5_(constness, ct, Method, ...) \
  static_assert(5 == \
      ::testing::internal::Function<__VA_ARGS__>::ArgumentCount, \
      "MOCK_METHOD<N> must match argument count.");\
  GMOCK_RESULT_(__VA_ARGS__) ct Method( \
      GMOCK_ARG_(1, __VA_ARGS__) gmock_a1, GMOCK_ARG_(2, \
          __VA_ARGS__) gmock_a2, GMOCK_ARG_(3, __VA_ARGS__) gmock_a3, \
          GMOCK_ARG_(4, __VA_ARGS__) gmock_a4, GMOCK_ARG_(5, \
          __VA_ARGS__) gmock_a5) constness { \
    GMOCK_MOCKER_(5, constness, Method).SetOwnerAndName(this, #Method); \
    return GMOCK_MOCKER_(5, constness, \
        Method).Invoke(::std::forward<GMOCK_ARG_(1, __VA_ARGS__)>(gmock_a1), \
        \
  ::std::forward<GMOCK_ARG_(2, __VA_ARGS__)>(gmock_a2), \
  ::std::forward<GMOCK_ARG_(3, __VA_ARGS__)>(gmock_a3), \
  ::std::forward<GMOCK_ARG_(4, __VA_ARGS__)>(gmock_a4), \
  ::std::forward<GMOCK_ARG_(5, __VA_ARGS__)>(gmock_a5)); \
  } \
  ::testing::MockSpec<__VA_ARGS__> \
      gmock_##Method(GMOCK_MATCHER_(1, __VA_ARGS__) gmock_a1, \
                     GMOCK_MATCHER_(2, __VA_ARGS__) gmock_a2, \
                     GMOCK_MATCHER_(3, __VA_ARGS__) gmock_a3, \
                     GMOCK_MATCHER_(4, __VA_ARGS__) gmock_a4, \
                     GMOCK_MATCHER_(5, __VA_ARGS__) gmock_a5) constness { \
    GMOCK_MOCKER_(5, constness, Method).RegisterOwner(this); \
    return GMOCK_MOCKER_(5, constness, Method).With(gmock_a1, gmock_a2, \
        gmock_a3, gmock_a4, gmock_a5); \
  } \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method( \
      const ::testing::internal::WithoutMatchers&, \
      constness ::testing::internal::Function<__VA_ARGS__>* ) const { \
        return ::testing::internal::AdjustConstness_##constness(this)-> \
            gmock_##Method(::testing::A<GMOCK_ARG_(1, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(2, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(3, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(4, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(5, __VA_ARGS__)>()); \
      } \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(5, constness, \
      Method)

// INTERNAL IMPLEMENTATION - DON'T USE IN USER CODE!!!
#define GMOCK_METHOD6_(constness, ct, Method, ...) \
  static_assert(6 == \
      ::testing::internal::Function<__VA_ARGS__>::ArgumentCount, \
      "MOCK_METHOD<N> must match argument count.");\
  GMOCK_RESULT_(__VA_ARGS__) ct Method( \
      GMOCK_ARG_(1, __VA_ARGS__) gmock_a1, GMOCK_ARG_(2, \
          __VA_ARGS__) gmock_a2, GMOCK_ARG_(3, __VA_ARGS__) gmock_a3, \
          GMOCK_ARG_(4, __VA_ARGS__) gmock_a4, GMOCK_ARG_(5, \
          __VA_ARGS__) gmock_a5, GMOCK_ARG_(6, \
          __VA_ARGS__) gmock_a6) constness { \
    GMOCK_MOCKER_(6, constness, Method).SetOwnerAndName(this, #Method); \
    return GMOCK_MOCKER_(6, constness, \
        Method).Invoke(::std::forward<GMOCK_ARG_(1, __VA_ARGS__)>(gmock_a1), \
        \
  ::std::forward<GMOCK_ARG_(2, __VA_ARGS__)>(gmock_a2), \
  ::std::forward<GMOCK_ARG_(3, __VA_ARGS__)>(gmock_a3), \
  ::std::forward<GMOCK_ARG_(4, __VA_ARGS__)>(gmock_a4), \
  ::std::forward<GMOCK_ARG_(5, __VA_ARGS__)>(gmock_a5), \
  ::std::forward<GMOCK_ARG_(6, __VA_ARGS__)>(gmock_a6)); \
  } \
  ::testing::MockSpec<__VA_ARGS__> \
      gmock_##Method(GMOCK_MATCHER_(1, __VA_ARGS__) gmock_a1, \
                     GMOCK_MATCHER_(2, __VA_ARGS__) gmock_a2, \
                     GMOCK_MATCHER_(3, __VA_ARGS__) gmock_a3, \
                     GMOCK_MATCHER_(4, __VA_ARGS__) gmock_a4, \
                     GMOCK_MATCHER_(5, __VA_ARGS__) gmock_a5, \
                     GMOCK_MATCHER_(6, __VA_ARGS__) gmock_a6) constness { \
    GMOCK_MOCKER_(6, constness, Method).RegisterOwner(this); \
    return GMOCK_MOCKER_(6, constness, Method).With(gmock_a1, gmock_a2, \
        gmock_a3, gmock_a4, gmock_a5, gmock_a6); \
  } \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method( \
      const ::testing::internal::WithoutMatchers&, \
      constness ::testing::internal::Function<__VA_ARGS__>* ) const { \
        return ::testing::internal::AdjustConstness_##constness(this)-> \
            gmock_##Method(::testing::A<GMOCK_ARG_(1, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(2, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(3, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(4, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(5, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(6, __VA_ARGS__)>()); \
      } \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(6, constness, \
      Method)

// INTERNAL IMPLEMENTATION - DON'T USE IN USER CODE!!!
#define GMOCK_METHOD7_(constness, ct, Method, ...) \
  static_assert(7 == \
      ::testing::internal::Function<__VA_ARGS__>::ArgumentCount, \
      "MOCK_METHOD<N> must match argument count.");\
  GMOCK_RESULT_(__VA_ARGS__) ct Method( \
      GMOCK_ARG_(1, __VA_ARGS__) gmock_a1, GMOCK_ARG_(2, \
          __VA_ARGS__) gmock_a2, GMOCK_ARG_(3, __VA_ARGS__) gmock_a3, \
          GMOCK_ARG_(4, __VA_ARGS__) gmock_a4, GMOCK_ARG_(5, \
          __VA_ARGS__) gmock_a5, GMOCK_ARG_(6, __VA_ARGS__) gmock_a6, \
          GMOCK_ARG_(7, __VA_ARGS__) gmock_a7) constness { \
    GMOCK_MOCKER_(7, constness, Method).SetOwnerAndName(this, #Method); \
    return GMOCK_MOCKER_(7, constness, \
        Method).Invoke(::std::forward<GMOCK_ARG_(1, __VA_ARGS__)>(gmock_a1), \
        \
  ::std::forward<GMOCK_ARG_(2, __VA_ARGS__)>(gmock_a2), \
  ::std::forward<GMOCK_ARG_(3, __VA_ARGS__)>(gmock_a3), \
  ::std::forward<GMOCK_ARG_(4, __VA_ARGS__)>(gmock_a4), \
  ::std::forward<GMOCK_ARG_(5, __VA_ARGS__)>(gmock_a5), \
  ::std::forward<GMOCK_ARG_(6, __VA_ARGS__)>(gmock_a6), \
  ::std::forward<GMOCK_ARG_(7, __VA_ARGS__)>(gmock_a7)); \
  } \
  ::testing::MockSpec<__VA_ARGS__> \
      gmock_##Method(GMOCK_MATCHER_(1, __VA_ARGS__) gmock_a1, \
                     GMOCK_MATCHER_(2, __VA_ARGS__) gmock_a2, \
                     GMOCK_MATCHER_(3, __VA_ARGS__) gmock_a3, \
                     GMOCK_MATCHER_(4, __VA_ARGS__) gmock_a4, \
                     GMOCK_MATCHER_(5, __VA_ARGS__) gmock_a5, \
                     GMOCK_MATCHER_(6, __VA_ARGS__) gmock_a6, \
                     GMOCK_MATCHER_(7, __VA_ARGS__) gmock_a7) constness { \
    GMOCK_MOCKER_(7, constness, Method).RegisterOwner(this); \
    return GMOCK_MOCKER_(7, constness, Method).With(gmock_a1, gmock_a2, \
        gmock_a3, gmock_a4, gmock_a5, gmock_a6, gmock_a7); \
  } \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method( \
      const ::testing::internal::WithoutMatchers&, \
      constness ::testing::internal::Function<__VA_ARGS__>* ) const { \
        return ::testing::internal::AdjustConstness_##constness(this)-> \
            gmock_##Method(::testing::A<GMOCK_ARG_(1, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(2, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(3, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(4, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(5, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(6, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(7, __VA_ARGS__)>()); \
      } \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(7, constness, \
      Method)

// INTERNAL IMPLEMENTATION - DON'T USE IN USER CODE!!!
#define GMOCK_METHOD8_(constness, ct, Method, ...) \
  static_assert(8 == \
      ::testing::internal::Function<__VA_ARGS__>::ArgumentCount, \
      "MOCK_METHOD<N> must match argument count.");\
  GMOCK_RESULT_(__VA_ARGS__) ct Method( \
      GMOCK_ARG_(1, __VA_ARGS__) gmock_a1, GMOCK_ARG_(2, \
          __VA_ARGS__) gmock_a2, GMOCK_ARG_(3, __VA_ARGS__) gmock_a3, \
          GMOCK_ARG_(4, __VA_ARGS__) gmock_a4, GMOCK_ARG_(5, \
          __VA_ARGS__) gmock_a5, GMOCK_ARG_(6, __VA_ARGS__) gmock_a6, \
          GMOCK_ARG_(7, __VA_ARGS__) gmock_a7, GMOCK_ARG_(8, \
          __VA_ARGS__) gmock_a8) constness { \
    GMOCK_MOCKER_(8, constness, Method).SetOwnerAndName(this, #Method); \
    return GMOCK_MOCKER_(8, constness, \
        Method).Invoke(::std::forward<GMOCK_ARG_(1, __VA_ARGS__)>(gmock_a1), \
        \
  ::std::forward<GMOCK_ARG_(2, __VA_ARGS__)>(gmock_a2), \
  ::std::forward<GMOCK_ARG_(3, __VA_ARGS__)>(gmock_a3), \
  ::std::forward<GMOCK_ARG_(4, __VA_ARGS__)>(gmock_a4), \
  ::std::forward<GMOCK_ARG_(5, __VA_ARGS__)>(gmock_a5), \
  ::std::forward<GMOCK_ARG_(6, __VA_ARGS__)>(gmock_a6), \
  ::std::forward<GMOCK_ARG_(7, __VA_ARGS__)>(gmock_a7), \
  ::std::forward<GMOCK_ARG_(8, __VA_ARGS__)>(gmock_a8)); \
  } \
  ::testing::MockSpec<__VA_ARGS__> \
      gmock_##Method(GMOCK_MATCHER_(1, __VA_ARGS__) gmock_a1, \
                     GMOCK_MATCHER_(2, __VA_ARGS__) gmock_a2, \
                     GMOCK_MATCHER_(3, __VA_ARGS__) gmock_a3, \
                     GMOCK_MATCHER_(4, __VA_ARGS__) gmock_a4, \
                     GMOCK_MATCHER_(5, __VA_ARGS__) gmock_a5, \
                     GMOCK_MATCHER_(6, __VA_ARGS__) gmock_a6, \
                     GMOCK_MATCHER_(7, __VA_ARGS__) gmock_a7, \
                     GMOCK_MATCHER_(8, __VA_ARGS__) gmock_a8) constness { \
    GMOCK_MOCKER_(8, constness, Method).RegisterOwner(this); \
    return GMOCK_MOCKER_(8, constness, Method).With(gmock_a1, gmock_a2, \
        gmock_a3, gmock_a4, gmock_a5, gmock_a6, gmock_a7, gmock_a8); \
  } \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method( \
      const ::testing::internal::WithoutMatchers&, \
      constness ::testing::internal::Function<__VA_ARGS__>* ) const { \
        return ::testing::internal::AdjustConstness_##constness(this)-> \
            gmock_##Method(::testing::A<GMOCK_ARG_(1, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(2, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(3, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(4, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(5, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(6, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(7, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(8, __VA_ARGS__)>()); \
      } \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(8, constness, \
      Method)

// INTERNAL IMPLEMENTATION - DON'T USE IN USER CODE!!!
#define GMOCK_METHOD9_(constness, ct, Method, ...) \
  static_assert(9 == \
      ::testing::internal::Function<__VA_ARGS__>::ArgumentCount, \
      "MOCK_METHOD<N> must match argument count.");\
  GMOCK_RESULT_(__VA_ARGS__) ct Method( \
      GMOCK_ARG_(1, __VA_ARGS__) gmock_a1, GMOCK_ARG_(2, \
          __VA_ARGS__) gmock_a2, GMOCK_ARG_(3, __VA_ARGS__) gmock_a3, \
          GMOCK_ARG_(4, __VA_ARGS__) gmock_a4, GMOCK_ARG_(5, \
          __VA_ARGS__) gmock_a5, GMOCK_ARG_(6, __VA_ARGS__) gmock_a6, \
          GMOCK_ARG_(7, __VA_ARGS__) gmock_a7, GMOCK_ARG_(8, \
          __VA_ARGS__) gmock_a8, GMOCK_ARG_(9, \
          __VA_ARGS__) gmock_a9) constness { \
    GMOCK_MOCKER_(9, constness, Method).SetOwnerAndName(this, #Method); \
    return GMOCK_MOCKER_(9, constness, \
        Method).Invoke(::std::forward<GMOCK_ARG_(1, __VA_ARGS__)>(gmock_a1), \
        \
  ::std::forward<GMOCK_ARG_(2, __VA_ARGS__)>(gmock_a2), \
  ::std::forward<GMOCK_ARG_(3, __VA_ARGS__)>(gmock_a3), \
  ::std::forward<GMOCK_ARG_(4, __VA_ARGS__)>(gmock_a4), \
  ::std::forward<GMOCK_ARG_(5, __VA_ARGS__)>(gmock_a5), \
  ::std::forward<GMOCK_ARG_(6, __VA_ARGS__)>(gmock_a6), \
  ::std::forward<GMOCK_ARG_(7, __VA_ARGS__)>(gmock_a7), \
  ::std::forward<GMOCK_ARG_(8, __VA_ARGS__)>(gmock_a8), \
  ::std::forward<GMOCK_ARG_(9, __VA_ARGS__)>(gmock_a9)); \
  } \
  ::testing::MockSpec<__VA_ARGS__> \
      gmock_##Method(GMOCK_MATCHER_(1, __VA_ARGS__) gmock_a1, \
                     GMOCK_MATCHER_(2, __VA_ARGS__) gmock_a2, \
                     GMOCK_MATCHER_(3, __VA_ARGS__) gmock_a3, \
                     GMOCK_MATCHER_(4, __VA_ARGS__) gmock_a4, \
                     GMOCK_MATCHER_(5, __VA_ARGS__) gmock_a5, \
                     GMOCK_MATCHER_(6, __VA_ARGS__) gmock_a6, \
                     GMOCK_MATCHER_(7, __VA_ARGS__) gmock_a7, \
                     GMOCK_MATCHER_(8, __VA_ARGS__) gmock_a8, \
                     GMOCK_MATCHER_(9, __VA_ARGS__) gmock_a9) constness { \
    GMOCK_MOCKER_(9, constness, Method).RegisterOwner(this); \
    return GMOCK_MOCKER_(9, constness, Method).With(gmock_a1, gmock_a2, \
        gmock_a3, gmock_a4, gmock_a5, gmock_a6, gmock_a7, gmock_a8, \
        gmock_a9); \
  } \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method( \
      const ::testing::internal::WithoutMatchers&, \
      constness ::testing::internal::Function<__VA_ARGS__>* ) const { \
        return ::testing::internal::AdjustConstness_##constness(this)-> \
            gmock_##Method(::testing::A<GMOCK_ARG_(1, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(2, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(3, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(4, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(5, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(6, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(7, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(8, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(9, __VA_ARGS__)>()); \
      } \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(9, constness, \
      Method)

// INTERNAL IMPLEMENTATION - DON'T USE IN USER CODE!!!
#define GMOCK_METHOD10_(constness, ct, Method, ...) \
  static_assert(10 == \
      ::testing::internal::Function<__VA_ARGS__>::ArgumentCount, \
      "MOCK_METHOD<N> must match argument count.");\
  GMOCK_RESULT_(__VA_ARGS__) ct Method( \
      GMOCK_ARG_(1, __VA_ARGS__) gmock_a1, GMOCK_ARG_(2, \
          __VA_ARGS__) gmock_a2, GMOCK_ARG_(3, __VA_ARGS__) gmock_a3, \
          GMOCK_ARG_(4, __VA_ARGS__) gmock_a4, GMOCK_ARG_(5, \
          __VA_ARGS__) gmock_a5, GMOCK_ARG_(6, __VA_ARGS__) gmock_a6, \
          GMOCK_ARG_(7, __VA_ARGS__) gmock_a7, GMOCK_ARG_(8, \
          __VA_ARGS__) gmock_a8, GMOCK_ARG_(9, __VA_ARGS__) gmock_a9, \
          GMOCK_ARG_(10, __VA_ARGS__) gmock_a10) constness { \
    GMOCK_MOCKER_(10, constness, Method).SetOwnerAndName(this, #Method); \
    return GMOCK_MOCKER_(10, constness, \
        Method).Invoke(::std::forward<GMOCK_ARG_(1, __VA_ARGS__)>(gmock_a1), \
        \
  ::std::forward<GMOCK_ARG_(2, __VA_ARGS__)>(gmock_a2), \
  ::std::forward<GMOCK_ARG_(3, __VA_ARGS__)>(gmock_a3), \
  ::std::forward<GMOCK_ARG_(4, __VA_ARGS__)>(gmock_a4), \
  ::std::forward<GMOCK_ARG_(5, __VA_ARGS__)>(gmock_a5), \
  ::std::forward<GMOCK_ARG_(6, __VA_ARGS__)>(gmock_a6), \
  ::std::forward<GMOCK_ARG_(7, __VA_ARGS__)>(gmock_a7), \
  ::std::forward<GMOCK_ARG_(8, __VA_ARGS__)>(gmock_a8), \
  ::std::forward<GMOCK_ARG_(9, __VA_ARGS__)>(gmock_a9), \
  ::std::forward<GMOCK_ARG_(10, __VA_ARGS__)>(gmock_a10)); \
  } \
  ::testing::MockSpec<__VA_ARGS__> \
      gmock_##Method(GMOCK_MATCHER_(1, __VA_ARGS__) gmock_a1, \
                     GMOCK_MATCHER_(2, __VA_ARGS__) gmock_a2, \
                     GMOCK_MATCHER_(3, __VA_ARGS__) gmock_a3, \
                     GMOCK_MATCHER_(4, __VA_ARGS__) gmock_a4, \
                     GMOCK_MATCHER_(5, __VA_ARGS__) gmock_a5, \
                     GMOCK_MATCHER_(6, __VA_ARGS__) gmock_a6, \
                     GMOCK_MATCHER_(7, __VA_ARGS__) gmock_a7, \
                     GMOCK_MATCHER_(8, __VA_ARGS__) gmock_a8, \
                     GMOCK_MATCHER_(9, __VA_ARGS__) gmock_a9, \
                     GMOCK_MATCHER_(10, __VA_ARGS__) gmock_a10) constness { \
    GMOCK_MOCKER_(10, constness, Method).RegisterOwner(this); \
    return GMOCK_MOCKER_(10, constness, Method).With(gmock_a1, gmock_a2, \
        gmock_a3, gmock_a4, gmock_a5, gmock_a6, gmock_a7, gmock_a8, gmock_a9, \
        gmock_a10); \
  } \
  ::testing::MockSpec<__VA_ARGS__> gmock_##Method( \
      const ::testing::internal::WithoutMatchers&, \
      constness ::testing::internal::Function<__VA_ARGS__>* ) const { \
        return ::testing::internal::AdjustConstness_##constness(this)-> \
            gmock_##Method(::testing::A<GMOCK_ARG_(1, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(2, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(3, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(4, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(5, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(6, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(7, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(8, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(9, __VA_ARGS__)>(), \
                     ::testing::A<GMOCK_ARG_(10, __VA_ARGS__)>()); \
      } \
  mutable ::testing::FunctionMocker<__VA_ARGS__> GMOCK_MOCKER_(10, constness, \
      Method)


}  // namespace testing

#endif  // GMOCK_INCLUDE_GMOCK_GMOCK_GENERATED_FUNCTION_MOCKERS_H_