//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>
#include <cstdio>

extern "C" void print_fortran_();
void print_cxx();

struct CountEvenIntegers {
  KOKKOS_FUNCTION void operator()(const long i, long& lcount) const {
    lcount += (i % 2) == 0;
  }
};

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  Kokkos::DefaultExecutionSpace().print_configuration(std::cout);

  const long n = argc > 1 ? atoi(argv[1]) : 10;

  printf("Number of even integers from 0 to %ld\n", n - 1);

  Kokkos::Timer timer;
  timer.reset();

  // Compute the number of even integers from 0 to n-1, in parallel.
  long count = 0;
  CountEvenIntegers functor;
  Kokkos::parallel_reduce(n, functor, count);

  double count_time = timer.seconds();
  printf("  Parallel: %ld    %10.6f\n", count, count_time);

  timer.reset();

  // Compare to a sequential loop.
  long seq_count = 0;
  for (long i = 0; i < n; ++i) {
    seq_count += (i % 2) == 0;
  }

  count_time = timer.seconds();
  printf("Sequential: %ld    %10.6f\n", seq_count, count_time);

  print_fortran_();

  return (count == seq_count) ? 0 : -1;
}