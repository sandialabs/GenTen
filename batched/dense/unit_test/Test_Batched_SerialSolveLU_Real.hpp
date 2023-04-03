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

#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F(TestCategory, batched_scalar_serial_solvelu_float) {
  // printf("Batched serial solveLU - float - algorithm type: Unblocked\n");
  test_batched_solvelu<TestExecSpace, float, Algo::SolveLU::Unblocked>();
  // printf("Batched serial solveLU - float - algorithm type: Blocked\n");
  test_batched_solvelu<TestExecSpace, float, Algo::SolveLU::Blocked>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, batched_scalar_serial_solvelu_double) {
  // printf("Batched serial solveLU - double - algorithm type: Unblocked\n");
  test_batched_solvelu<TestExecSpace, double, Algo::SolveLU::Unblocked>();
  // printf("Batched serial solveLU - double - algorithm type: Blocked\n");
  test_batched_solvelu<TestExecSpace, double, Algo::SolveLU::Blocked>();
}
#endif
