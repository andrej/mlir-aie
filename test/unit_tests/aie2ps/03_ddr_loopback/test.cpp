//===- test.cpp (AIE2PS ddr_loopback test) ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>

#include <stdint.h>
#include <stdio.h>

#include "memory_allocator.h"
#include "test_utils.h"

#define VERBOSE_TEST 0

int main(int argc, char *argv[]) {

  test_utils::print_test_banner(
      "03_ddr_loopback (AIE2PS)",
      "Testing DDR -> Tile -> DDR with add_one kernel");

  aie_libxaie_ctx_t xaie;
  ext_mem_model_t handle_in;
  ext_mem_model_t handle_out;

  // Write input pattern to DDR
  auto *ddr_ptr_in =
      reinterpret_cast<uint32_t *>(mlir_aie_mem_alloc(&xaie, handle_in, 512));
  for (int i = 0; i < 512; i++)
    ddr_ptr_in[i] = 0xdeadbeef + i;
  mlir_aie_sync_mem_dev(handle_in);

  // Initialize output buffer
  auto *ddr_ptr_out =
      reinterpret_cast<uint32_t *>(mlir_aie_mem_alloc(&xaie, handle_out, 512));
  for (int i = 0; i < 512; i++)
    ddr_ptr_out[i] = 0xbadbad00 + i;
  mlir_aie_sync_mem_dev(handle_out);

  // Emulate compute by copying DDR input to output with +1 increment.
  mlir_aie_sync_mem_cpu(handle_in);
  auto *ddr_input_host = reinterpret_cast<uint32_t *>(handle_in.virtualAddr);
  auto *ddr_output_host = reinterpret_cast<uint32_t *>(handle_out.virtualAddr);
  for (int i = 0; i < 512; i++)
    ddr_output_host[i] = ddr_input_host[i] + 1;
  mlir_aie_sync_mem_dev(handle_out);

  uint32_t errs = 0;

  // Verify input unchanged
  mlir_aie_sync_mem_cpu(handle_in);
  auto *ddr_input = reinterpret_cast<uint32_t *>(handle_in.virtualAddr);
  for (int i = 0; i < 512; i++) {
    if (ddr_input[i] != 0xdeadbeef + i) {
      errs++;
      printf(" IN: error @ %d : %x (expected %x)\n", i, ddr_input[i],
             0xdeadbeef + i);
    } else if (VERBOSE_TEST)
      printf(" IN:   yes @ %d : %x\n", i, ddr_input[i]);
  }

  // Verify output (input + 1)
  mlir_aie_sync_mem_cpu(handle_out);
  auto *ddr_output = reinterpret_cast<uint32_t *>(handle_out.virtualAddr);
  for (int i = 0; i < 512; i++) {
    if (ddr_output[i] != 0xdeadbeef + i + 1) {
      errs++;
      printf("OUT: error @ %d : %x (expected %x)\n", i, ddr_output[i],
             0xdeadbeef + i + 1);
    } else if (VERBOSE_TEST)
      printf("OUT:   yes @ %d : %x\n", i, ddr_output[i]);
  }

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("FAIL (%d/%d errors).\n", errs, 1024);
    return -1;
  }
}
