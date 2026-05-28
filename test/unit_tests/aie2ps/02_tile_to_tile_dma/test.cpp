//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "memory_allocator.h"
#include "test_utils.h"

int main(int argc, char *argv[]) {
  uint64_t col = 0;
  printf("Starting test for tile-to-tile DMA\n");

  // Write data to source tile (0, 3)
  uint64_t row_src = 3;
  uint64_t tile_0_3_data_mem = ((col << 25) | (row_src << 20)) + 2048;
  for (int i = 0; i < 256; i++)
    mlir_aie_sim_write32(tile_0_3_data_mem + (4 * i), 0xdeadbeef + i);

  // Execute the transaction
  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  printf("done writing configuration\n");

  // write 2 to lock_0_3
  uint64_t tile_0_3_lock_0 = ((col << 25) | (row_src << 20)) + 0x0001F000;
  mlir_aie_sim_write32(tile_0_3_lock_0, 2);

  // Wait for DMA completion at destination (0, 2)
  uint64_t row_dst = 2;
  uint64_t tile_0_2_lock_1 = ((col << 25) | (row_dst << 20)) + 0x0001F010;
  while (1) {
    uint32_t l = mlir_aie_sim_read32(tile_0_2_lock_1);
    // printf("lock value: %d\n", l);
    if (l >= 1)
      break;
    sleep(1);
  }

  // Check data in source tile (0, 3) - should still be there
  printf("Checking source tile (0, 3) data:\n");
  uint32_t output_src[256];
  int errs_src = 0;
  for (int i = 0; i < 256; i++) {
    output_src[i] = mlir_aie_sim_read32(tile_0_3_data_mem + (4 * i));
    if (output_src[i] != 0xdeadbeef + i) {
      errs_src++;
      if (errs_src < 10) // Only print first 10 errors
        printf("  error @ %d : expected 0x%x, got 0x%x\n", i, 0xdeadbeef + i,
               output_src[i]);
    }
  }

  if (!errs_src) {
    printf("  Source tile (0, 3): PASS! All %d values correct.\n", 256);
  } else {
    printf("  Source tile (0, 3): failed %d errors out of %d values.\n",
           errs_src, 256);
  }

  // Check data in destination tile (0, 2)
  printf("Checking destination tile (0, 2) data:\n");
  uint64_t tile_0_2_data_mem = ((col << 25) | (row_dst << 20)) + 2048;
  uint32_t output_dst[256];
  int errs_dst = 0;
  for (int i = 0; i < 256; i++) {
    output_dst[i] = mlir_aie_sim_read32(tile_0_2_data_mem + (4 * i));
    if (output_dst[i] != 0xdeadbeef + i) {
      errs_dst++;
      if (errs_dst < 10) // Only print first 10 errors
        printf("  error @ %d : expected 0x%x, got 0x%x\n", i, 0xdeadbeef + i,
               output_dst[i]);
      if (errs_dst < 10) // Only print first 10 errors
        printf("  error @ %d : expected 0x%x, got 0x%x\n", i, 0xdeadbeef + i,
               output_dst[i]);
    }
  }

  if (!errs_dst) {
    printf("  Destination tile (0, 2): PASS! All %d values correct.\n", 256);
  } else {
    printf("  Destination tile (0, 2): failed %d errors out of %d values.\n",
           errs_dst, 256);
  }

  // Overall result
  int total_errs = errs_src + errs_dst;
  if (!total_errs) {
    printf("\nOverall: PASS!\n");
    return 0;
  } else {
    printf("\nOverall: failed with %d errors.\n", total_errs);
    return -1;
  }
}
