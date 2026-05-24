//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <unistd.h>
#include <vector>

#include "memory_allocator.h"
#include "test_utils.h"

int main(int argc, char *argv[]) {
  printf("Starting simple DMA loopback test\n");

  uint64_t col = 0;
  uint64_t row = 2;
  uint64_t tile_base = (col << 25) | (row << 20);

  // Write test pattern to source buffer (address 2048)
  uint64_t src_addr = tile_base + 2048;
  for (int i = 0; i < 64; i++)
    mlir_aie_sim_write32(src_addr + (4 * i), 0xA0000000 + i);

  // Load and execute the transaction binary
  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));
  printf("Configuration written\n");

  // Signal source lock (lock 1) to start the DMA send
  uint64_t lock_1 = tile_base + 0x0001F010; // lock 1
  mlir_aie_sim_write32(lock_1, 1);

  // Wait for destination lock (lock 3) to signal completion
  uint64_t lock_3 = tile_base + 0x0001F030; // lock 3
  for (int timeout = 0; timeout < 30; timeout++) {
    uint32_t l = mlir_aie_sim_read32(lock_3);
    if (l >= 1)
      break;
    sleep(1);
  }

  // Verify destination buffer (address 4096) matches source pattern
  uint64_t dst_addr = tile_base + 4096;
  int errs = 0;
  for (int i = 0; i < 64; i++) {
    uint32_t val = mlir_aie_sim_read32(dst_addr + (4 * i));
    uint32_t expected = 0xA0000000 + i;
    if (val != expected) {
      errs++;
      if (errs < 10)
        printf("  error @ %d : expected 0x%x, got 0x%x\n", i, expected, val);
    }
  }

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("FAIL: %d errors out of 64\n", errs);
    return -1;
  }
}
