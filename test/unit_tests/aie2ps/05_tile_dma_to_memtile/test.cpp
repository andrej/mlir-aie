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
  printf("Core tile to MemTile DMA test\n");

  uint64_t col = 0;
  uint64_t core_row = 2;

  // Write test pattern to core tile (0,2) source buffer at address 2048
  uint64_t tile_data_mem = ((col << 25) | (core_row << 20)) + 2048;
  for (int i = 0; i < 256; i++)
    mlir_aie_sim_write32(tile_data_mem + (i * 4), 0xdeadbeef + i);

  // Load and execute transaction binary
  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  // Signal core tile lock 0 to start DMA send
  uint64_t tile_lock_0 = ((col << 25) | (core_row << 20)) + 0x0001F000;
  mlir_aie_sim_write32(tile_lock_0, 2);

  // Wait for memtile (0,1) lock 1 to signal completion
  uint64_t memtile_row = 1;
  uint64_t memtile_lock_1 =
      ((col << 25) | (memtile_row << 20)) + 0x000C0010; // memtile lock 1
  for (int timeout = 0; timeout < 30; timeout++) {
    uint32_t l = mlir_aie_sim_read32(memtile_lock_1);
    if (l >= 1)
      break;
    sleep(1);
  }

  // Verify data arrived at memtile (0,1) buffer at address 2048
  // MemTile data memory starts at base 0x0 within the tile
  uint64_t memtile_data_mem =
      ((col << 25) | (memtile_row << 20)) + 2048;
  int errs = 0;
  for (int i = 0; i < 256; i++) {
    uint32_t val = mlir_aie_sim_read32(memtile_data_mem + (4 * i));
    if (val != 0xdeadbeef + i) {
      errs++;
      if (errs < 10)
        printf("error @ %d : expected 0x%x, got 0x%x\n", i,
               (uint32_t)(0xdeadbeef + i), val);
    }
  }

  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("FAIL: %d errors out of 256\n", errs);
    return -1;
  }
}
