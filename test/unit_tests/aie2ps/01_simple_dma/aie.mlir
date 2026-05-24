//===- aie.mlir (AIE2PS simple DMA) ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator, chess
//
// RUN: %python aiecc.py --alloc-scheme=basic-sequential --xchesscc --xbridge --no-compile-host --aie-generate-txn --txn-name=transaction.bin --aiesim %s %test_utils_flags %S/test.cpp
// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s
//
// CHECK: AIE2PS
// CHECK: PASS!
//
// Simple single-tile DMA test for AIE2PS. Tile (0,2) does a DMA loopback:
// MM2S channel 0 sends data, S2MM channel 0 receives it back into another buffer.
// Validates BD allocation and DMA channel configuration work on xcve3558.

module @test01_simple_dma_aie2ps {
  aie.device(xcve3558) {
    %tile_0_2 = aie.tile(0, 2)

    %buf_src = aie.buffer(%tile_0_2) {address = 2048 : i32, sym_name = "buf_src"} : memref<64xi32>
    %buf_dst = aie.buffer(%tile_0_2) {address = 4096 : i32, sym_name = "buf_dst"} : memref<64xi32>

    // Locks for src buffer (producer/consumer)
    %lock_src = aie.lock(%tile_0_2, 0) { init = 1 : i32 }
    %lock_src_done = aie.lock(%tile_0_2, 1) { init = 0 : i32 }
    // Locks for dst buffer (producer/consumer)
    %lock_dst = aie.lock(%tile_0_2, 2) { init = 1 : i32 }
    %lock_dst_done = aie.lock(%tile_0_2, 3) { init = 0 : i32 }

    %mem_0_2 = aie.mem(%tile_0_2) {
      // MM2S: send from buf_src
      %0 = aie.dma_start(MM2S, 0, ^send, ^recv_start)
    ^send:
      aie.use_lock(%lock_src_done, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_src : memref<64xi32>, 0, 64) {bd_id = 0 : i32}
      aie.use_lock(%lock_src, Release, 1)
      aie.next_bd ^send
    ^recv_start:
      // S2MM: receive into buf_dst
      %1 = aie.dma_start(S2MM, 0, ^recv, ^end)
    ^recv:
      aie.use_lock(%lock_dst, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_dst : memref<64xi32>, 0, 64) {bd_id = 1 : i32}
      aie.use_lock(%lock_dst_done, Release, 1)
      aie.next_bd ^recv
    ^end:
      aie.end
    }

    // Loopback: tile DMA MM2S ch0 -> S2MM ch0
    aie.flow(%tile_0_2, DMA : 0, %tile_0_2, DMA : 0)
  }
}
