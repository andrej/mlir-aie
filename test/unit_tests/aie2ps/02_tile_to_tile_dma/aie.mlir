//===- aie.mlir (AIE2PS tile_to_tile_dma) ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
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
// Test tile-to-tile DMA transfers for AIE2PS
// This verifies that core tiles can communicate via DMA using ping-pong buffers
//
module @test02_tile_to_tile_dma_aie2ps {
  aie.device(xcve3558) {
    // Receiving tile (0, 2)
    %tile_0_2 = aie.tile(0, 2)
    %buf02_0 = aie.buffer(%tile_0_2) {address = 2048 : i32, sym_name = "buf02_0"} : memref<256xi32>
    %buf02_1 = aie.buffer(%tile_0_2) {address = 4096 : i32, sym_name = "buf02_1"} : memref<256xi32>
    %lock_0_2 = aie.lock(%tile_0_2, 0) { init = 2 : i32 }
    %lock_0_2_0 = aie.lock(%tile_0_2, 1) { init = 0 : i32 }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf02_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf02_1 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }

    // Sending tile (0, 3)
    %tile_0_3 = aie.tile(0, 3)
    %buf03_0 = aie.buffer(%tile_0_3) {address = 2048 : i32, sym_name = "buf03_0"} : memref<256xi32>
    %buf03_1 = aie.buffer(%tile_0_3) {address = 4096 : i32, sym_name = "buf03_1"} : memref<256xi32>

    %lock_0_3 = aie.lock(%tile_0_3, 0) { init = 0 : i32 }
    %lock_0_3_0 = aie.lock(%tile_0_3, 1)

    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf03_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_0_3_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf03_1 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_0_3_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }

    // Flow connection from tile (0,3) to tile (0,2)
    aie.flow(%tile_0_3, DMA : 0, %tile_0_2, DMA : 0)
  }
}
