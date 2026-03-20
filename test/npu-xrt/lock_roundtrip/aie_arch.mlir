//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Minimal test case for lock lifting in xclbin decompiler
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Buffers
    %buf_in = aie.buffer(%tile_0_2) {sym_name = "buf_in"} : memref<16xi32>
    %buf_out = aie.buffer(%tile_0_2) {sym_name = "buf_out"} : memref<16xi32>

    // Locks for input buffer (producer/consumer pattern)
    %lock_in_prod = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "lock_in_prod"}
    %lock_in_cons = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "lock_in_cons"}

    // Locks for output buffer (producer/consumer pattern)
    %lock_out_prod = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "lock_out_prod"}
    %lock_out_cons = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "lock_out_cons"}

    // Flow configuration
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)

    // Core - simple passthrough
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index

      // Acquire input lock (wait for data)
      aie.use_lock(%lock_in_cons, AcquireGreaterEqual, 1)
      // Acquire output lock (wait for space)
      aie.use_lock(%lock_out_prod, AcquireGreaterEqual, 1)

      // Copy data
      scf.for %i = %c0 to %c16 step %c1 {
        %val = memref.load %buf_in[%i] : memref<16xi32>
        memref.store %val, %buf_out[%i] : memref<16xi32>
      }

      // Release locks
      aie.use_lock(%lock_in_prod, Release, 1)
      aie.use_lock(%lock_out_cons, Release, 1)

      aie.end
    }

    // Shim DMA allocations
    aie.shim_dma_allocation @in_fifo (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out_fifo (%tile_0_0, S2MM, 0)

    // Runtime sequence
    aie.runtime_sequence(%arg_in: memref<16xi32>, %arg_out: memref<16xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c16_i64 = arith.constant 16 : i64

      aiex.npu.dma_memcpy_nd(%arg_in[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c16_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, issue_token = true, metadata = @in_fifo} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%arg_out[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c16_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @out_fifo} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @in_fifo}
      aiex.npu.dma_wait {symbol = @out_fifo}
    }

    // Memory tile DMA with locks
    %mem_0_2 = aie.mem(%tile_0_2) {
      // S2MM channel (input)
      %dma_start_in = aie.dma_start(S2MM, 0, ^bd_in, ^dma_out)
    ^bd_in:
      aie.use_lock(%lock_in_prod, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_in : memref<16xi32>, 0, 16)
      aie.use_lock(%lock_in_cons, Release, 1)
      aie.next_bd ^bd_in

    ^dma_out:
      // MM2S channel (output)
      %dma_start_out = aie.dma_start(MM2S, 0, ^bd_out, ^end)
    ^bd_out:
      aie.use_lock(%lock_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_out : memref<16xi32>, 0, 16)
      aie.use_lock(%lock_out_prod, Release, 1)
      aie.next_bd ^bd_out

    ^end:
      aie.end
    }
  }
}
