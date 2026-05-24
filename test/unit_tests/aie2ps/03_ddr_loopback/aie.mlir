//===- aie.mlir (AIE2PS ddr_loopback) ------------------------*- MLIR -*-===//
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
//
// RUN: xchesscc_wrapper aie2ps -c %S/kernel.cc
// RUN: %python aiecc.py --alloc-scheme=basic-sequential --no-compile --aiesim %s %test_utils_flags %S/test.cpp
// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s
//
// CHECK: AIE2PS
// CHECK: PASS!
//
// Test DDR loopback for AIE2PS: data flows from DDR -> Core Tile -> DDR
// The core performs a simple add_one operation on the data
//
module @test03_ddr_loopback_aie2ps {
  aie.device(xcve3558) {
    %tile_0_2 = aie.tile(0, 2)

    // Input and output buffers on core tile
    %ping_in_buffer  = aie.buffer(%tile_0_2) {address = 2048 : i32, sym_name = "ping_in_buffer"} : memref<64xi32>
    %pong_in_buffer  = aie.buffer(%tile_0_2) {address = 2304 : i32, sym_name = "pong_in_buffer"} : memref<64xi32>
    %ping_out_buffer = aie.buffer(%tile_0_2) {address = 2560 : i32, sym_name = "ping_out_buffer"} : memref<64xi32>
    %pong_out_buffer = aie.buffer(%tile_0_2) {address = 2816 : i32, sym_name = "pong_out_buffer"} : memref<64xi32>

    // Locks for input and output coordination
    %tile_input_read_lock   = aie.lock(%tile_0_2, 0) { init = 0 : i32 } 
    %tile_input_write_lock  = aie.lock(%tile_0_2, 1) { init = 1 : i32 } 
    %tile_output_read_lock  = aie.lock(%tile_0_2, 2) { init = 0 : i32 } 
    %tile_output_write_lock = aie.lock(%tile_0_2, 3) { init = 1 : i32 } 

    func.func private @add_one(%A: memref<64xi32>, %B: memref<64xi32>) -> ()

    // Core logic: read input, call kernel, write output
    %core02 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c1 = arith.constant 1 : index  

      scf.for %i = %c0 to %c256 step %c1 {
        // Process ping buffers
        aie.use_lock(%tile_input_read_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%tile_output_write_lock, AcquireGreaterEqual, 1)
        func.call @add_one(%ping_in_buffer, %ping_out_buffer) : (memref<64xi32>, memref<64xi32>) -> ()
        aie.use_lock(%tile_input_write_lock, Release, 1)
        aie.use_lock(%tile_output_read_lock, Release, 1)

        // Process pong buffers
        aie.use_lock(%tile_input_read_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%tile_output_write_lock, AcquireGreaterEqual, 1)
        func.call @add_one(%pong_in_buffer, %pong_out_buffer) : (memref<64xi32>, memref<64xi32>) -> ()
        aie.use_lock(%tile_input_write_lock, Release, 1)
        aie.use_lock(%tile_output_read_lock, Release, 1)
      }
      aie.end
    } { link_with="kernel.o" }
      
    // DMA operations on core tile
    %mem_0_2 = aie.mem(%tile_0_2) {
        %srcDma = aie.dma_start("MM2S", 0, ^bb1, ^dma0)
      ^dma0:
        %dstDma = aie.dma_start("S2MM", 0, ^bb4, ^end)
    ^bb1: 
      aie.use_lock(%tile_output_read_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_out_buffer : memref<64xi32>, 0, 64) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%tile_output_write_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2: 
      aie.use_lock(%tile_output_read_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_out_buffer : memref<64xi32>, 0, 64) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%tile_output_write_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb4: 
      aie.use_lock(%tile_input_write_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_in_buffer : memref<64xi32>, 0, 64) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%tile_input_read_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5: 
      aie.use_lock(%tile_input_write_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_in_buffer : memref<64xi32>, 0, 64) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%tile_input_read_lock, Release, 1)
      aie.next_bd ^bb4
    ^end:   
      aie.end
    }

    // Shim tile configuration
    %tile_0_0 = aie.tile(0, 0)
    %ext_input_buffer  = aie.external_buffer{sym_name = "ext_input_buffer"}  : memref<512xi32>
    %ext_output_buffer = aie.external_buffer{sym_name = "ext_output_buffer"} : memref<512xi32>

    %shim_tile_input_write_lock  = aie.lock(%tile_0_0, 0) { init = 1 : i32 } 
    %shim_tile_input_read_lock   = aie.lock(%tile_0_0, 1) { init = 0 : i32 } 
    %shim_tile_output_write_lock = aie.lock(%tile_0_0, 2) { init = 1 : i32 } 
    %shim_tile_output_read_lock  = aie.lock(%tile_0_0, 3) { init = 0 : i32 } 

    %mem_0_0 = aie.shim_dma(%tile_0_0) {
      %mm2s = aie.dma_start(MM2S, 0, ^bb_mm2s_0, ^bb_s2mm)
    ^bb_s2mm:
      %s2mm = aie.dma_start(S2MM, 0, ^bb_s2mm_0, ^bb_end)
    ^bb_mm2s_0:
      aie.use_lock(%shim_tile_input_read_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_input_buffer : memref<512xi32>, 0, 512) {bd_id = 0 : i32}
      aie.use_lock(%shim_tile_input_write_lock, Release, 1)
      aie.next_bd ^bb_end
    ^bb_s2mm_0:
      aie.use_lock(%shim_tile_output_write_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_output_buffer : memref<512xi32>, 0, 512) {bd_id = 1 : i32}
      aie.use_lock(%shim_tile_output_read_lock, Release, 1)
      aie.next_bd ^bb_end
    ^bb_end:
      aie.end
    }

    // Flow connections
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)
  }
}
