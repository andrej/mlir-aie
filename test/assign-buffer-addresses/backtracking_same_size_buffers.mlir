//===- backtracking_same_size_buffers.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// One MobileNet bottleneck core under the SA placer: fourteen buffers in three
// sizes fill 64064 of 65536 bytes, and the layout the ranking tries first
// strands the last 2576-byte buffer. Seven buffers share a size, so every order
// they can be placed in leaves the same free runs; the search has to remember
// the states it already exhausted, or it retries each order and runs out of
// budget before it gets back to the choices that matter.

// RUN: aie-opt --aie-assign-buffer-addresses %s | FileCheck %s

// CHECK-COUNT-7: aie.buffer @a{{[0-6]}}({{.*}}) {address = {{[0-9]+}} : i32, mem_bank = {{[0-9]}} : i32}
// CHECK-COUNT-6: aie.buffer @b{{[0-5]}}({{.*}}) {address = {{[0-9]+}} : i32, mem_bank = {{[0-9]}} : i32}
// CHECK: aie.buffer @w({{.*}}) {address = {{[0-9]+}} : i32, mem_bank = {{[0-9]}} : i32}

module @backtracking_same_size_buffers {
  aie.device(npu2) {
    %t = aie.tile(7, 4)
    aie.buffer @a0(%t)  : memref<2576xui8>
    %a0 = aie.get_buffer @a0 : memref<2576xui8>
    aie.buffer @a1(%t)  : memref<2576xui8>
    %a1 = aie.get_buffer @a1 : memref<2576xui8>
    aie.buffer @a2(%t)  : memref<2576xui8>
    %a2 = aie.get_buffer @a2 : memref<2576xui8>
    aie.buffer @a3(%t)  : memref<2576xui8>
    %a3 = aie.get_buffer @a3 : memref<2576xui8>
    aie.buffer @a4(%t)  : memref<2576xui8>
    %a4 = aie.get_buffer @a4 : memref<2576xui8>
    aie.buffer @a5(%t)  : memref<2576xui8>
    %a5 = aie.get_buffer @a5 : memref<2576xui8>
    aie.buffer @a6(%t)  : memref<2576xui8>
    %a6 = aie.get_buffer @a6 : memref<2576xui8>
    aie.buffer @b0(%t)  : memref<6720xui8>
    %b0 = aie.get_buffer @b0 : memref<6720xui8>
    aie.buffer @b1(%t)  : memref<6720xui8>
    %b1 = aie.get_buffer @b1 : memref<6720xui8>
    aie.buffer @b2(%t)  : memref<6720xui8>
    %b2 = aie.get_buffer @b2 : memref<6720xui8>
    aie.buffer @b3(%t)  : memref<6720xui8>
    %b3 = aie.get_buffer @b3 : memref<6720xui8>
    aie.buffer @b4(%t)  : memref<6720xui8>
    %b4 = aie.get_buffer @b4 : memref<6720xui8>
    aie.buffer @b5(%t)  : memref<6720xui8>
    %b5 = aie.get_buffer @b5 : memref<6720xui8>
    aie.buffer @w(%t)  : memref<4320xi8>
    %w = aie.get_buffer @w : memref<4320xi8>
    aie.core(%t) {
      aie.end
    }
  }
}
