//===- partial_segments.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core endpoint covering one segment of a shared object reaches it through a
// memref.subview, which is what lets a core take one side of a join.

// RUN: aie-opt --aie-objectfifo-lower-cores %s | FileCheck %s

module {
  aie.device(xcve2302) {
    %t = aie.tile(1, 2)
    aie.buffer @b0(%t)  : memref<32xi32>
    %b0 = aie.get_buffer @b0 : memref<32xi32>
    aie.buffer @b1(%t)  : memref<32xi32>
    %b1 = aie.get_buffer @b1 : memref<32xi32>
    aie.lock @pl(%t) {init = 2 : i32 }
    %pl = aie.get_lock @pl
    aie.lock @cl(%t) {init = 0 : i32 }
    %cl = aie.get_lock @cl
    aie.objectfifo.pool @p(%t) {depth = 2 : i32, buffers = [@b0, @b1]} : memref<32xi32> {
      aie.objectfifo.segment @s0 {consumeLock = @cl, offset = 0 : i32, produceLock = @pl, size = 16 : i32}
      aie.objectfifo.segment @s1 {consumeLock = @cl, offset = 16 : i32, produceLock = @pl, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @half(%t) fills @p {segments = [@s1]}
    %c = aie.core(%t) {
      %e = aie.objectfifo.acquire @half (1) : memref<16xi32, strided<[1], offset: 16>>
      aie.objectfifo.release @half (1)
      aie.end
    }
  }
}

// CHECK-LABEL: aie.core
// CHECK:   %[[B0:.*]] = aie.get_buffer @b0
// CHECK:   %[[B1:.*]] = aie.get_buffer @b1
// CHECK:   %[[V0:.*]] = memref.subview %[[B0]][16] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 16>>
// CHECK:   %[[V1:.*]] = memref.subview %[[B1]][16] [16] [1] : memref<32xi32> to memref<16xi32, strided<[1], offset: 16>>
// CHECK:   scf.index_switch
// CHECK:     scf.yield %[[V0]]
// CHECK:     scf.yield %[[V1]]
