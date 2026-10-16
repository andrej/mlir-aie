//===- bad_buffer.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A named aie.buffer op is a symbol, so its parent region must be a symbol
// table (the device scope). Nesting one inside a core body fails op
// verification before aie-prepare-buffers runs.
// RUN: not aie-opt --aie-prepare-buffers %s 2>&1 | FileCheck %s
// CHECK: error: 'aie.buffer' op symbol's parent must have the SymbolTable trait

module @test {
  aie.device(xcvc1902) {
    %0 = aie.tile(3, 3)
    aie.buffer @a(%0)  : memref<16xi8>
    %b1 = aie.get_buffer @a : memref<16xi8>
    %3 = aie.tile(4, 4)
    aie.buffer @buffer_4(%3) : memref<500xi32>
    %4 = aie.get_buffer @buffer_4 : memref<500xi32>
    aie.core(%0) {
     aie.buffer @c(%0)  : memref<16xi16>
     %b2 = aie.get_buffer @c : memref<16xi16>
     aie.buffer @b(%0)  : memref<3x3xi16> = dense<[[0, 4096, 0], [4096, -16384, 4096], [0, 4096, 0]]>
     %1 = aie.get_buffer @b : memref<3x3xi16>
      aie.end
    }
    aie.core(%3) {
      aie.end
    }
  }
}
