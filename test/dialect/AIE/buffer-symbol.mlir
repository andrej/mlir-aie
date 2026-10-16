//===- buffer-symbol.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

module {
  aie.device(npu1) {
    %tile = aie.tile(1, 2)
    aie.buffer @data(%tile) : memref<16xi32>
    aie.core(%tile) {
      %data = aie.get_buffer @data : memref<16xi32>
      %c0 = arith.constant 0 : index
      %value = memref.load %data[%c0] : memref<16xi32>
      memref.store %value, %data[%c0] : memref<16xi32>
      aie.end
    }
  }
}

// CHECK: aie.buffer @data(%[[TILE:.*]]) : memref<16xi32>
// CHECK: %[[DATA:.*]] = aie.get_buffer @data : memref<16xi32>
