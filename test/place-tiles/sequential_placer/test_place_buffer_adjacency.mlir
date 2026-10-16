//===- test_place_buffer_adjacency.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// CHECK-LABEL: @adjacency_both_pinned_compatible
module @adjacency_both_pinned_compatible {
  aie.device(npu1) {
    // CHECK-DAG: aie.tile(0, 2)
    // CHECK-DAG: aie.tile(0, 3)
    %owner = aie.logical_tile<CoreTile>(0, 2)
    aie.buffer @buf(%owner) : memref<16xi32>
    %buf = aie.get_buffer @buf : memref<16xi32>
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(0, 3)
    aie.core(%consumer) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Both endpoints pinned at compatible positions: owner at (0, 2), consumer
// at (0, 3) (consumer's S-affinity slot). Placer is a pass-through.
// CHECK-LABEL: @adjacency_self_reference_no_constraint
module @adjacency_self_reference_no_constraint {
  aie.device(npu1) {
    // CHECK-DAG: aie.tile(0, 2)
    %lt = aie.logical_tile<CoreTile>(0, 2)
    aie.buffer @buf(%lt) : memref<16xi32>
    %buf = aie.get_buffer @buf : memref<16xi32>
    aie.core(%lt) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Buffer accessed via memref.subview is still traced back to the owning
// LogicalTileOp. This setup has no reachable adjacency slot for the
// unconstrained consumer, so placement fails.
module @adjacency_through_view_op {
  aie.device(npu1) {
    %owner = aie.logical_tile<CoreTile>(2, 3)
    aie.buffer @buf(%owner) : memref<16xi32>
    %buf = aie.get_buffer @buf : memref<16xi32>
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%consumer) {
      %sub = memref.subview %buf[0][8][1] : memref<16xi32> to memref<8xi32, strided<[1]>>
      %i = arith.constant 0 : index
      %v = memref.load %sub[%i] : memref<8xi32, strided<[1]>>
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// AIE2P (npu2) target-model dispatch: this setup also has no reachable
// adjacency slot for the unconstrained consumer, so placement fails.
module @adjacency_npu2_target_model {
  aie.device(npu2) {
    %owner = aie.logical_tile<CoreTile>(2, 3)
    aie.buffer @buf(%owner) : memref<16xi32>
    %buf = aie.get_buffer @buf : memref<16xi32>
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%consumer) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    // CHECK-NOT: aie.logical_tile
  }
}
