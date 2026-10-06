//===- requires-vias.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aie-visualize
// RUN: not aie-visualize --emit-dot %s 2>&1 | FileCheck %s

// CHECK: error: 'aie.flow' op requires vias; run aie-find-flows with emit-vias=true

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.flow(%t02, DMA : 0, %t03, DMA : 0)
  }
}