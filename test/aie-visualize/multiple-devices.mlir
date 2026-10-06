//===- multiple-devices.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aie-visualize
// RUN: not aie-visualize --emit-dot --topology-only %s 2>&1 | FileCheck %s --check-prefix=AMBIGUOUS
// RUN: aie-visualize --emit-dot --topology-only --device=left %s | FileCheck %s --check-prefix=LEFT
// RUN: aie-visualize --emit-dot --topology-only --device=right %s | FileCheck %s --check-prefix=RIGHT
// RUN: not aie-visualize --emit-dot --topology-only --device=missing %s 2>&1 | FileCheck %s --check-prefix=MISSING

// AMBIGUOUS: input contains multiple aie.device operations; select one with --device=<symbol>

// LEFT: tile_0_2
// LEFT: tile_0_3
// LEFT-NOT: tile_0_4
// LEFT: tile_0_2 -> tile_0_3

// RIGHT-NOT: tile_0_2
// RIGHT: tile_0_3
// RIGHT: tile_0_4
// RIGHT: tile_0_4 -> tile_0_3

// MISSING: no aie.device named 'missing'

module {
  aie.device(npu1_1col) @left {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.flow(%t02, DMA : 0, %t03, DMA : 0)
  }
  aie.device(npu1_1col) @right {
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    aie.flow(%t04, DMA : 0, %t03, DMA : 0)
  }
}
