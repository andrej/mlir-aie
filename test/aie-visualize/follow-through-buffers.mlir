//===- follow-through-buffers.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --follow-through-buffers --show-buffers %s | FileCheck %s --check-prefix=ALL
// RUN: aie-visualize --emit-dot --follow-through-buffers --only-flow=0 --show-buffers %s | FileCheck %s --check-prefix=GROUP
// RUN: rm -rf %t && aie-visualize --emit-dot-per-flow=%t --follow-through-buffers --show-buffers %s
// RUN: ls %t/flow-*.dot | count 2
// RUN: FileCheck %s --check-prefix=GROUP < %t/flow-0.dot
// RUN: FileCheck %s --check-prefix=OTHER < %t/flow-1.dot
// RUN: rm -rf %t && aie-visualize --emit-dot-per-flow=%t %s
// RUN: ls %t/flow-*.dot | count 3

// ALL: p_0_2_1_0 -> p_0_2_5_0 [color="#d73027"
// ALL: p_0_2_5_0 -> p_0_3_3_0 {{.*}} label="F0"
// ALL: p_0_3_1_1 -> p_0_3_5_1 [color="#d73027"
// ALL: p_0_3_5_1 -> p_0_4_3_1 {{.*}} label="F0"
// ALL: p_0_4_0_0 -> p_0_4_3_0 [color="#4575b4"
// ALL: p_0_4_3_0 -> p_0_3_5_0 {{.*}} label="F1"

// GROUP: label="bridge\nmemref<16xi32>"
// GROUP: p_0_2_1_0 -> p_0_2_5_0
// GROUP: p_0_3_1_1 -> p_0_3_5_1
// GROUP: p_0_3_1_0 -> buffer_0 {{.*}} label="S2MM"
// GROUP: buffer_0 -> p_0_3_1_1 {{.*}} label="MM2S"
// GROUP-NOT: p_0_4_0_0

// OTHER-NOT: label="bridge\nmemref<16xi32>"
// OTHER: p_0_4_0_0 -> p_0_4_3_0
// OTHER: p_0_4_3_0 -> p_0_3_5_0 {{.*}} label="F1"

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)

    %bridge = aie.buffer(%t03) {sym_name = "bridge"} : memref<16xi32>

    aie.mem(%t03) {
      %0 = aie.dma_start(S2MM, 0, ^s2mm, ^mm2s_start)
    ^s2mm:
      aie.dma_bd(%bridge : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^s2mm
    ^mm2s_start:
      %1 = aie.dma_start(MM2S, 1, ^mm2s, ^end)
    ^mm2s:
      aie.dma_bd(%bridge : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^mm2s
    ^end:
      aie.end
    }

    aie.flow(%t02, DMA : 0, %t03, DMA : 0)
      via (%t02 : DMA : 0 -> North : 0,
           %t03 : South : 0 -> DMA : 0)
    aie.flow(%t03, DMA : 1, %t04, DMA : 0)
      via (%t03 : DMA : 1 -> North : 1,
           %t04 : South : 1 -> DMA : 0)
    aie.flow(%t04, Core : 0, %t03, Core : 0)
      via (%t04 : Core : 0 -> South : 0,
           %t03 : North : 0 -> Core : 0)
  }
}