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
// RUN: ls %t/flow-*.dot | count 4
// RUN: FileCheck %s --check-prefix=GROUP < %t/flow-0.dot
// RUN: FileCheck %s --check-prefix=OTHER < %t/flow-1.dot
// RUN: FileCheck %s --check-prefix=FANOUT < %t/flow-2.dot
// RUN: FileCheck %s --check-prefix=JOIN < %t/flow-3.dot
// RUN: rm -rf %t && aie-visualize --emit-dot-per-flow=%t %s
// RUN: ls %t/flow-*.dot | count 9

// ALL: p_0_2_1_0_m -> p_0_2_5_0 [color="#d73027"
// ALL: p_0_2_5_0 -> p_0_3_3_0 {{.*}} label="F0"
// ALL: p_0_3_1_0_m -> p_0_3_5_1 [color="#d73027"
// ALL: p_0_3_1_3_m -> p_0_3_3_3 [color="#d73027"
// ALL: p_0_3_5_1 -> p_0_4_3_1 {{.*}} label="F0"
// ALL: p_0_3_5_2 -> p_0_3_1_2_s [color="#d73027"
// ALL: p_0_4_0_1 -> p_0_4_3_0 [color="#4575b4"
// ALL: p_0_4_3_0 -> p_0_3_5_0 {{.*}} label="F1"

// GROUP: p_0_3_1_0_s {{.*}} xlabel="S2MM0"
// GROUP: p_0_3_1_0_m {{.*}} xlabel="MM2S0"
// GROUP: label="bridge"
// GROUP-NOT: memref
// GROUP: p_0_2_1_0_m -> p_0_2_5_0
// GROUP: p_0_3_1_0_m -> p_0_3_5_1
// GROUP: p_0_3_1_3_m -> p_0_3_3_3
// GROUP: p_0_3_5_2 -> p_0_3_1_2_s
// GROUP: p_0_3_1_0_s -> buffer_0 {{.*}}style=dashed];
// GROUP: buffer_0 -> p_0_3_1_0_m {{.*}}style=dashed];
// GROUP: p_0_3_1_2_s -> buffer_0 {{.*}}style=dashed];
// GROUP: buffer_0 -> p_0_3_1_3_m {{.*}}style=dashed];
// GROUP-NOT: S2MM
// GROUP-NOT: MM2S
// GROUP-NOT: p_0_4_0_1

// OTHER-NOT: label="bridge"
// OTHER: p_0_4_0_1 -> p_0_4_3_0
// OTHER: p_0_4_3_0 -> p_0_3_5_0 {{.*}} label="F1"

// FANOUT: label="fanout"
// FANOUT: buffer_1 -> p_0_2_1_2_m {{.*}}style=dashed];
// FANOUT: buffer_1 -> p_0_2_1_3_m {{.*}}style=dashed];

// JOIN: label="join"
// JOIN: p_0_4_1_2_s -> buffer_2 {{.*}}style=dashed];
// JOIN: p_0_4_1_3_s -> buffer_2 {{.*}}style=dashed];

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)

    %bridge = aie.buffer(%t03) {sym_name = "bridge"} : memref<16xi32>
    %fanout = aie.buffer(%t02) {sym_name = "fanout"} : memref<16xi32>
    %join = aie.buffer(%t04) {sym_name = "join"} : memref<16xi32>

    aie.mem(%t02) {
      %0 = aie.dma_start(MM2S, 2, ^mm2s2, ^mm2s3_start)
    ^mm2s2:
      aie.dma_bd(%fanout : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^mm2s2
    ^mm2s3_start:
      %1 = aie.dma_start(MM2S, 3, ^mm2s3, ^end)
    ^mm2s3:
      aie.dma_bd(%fanout : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^mm2s3
    ^end:
      aie.end
    }

    aie.mem(%t03) {
      %0 = aie.dma_start(S2MM, 0, ^s2mm0, ^s2mm2_start)
    ^s2mm0:
      aie.dma_bd(%bridge : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^s2mm0
    ^s2mm2_start:
      %1 = aie.dma_start(S2MM, 2, ^s2mm2, ^mm2s0_start)
    ^s2mm2:
      aie.dma_bd(%bridge : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^s2mm2
    ^mm2s0_start:
      %2 = aie.dma_start(MM2S, 0, ^mm2s0, ^mm2s3_start)
    ^mm2s0:
      aie.dma_bd(%bridge : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^mm2s0
    ^mm2s3_start:
      %3 = aie.dma_start(MM2S, 3, ^mm2s3, ^end)
    ^mm2s3:
      aie.dma_bd(%bridge : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^mm2s3
    ^end:
      aie.end
    }

    aie.mem(%t04) {
      %0 = aie.dma_start(S2MM, 2, ^s2mm2, ^s2mm3_start)
    ^s2mm2:
      aie.dma_bd(%join : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^s2mm2
    ^s2mm3_start:
      %1 = aie.dma_start(S2MM, 3, ^s2mm3, ^end)
    ^s2mm3:
      aie.dma_bd(%join : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^s2mm3
    ^end:
      aie.end
    }

    aie.flow(%t02, DMA : 0, %t03, DMA : 0)
      via (%t02 : DMA : 0 -> North : 0,
           %t03 : South : 0 -> DMA : 0)
    aie.flow(%t03, DMA : 0, %t04, DMA : 0)
      via (%t03 : DMA : 0 -> North : 1,
           %t04 : South : 1 -> DMA : 0)
    aie.flow(%t04, Core : 0, %t03, DMA : 2)
      via (%t04 : Core : 0 -> South : 2,
           %t03 : North : 2 -> DMA : 2)
    aie.flow(%t03, DMA : 3, %t02, Core : 0)
      via (%t03 : DMA : 3 -> South : 3,
           %t02 : North : 3 -> Core : 0)
    aie.flow(%t04, Core : 1, %t03, Core : 1)
       via (%t04 : Core : 1 -> South : 0,
         %t03 : North : 0 -> Core : 1)
    aie.flow(%t02, DMA : 2, %t03, Core : 2)
      via (%t02 : DMA : 2 -> North : 2,
           %t03 : South : 2 -> Core : 2)
    aie.flow(%t02, DMA : 3, %t04, Core : 2)
      via (%t02 : DMA : 3 -> North : 3,
           %t03 : South : 3 -> North : 3,
           %t04 : South : 3 -> Core : 2)
    aie.flow(%t02, Core : 2, %t04, DMA : 2)
      via (%t02 : Core : 2 -> North : 4,
           %t03 : South : 4 -> North : 4,
           %t04 : South : 4 -> DMA : 2)
    aie.flow(%t03, Core : 2, %t04, DMA : 3)
      via (%t03 : Core : 2 -> North : 5,
           %t04 : South : 5 -> DMA : 3)
  }
}