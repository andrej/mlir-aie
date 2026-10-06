//===- routes.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --no-follow-buffers --show-buffers --show-packet-ids %s | FileCheck %s --check-prefix=ALL
// RUN: aie-visualize --emit-dot --no-follow-buffers %s | FileCheck %s --check-prefix=GUIDED
// RUN: aie-visualize --emit-dot --no-follow-buffers --show-vias %s | FileCheck %s --check-prefix=VIAS
// RUN: aie-visualize --emit-dot --no-follow-buffers --topology-only --show-buffers %s | FileCheck %s --check-prefix=TOPOLOGY
// RUN: aie-visualize --emit-dot --no-follow-buffers --topology-only --device=main %s | FileCheck %s --check-prefix=TOPOLOGY
// RUN: aie-visualize --emit-dot --no-follow-buffers --show-buffers %s | FileCheck %s --check-prefix=NO-IDS
// RUN: aie-visualize --emit-dot --no-follow-buffers %s | FileCheck %s --check-prefix=NO-BUFFER-LINK
// RUN: aie-visualize --emit-dot --no-follow-buffers --highlight-flow=1 --show-packet-ids %s | FileCheck %s --check-prefix=HIGHLIGHT
// RUN: aie-visualize --emit-dot --no-follow-buffers --only-flow=1 --show-packet-ids %s | FileCheck %s --check-prefix=ONLY
// RUN: not aie-visualize --emit-dot --no-follow-buffers --only-flow=9 %s 2>&1 | FileCheck %s --check-prefix=BAD-ID
// RUN: rm -rf %t && aie-visualize --emit-dot-per-flow=%t --no-follow-buffers --show-buffers --show-packet-ids %s
// RUN: FileCheck %s --check-prefix=FLOW0 < %t/flow-0.dot
// RUN: FileCheck %s --check-prefix=FLOW2 < %t/flow-2.dot
// RUN: ls %t/flow-*.dot | count 5
// RUN: not aie-visualize --emit-dot --emit-dot-per-flow=%t %s 2>&1 | FileCheck %s --check-prefix=BAD-MODE

// ALL: digraph aie_routes
// ALL: tile_0_2 {{.*}} pos="0.000000e+00,6.000000e+00!"
// ALL: tile_0_3 {{.*}} pos="0.000000e+00,9.000000e+00!"
// ALL: buffer_0 {{.*}} label="source"
// ALL: buffer_1 {{.*}} label="dest"
// ALL-NOT: memref
// ALL: p_0_2_5_0 -> p_0_3_3_0 [color="#d73027:#4575b4"
// ALL-SAME: label=<<FONT COLOR="#d73027">F0 pkt=3/31</FONT><BR/><FONT COLOR="#4575b4">F1 pkt=4/31</FONT>>
// ALL: p_0_3_0_0 -> p_0_3_0_1 {{.*}} label=<<FONT COLOR="#984ea3">F3 pkt=6/31</FONT>>
// ALL-DAG: p_0_4_0_1 -> p_0_4_3_2
// ALL-DAG: p_0_4_3_2 -> p_0_3_5_2 {{.*}} label=<<FONT COLOR="#ff7f00">F4</FONT>>
// ALL-DAG: p_0_3_5_2 -> p_0_3_0_0
// ALL-NOT: p_0_2_0_0
// ALL: buffer_0 -> p_0_2_1_0_m {{.*}}style=dashed];
// ALL: p_0_3_1_0_s -> buffer_1 {{.*}}style=dashed];
// ALL-NOT: MM2S
// ALL-NOT: S2MM

// GUIDED: graph [layout=neato
// GUIDED: p_0_2_1_0_m [shape=box, fixedsize=true,{{.*}} label="MM2S0"
// GUIDED: p_0_2_5_0 [shape=point, width=0, height=0, {{.*}} label=""];
// GUIDED: p_0_2_1_0_m -> p_0_2_5_0 {{.*}} arrowhead=none];
// GUIDED: p_0_2_5_0 -> p_0_3_3_0 {{.*}} arrowhead=none];
// GUIDED: p_0_3_3_0 -> p_0_3_1_0_s
// GUIDED-NOT: p_0_3_3_0 {{.*}} xlabel=

// VIAS: p_0_2_5_0 [shape=point, width=0.09, {{.*}} xlabel="N0"];
// VIAS: p_0_3_3_0 [shape=point, width=0.09, {{.*}} xlabel="S0"];
// VIAS: p_0_2_1_0_m -> p_0_2_5_0
// VIAS-NOT: arrowhead=none

// TOPOLOGY: graph [layout=neato
// TOPOLOGY: tile_0_2 {{.*}} pos="0.000000e+00,6.000000e+00!"
// TOPOLOGY: tile_0_3 {{.*}} pos="0.000000e+00,9.000000e+00!"
// TOPOLOGY-NOT: p_0_2_5_0
// TOPOLOGY-NOT: buffer_0
// TOPOLOGY: tile_0_2 -> tile_0_3
// TOPOLOGY: tile_0_4 -> tile_0_3

// NO-IDS: p_0_2_5_0 -> p_0_3_3_0
// NO-IDS-NOT: pkt=
// NO-IDS-NOT: label=<

// NO-BUFFER-LINK: p_0_2_5_0 -> p_0_3_3_0
// NO-BUFFER-LINK-NOT: buffer_0

// HIGHLIGHT: p_0_2_1_0_m -> p_0_2_5_0 [color="#c2c2c2:#4575b4", penwidth="2.4"
// HIGHLIGHT: p_0_2_5_0 -> p_0_3_3_0 {{.*}} label=<<FONT COLOR="#c2c2c2">F0 pkt=3/31</FONT><BR/><FONT COLOR="#4575b4">F1 pkt=4/31</FONT>>
// HIGHLIGHT: p_0_4_1_0_m -> p_0_4_3_1 [color="#c2c2c2", penwidth="1.2"

// ONLY-NOT: F0 pkt=3
// ONLY: F1 pkt=4/31
// ONLY-NOT: F2 pkt=5

// BAD-ID: --only-flow references unknown flow 9; valid IDs are 0 through 4
// BAD-MODE: --emit-dot and --emit-dot-per-flow are mutually exclusive

// FLOW0: buffer_0 {{.*}} label="source"
// FLOW0: buffer_1 {{.*}} label="dest"
// FLOW0-NOT: label="other"
// FLOW0: F0 pkt=3/31
// FLOW0-NOT: F1 pkt=4/31

// FLOW2-NOT: label="source"
// FLOW2: buffer_2 {{.*}} label="other"
// FLOW2: F2 pkt=5/31
// FLOW2-NOT: F3 pkt=6/31

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)

    %source = aie.buffer(%t02) {sym_name = "source"} : memref<16xi32>
    %dest = aie.buffer(%t03) {sym_name = "dest"} : memref<16xi32>
    %other = aie.buffer(%t04) {sym_name = "other"} : memref<16xi32>

    aie.mem(%t02) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%source : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.mem(%t03) {
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%dest : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.mem(%t04) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%other : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    aie.packet_flow(3, mask = 31) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    } via (%t02 : DMA : 0 -> North : 0,
           %t03 : South : 0 -> DMA : 0)
    aie.packet_flow(4, mask = 31) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    } via (%t02 : DMA : 0 -> North : 0,
           %t03 : South : 0 -> DMA : 0)
    aie.packet_flow(5, mask = 31) {
      aie.packet_source<%t04, DMA : 0>
      aie.packet_dest<%t03, DMA : 1>
    } via (%t04 : DMA : 0 -> South : 1,
           %t03 : North : 1 -> DMA : 1)
    aie.packet_flow(6, mask = 31) {
      aie.packet_source<%t03, Core : 0>
      aie.packet_dest<%t03, Core : 1>
    } via (%t03 : Core : 0 -> Core : 1)

    // This materialized connection differs from the pinned flow below. The
    // visualizer must use the flow's vias as its only route description.
    aie.switchbox(%t02) {
      aie.connect<Core : 0, DMA : 0>
    }
    aie.flow(%t04, Core : 1, %t03, Core : 0)
      via (%t04 : Core : 1 -> South : 2,
           %t03 : North : 2 -> Core : 0)
  }
}