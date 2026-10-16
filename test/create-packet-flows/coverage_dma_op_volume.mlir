//===- coverage_dma_op_volume.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s
// RUN: sed 's/repeat_count = 0/repeat_count = 1/' %s | not aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" 2>&1 | FileCheck %s --check-prefix=STALL
// RUN: sed 's/loop = false/loop = true/' %s | aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" 2>&1 | FileCheck %s
// RUN: sed -e 's/loop = false/loop = true/' -e 's/sd\([0-5]\), Release/sl\1, Release/' %s | not aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" 2>&1 | FileCheck %s --check-prefix=STALL

// arbiter_keep_pkt_header_receive.mlir written with aie.dma instead of
// aie.dma_start. Each receiver takes 64 bytes before its lock blocks. Sent
// once, every packet fits and the seven master ports at (0,1) share six
// arbiters; sent twice, or forever, each receiver fills and they cannot. A
// looping sender whose lock nothing refills still sends once.

// CHECK-NOT:   error
// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-COUNT-7: aie.masterset

// STALL: error: Unable to find a legal routing: at tile (0, 1), no two of
// STALL-SAME: but the switchbox has 6 free.

module {
  aie.device(npu2) {
    %m  = aie.tile(0, 1)
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(1, 2)
    %t2 = aie.tile(2, 2)
    %t3 = aie.tile(3, 2)
    %t4 = aie.tile(4, 2)
    %t5 = aie.tile(5, 2)
    %t6 = aie.tile(0, 3)

    aie.packet_flow(0) { aie.packet_source<%t0, DMA : 0> aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t1, DMA : 0> aie.packet_dest<%m, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%t2, DMA : 0> aie.packet_dest<%m, DMA : 2> }
    aie.packet_flow(3) { aie.packet_source<%t3, DMA : 0> aie.packet_dest<%m, DMA : 3> }
    aie.packet_flow(4) { aie.packet_source<%t4, DMA : 0> aie.packet_dest<%m, DMA : 4> }
    aie.packet_flow(5) { aie.packet_source<%t5, DMA : 0> aie.packet_dest<%m, DMA : 5> }
    aie.packet_flow(6) { aie.packet_source<%m, DMA : 0> aie.packet_dest<%t6, DMA : 1> }

    aie.buffer @b0(%t0) : memref<16xi32>
    %b0 = aie.get_buffer @b0 : memref<16xi32>
    aie.lock @sl0(%t0, 0) {init = 1 : i32}
    %sl0 = aie.get_lock @sl0
    aie.lock @sd0(%t0, 1) {init = 0 : i32}
    %sd0 = aie.get_lock @sd0
    aie.mem(%t0) {
      %one = arith.constant 1 : i32
      %0 = aie.dma(MM2S, 0) {loop = false, repeat_count = 0 : i32} [{
        aie.use_lock(@sl0, AcquireGreaterEqual, %one)
        aie.dma_bd(%b0 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 0, pkt_type = 0>}
        aie.use_lock(@sd0, Release, %one)
      }]
      aie.end
    }
    aie.buffer @b1(%t1) : memref<16xi32>
    %b1 = aie.get_buffer @b1 : memref<16xi32>
    aie.lock @sl1(%t1, 0) {init = 1 : i32}
    %sl1 = aie.get_lock @sl1
    aie.lock @sd1(%t1, 1) {init = 0 : i32}
    %sd1 = aie.get_lock @sd1
    aie.mem(%t1) {
      %one = arith.constant 1 : i32
      %0 = aie.dma(MM2S, 0) {loop = false, repeat_count = 0 : i32} [{
        aie.use_lock(@sl1, AcquireGreaterEqual, %one)
        aie.dma_bd(%b1 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 1, pkt_type = 0>}
        aie.use_lock(@sd1, Release, %one)
      }]
      aie.end
    }
    aie.buffer @b2(%t2) : memref<16xi32>
    %b2 = aie.get_buffer @b2 : memref<16xi32>
    aie.lock @sl2(%t2, 0) {init = 1 : i32}
    %sl2 = aie.get_lock @sl2
    aie.lock @sd2(%t2, 1) {init = 0 : i32}
    %sd2 = aie.get_lock @sd2
    aie.mem(%t2) {
      %one = arith.constant 1 : i32
      %0 = aie.dma(MM2S, 0) {loop = false, repeat_count = 0 : i32} [{
        aie.use_lock(@sl2, AcquireGreaterEqual, %one)
        aie.dma_bd(%b2 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 2, pkt_type = 0>}
        aie.use_lock(@sd2, Release, %one)
      }]
      aie.end
    }
    aie.buffer @b3(%t3) : memref<16xi32>
    %b3 = aie.get_buffer @b3 : memref<16xi32>
    aie.lock @sl3(%t3, 0) {init = 1 : i32}
    %sl3 = aie.get_lock @sl3
    aie.lock @sd3(%t3, 1) {init = 0 : i32}
    %sd3 = aie.get_lock @sd3
    aie.mem(%t3) {
      %one = arith.constant 1 : i32
      %0 = aie.dma(MM2S, 0) {loop = false, repeat_count = 0 : i32} [{
        aie.use_lock(@sl3, AcquireGreaterEqual, %one)
        aie.dma_bd(%b3 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 3, pkt_type = 0>}
        aie.use_lock(@sd3, Release, %one)
      }]
      aie.end
    }
    aie.buffer @b4(%t4) : memref<16xi32>
    %b4 = aie.get_buffer @b4 : memref<16xi32>
    aie.lock @sl4(%t4, 0) {init = 1 : i32}
    %sl4 = aie.get_lock @sl4
    aie.lock @sd4(%t4, 1) {init = 0 : i32}
    %sd4 = aie.get_lock @sd4
    aie.mem(%t4) {
      %one = arith.constant 1 : i32
      %0 = aie.dma(MM2S, 0) {loop = false, repeat_count = 0 : i32} [{
        aie.use_lock(@sl4, AcquireGreaterEqual, %one)
        aie.dma_bd(%b4 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 4, pkt_type = 0>}
        aie.use_lock(@sd4, Release, %one)
      }]
      aie.end
    }
    aie.buffer @b5(%t5) : memref<16xi32>
    %b5 = aie.get_buffer @b5 : memref<16xi32>
    aie.lock @sl5(%t5, 0) {init = 1 : i32}
    %sl5 = aie.get_lock @sl5
    aie.lock @sd5(%t5, 1) {init = 0 : i32}
    %sd5 = aie.get_lock @sd5
    aie.mem(%t5) {
      %one = arith.constant 1 : i32
      %0 = aie.dma(MM2S, 0) {loop = false, repeat_count = 0 : i32} [{
        aie.use_lock(@sl5, AcquireGreaterEqual, %one)
        aie.dma_bd(%b5 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 5, pkt_type = 0>}
        aie.use_lock(@sd5, Release, %one)
      }]
      aie.end
    }

    aie.buffer @mb0(%m) : memref<16xi32>
    %mb0 = aie.get_buffer @mb0 : memref<16xi32>
    aie.buffer @mb1(%m) : memref<16xi32>
    %mb1 = aie.get_buffer @mb1 : memref<16xi32>
    aie.buffer @mb2(%m) : memref<16xi32>
    %mb2 = aie.get_buffer @mb2 : memref<16xi32>
    aie.buffer @mb3(%m) : memref<16xi32>
    %mb3 = aie.get_buffer @mb3 : memref<16xi32>
    aie.buffer @mb4(%m) : memref<16xi32>
    %mb4 = aie.get_buffer @mb4 : memref<16xi32>
    aie.buffer @mb5(%m) : memref<16xi32>
    %mb5 = aie.get_buffer @mb5 : memref<16xi32>
    aie.lock @p0(%m, 0) {init = 1 : i32}
    %p0 = aie.get_lock @p0
    aie.lock @c0(%m, 1) {init = 0 : i32}
    %c0 = aie.get_lock @c0
    aie.lock @p1(%m, 2) {init = 1 : i32}
    %p1 = aie.get_lock @p1
    aie.lock @c1(%m, 3) {init = 0 : i32}
    %c1 = aie.get_lock @c1
    aie.lock @p2(%m, 4) {init = 1 : i32}
    %p2 = aie.get_lock @p2
    aie.lock @c2(%m, 5) {init = 0 : i32}
    %c2 = aie.get_lock @c2
    aie.lock @p3(%m, 6) {init = 1 : i32}
    %p3 = aie.get_lock @p3
    aie.lock @c3(%m, 7) {init = 0 : i32}
    %c3 = aie.get_lock @c3
    aie.lock @p4(%m, 8) {init = 1 : i32}
    %p4 = aie.get_lock @p4
    aie.lock @c4(%m, 9) {init = 0 : i32}
    %c4 = aie.get_lock @c4
    aie.lock @p5(%m, 10) {init = 1 : i32}
    %p5 = aie.get_lock @p5
    aie.lock @c5(%m, 11) {init = 0 : i32}
    %c5 = aie.get_lock @c5
    aie.memtile_dma(%m) {
      %one = arith.constant 1 : i32
      %r0 = aie.dma(S2MM, 0) [{
        aie.use_lock(@p0, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb0 : memref<16xi32> offset = 0 len = 16)
        aie.use_lock(@c0, Release, %one)
      }]
      %r1 = aie.dma(S2MM, 1) [{
        aie.use_lock(@p1, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb1 : memref<16xi32> offset = 0 len = 16)
        aie.use_lock(@c1, Release, %one)
      }]
      %r2 = aie.dma(S2MM, 2) [{
        aie.use_lock(@p2, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb2 : memref<16xi32> offset = 0 len = 16)
        aie.use_lock(@c2, Release, %one)
      }]
      %r3 = aie.dma(S2MM, 3) [{
        aie.use_lock(@p3, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb3 : memref<16xi32> offset = 0 len = 16)
        aie.use_lock(@c3, Release, %one)
      }]
      %r4 = aie.dma(S2MM, 4) [{
        aie.use_lock(@p4, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb4 : memref<16xi32> offset = 0 len = 16)
        aie.use_lock(@c4, Release, %one)
      }]
      %r5 = aie.dma(S2MM, 5) [{
        aie.use_lock(@p5, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb5 : memref<16xi32> offset = 0 len = 16)
        aie.use_lock(@c5, Release, %one)
      }]
      %join = aie.dma(MM2S, 0) [{
        aie.use_lock(@c0, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb0 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
        aie.use_lock(@p0, Release, %one)
      }, {
        aie.use_lock(@c1, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb1 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
        aie.use_lock(@p1, Release, %one)
      }, {
        aie.use_lock(@c2, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb2 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
        aie.use_lock(@p2, Release, %one)
      }, {
        aie.use_lock(@c3, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb3 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
        aie.use_lock(@p3, Release, %one)
      }, {
        aie.use_lock(@c4, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb4 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
        aie.use_lock(@p4, Release, %one)
      }, {
        aie.use_lock(@c5, AcquireGreaterEqual, %one)
        aie.dma_bd(%mb5 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 6, pkt_type = 0>}
        aie.use_lock(@p5, Release, %one)
      }]
      aie.end
    }
  }
}
