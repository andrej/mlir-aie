//===- basic.mlir ----------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --split-input-file %s | FileCheck %s

// CHECK-LABEL: aie.device(npu1) {
// CHECK:  %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:  %[[VAL_1:.*]] = aie.tile(0, 1)
// CHECK:  %[[VAL_2:.*]] = aie.tile(0, 2)
// CHECK:  aie.buffer @double_buffer(%[[VAL_2]]) : memref<32xi32>
// CHECK:  %[[VAL_3:.*]] = aie.get_buffer @double_buffer : memref<32xi32>
// CHECK:  aie.buffer @buffer_0_1(%[[VAL_1]]) : memref<32xi32>
// CHECK:  %[[VAL_4:.*]] = aie.get_buffer @buffer_0_1 : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32> offset = {{.*}}) {bd_id = 0 : i32, next_bd_id = 1 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 1 : i32, next_bd_id = 2 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 2 : i32, next_bd_id = 0 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32> offset = {{.*}}) {bd_id = 3 : i32, next_bd_id = 4 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 4 : i32, next_bd_id = 5 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 5 : i32, next_bd_id = 3 : i32}
// CHECK:  aie.dma_bd(%[[VAL_4]] : memref<32xi32>) {bd_id = 0 : i32}
// CHECK:  aie.dma_bd(%[[VAL_4]] : memref<32xi32>) {bd_id = 1 : i32}
// CHECK:  aie.dma_bd(%[[VAL_4]] : memref<32xi32>) {bd_id = 24 : i32}
// CHECK:  aie.dma_bd(%[[VAL_4]] : memref<32xi32>) {bd_id = 25 : i32}

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @double_buffer(%tile_0_2)  : memref<32xi32>
    %double_buffer = aie.get_buffer @double_buffer : memref<32xi32>
    aie.buffer @buffer_0_1(%tile_0_1) : memref<32xi32>
    %buffer_0_1 = aie.get_buffer @buffer_0_1 : memref<32xi32>
    aie.lock @lock_X(%tile_0_2) {init = 1 : i32 }
    %lock_X = aie.get_lock @lock_X
    aie.lock @lock_Y(%tile_0_2) {init = 0 : i32 }
    %lock_Y = aie.get_lock @lock_Y
    %mem_0_2 = aie.mem(%tile_0_2) {
      %c0_i32 = arith.constant 0 : i32
      %player_a = aie.dma(S2MM, 0)  [{
        %c0_ul1 = arith.constant 0 : i32
        aie.use_lock(@lock_Y, Acquire, %c0_ul1)
        aie.dma_bd(%double_buffer : memref<32xi32> offset = 0) {bd_id = 0 : i32}
        %c0_ul2 = arith.constant 0 : i32
        aie.use_lock(@lock_Y, Release, %c0_ul2)
      }, {
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(@lock_X, Acquire, %c1_ul3)
        aie.dma_bd(%double_buffer : memref<32xi32>) {bd_id = 1 : i32}
        %cm1_ul4 = arith.constant -1 : i32
        aie.use_lock(@lock_X, Release, %cm1_ul4)
      }, {
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(@lock_Y, Acquire, %c1_ul1) {acq_en = false}
        aie.dma_bd(%double_buffer : memref<32xi32>)
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(@lock_Y, Release, %c1_ul5)
      }]
      %player_b = aie.dma(S2MM, 1)  [{
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(@lock_Y, Acquire, %c1_ul6)
        aie.dma_bd(%double_buffer : memref<32xi32> offset = 0)
        %c0_ul7 = arith.constant 0 : i32
        aie.use_lock(@lock_Y, Release, %c0_ul7)
      }, {
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(@lock_X, Acquire, %c1_ul8)
        aie.dma_bd(%double_buffer : memref<32xi32>) {bd_id = 4 : i32}
        %cm1_ul9 = arith.constant -1 : i32
        aie.use_lock(@lock_X, Release, %cm1_ul9)
      }, {
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(@lock_Y, Acquire, %c1_ul2) {acq_en = false}
        aie.dma_bd(%double_buffer : memref<32xi32>)
        %cm1_ul10 = arith.constant -1 : i32
        aie.use_lock(@lock_Y, Release, %cm1_ul10)
      }]
      aie.end
    }
    aie.lock @lock_0_1(%tile_0_1) {init = 1 : i32}
    %lock_0_1 = aie.get_lock @lock_0_1
    aie.lock @lock_0_1_0(%tile_0_1) {init = 0 : i32}
    %lock_0_1_0 = aie.get_lock @lock_0_1_0
    aie.lock @lock_0_1_1(%tile_0_1) {init = 1 : i32}
    %lock_0_1_1 = aie.get_lock @lock_0_1_1
    aie.lock @lock_0_1_2(%tile_0_1) {init = 0 : i32}
    %lock_0_1_2 = aie.get_lock @lock_0_1_2
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma(S2MM, 0) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul11 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1, AcquireGreaterEqual, %c1_ul11)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul12 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_0, Release, %c1_ul12)
      }]
      %1 = aie.dma(MM2S, 0) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul13 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_0, AcquireGreaterEqual, %c1_ul13)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul14 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1, Release, %c1_ul14)
      }]
      %2 = aie.dma(S2MM, 1) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul15 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_1, AcquireGreaterEqual, %c1_ul15)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>) {bd_id = 24 : i32}
        %c1_ul16 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_2, Release, %c1_ul16)
      }]
      %3 = aie.dma(MM2S, 1) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul17 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_2, AcquireGreaterEqual, %c1_ul17)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul18 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_1, Release, %c1_ul18)
      }]
      aie.end
    }
  }
}



// -----

// CHECK-LABEL: aie.device(xcve2302) {
// CHECK:  %[[VAL_0:.*]] = aie.tile(2, 1)
// CHECK:  aie.buffer @in(%[[VAL_0]]) {address = 8192 : i32} : memref<16xi32>
// CHECK:  aie.buffer @out(%[[VAL_0]]) {address = 1824 : i32} : memref<16xi32>
// CHECK:  aie.dma_bd(%[[VAL_1]] : memref<16xi32> offset = {{.*}} len = {{.*}}) {bd_id = 0 : i32, next_bd_id = 0 : i32}
// CHECK:  aie.dma_bd(%[[VAL_1]] : memref<16xi32> offset = {{.*}} len = {{.*}}) {bd_id = 24 : i32, next_bd_id = 24 : i32}
// CHECK:  aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = {{.*}} len = {{.*}}) {bd_id = 25 : i32, next_bd_id = 25 : i32}
// CHECK:  aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = {{.*}} len = {{.*}}) {bd_id = 1 : i32, next_bd_id = 1 : i32}

module @aie_module  {
  aie.device(xcve2302) {
    %t01 = aie.tile(2, 1)
    aie.buffer @in(%t01) { address = 8192 : i32 } : memref<16xi32>
    %buf01_0 = aie.get_buffer @in : memref<16xi32>
    aie.buffer @out(%t01) { address = 1824 : i32 } : memref<16xi32>
    %buf01_1 = aie.get_buffer @out : memref<16xi32>

    aie.lock @l01_0(%t01, 0) { init = 1 : i32 }
    %l01_0 = aie.get_lock @l01_0
    aie.lock @l01_1(%t01, 1)
    %l01_1 = aie.get_lock @l01_1
    aie.lock @l01_2(%t01, 2) { init = 1 : i32 }
    %l01_2 = aie.get_lock @l01_2
    aie.lock @l01_3(%t01, 3)
    %l01_3 = aie.get_lock @l01_3

    %m01 = aie.memtile_dma(%t01) {
      %c0_i32 = arith.constant 0 : i32
        %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^dma0)
      ^dma0:
        %memSrcDma = aie.dma_start(MM2S, 1, ^bd1, ^dma1)
      ^dma1:
        %memDstDma = aie.dma_start(S2MM, 1, ^bd2, ^dma2)
      ^dma2:
        %dstDma = aie.dma_start(MM2S, 0, ^bd3, ^end)
      ^bd0:
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(@l01_0, "AcquireGreaterEqual", %c1_ul1)
        aie.dma_bd(%buf01_0 : memref<16xi32> offset = 0 len = 128 sizes = [2, 3, 2, 1] strides = [1, 2, 4, 1])
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(@l01_1, "Release", %c1_ul2)
        aie.next_bd ^bd0
      ^bd1:
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(@l01_1, "AcquireGreaterEqual", %c1_ul3)
        aie.dma_bd(%buf01_0 : memref<16xi32> offset = 0 len = 16) {bd_id = 24 : i32}
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(@l01_0, "Release", %c1_ul4)
        aie.next_bd ^bd1
      ^bd2:
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(@l01_2, "AcquireGreaterEqual", %c1_ul5)
        aie.dma_bd(%buf01_1 : memref<16xi32> offset = 0 len = 16)
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(@l01_3, "Release", %c1_ul6)
        aie.next_bd ^bd2
      ^bd3:
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(@l01_3, "AcquireGreaterEqual", %c1_ul7)
        aie.dma_bd(%buf01_1 : memref<16xi32> offset = 0 len = 16) {bd_id = 1 : i32}
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(@l01_2, "Release", %c1_ul8)
        aie.next_bd ^bd3
      ^end:
        aie.end
    }
  }
}



// -----

// CHECK-LABEL: aie.device(npu1) {
// CHECK:  %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:  %[[VAL_1:.*]] = aie.tile(0, 1)
// CHECK:  %[[VAL_2:.*]] = aie.tile(0, 2)
// CHECK:  aie.buffer @double_buffer(%[[VAL_2]]) : memref<32xi32>
// CHECK:  %[[VAL_3:.*]] = aie.get_buffer @double_buffer : memref<32xi32>
// CHECK:  aie.buffer @buffer_0_1(%[[VAL_1]]) : memref<32xi32>
// CHECK:  %[[VAL_4:.*]] = aie.get_buffer @buffer_0_1 : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32> offset = {{.*}}) {bd_id = 5 : i32, next_bd_id = 4 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 4 : i32, next_bd_id = 3 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 3 : i32, next_bd_id = 5 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32> offset = {{.*}}) {bd_id = 2 : i32, next_bd_id = 1 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 1 : i32, next_bd_id = 0 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 0 : i32, next_bd_id = 2 : i32}
// CHECK:  aie.dma_bd(%[[VAL_4]] : memref<32xi32>) {bd_id = 0 : i32}
// CHECK:  aie.dma_bd(%[[VAL_4]] : memref<32xi32>) {bd_id = 1 : i32}
// CHECK:  aie.dma_bd(%[[VAL_4]] : memref<32xi32>) {bd_id = 24 : i32}
// CHECK:  aie.dma_bd(%[[VAL_4]] : memref<32xi32>) {bd_id = 25 : i32}

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @double_buffer(%tile_0_2)  : memref<32xi32>
    %double_buffer = aie.get_buffer @double_buffer : memref<32xi32>
    aie.buffer @buffer_0_1(%tile_0_1) : memref<32xi32>
    %buffer_0_1 = aie.get_buffer @buffer_0_1 : memref<32xi32>
    aie.lock @lock_X(%tile_0_2) {init = 1 : i32 }
    %lock_X = aie.get_lock @lock_X
    aie.lock @lock_Y(%tile_0_2) {init = 0 : i32 }
    %lock_Y = aie.get_lock @lock_Y
    %mem_0_2 = aie.mem(%tile_0_2) {
      %c0_i32 = arith.constant 0 : i32
      %player_a = aie.dma(S2MM, 0)  [{
        %c0_ul19 = arith.constant 0 : i32
        aie.use_lock(@lock_Y, Acquire, %c0_ul19)
        aie.dma_bd(%double_buffer : memref<32xi32> offset = 0) {bd_id = 5 : i32}
        %c0_ul20 = arith.constant 0 : i32
        aie.use_lock(@lock_Y, Release, %c0_ul20)
      }, {
        %c1_ul21 = arith.constant 1 : i32
        aie.use_lock(@lock_X, Acquire, %c1_ul21)
        aie.dma_bd(%double_buffer : memref<32xi32>) {bd_id = 4 : i32}
        %cm1_ul22 = arith.constant -1 : i32
        aie.use_lock(@lock_X, Release, %cm1_ul22)
      }, {
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(@lock_Y, Acquire, %c1_ul3) {acq_en = false}
        aie.dma_bd(%double_buffer : memref<32xi32>) {bd_id = 3 : i32}
        %c1_ul23 = arith.constant 1 : i32
        aie.use_lock(@lock_Y, Release, %c1_ul23)
      }]
      %player_b = aie.dma(S2MM, 1)  [{
        %c1_ul24 = arith.constant 1 : i32
        aie.use_lock(@lock_Y, Acquire, %c1_ul24)
        aie.dma_bd(%double_buffer : memref<32xi32> offset = 0) {bd_id = 2 : i32}
        %c0_ul25 = arith.constant 0 : i32
        aie.use_lock(@lock_Y, Release, %c0_ul25)
      }, {
        %c1_ul26 = arith.constant 1 : i32
        aie.use_lock(@lock_X, Acquire, %c1_ul26)
        aie.dma_bd(%double_buffer : memref<32xi32>) {bd_id = 1 : i32}
        %cm1_ul27 = arith.constant -1 : i32
        aie.use_lock(@lock_X, Release, %cm1_ul27)
      }, {
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(@lock_Y, Acquire, %c1_ul4) {acq_en = false}
        aie.dma_bd(%double_buffer : memref<32xi32>) {bd_id = 0 : i32}
        %cm1_ul28 = arith.constant -1 : i32
        aie.use_lock(@lock_Y, Release, %cm1_ul28)
      }]
      aie.end
    }
    aie.lock @lock_0_1(%tile_0_1) {init = 1 : i32}
    %lock_0_1 = aie.get_lock @lock_0_1
    aie.lock @lock_0_1_0(%tile_0_1) {init = 0 : i32}
    %lock_0_1_0 = aie.get_lock @lock_0_1_0
    aie.lock @lock_0_1_1(%tile_0_1) {init = 1 : i32}
    %lock_0_1_1 = aie.get_lock @lock_0_1_1
    aie.lock @lock_0_1_2(%tile_0_1) {init = 0 : i32}
    %lock_0_1_2 = aie.get_lock @lock_0_1_2
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma(S2MM, 0) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul29 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1, AcquireGreaterEqual, %c1_ul29)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul30 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_0, Release, %c1_ul30)
      }]
      %1 = aie.dma(MM2S, 0) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul31 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_0, AcquireGreaterEqual, %c1_ul31)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul32 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1, Release, %c1_ul32)
      }]
      %2 = aie.dma(S2MM, 1) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul33 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_1, AcquireGreaterEqual, %c1_ul33)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>) {bd_id = 24 : i32}
        %c1_ul34 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_2, Release, %c1_ul34)
      }]
      %3 = aie.dma(MM2S, 1) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul35 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_2, AcquireGreaterEqual, %c1_ul35)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul36 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_1, Release, %c1_ul36)
      }]
      aie.end
    }
  }
}
