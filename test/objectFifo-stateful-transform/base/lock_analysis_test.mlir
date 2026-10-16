//===- lock_analysis_test.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @lockAnalysis {
// CHECK:   aie.device(xcve2302) {
// CHECK-DAG:     %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK-DAG:     %{{.*}}tile_3_3 = aie.tile(3, 3)
// CHECK-DAG:     aie.buffer @of1_cons_buff_0(%{{.*}}tile_3_3) : memref<16xi32>
// CHECK-DAG:     aie.buffer @of1_cons_buff_1(%{{.*}}tile_3_3) : memref<16xi32>
// CHECK-DAG:     aie.lock @of1_cons_prod_lock_0(%{{.*}}tile_3_3) {init = 2 : i32}
// CHECK-DAG:     aie.lock @of1_cons_cons_lock_0(%{{.*}}tile_3_3) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @of1_buff_0(%{{.*}}tile_1_2) : memref<16xi32>
// CHECK-DAG:     aie.buffer @of1_buff_1(%{{.*}}tile_1_2) : memref<16xi32>
// CHECK-DAG:     aie.lock @of1_prod_lock_0(%{{.*}}tile_1_2) {init = 2 : i32}
// CHECK-DAG:     aie.lock @of1_cons_lock_0(%{{.*}}tile_1_2) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @test_buff(%{{.*}}tile_1_2) : memref<16xi32>
// CHECK-DAG:     %[[TEST_BUFF:.*]] = aie.get_buffer @test_buff : memref<16xi32>
// CHECK-DAG:     aie.lock @test_prod_lock(%{{.*}}tile_1_2, 0) {init = 1 : i32}
// CHECK-DAG:     %{{.*}} = aie.get_lock @test_prod_lock
// CHECK-DAG:     aie.lock @test_cons_lock(%{{.*}}tile_1_2, 1) {init = 0 : i32}
// CHECK-DAG:     %{{.*}} = aie.get_lock @test_cons_lock
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_2, DMA : 0, %{{.*}}tile_3_3, DMA : 0)
// CHECK:     %[[OF1_BUFF_0:.*]] = aie.get_buffer @of1_buff_0 : memref<16xi32>
// CHECK:     %[[OF1_BUFF_1:.*]] = aie.get_buffer @of1_buff_1 : memref<16xi32>
// CHECK:     %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK:       %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(@test_prod_lock, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[TEST_BUFF]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@test_cons_lock, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       %{{.*}} = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
// CHECK:     ^bb3:  // 2 preds: ^bb2, ^bb4
// CHECK:       aie.use_lock(@of1_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[OF1_BUFF_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@of1_prod_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(@of1_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[OF1_BUFF_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@of1_prod_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb5:  // pred: ^bb2
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %[[OF1_CONS_BUFF_0:.*]] = aie.get_buffer @of1_cons_buff_0 : memref<16xi32>
// CHECK:     %[[OF1_CONS_BUFF_1:.*]] = aie.get_buffer @of1_cons_buff_1 : memref<16xi32>
// CHECK:     %mem_3_3 = aie.mem(%{{.*}}tile_3_3) {
// CHECK:       %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(@of1_cons_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[OF1_CONS_BUFF_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@of1_cons_cons_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@of1_cons_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[OF1_CONS_BUFF_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@of1_cons_cons_lock_0, Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @lockAnalysis {
   aie.device(xcve2302) {
      %tile12 = aie.tile(1, 2)
      %tile33 = aie.tile(3, 3)

      aie.buffer @test_buff(%tile12)  : memref<16xi32>
      %test_buff = aie.get_buffer @test_buff : memref<16xi32>
      aie.lock @test_prod_lock(%tile12, 0) {init = 1 : i32 }
      %test_prod_lock = aie.get_lock @test_prod_lock
      aie.lock @test_cons_lock(%tile12, 1) {init = 0 : i32 }
      %test_cons_lock = aie.get_lock @test_cons_lock

      aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

      %mem_1_2 = aie.mem(%tile12) {
         %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
         ^bb1:
            %c1_ul1 = arith.constant 1 : i32
            aie.use_lock(@test_prod_lock, AcquireGreaterEqual, %c1_ul1)
            aie.dma_bd(%test_buff : memref<16xi32> offset = 0 len = 16)
            %c1_ul2 = arith.constant 1 : i32
            aie.use_lock(@test_cons_lock, Release, %c1_ul2)
            aie.next_bd ^bb1
         ^bb2:
            aie.end
      }
   }
}
