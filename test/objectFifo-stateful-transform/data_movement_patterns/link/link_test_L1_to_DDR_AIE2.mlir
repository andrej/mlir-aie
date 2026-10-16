//===- link_test_L1_to_DDR.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: June 30th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK-DAG:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK-DAG:           %[[VAL_1:.*]] = aie.tile(2, 1)
// CHECK-DAG:           %[[VAL_2:.*]] = aie.tile(2, 2)
// CHECK-DAG:           aie.lock @from_memTile_cons_prod_lock_0(%[[VAL_0]]) {init = 1 : i32}
// CHECK-DAG:           aie.lock @from_memTile_cons_cons_lock_0(%[[VAL_0]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @from_memTile_buff_0(%[[VAL_1]]) : memref<48xi32>
// CHECK-DAG:           aie.buffer @from_memTile_buff_1(%[[VAL_1]]) : memref<48xi32>
// CHECK-DAG:           aie.lock @from_memTile_prod_lock_0(%[[VAL_1]]) {init = 2 : i32}
// CHECK-DAG:           aie.lock @from_memTile_cons_lock_0(%[[VAL_1]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @to_memTile_buff_0(%[[VAL_2]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @to_memTile_buff_1(%[[VAL_2]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @to_memTile_prod_lock_0(%[[VAL_2]]) {init = 2 : i32}
// CHECK-DAG:           aie.lock @to_memTile_cons_lock_0(%[[VAL_2]]) {init = 0 : i32}
// CHECK-DAG:           aie.flow(%[[VAL_2]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK-DAG:           aie.external_buffer @ext_buff_in : memref<48xi32>
// CHECK-DAG:           %{{.*}} = aie.get_external_buffer @ext_buff_in : memref<48xi32>
// CHECK-DAG:           aie.shim_dma_allocation @from_memTile_shim_alloc(%[[VAL_0]], S2MM, 0)
// CHECK:           %[[TO_BUFF_0:.*]] = aie.get_buffer @to_memTile_buff_0 : memref<16xi32>
// CHECK:           %[[TO_BUFF_1:.*]] = aie.get_buffer @to_memTile_buff_1 : memref<16xi32>
// CHECK:           %[[VAL_14:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_15:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(@to_memTile_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[TO_BUFF_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@to_memTile_prod_lock_0, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@to_memTile_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[TO_BUFF_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@to_memTile_prod_lock_0, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[FROM_S2MM_0:.*]] = aie.get_buffer @from_memTile_buff_0 : memref<48xi32>
// CHECK:           %[[FROM_S2MM_1:.*]] = aie.get_buffer @from_memTile_buff_1 : memref<48xi32>
// CHECK:           %[[FROM_MM2S_0:.*]] = aie.get_buffer @from_memTile_buff_0 : memref<48xi32>
// CHECK:           %[[FROM_MM2S_1:.*]] = aie.get_buffer @from_memTile_buff_1 : memref<48xi32>
// CHECK:           %[[VAL_16:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_17:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(@from_memTile_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[FROM_S2MM_0]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@from_memTile_cons_lock_0, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@from_memTile_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[FROM_S2MM_1]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@from_memTile_cons_lock_0, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_18:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:             aie.use_lock(@from_memTile_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[FROM_MM2S_0]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@from_memTile_prod_lock_0, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:  // pred: ^bb4
// CHECK:             aie.use_lock(@from_memTile_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[FROM_MM2S_1]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@from_memTile_prod_lock_0, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:  // pred: ^bb3
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[EXT_BUFF_IN:.*]] = aie.get_external_buffer @ext_buff_in : memref<48xi32>
// CHECK:           %[[VAL_19:.*]] = aie.shim_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_20:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:             aie.use_lock(@from_memTile_cons_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[EXT_BUFF_IN]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@from_memTile_cons_cons_lock_0, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_L1_DDR {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)

        aie.objectfifo @to_memTile (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @from_memTile (%tile21, {%tile20}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

        aie.objectfifo.link [@to_memTile] -> [@from_memTile] ([] [])

        aie.external_buffer @ext_buff_in : memref<48xi32>
        %ext_buff_in = aie.get_external_buffer @ext_buff_in : memref<48xi32>
        aie.objectfifo.register_external_buffers @from_memTile (%tile20, {%ext_buff_in}) : (memref<48xi32>)
    }
}
