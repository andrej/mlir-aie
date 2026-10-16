//===- shimtileDMA_test.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK-DAG:           %{{.*}}tile_2_0 = aie.tile(2, 0)
// CHECK-DAG:           %{{.*}}tile_3_3 = aie.tile(3, 3)
// CHECK-DAG:           aie.buffer @objfifo_cons_buff_0(%{{.*}}tile_3_3) : memref<16xi32>
// CHECK-DAG:           aie.buffer @objfifo_cons_buff_1(%{{.*}}tile_3_3) : memref<16xi32>
// CHECK-DAG:           aie.lock @objfifo_cons_prod_lock_0(%{{.*}}tile_3_3) {init = 2 : i32}
// CHECK-DAG:           aie.lock @objfifo_cons_cons_lock_0(%{{.*}}tile_3_3) {init = 0 : i32}
// CHECK-DAG:           aie.lock @objfifo_prod_lock_0(%{{.*}}tile_2_0) {init = 1 : i32}
// CHECK-DAG:           aie.lock @objfifo_cons_lock_0(%{{.*}}tile_2_0) {init = 0 : i32}
// CHECK-DAG:           aie.external_buffer @buff0 : memref<16xi32>
// CHECK-DAG:           %[[BUFF0:.*]] = aie.get_external_buffer @buff0 : memref<16xi32>
// CHECK-DAG:           aie.lock @lock0(%{{.*}}tile_2_0, 0)
// CHECK-DAG:           aie.external_buffer @buff1 : memref<16xi32>
// CHECK-DAG:           %[[BUFF1:.*]] = aie.get_external_buffer @buff1 : memref<16xi32>
// CHECK-DAG:           aie.lock @lock1(%{{.*}}tile_2_0, 1)
// CHECK-DAG:           aie.external_buffer @buff2 : memref<16xi32>
// CHECK-DAG:           %[[BUFF2:.*]] = aie.get_external_buffer @buff2 : memref<16xi32>
// CHECK-DAG:           aie.lock @lock2(%{{.*}}tile_2_0, 2)
// CHECK-DAG:           aie.flow(%{{.*}}tile_2_0, DMA : 0, %{{.*}}tile_3_3, DMA : 0)
// CHECK-DAG:           aie.external_buffer @ext_buffer_in : memref<16xi32>
// CHECK-DAG:           %[[EXT_BUFFER_IN_UNUSED:.*]] = aie.get_external_buffer @ext_buffer_in : memref<16xi32>
// CHECK-DAG:           %[[EXT_BUFFER_IN:.*]] = aie.get_external_buffer @ext_buffer_in : memref<16xi32>
// CHECK:           %shim_dma_2_0 = aie.shim_dma(%{{.*}}tile_2_0) {
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 1, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(@lock0, Acquire, %{{.*}})
// CHECK:             aie.dma_bd(%[[BUFF0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@lock0, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@lock1, Acquire, %{{.*}})
// CHECK:             aie.dma_bd(%[[BUFF1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@lock1, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb4
// CHECK:             aie.use_lock(@lock2, Acquire, %{{.*}})
// CHECK:             aie.dma_bd(%[[BUFF2]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@lock2, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:  // pred: ^bb3
// CHECK:             %{{.*}} = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
// CHECK:           ^bb6:  // 2 preds: ^bb5, ^bb6
// CHECK:             aie.use_lock(@objfifo_cons_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[EXT_BUFFER_IN]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@objfifo_prod_lock_0, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb7:  // pred: ^bb5
// CHECK:             aie.end
// CHECK:           }
// CHECK-DAG:           aie.shim_dma_allocation @objfifo_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
// CHECK:           %[[CONS_BUFF0:.*]] = aie.get_buffer @objfifo_cons_buff_0 : memref<16xi32>
// CHECK:           %[[CONS_BUFF1:.*]] = aie.get_buffer @objfifo_cons_buff_1 : memref<16xi32>
// CHECK:           %mem_3_3 = aie.mem(%{{.*}}tile_3_3) {
// CHECK:             %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(@objfifo_cons_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[CONS_BUFF0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@objfifo_cons_cons_lock_0, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@objfifo_cons_prod_lock_0, AcquireGreaterEqual, %{{.*}})
// CHECK:             aie.dma_bd(%[[CONS_BUFF1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@objfifo_cons_cons_lock_0, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @shimtileDMA_channels {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile33 = aie.tile(3, 3)

        aie.external_buffer @buff0 : memref<16xi32>
        %buff0 = aie.get_external_buffer @buff0 : memref<16xi32>
        aie.lock @lock0(%tile20, 0)
        %lock0 = aie.get_lock @lock0
        aie.external_buffer @buff1 : memref<16xi32>
        %buff1 = aie.get_external_buffer @buff1 : memref<16xi32>
        aie.lock @lock1(%tile20, 1)
        %lock1 = aie.get_lock @lock1
        aie.external_buffer @buff2 : memref<16xi32>
        %buff2 = aie.get_external_buffer @buff2 : memref<16xi32>
        aie.lock @lock2(%tile20, 2)
        %lock2 = aie.get_lock @lock2

        aie.objectfifo @objfifo (%tile20, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.external_buffer @ext_buffer_in : memref<16xi32>
        %ext_buffer_in = aie.get_external_buffer @ext_buffer_in : memref<16xi32>
        aie.objectfifo.register_external_buffers @objfifo (%tile20, {%ext_buffer_in}) : (memref<16xi32>)

        %mem12 = aie.shim_dma(%tile20) {
            %dma1 = aie.dma_start(MM2S, 1, ^bb1, ^bb3)
        ^bb1:
            %c1_ul1 = arith.constant 1 : i32
            aie.use_lock(@lock0, Acquire, %c1_ul1)
            aie.dma_bd(%buff0 : memref<16xi32> offset = 0 len = 16)
            %c0_ul2 = arith.constant 0 : i32
            aie.use_lock(@lock0, Release, %c0_ul2)
            aie.next_bd ^bb2
        ^bb2:
            %c1_ul3 = arith.constant 1 : i32
            aie.use_lock(@lock1, Acquire, %c1_ul3)
            aie.dma_bd(%buff1 : memref<16xi32> offset = 0 len = 16)
            %c0_ul4 = arith.constant 0 : i32
            aie.use_lock(@lock1, Release, %c0_ul4)
            aie.next_bd ^bb1
        ^bb3:
            %dma2 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
        ^bb4:
            %c0_ul5 = arith.constant 0 : i32
            aie.use_lock(@lock2, Acquire, %c0_ul5)
            aie.dma_bd(%buff2 : memref<16xi32> offset = 0 len = 16)
            %c1_ul6 = arith.constant 1 : i32
            aie.use_lock(@lock2, Release, %c1_ul6)
            aie.next_bd ^bb4
        ^bb5:
            aie.end
        }
    }
}
