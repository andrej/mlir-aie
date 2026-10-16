//===- register_external_buffers_test.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: January 27th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK-DAG:           %[[VAL_0:.*]] = aie.tile(7, 1)
// CHECK-DAG:           %[[VAL_1:.*]] = aie.tile(7, 0)
// CHECK-DAG:           aie.buffer @ext_of_cons_buff_0(%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @ext_of_cons_buff_1(%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @ext_of_cons_buff_2(%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @ext_of_cons_lock_0(%[[VAL_0]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @ext_of_cons_lock_1(%[[VAL_0]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @ext_of_cons_lock_2(%[[VAL_0]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @ext_of_lock_0(%[[VAL_1]]) {init = 0 : i32}
// CHECK-DAG:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK-DAG:           aie.external_buffer @ext_buffer_in : memref<64xi32>
// CHECK:             aie.use_lock(@ext_of_cons_lock_0, Acquire, %{{.*}})
// CHECK:             aie.use_lock(@ext_of_cons_lock_1, Acquire, %{{.*}})
// CHECK:             %[[CORE_0:.*]] = aie.get_buffer @ext_of_cons_buff_0 : memref<16xi32>
// CHECK:             %[[CORE_1:.*]] = aie.get_buffer @ext_of_cons_buff_1 : memref<16xi32>
// CHECK:             func.call @some_work(%[[CORE_0]], %[[CORE_1]]) : (memref<16xi32>, memref<16xi32>) -> ()
// CHECK:             aie.use_lock(@ext_of_cons_lock_0, Release, %{{.*}})
// CHECK:             aie.end
// CHECK-DAG:           aie.shim_dma_allocation @ext_of_shim_alloc(%[[VAL_1]], MM2S, 0)
// CHECK:           %[[VAL_9:.*]] = aie.get_external_buffer @ext_buffer_in : memref<64xi32>
// CHECK:           %[[VAL_16:.*]] = aie.shim_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_17:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:             aie.use_lock(@ext_of_lock_0, Acquire, %{{.*}})
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<64xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@ext_of_lock_0, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_2:.*]] = aie.get_buffer @ext_of_cons_buff_0 : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.get_buffer @ext_of_cons_buff_1 : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.get_buffer @ext_of_cons_buff_2 : memref<16xi32>
// CHECK:           %[[VAL_18:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_19:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:             aie.use_lock(@ext_of_cons_lock_0, Acquire, %{{.*}})
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@ext_of_cons_lock_0, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@ext_of_cons_lock_1, Acquire, %{{.*}})
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@ext_of_cons_lock_1, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:             aie.use_lock(@ext_of_cons_lock_2, Acquire, %{{.*}})
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@ext_of_cons_lock_2, Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @register_external_buffers {
 aie.device(xcvc1902) {
    %tile71 = aie.tile(7, 1)
    %tile70 = aie.tile(7, 0)

    aie.objectfifo @ext_of (%tile70, {%tile71}, 3 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.external_buffer @ext_buffer_in : memref<64xi32>
    %ext_buffer_in = aie.get_external_buffer @ext_buffer_in : memref<64xi32>
    aie.objectfifo.register_external_buffers @ext_of (%tile70, {%ext_buffer_in}) : (memref<64xi32>)

    func.func @some_work(%a : memref<16xi32>, %b : memref<16xi32>) -> () {
        return
    }

    %core71 = aie.core(%tile71) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        %elem0, %elem1 = aie.objectfifo.acquire @ext_of (Consume, 2) : memref<16xi32>, memref<16xi32>
        func.call @some_work(%elem0, %elem1) : (memref<16xi32>, memref<16xi32>) -> ()
        aie.objectfifo.release @ext_of (Consume, 1)

        aie.end
    }
 }
}
