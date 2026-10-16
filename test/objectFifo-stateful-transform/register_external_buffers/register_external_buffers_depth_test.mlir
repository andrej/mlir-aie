//===- register_external_buffers_depth_test.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK-DAG:               %{{.*}}tile_7_1 = aie.tile(7, 1)
// CHECK-DAG:               %{{.*}}tile_7_0 = aie.tile(7, 0)
// CHECK-DAG:               aie.buffer @ext_of_cons_buff_0(%{{.*}}tile_7_1) : memref<16xi32>
// CHECK-DAG:               aie.lock @ext_of_cons_lock_0(%{{.*}}tile_7_1) {init = 0 : i32}
// CHECK-DAG:               aie.lock @ext_of_lock_0(%{{.*}}tile_7_0) {init = 0 : i32}
// CHECK-DAG:               aie.lock @ext_of_lock_1(%{{.*}}tile_7_0) {init = 0 : i32}
// CHECK-DAG:               aie.flow(%{{.*}}tile_7_0, DMA : 0, %{{.*}}tile_7_1, DMA : 0)
// CHECK-DAG:               aie.external_buffer @ext_buffer_in0 : memref<64xi32>
// CHECK-DAG:               %{{.*}} = aie.get_external_buffer @ext_buffer_in0 : memref<64xi32>
// CHECK-DAG:               aie.external_buffer @ext_buffer_in1 : memref<64xi32>
// CHECK-DAG:               %{{.*}} = aie.get_external_buffer @ext_buffer_in1 : memref<64xi32>
// CHECK-DAG:               aie.shim_dma_allocation @ext_of_shim_alloc(%shim_noc_tile_7_0, MM2S, 0)
// CHECK:               %[[EXT_BUFFER_IN0:.*]] = aie.get_external_buffer @ext_buffer_in0 : memref<64xi32>
// CHECK:               %[[EXT_BUFFER_IN1:.*]] = aie.get_external_buffer @ext_buffer_in1 : memref<64xi32>
// CHECK:               %shim_dma_7_0 = aie.shim_dma(%{{.*}}tile_7_0) {
// CHECK:                 %{{.*}} = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:               ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:                 aie.use_lock(@ext_of_lock_0, Acquire, %{{.*}})
// CHECK:                 aie.dma_bd(%[[EXT_BUFFER_IN0]] : memref<64xi32> offset = {{.*}} len = {{.*}})
// CHECK:                 aie.use_lock(@ext_of_lock_0, Release, %{{.*}})
// CHECK:                 aie.next_bd ^bb2
// CHECK:               ^bb2:  // pred: ^bb1
// CHECK:                 aie.use_lock(@ext_of_lock_1, Acquire, %{{.*}})
// CHECK:                 aie.dma_bd(%[[EXT_BUFFER_IN1]] : memref<64xi32> offset = {{.*}} len = {{.*}})
// CHECK:                 aie.use_lock(@ext_of_lock_1, Release, %{{.*}})
// CHECK:                 aie.next_bd ^bb1
// CHECK:               ^bb3:  // pred: ^bb0
// CHECK:                 aie.end
// CHECK:               }
// CHECK:               %[[CONS_BUFF:.*]] = aie.get_buffer @ext_of_cons_buff_0 : memref<16xi32>
// CHECK:               %mem_7_1 = aie.mem(%{{.*}}tile_7_1) {
// CHECK:                 %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:               ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:                 aie.use_lock(@ext_of_cons_lock_0, Acquire, %{{.*}})
// CHECK:                 aie.dma_bd(%[[CONS_BUFF]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:                 aie.use_lock(@ext_of_cons_lock_0, Release, %{{.*}})
// CHECK:                 aie.next_bd ^bb1
// CHECK:               ^bb2:  // pred: ^bb0
// CHECK:                 aie.end
// CHECK:               }
// CHECK:             }

module @register_external_buffers_depth {
 aie.device(xcvc1902) {
    %tile71 = aie.tile(7, 1)
    %tile70 = aie.tile(7, 0)

    aie.objectfifo @ext_of (%tile70, {%tile71}, 1 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.external_buffer @ext_buffer_in0 : memref<64xi32>
    %ext_buffer_in0 = aie.get_external_buffer @ext_buffer_in0 : memref<64xi32>
    aie.external_buffer @ext_buffer_in1 : memref<64xi32>
    %ext_buffer_in1 = aie.get_external_buffer @ext_buffer_in1 : memref<64xi32>
    aie.objectfifo.register_external_buffers @ext_of (%tile70, {%ext_buffer_in0, %ext_buffer_in1}) : (memref<64xi32>, memref<64xi32>)
 }
}
