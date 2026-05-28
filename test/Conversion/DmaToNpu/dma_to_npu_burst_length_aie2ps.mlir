//===- dma_to_npu_burst_length_aie2ps.mlir ----------------------*- MLIR -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verify 512B burst length works for AIE2PS (xcve3558).
// AIE2PS supports burst lengths 64/128/256/512B, same as AIE2P/NPU2.

// RUN: aie-opt --split-input-file --aie-dma-to-npu %s | FileCheck %s

// 512B burst via writebd
module {
  aie.device(xcve3558) {
    aie.runtime_sequence(%in : memref<4x2x8xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
        // Burst length 512 encoding = 3 (0xC0000000 in BD word 4).
        // CHECK: memref.global "private" constant {{.*}} = dense<[{{[0-9]*}}, {{[0-9]*}}, {{[0-9]*}}, {{[0-9]*}}, -1073741824, {{[0-9]*}}, {{[0-9]*}}, {{[0-9]*}}]>
        aiex.npu.writebd { burst_length = 512 : i32, bd_id = 7 : i32, buffer_length = 4 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @of_toMem (%tile_0_0, S2MM, 0)
  }
}

// -----

// 512B burst via dma_memcpy_nd
module {
  aie.device(xcve3558) {
    aie.runtime_sequence(%in : memref<4x2x8xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
        // 64B burst
        // CHECK: memref.global "private" constant {{.*}} = dense<[{{[0-9]*}}, {{[0-9]*}}, {{[0-9]*}}, {{[0-9]*}}, 2097152, {{[0-9]*}}, {{[0-9]*}}, {{[0-9]*}}]>
        aiex.npu.dma_memcpy_nd (%in[0,2,0,0][1,2,2,8][0,16,1,1]) { metadata = @of_fromMem, id = 0 : i64, burst_length = 64 : i64} : memref<4x2x8xi32>
        // 512B burst (0xC0000000 with no stride interference)
        // CHECK: memref.global "private" constant {{.*}} = dense<[{{[0-9]*}}, {{[0-9]*}}, {{[0-9]*}}, {{[0-9]*}}, -1073741824, {{[0-9]*}}, {{[0-9]*}}, {{[0-9]*}}]>
        aiex.npu.dma_memcpy_nd (%out[0,0,0,0][1,1,1,32][0,0,0,1]) { metadata = @of_toMem, id = 1 : i64, burst_length = 512 : i64} : memref<64xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @of_toMem (%tile_0_0, S2MM, 0)
  }
}
