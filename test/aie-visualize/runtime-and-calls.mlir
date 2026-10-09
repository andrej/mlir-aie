//===- runtime-and-calls.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --show-calls --show-buffers %s | FileCheck %s --check-prefix=ALL
// RUN: aie-visualize --emit-dot --skip-runtime %s | FileCheck %s --check-prefix=SKIP
// RUN: aie-visualize --emit-dot %s | FileCheck %s --check-prefix=NO-CALLS

// ALL: p_0_0_1_0_m {{.*}} label="MM2S0\n%allocated_input\n%direct_input\n%endpoint_input"
// ALL: p_0_0_1_1_s {{.*}} label="S2MM1\n%output"
// ALL: buffer_0 {{.*}} label="core_buffer"
// ALL: call_0 {{.*}} label="kernel"
// ALL: buffer_0 -> call_0

// SKIP: label="MM2S0"
// SKIP: label="S2MM1"
// SKIP-NOT: %direct_input
// SKIP-NOT: %allocated_input
// SKIP-NOT: %endpoint_input
// SKIP-NOT: %output

// NO-CALLS-NOT: call_0
// NO-CALLS-NOT: label="kernel"

module {
  aie.device(npu1_1col) {
    func.func private @kernel(memref<16xi32>)

    %shim = aie.tile(0, 0)
    %core_tile = aie.tile(0, 2)
    %core_buffer = aie.buffer(%core_tile) {sym_name = "core_buffer"} : memref<16xi32>

    aie.core(%core_tile) {
      func.call @kernel(%core_buffer) : (memref<16xi32>) -> ()
      aie.end
    }

    aie.flow(%shim, DMA : 0, %core_tile, Core : 0)
      via (%shim : DMA : 0 -> North : 0,
           %core_tile : South : 0 -> Core : 0)
    aie.flow(%core_tile, Core : 0, %shim, DMA : 1)
      via (%core_tile : Core : 0 -> South : 1,
           %shim : North : 1 -> DMA : 1)

    aie.shim_dma_allocation @input_alloc(%shim, MM2S, 0)
    aie.shim_dma_allocation @output_alloc(%shim, S2MM, 1)
    aie.route_endpoint @input_endpoint(%shim) DMA {channelIndex = 0 : i32}
    aie.runtime_sequence @run(
        %direct_input: memref<16xi32>,
        %allocated_input: memref<16xi32>,
      %endpoint_input: memref<16xi32>,
        %output: memref<16xi32>) {
      %direct = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%direct_input : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      %allocated = aiex.dma_configure_task_for @input_alloc {
        aie.dma_bd(%allocated_input : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      %endpoint = aiex.dma_configure_task_for @input_endpoint {
        aie.dma_bd(%endpoint_input : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      aiex.npu.dma_memcpy_nd(%output[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {
        metadata = @output_alloc, id = 0 : i64
      } : memref<16xi32>
    }
  }
}
