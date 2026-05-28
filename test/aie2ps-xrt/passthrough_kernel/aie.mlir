//===- aie.mlir --------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This file is used by the Makefile (run_chess.lit / run_peano.lit).
// Do not run directly with lit — use the Makefile-driven lit files instead.

module {
  aie.device(xcve3858) {
    func.func private @passThroughLine(memref<1024xui8>, memref<1024xui8>, i32) attributes {link_with = "passthrough.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_3 = aie.tile(0, 3)
    aie.objectfifo @in(%shim_noc_tile_0_0, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<1024xui8>>
    aie.objectfifo @out(%tile_0_3, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xui8>>
    %core_0_2 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
        %2 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<1024xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
        %c1024_i32 = arith.constant 1024 : i32
        func.call @passThroughLine(%3, %1, %c1024_i32) : (memref<1024xui8>, memref<1024xui8>, i32) -> ()
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<4096xui8>, %arg1: memref<4096xui8>, %arg2: memref<4096xui8>) {
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xui8>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      %1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<4096xui8>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%1)
    }
  }
}
