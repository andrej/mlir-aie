//===- aie.mlir --------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(xcve3858) {
    %shim = aie.tile(0, 0)
    %mem  = aie.tile(0, 1)

    aie.objectfifo @in(%shim, {%mem}, 2 : i32)
        : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%mem, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@in] -> [@out]([] [])

    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      %1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg2 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%1)
    }
  }
}
