//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

module @test {
    
  AIE.device(xcvc1902) {

    %tile30 = AIE.tile(3, 0)
    %lock30_0 = AIE.lock(%tile30) {init = 0 : i32, sym_name = "lock30_0"}
    %tile34 = AIE.tile(3, 4)
    %lock34_0 = AIE.lock(%tile34) {init = 0 : i32, sym_name = "lock34_0"}
    %buf34 = AIE.buffer(%tile34) : memref<256xi32>
    %extbuf  = AIE.external_buffer {sym_name = "extbuf"}: memref<256xi32> 
    AIE.flow(%tile34, DMA : 0, %tile30, DMA : 0)

    %core34 = AIE.core(%tile34) {
        %i0 = arith.constant 0 : index
        %c123456 = arith.constant 123456 : i32
        AIE.useLock(%lock34_0, "Acquire", 0)
        memref.store %c123456, %buf34[%i0] : memref<256xi32>
        AIE.useLock(%lock34_0, "Release", 1)
        AIE.end
    } 

    %mem = AIE.mem(%tile34) {
        %dma = AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
        AIE.useLock(%lock34_0, "Acquire", 1)
        AIE.dmaBd(<%buf34 : memref<256xi32>, 0, 256>, 0)
        AIE.useLock(%lock34_0, "Release", 0)
        AIE.nextBd ^bb1
    ^bb2:
        AIE.end
    }

    %shim = AIE.shimDMA(%tile30) {
        %dma = AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
    ^bb1:
        AIE.useLock(%lock30_0, Acquire, 0)
        AIE.dmaBd(<%extbuf : memref<256xi32>, 0, 256>, 0)
        AIE.useLock(%lock30_0, Release, 1)
        AIE.nextBd ^bb1
    ^bb2:
        AIE.end
    }
    
  }

  AIE.shimDMAAllocation("of_out", S2MM, 0, 3)

}
