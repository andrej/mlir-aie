//===- local_locks.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering="tilecol=3 tilerow=3" %s | FileCheck --check-prefix=CHECK33 %s

// CHECK33:  func.func @core_3_3() {
// CHECK33:    %c0_i32 = arith.constant 0 : i32
// CHECK33:    %c8_i32 = arith.constant 8 : i32
// CHECK33:    call @llvm.aie.lock.acquire.reg(%c8_i32, %c0_i32) : (i32, i32) -> ()
// CHECK33:    %c1_i32 = arith.constant 1 : i32
// CHECK33:    %c8_i32_0 = arith.constant 8 : i32
// CHECK33:    call @llvm.aie.lock.release.reg(%c8_i32_0, %c1_i32) : (i32, i32) -> ()
// CHECK33:    return
// CHECK33:  }

module @local_locks {
 aie.device(xcvc1902) {
  %3 = aie.tile(3, 3)
  aie.lock @lock(%3, 8)
  %11 = aie.core(%3)  {
    %c0_ul0 = arith.constant 0 : i32
    aie.use_lock(@lock, Acquire, %c0_ul0)
    %c1_ul1 = arith.constant 1 : i32
    aie.use_lock(@lock, Release, %c1_ul1)
    aie.end
  }
 }
}
