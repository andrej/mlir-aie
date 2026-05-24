//===- aie.mlir (AIE2PS itsalive) ----------------------------*- MLIR -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//

// REQUIRES: aiesimulator

// RUN: %python aiecc.py --no-compile --aiesim %s %S/test.cpp
// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: AIE2PS
// CHECK: Hello, world.

// Basic smoke test that the xcve3558 target builds and simulates

module @test00_itsalive_aie2ps {
  aie.device(xcve3558) {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) : memref<16xi32>
    %c = aie.core(%t) {
      %zero = arith.constant 0 : i32
      %idx  = arith.constant 0 : index
      memref.store %zero, %buf[%idx] : memref<16xi32>
      aie.end
    }
  }
}
