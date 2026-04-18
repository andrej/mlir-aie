//===- npu2_instgen_offsets_rtp_name.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
//
// Test that when an aiex.npu.rtp_write is compiled with --npu-insts-offsets-name,
// the resulting JSON contains the expected "name" field from the RTP buffer.
//
// REQUIRES: peano
//
// RUN: aiecc --no-xbridge --no-xchesscc --aie-generate-npu-insts \
// RUN:   --npu-insts-name=rtp_offsets_test.bin \
// RUN:   --npu-insts-offsets-name=rtp_offsets_test.json %s
// RUN: cat rtp_offsets_test.json | FileCheck %s
//
// JSON keys are alphabetically ordered by llvm::json.
// CHECK: "instructions"
// CHECK:      "name": "rtp2"
// CHECK:      "offset_bytes": 16
// CHECK:      "type": "write32"
// CHECK:      "value_field_offset_bytes": 32

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %buf = aie.buffer(%tile) {address = 1024 : i32, sym_name = "rtp2"} : memref<16xi32>
    aie.runtime_sequence(%arg0: memref<16xf32>) {
      aiex.npu.rtp_write(@rtp2, 0, 42)
    }
  }
}
