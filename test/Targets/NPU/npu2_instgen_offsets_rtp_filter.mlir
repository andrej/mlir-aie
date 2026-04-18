//===- npu2_instgen_offsets_rtp_filter.mlir ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
//
// Test that when both RTP and non-RTP write32s are present, only the RTP
// write32 appears in the JSON offset file.
//
// REQUIRES: peano
//
// RUN: aiecc --no-xbridge --no-xchesscc --aie-generate-npu-insts \
// RUN:   --npu-insts-name=rtp_filter_test.bin \
// RUN:   --npu-insts-offsets-name=rtp_filter_test.json %s
// RUN: cat rtp_filter_test.json | FileCheck %s
//
// The sequence has: non-RTP write32, RTP write32, load_pdi.
// Only the RTP write32 and load_pdi should appear in the JSON.
//
// The JSON should contain exactly one write32 (the RTP one with name) and one
// load_pdi. The non-RTP write32 at offset 16 should be absent.
//
// JSON keys are alphabetically ordered by llvm::json.
// CHECK:       "instructions"
// CHECK:       "name": "rtp_buf"
// CHECK:       "type": "write32"
// CHECK:       "type": "load_pdi"
// CHECK-NOT:   "type": "write32"

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %buf = aie.buffer(%tile) {address = 1024 : i32, sym_name = "rtp_buf"} : memref<16xi32>
    aie.runtime_sequence(%arg0: memref<16xf32>) {
      // Non-RTP write32 — should NOT appear in JSON
      aiex.npu.write32 { column = 0 : i32, row = 0 : i32, address = 0x000001D0 : ui32, value = 0x1234 : ui32 }
      // RTP write32 — SHOULD appear in JSON
      aiex.npu.rtp_write(@rtp_buf, 0, 99)
      // load_pdi — SHOULD appear in JSON
      aiex.npu.load_pdi { id = 2 : i32, size = 0 : i32, address = 0 : ui64 }
    }
  }
}
