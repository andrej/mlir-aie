//===- npu2_instgen_offsets.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
//
// Test that aiecc emits a JSON offset file for LOAD_PDI instructions
// when --npu-insts-offsets-name is specified, and that non-RTP write32
// instructions are excluded from the JSON.
//
// REQUIRES: peano
//
// RUN: aiecc --no-xbridge --no-xchesscc --aie-generate-npu-insts \
// RUN:   --npu-insts-name=offsets_test.bin \
// RUN:   --npu-insts-offsets-name=offsets_test.json %s
// RUN: cat offsets_test.json | FileCheck %s
//
// The instruction stream contains a non-RTP write32 followed by a load_pdi.
// Only the load_pdi should appear in the JSON (non-RTP write32s are filtered).
//
// JSON keys are alphabetically ordered by llvm::json.
// CHECK:       "instructions"
// CHECK:       "address_field_offset_bytes":
// CHECK:       "type": "load_pdi"
// CHECK-NOT:   "type": "write32"

module {
  aie.device(npu2) {
    aie.runtime_sequence(%arg0: memref<16xf32>) {
      aiex.npu.write32 { column = 0 : i32, row = 0 : i32, address = 0x000001D0 : ui32, value = 0x1234 : ui32 }
      aiex.npu.load_pdi { id = 1 : i32, size = 0 : i32, address = 0 : ui64 }
    }
  }
}
