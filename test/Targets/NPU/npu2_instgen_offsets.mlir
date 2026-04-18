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
// Test that aiecc emits a JSON offset file for LOAD_PDI and write32
// instructions when --npu-insts-offsets-name is specified.
//
// REQUIRES: peano
//
// RUN: aiecc --no-xbridge --no-xchesscc --aie-generate-npu-insts \
// RUN:   --npu-insts-name=offsets_test.bin \
// RUN:   --npu-insts-offsets-name=offsets_test.json %s
// RUN: cat offsets_test.json | FileCheck %s
//
// The instruction stream layout (after 4-word TXN header = 16 bytes):
//   write32  at offset 16  (6 words = 24 bytes), value field at word[4] = offset 32
//   load_pdi at offset 40  (4 words = 16 bytes), size at word[1] = offset 44, addr at word[2] = offset 48
//
// JSON keys are alphabetically ordered by llvm::json.
// CHECK: "instructions"
// CHECK:      "offset_bytes": 16
// CHECK:      "type": "write32"
// CHECK:      "value_field_offset_bytes": 32
// CHECK:      "address_field_offset_bytes": 48
// CHECK:      "offset_bytes": 40
// CHECK:      "pdi_id": 1
// CHECK:      "size_field_offset_bytes": 44
// CHECK:      "type": "load_pdi"

module {
  aie.device(npu2) {
    aie.runtime_sequence(%arg0: memref<16xf32>) {
      aiex.npu.write32 { column = 0 : i32, row = 0 : i32, address = 0x000001D0 : ui32, value = 0x1234 : ui32 }
      aiex.npu.load_pdi { id = 1 : i32, size = 0 : i32, address = 0 : ui64 }
    }
  }
}
