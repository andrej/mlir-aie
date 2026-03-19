// RUN: aie-translate --xclbin-to-mlir %S/../npu-xrt/add_blockwrite/aie.xclbin | FileCheck %s --check-prefix=CHECK-RAW
// RUN: aie-translate --xclbin-to-mlir --emit-lifted %S/../npu-xrt/add_blockwrite/aie.xclbin | FileCheck %s --check-prefix=CHECK-LIFTED

// This test decompiles a real xclbin file from the test suite and verifies
// that basic MLIR operations are generated.

// Test raw (default) mode - should emit raw register writes
// CHECK-RAW: module
// CHECK-RAW: aie.device(npu1_1col)
// CHECK-RAW: memref.global "private" constant @cdo_blockwrite_
// CHECK-RAW: aiex.runtime_sequence
// CHECK-RAW: aiex.npu.write32

// Test lifted mode - should emit high-level AIE operations
// CHECK-LIFTED: module
// CHECK-LIFTED: aie.device(npu1_1col)
// CHECK-LIFTED-DAG: aie.tile
// CHECK-LIFTED-DAG: aie.buffer
// CHECK-LIFTED-DAG: aie.dma_bd
// CHECK-LIFTED: aiex.runtime_sequence
