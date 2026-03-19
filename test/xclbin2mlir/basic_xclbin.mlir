// RUN: aie-translate --xclbin-to-mlir %S/../npu-xrt/add_blockwrite/aie.xclbin | FileCheck %s

// This test decompiles a real xclbin file from the test suite and verifies
// that basic MLIR operations are generated.

// CHECK: module
// CHECK: aie.device(npu1_1col)
// CHECK: memref.global "private" constant @cdo_blockwrite_
// CHECK: aiex.runtime_sequence
// CHECK: aiex.npu.write32
