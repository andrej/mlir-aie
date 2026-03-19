// RUN: aie-translate --xclbin-to-mlir --emit-lifted %S/../npu-xrt/add_blockwrite/aie.xclbin | FileCheck %s

// This integration test verifies that the lifted mode properly emits
// switchbox routing operations, including:
// - aie.switchbox operations for each tile with routing
// - aie.connect operations inside switchboxes

// CHECK: module
// CHECK: aie.device(npu1_1col)

// Verify tile operations are created for switchboxes
// CHECK-DAG: [[TILE:%.*]] = aie.tile({{[0-9]+}}, {{[0-9]+}})

// Verify switchbox operations are created
// CHECK-DAG: aie.switchbox([[TILE]]) {
// CHECK-NEXT:   aie.connect<

// Alternative format check - sometimes connects might be on separate lines
// CHECK-DAG: aie.connect<{{[A-Za-z]+}} : {{[0-9]+}}, {{[A-Za-z]+}} : {{[0-9]+}}>
