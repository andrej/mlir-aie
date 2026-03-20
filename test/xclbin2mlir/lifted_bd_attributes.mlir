// RUN: aie-translate --xclbin-to-mlir --emit-lifted %S/../npu-xrt/add_blockwrite/aie.xclbin | FileCheck %s

// This test verifies that aie.dma_bd operations include the bd_id attribute.
// Note: Advanced BD features like dimensions, locks, and chaining are only
// emitted when the xclbin contains BD configurations with those features
// enabled. This test xclbin has minimal BD configuration.

// CHECK: module
// CHECK: aie.device(npu1_1col)

// Verify that aie.mem operation is created
// CHECK: aie.mem({{%.*}})

// Verify that dma_bd operations have bd_id attribute
// CHECK-DAG: aie.dma_bd({{%.*}} : memref<{{[0-9]+}}xi32>, 0, 0) {bd_id = 0 : i32}
// CHECK-DAG: aie.dma_bd({{%.*}} : memref<{{[0-9]+}}xi32>, 0, 0) {bd_id = 1 : i32}
// CHECK-DAG: aie.dma_bd({{%.*}} : memref<{{[0-9]+}}xi32>, 0, 0) {bd_id = 2 : i32}
// CHECK-DAG: aie.dma_bd({{%.*}} : memref<{{[0-9]+}}xi32>, 0, 0) {bd_id = 3 : i32}

// Verify the runtime sequence still exists
// CHECK: aie.runtime_sequence
