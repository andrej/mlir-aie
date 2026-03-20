// RUN: aie-translate --xclbin-to-mlir --emit-lifted %S/../npu-xrt/add_blockwrite/aie.xclbin | FileCheck %s

// This integration test verifies that the lifted mode properly emits
// high-level AIE operations for buffer descriptors, including:
// - aie.tile operations for each tile referenced
// - aie.buffer operations for BD buffers
// - aie.mem operations containing DMA BD blocks
// - aie.dma_bd operations with proper attributes
// Note: aie.lock and aie.use_lock are only emitted when BDs have lock
// acquire/release configured. This test xclbin doesn't use locks.

// CHECK: module
// CHECK: aie.device(npu1_1col)

// Verify tile operations are created
// CHECK-DAG: [[TILE:%.*]] = aie.tile({{[0-9]+}}, {{[0-9]+}})

// Verify buffer operations are created for BDs
// CHECK-DAG: [[BUFFER:%.*]] = aie.buffer([[TILE]]) {{.*}} : memref<{{[0-9]+}}xi32>

// Verify memory operation is created containing BDs
// CHECK: aie.mem([[TILE]])

// Verify DMA BD operations are created
// CHECK-DAG: aie.dma_bd({{%.*}} : memref<{{[0-9]+}}xi32>, {{[0-9]+}}, {{[0-9]+}})

// Verify the runtime sequence still exists
// CHECK: aie.runtime_sequence
