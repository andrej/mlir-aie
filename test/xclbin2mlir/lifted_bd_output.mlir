// RUN: aie-translate --xclbin-to-mlir --emit-lifted %S/../npu-xrt/add_blockwrite/aie.xclbin | FileCheck %s

// This integration test verifies that the lifted mode properly emits
// high-level AIE operations for buffer descriptors, including:
// - aie.tile operations for each tile referenced
// - aie.buffer operations for BD buffers
// - aie.lock operations for BD lock acquire/release
// - aie.dma_bd operations with proper attributes

// CHECK: module
// CHECK: aie.device(npu1_1col)

// Verify tile operations are created
// CHECK-DAG: [[TILE:%.*]] = aie.tile({{[0-9]+}}, {{[0-9]+}})

// Verify buffer operations are created for BDs
// CHECK-DAG: [[BUFFER:%.*]] = aie.buffer([[TILE]]) {{.*}} : memref<{{[0-9]+}}xi32>

// Verify DMA BD operations are created
// CHECK-DAG: aie.dma_bd([[BUFFER]] : memref<{{[0-9]+}}xi32>, {{[0-9]+}}, {{[0-9]+}})

// Verify lock operations are created when BDs use locks
// CHECK-DAG: [[LOCK:%.*]] = aie.lock([[TILE]], {{[0-9]+}})

// Verify the runtime sequence still exists
// CHECK: aiex.runtime_sequence
