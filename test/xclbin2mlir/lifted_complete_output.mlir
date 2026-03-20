// RUN: aie-translate --xclbin-to-mlir --emit-lifted %S/../npu-xrt/add_blockwrite/aie.xclbin | FileCheck %s

// Integration test for lifted xclbin decompilation.
// This test verifies that the following high-level operations are emitted:
// 1. aie.tile - tile references
// 2. aie.buffer - BD buffer allocations
// 3. aie.mem - memory operation containing BD blocks
// 4. aie.dma_bd - DMA buffer descriptors with attributes
// 5. aie.runtime_sequence - runtime configuration sequence
//
// Note: aie.lock and aie.switchbox are only emitted when the xclbin
// contains lock configurations or switchbox routing respectively.
// This test xclbin doesn't have those features configured.

// CHECK: module {

// Must have device declaration
// CHECK: aie.device(npu1_1col)

// Verify tile operations exist
// These should be created for any tile that has BDs
// CHECK-DAG: {{%.*}} = aie.tile({{[0-9]+}}, {{[0-9]+}})

// Verify buffer operations exist
// These represent the memory regions referenced by buffer descriptors
// CHECK-DAG: {{%.*}} = aie.buffer({{%.*}}) {{.*}} : memref<{{[0-9]+}}xi32>

// Verify mem operation exists containing BDs
// CHECK: aie.mem({{%.*}}) {

// Verify DMA BD operations exist with proper structure
// Format: aie.dma_bd(buffer, offset, length) with optional attributes
// CHECK-DAG: aie.dma_bd({{%.*}} : memref<{{[0-9]+}}xi32>, {{[0-9]+}}, {{[0-9]+}})

// Verify runtime sequence is still present
// This contains the low-level configuration that wasn't lifted
// CHECK: aie.runtime_sequence

// The module should close properly
// CHECK: }
