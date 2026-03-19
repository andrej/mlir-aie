// RUN: aie-translate --xclbin-to-mlir --emit-lifted %S/../npu-xrt/add_blockwrite/aie.xclbin | FileCheck %s

// Comprehensive integration test for lifted xclbin decompilation.
// This test verifies that ALL the following high-level operations are emitted:
// 1. aie.tile - tile references
// 2. aie.buffer - BD buffer allocations
// 3. aie.lock - lock declarations for BD synchronization
// 4. aie.dma_bd - DMA buffer descriptors with attributes
// 5. aie.switchbox - switchbox declarations
// 6. aie.connect - routing connections within switchboxes
// 7. aiex.runtime_sequence - runtime configuration sequence

// CHECK: module {

// Must have device declaration
// CHECK: aie.device(npu1_1col) {

// Verify tile operations exist
// These should be created for any tile that has BDs or switchbox config
// CHECK-DAG: {{%.*}} = aie.tile({{[0-9]+}}, {{[0-9]+}})

// Verify buffer operations exist
// These represent the memory regions referenced by buffer descriptors
// CHECK-DAG: {{%.*}} = aie.buffer({{%.*}}) {{.*}} : memref<{{[0-9]+}}xi32>

// Verify lock operations exist
// These are created when BDs have lock acquire or release configured
// CHECK-DAG: {{%.*}} = aie.lock({{%.*}}, {{[0-9]+}})

// Verify DMA BD operations exist with proper structure
// Format: aie.dma_bd(buffer, offset, length) with optional attributes
// CHECK-DAG: aie.dma_bd({{%.*}} : memref<{{[0-9]+}}xi32>, {{[0-9]+}}, {{[0-9]+}})

// Verify switchbox operations exist
// These define the routing configuration for each tile
// CHECK-DAG: aie.switchbox({{%.*}}) {

// Verify connect operations inside switchboxes
// Format: aie.connect<SourceBundle : channel, DestBundle : channel>
// CHECK-DAG: aie.connect<{{[A-Z][A-Za-z]*}} : {{[0-9]+}}, {{[A-Z][A-Za-z]*}} : {{[0-9]+}}>

// Verify runtime sequence is still present
// This contains the low-level configuration that wasn't lifted
// CHECK: aiex.runtime_sequence

// The module should close properly
// CHECK: }
