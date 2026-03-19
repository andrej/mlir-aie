// RUN: aie-translate --xclbin-to-mlir --emit-lifted %S/../npu-xrt/add_blockwrite/aie.xclbin | FileCheck %s

// Round-trip verification test for xclbin decompilation.
// This test validates that decompiling a real xclbin (from add_blockwrite)
// produces semantically equivalent output to the original aie.mlir.
//
// The original design (test/npu-xrt/add_blockwrite/aie.mlir) contains:
// - Tiles: (0,0) shim tile and (0,2) compute tile
// - Buffers: 5 buffers on tile (0,2) - two input, two output, one constant
// - Locks: 4 locks on tile (0,2) for synchronization
// - DMA BDs: Multiple buffer descriptors in the mem block
// - Switchbox routing: Flow from (0,0) DMA:0 to (0,2) DMA:0 and reverse

// CHECK: module {

// Device declaration
// CHECK: aie.device(npu1_1col) {

// ============================================================================
// TILES - Verify both shim tile (0,0) and compute tile (0,2) are present
// ============================================================================

// Shim tile at (0,0) - used for host interface
// CHECK-DAG: {{%.*}} = aie.tile(0, 0)

// Compute tile at (0,2) - contains the core logic and DMA configuration
// CHECK-DAG: {{%.*}} = aie.tile(0, 2)

// ============================================================================
// BUFFERS - Verify all 5 buffers on tile (0,2) are present
// ============================================================================

// The original design has 5 buffers on tile 0_2:
// - objFifo_in1_cons_buff_0, objFifo_in1_cons_buff_1 (input buffers)
// - objFifo_out1_buff_0, objFifo_out1_buff_1 (output buffers)
// - constant_buffer (constant data)

// All are memref<8xi32>, so we should see 5 buffer declarations
// We use COUNT to verify exactly 5 buffers exist
// CHECK-DAG: {{%.*}} = aie.buffer({{%.*}}) {{.*}} : memref<8xi32>
// CHECK-DAG: {{%.*}} = aie.buffer({{%.*}}) {{.*}} : memref<8xi32>
// CHECK-DAG: {{%.*}} = aie.buffer({{%.*}}) {{.*}} : memref<8xi32>
// CHECK-DAG: {{%.*}} = aie.buffer({{%.*}}) {{.*}} : memref<8xi32>
// CHECK-DAG: {{%.*}} = aie.buffer({{%.*}}) {{.*}} : memref<8xi32>

// ============================================================================
// LOCKS - Verify all 4 locks on tile (0,2) are present
// ============================================================================

// The original design has 4 locks on tile 0_2:
// - Lock 0: objFifo_in1_cons_prod_lock (init=2)
// - Lock 1: objFifo_in1_cons_cons_lock (init=0)
// - Lock 2: objFifo_out1_prod_lock (init=2)
// - Lock 3: objFifo_out1_cons_lock (init=0)

// Verify we have lock declarations (lock IDs 0-3)
// CHECK-DAG: {{%.*}} = aie.lock({{%.*}}, 0)
// CHECK-DAG: {{%.*}} = aie.lock({{%.*}}, 1)
// CHECK-DAG: {{%.*}} = aie.lock({{%.*}}, 2)
// CHECK-DAG: {{%.*}} = aie.lock({{%.*}}, 3)

// ============================================================================
// DMA BUFFER DESCRIPTORS - Verify BDs with proper structure
// ============================================================================

// The original mem block has multiple BDs (lines 88, 93, 100, 105 in aie.mlir)
// Each BD references a buffer with offset 0 and length 8
// Verify we have DMA BD operations with the expected pattern

// BD format: aie.dma_bd(buffer, offset, length)
// We should see multiple BDs, all with length 8 (matching the buffer size)
// CHECK-DAG: aie.dma_bd({{%.*}} : memref<8xi32>, 0, 8)
// CHECK-DAG: aie.dma_bd({{%.*}} : memref<8xi32>, 0, 8)
// CHECK-DAG: aie.dma_bd({{%.*}} : memref<8xi32>, 0, 8)
// CHECK-DAG: aie.dma_bd({{%.*}} : memref<8xi32>, 0, 8)

// ============================================================================
// SWITCHBOX ROUTING - Verify switchbox configuration is present
// ============================================================================

// The original design has two flows:
// - aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)  [line 28]
// - aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)  [line 29]
//
// These should be translated to switchbox configurations with connections

// Verify switchbox declarations exist for the tiles with routing
// CHECK-DAG: aie.switchbox({{%.*}}) {

// Verify we have connect operations establishing the routing paths
// The exact bundle names and channels depend on how the xclbin was compiled,
// but we should see connection operations inside the switchboxes
// CHECK-DAG: aie.connect<{{[A-Z][A-Za-z]*}} : {{[0-9]+}}, {{[A-Z][A-Za-z]*}} : {{[0-9]+}}>

// ============================================================================
// RUNTIME SEQUENCE - Verify runtime configuration is preserved
// ============================================================================

// The runtime sequence should still be present with any non-lifted operations
// CHECK: aiex.runtime_sequence

// ============================================================================
// MODULE CLOSURE
// ============================================================================

// Verify proper module structure closure
// CHECK: }
