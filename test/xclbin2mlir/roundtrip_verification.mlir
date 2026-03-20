// RUN: aie-translate --xclbin-to-mlir --emit-lifted %S/../npu-xrt/add_blockwrite/aie.xclbin | FileCheck %s

// Round-trip verification test for xclbin decompilation.
// This test validates that decompiling the add_blockwrite xclbin
// produces properly structured MLIR output.
//
// Note: The decompiled output may differ from the original aie.mlir
// because the xclbin only contains certain register writes and not all
// the high-level information from the original MLIR.

// CHECK: module {

// Device declaration
// CHECK: aie.device(npu1_1col)

// ============================================================================
// TILES - Verify compute tile (0,2) is present
// ============================================================================

// Compute tile at (0,2) - where the BDs are configured
// CHECK-DAG: {{%.*}} = aie.tile(0, 2)

// ============================================================================
// BUFFERS - Verify buffer operations are created for BDs
// ============================================================================

// The decompiled output creates buffers for each BD found in the xclbin.
// Buffer sizes may be 0 if the BD configuration didn't include length info.
// CHECK-DAG: {{%.*}} = aie.buffer({{%.*}}) {{.*}} : memref<{{[0-9]+}}xi32>

// ============================================================================
// MEMORY OPERATION - Verify mem operation with BD blocks
// ============================================================================

// The mem operation contains all the BD configurations for a tile
// CHECK: aie.mem({{%.*}}) {

// BD operations should be emitted inside the mem block
// CHECK-DAG: aie.dma_bd({{%.*}} : memref<{{[0-9]+}}xi32>, {{[0-9]+}}, {{[0-9]+}})

// Each BD block terminates with aie.end or aie.next_bd
// CHECK: aie.end

// ============================================================================
// RUNTIME SEQUENCE - Verify runtime configuration is preserved
// ============================================================================

// The runtime sequence should still be present with any non-lifted operations
// CHECK: aie.runtime_sequence

// ============================================================================
// MODULE CLOSURE
// ============================================================================

// Verify proper module structure closure
// CHECK: }
