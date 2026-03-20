// RUN: aie-translate --xclbin-to-mlir --emit-lifted %S/../../npu-xrt/add_blockwrite/aie.xclbin | FileCheck %s

// Round-trip verification test for xclbin decompilation in lifted mode.
// This test decompiles the add_blockwrite xclbin with the --emit-lifted flag
// and verifies that the operations are correctly generated.
//
// Lifted mode (--emit-lifted) now emits high-level semantic AIE operations
// (aie.tile, aie.buffer, aie.mem, aie.dma_bd) for compute tiles (row > 0).
// Shim tiles (row 0) continue to emit low-level NPU operations since they
// lack local memory and use different DMA infrastructure.

// ============================================================================
// MODULE STRUCTURE - Verify basic module structure
// ============================================================================

// CHECK: module {
// CHECK:   aie.device(npu1_1col)

// ============================================================================
// SEMANTIC AIE OPERATIONS - Verify semantic lifting for compute tiles
// ============================================================================

// Compute tile (0, 2) should be created
// CHECK:     %[[TILE_0_2:.*]] = aie.tile(0, 2)

// Buffers for BDs should be created
// CHECK:     %[[BUF_0:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "bd_buf_0_2_0"} : memref<0xi32>

// Memory operation should be created for compute tile
// CHECK:     %[[MEM:.*]] = aie.mem(%[[TILE_0_2]]) {

// DMA BD operations should be emitted inside aie.mem
// CHECK:       aie.dma_bd(%[[BUF_0]] : memref<0xi32>, 0, 0) {bd_id = 0 : i32}
// CHECK:       aie.end

// ============================================================================
// RUNTIME SEQUENCE - Verify runtime sequence block exists
// ============================================================================

// The decompiled output should contain a runtime_sequence block
// with the configuration function
// CHECK:     aie.runtime_sequence @configure() {

// ============================================================================
// NPU OPERATIONS - Verify presence of NPU operations for shim tiles
// ============================================================================

// Shim tile operations and non-BD operations still emit raw NPU operations
// CHECK:       aiex.npu.write32 {address = {{[0-9]+}} : ui32, value = {{[0-9]+}} : ui32}

// Lifted mode currently emits aiex.npu.maskwrite32 operations
// CHECK:       aiex.npu.maskwrite32 {address = {{[0-9]+}} : ui32, mask = {{[0-9]+}} : ui32, value = {{[0-9]+}} : ui32}

// ============================================================================
// SPECIFIC ADDRESS RANGE CHECKS - Verify hardware register regions
// ============================================================================

// Shim DMA BD configuration (0x220000 range) - still emitted as raw writes for shim tiles
// CHECK-DAG:   aiex.npu.write32 {address = 222{{[0-9][0-9][0-9][0-9]}} : ui32, value = {{[0-9]+}} : ui32}

// Note: Compute tile BD registers (0x21D000 range) are now lifted to aie.dma_bd operations
// and should NOT appear as aiex.npu.write32 operations

// Column control registers - lower address range (0x3F000 range, ~258048)
// CHECK-DAG:   aiex.npu.write32 {address = 25{{[7-9][0-9][0-9][0-9]}} : ui32, value = {{[0-9]+}} : ui32}

// Column reset registers - addresses around 0x232000 (2301952)
// CHECK-DAG:   aiex.npu.maskwrite32 {address = 230{{[0-9][0-9][0-9][0-9]}} : ui32, mask = {{[0-9]+}} : ui32, value = {{[0-9]+}} : ui32}

// ============================================================================
// TERMINATOR - Verify proper termination
// ============================================================================

// The runtime sequence must end with an aie.end terminator
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }
