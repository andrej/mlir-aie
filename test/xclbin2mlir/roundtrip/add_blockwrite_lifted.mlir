// RUN: aie-translate --xclbin-to-mlir --emit-lifted %S/../../npu-xrt/add_blockwrite/aie.xclbin | FileCheck %s

// Round-trip verification test for xclbin decompilation in lifted mode.
// This test decompiles the add_blockwrite xclbin with the --emit-lifted flag
// and verifies that the operations are correctly generated.
//
// NOTE: Currently, lifted mode (--emit-lifted) still emits low-level NPU operations
// (aiex.npu.write32 and aiex.npu.maskwrite32) rather than high-level semantic
// AIE operations (aie.tile, aie.buffer, aie.mem, aie.dma_bd, aie.switchbox).
// This test documents the current behavior and can be updated when lifted mode
// is enhanced to produce semantic operations.

// ============================================================================
// MODULE STRUCTURE - Verify basic module structure
// ============================================================================

// CHECK: module {
// CHECK:   aie.device(npu1_1col)

// ============================================================================
// RUNTIME SEQUENCE - Verify runtime sequence block exists
// ============================================================================

// The decompiled output should contain a runtime_sequence block
// with the configuration function
// CHECK:     aie.runtime_sequence @configure() {

// ============================================================================
// NPU OPERATIONS - Verify presence of NPU operations
// ============================================================================

// Lifted mode currently emits aiex.npu.write32 operations
// CHECK:       aiex.npu.write32 {address = {{[0-9]+}} : ui32, value = {{[0-9]+}} : ui32}

// Lifted mode currently emits aiex.npu.maskwrite32 operations
// CHECK:       aiex.npu.maskwrite32 {address = {{[0-9]+}} : ui32, mask = {{[0-9]+}} : ui32, value = {{[0-9]+}} : ui32}

// ============================================================================
// SPECIFIC ADDRESS RANGE CHECKS - Verify hardware register regions
// ============================================================================

// DMA Buffer Descriptor (BD) configuration registers - addresses in 0x220000 range (2220000+)
// CHECK-DAG:   aiex.npu.write32 {address = 222{{[0-9][0-9][0-9][0-9]}} : ui32, value = {{[0-9]+}} : ui32}

// Stream Switch configuration registers - addresses in 0x21D000-0x21E000 range (2215936-2224639)
// CHECK-DAG:   aiex.npu.maskwrite32 {address = 221{{[5-9][0-9][0-9][0-9]}} : ui32, mask = {{[0-9]+}} : ui32, value = {{[0-9]+}} : ui32}

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
