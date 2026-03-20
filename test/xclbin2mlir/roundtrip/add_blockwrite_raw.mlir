// RUN: aie-translate --xclbin-to-mlir %S/../../npu-xrt/add_blockwrite/aie.xclbin | FileCheck %s

// Round-trip verification test for xclbin decompilation in raw mode.
// This test decompiles the add_blockwrite xclbin and verifies that the
// raw register write operations are correctly generated.
//
// The raw mode (default) should emit aiex.npu.write32 and aiex.npu.maskwrite32
// operations that directly represent the register configuration commands.

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
// RAW REGISTER OPERATIONS - Verify presence of write operations
// ============================================================================

// Raw mode should emit aiex.npu.write32 operations for register writes
// These operations directly represent hardware configuration commands
// CHECK:       aiex.npu.write32 {address = {{[0-9]+}} : ui32, value = {{[0-9]+}} : ui32}

// Raw mode should also emit aiex.npu.maskwrite32 operations for masked writes
// These are used for updating specific bit fields without affecting other bits
// CHECK:       aiex.npu.maskwrite32 {address = {{[0-9]+}} : ui32, mask = {{[0-9]+}} : ui32, value = {{[0-9]+}} : ui32}

// ============================================================================
// SPECIFIC ADDRESS RANGE CHECKS - Verify hardware register regions
// ============================================================================

// DMA Buffer Descriptor (BD) configuration registers - typically in 0x21E000-0x21F000 range
// These writes configure buffer descriptors for DMA operations (addresses 2220000-2359999)
// CHECK-DAG:   aiex.npu.write32 {address = 222{{[0-9][0-9][0-9][0-9]}} : ui32, value = {{[0-9]+}} : ui32}

// Stream Switch configuration registers - typically in 0x21D000-0x21E000 range
// These writes configure stream routing and connections (addresses 2215936-2224639)
// CHECK-DAG:   aiex.npu.maskwrite32 {address = 221{{[5-9][0-9][0-9][0-9]}} : ui32, mask = {{[0-9]+}} : ui32, value = {{[0-9]+}} : ui32}

// Column control registers - lower address range (0x3F000 range, ~258048)
// These configure column-level settings (addresses 257000-269999)
// CHECK-DAG:   aiex.npu.write32 {address = 25{{[7-9][0-9][0-9][0-9]}} : ui32, value = {{[0-9]+}} : ui32}

// AXI MM registers - middle address range (0x1B0000 range, ~1769472)
// These configure memory-mapped AXI interfaces (addresses 1760000-1779999)
// CHECK-DAG:   aiex.npu.write32 {address = 176{{[0-9][0-9][0-9][0-9]}} : ui32, value = {{[0-9]+}} : ui32}

// ============================================================================
// TERMINATOR - Verify proper termination
// ============================================================================

// The runtime sequence must end with an aie.end terminator
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }
