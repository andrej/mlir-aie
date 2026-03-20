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
// TERMINATOR - Verify proper termination
// ============================================================================

// The runtime sequence must end with an aie.end terminator
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }
