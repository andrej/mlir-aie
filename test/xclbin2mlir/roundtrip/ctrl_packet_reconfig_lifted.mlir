// RUN: aie-translate --xclbin-to-mlir --emit-lifted %S/../../npu-xrt/ctrl_packet_reconfig/aie.xclbin | FileCheck %s

// Round-trip verification test for xclbin decompilation in lifted mode.
// This test decompiles the ctrl_packet_reconfig xclbin with the --emit-lifted flag
// and verifies that the operations are correctly generated.
//
// The ctrl_packet_reconfig design uses control packets for dynamic reconfiguration.
// This xclbin does not contain DMA BD configurations, so it only emits low-level
// NPU operations for register writes. When DMA BDs are present in compute tiles,
// lifted mode will emit semantic AIE operations (aie.tile, aie.buffer, aie.mem, aie.dma_bd).

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
// NPU OPERATIONS - Verify presence of specific operations
// ============================================================================

// Lifted mode currently emits aiex.npu.write32 operations for various registers
// Column control registers - addresses around 0x3F000 (258xxx)
// CHECK:       aiex.npu.write32 {address = {{[0-9]+}} : ui32, value = {{[0-9]+}} : ui32}

// AXI MM registers - addresses in 0x1B0000 range
// CHECK:       aiex.npu.write32 {address = {{[0-9]+}} : ui32, value = {{[0-9]+}} : ui32}

// Lifted mode currently emits aiex.npu.maskwrite32 operations
// Low-level control registers
// CHECK:       aiex.npu.maskwrite32 {address = {{[0-9]+}} : ui32, mask = {{[0-9]+}} : ui32, value = {{[0-9]+}} : ui32}

// ============================================================================
// TERMINATOR - Verify proper termination
// ============================================================================

// The runtime sequence must end with an aie.end terminator
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }
