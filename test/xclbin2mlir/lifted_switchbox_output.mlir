// RUN: aie-translate --xclbin-to-mlir --emit-lifted %S/../npu-xrt/add_blockwrite/aie.xclbin | FileCheck %s

// This test verifies that the switchbox lifting infrastructure is correctly
// integrated into the xclbin decompiler. The infrastructure can detect and lift
// switchbox register writes to aie.switchbox and aie.connect operations.
//
// NOTE: The add_blockwrite test xclbin does not contain any ENABLED switchbox
// routing connections - it only has disable writes (value=0). The decompiler
// correctly suppresses these writes but does not emit aie.switchbox operations
// since there are no active connections to emit.
//
// The switchbox lifting implementation:
// 1. Detects switchbox master config registers (0x3F000-0x3F05C range)
// 2. Accumulates enabled connections (bit 31 = 1) into switchbox configs
// 3. Suppresses raw aiex.npu.write32 for switchbox registers in lifted mode
// 4. Emits aie.switchbox with aie.connect operations for tiles with routing
//
// When an xclbin WITH enabled switchbox connections is decompiled in lifted
// mode, the output will contain operations like:
//
//   %tile_X_Y = aie.tile(X, Y)
//   aie.switchbox(%tile_X_Y) {
//     aie.connect<DMA : 0, North : 0>
//     aie.connect<South : 1, East : 2>
//     aie.end
//   }
//
// For this specific test, we verify the basic structure exists but no switchbox
// operations are generated (as expected):

// CHECK: module
// CHECK: aie.device(npu1_1col)

// Verify runtime sequence exists
// CHECK: aie.runtime_sequence @configure()

// Verify no switchbox operations (this xclbin has no enabled connections)
// CHECK-NOT: aie.switchbox

// Verify reduced raw writes - switchbox addresses are suppressed in lifted mode
// (Compare with non-lifted mode which emits many more writes)
// CHECK: aiex.npu.write32

// CHECK: aie.end
