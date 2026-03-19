// RUN: aie-opt --aie-coalesce-write32s %s | FileCheck %s

// Test that writes to special registers act as barriers for coalescing

// CHECK-LABEL: module
module {
  aie.device(npu2) {
    
    // CHECK: memref.global "private" constant @coalesced_write32
    // CHECK-SAME: memref<3xi32>
    
    aiex.runtime_sequence @test_special_regs() {
      // These three writes should be coalesced (addresses 1000, 1004, 1008)
      // CHECK: [[MEMREF:%.*]] = memref.get_global @coalesced_write32
      // CHECK-NEXT: aiex.npu.blockwrite([[MEMREF]]) {address = 1000 : ui32}
      aiex.npu.write32 {address = 1000 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 1004 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 1008 : ui32, value = 3 : ui32}
      
      // Write to a special register (Core_Control at tile (0,2) = 0x32000)
      // Address computation: (0 << 25) | (2 << 20) | 0x32000 = 0x232000
      // This should NOT be coalesced with neighboring writes
      // CHECK: aiex.npu.write32 {address = 2301952 : ui32, value = 100 : ui32}
      aiex.npu.write32 {address = 2301952 : ui32, value = 100 : ui32}
      
      // These two writes come after the special register write
      // They should be coalesced together but separate from the above
      // CHECK: memref.global "private" constant @coalesced_write32
      // CHECK-SAME: memref<2xi32>
      // CHECK: [[MEMREF2:%.*]] = memref.get_global @coalesced_write32
      // CHECK-NEXT: aiex.npu.blockwrite([[MEMREF2]]) {address = 2000 : ui32}
      aiex.npu.write32 {address = 2000 : ui32, value = 10 : ui32}
      aiex.npu.write32 {address = 2004 : ui32, value = 20 : ui32}
    }
  }
}
