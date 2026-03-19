// RUN: aie-opt --aie-coalesce-write32s %s | FileCheck %s

// CHECK-LABEL: module
module {
  aie.device(npu1_1col) {
    // Initial global for a blockwrite
    memref.global "private" constant @existing_data : memref<2xi32> = dense<[100, 200]>
    
    // CHECK: memref.global "private" constant @coalesced_write32
    // CHECK-SAME: memref<6xi32> = dense<[100, 200, 300, 400, 500, 600]>
    
    aiex.runtime_sequence @test_seq() {
      // Test 1: Coalesce blockwrite + write32s
      // This blockwrite at address 1000 with 2 words [100, 200]
      // followed by write32s at 1008, 1012, 1016, 1020 with values 300, 400, 500, 600
      // Should all be coalesced into one blockwrite of 6 words starting at 1000
      // CHECK-NOT: aiex.npu.write32 {address = 1008
      // CHECK-NOT: aiex.npu.write32 {address = 1012
      // CHECK-NOT: aiex.npu.write32 {address = 1016
      // CHECK-NOT: aiex.npu.write32 {address = 1020
      // CHECK: [[MEMREF:%.*]] = memref.get_global @coalesced_write32
      // CHECK-NEXT: aiex.npu.blockwrite([[MEMREF]]) {address = 1000 : ui32}
      %0 = memref.get_global @existing_data : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 1000 : ui32} : memref<2xi32>
      aiex.npu.write32 {address = 1008 : ui32, value = 300 : ui32}
      aiex.npu.write32 {address = 1012 : ui32, value = 400 : ui32}
      aiex.npu.write32 {address = 1016 : ui32, value = 500 : ui32}
      aiex.npu.write32 {address = 1020 : ui32, value = 600 : ui32}
      
      // Test 2: Coalesce two blockwrites
      // CHECK: memref.global "private" constant @coalesced_write32_2000
      // CHECK-SAME: memref<4xi32> = dense<[10, 20, 30, 40]>
      %1 = memref.get_global @existing_data : memref<2xi32>
      aiex.npu.blockwrite(%1) {address = 2000 : ui32} : memref<2xi32>
      %2 = memref.get_global @existing_data : memref<2xi32>
      aiex.npu.blockwrite(%2) {address = 2008 : ui32} : memref<2xi32>
      
      // Test 3: Non-contiguous should not be coalesced
      // CHECK: aiex.npu.write32 {address = 3000 : ui32, value = 1 : ui32}
      // CHECK-NEXT: aiex.npu.write32 {address = 3020 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 3000 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 3020 : ui32, value = 2 : ui32}
    }
  }
}
