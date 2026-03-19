// RUN: aie-opt --aie-coalesce-write32s %s | FileCheck %s

// CHECK-LABEL: module
module {
  aie.device(npu1_1col) {
    // CHECK: memref.global "private" constant @coalesced_write32
    // CHECK-SAME: memref<4xi32> = dense<[305419896, 2, 3, 4]>
    
    aiex.runtime_sequence @test_seq() {
      // These four write32 operations should be coalesced into one blockwrite
      // CHECK-NOT: aiex.npu.write32
      // CHECK: [[MEMREF:%.*]] = memref.get_global @coalesced_write32
      // CHECK-NEXT: aiex.npu.blockwrite([[MEMREF]])
      aiex.npu.write32 {address = 305419896 : ui32, value = 305419896 : ui32}
      aiex.npu.write32 {address = 305419900 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 305419904 : ui32, value = 3 : ui32}
      aiex.npu.write32 {address = 305419908 : ui32, value = 4 : ui32}
      
      // This single write32 should not be touched
      // CHECK: aiex.npu.write32 {address = 100 : ui32, value = 99 : ui32}
      aiex.npu.write32 {address = 100 : ui32, value = 99 : ui32}
      
      // These two write32 operations are not contiguous (gap in addresses)
      // CHECK: aiex.npu.write32 {address = 200 : ui32, value = 10 : ui32}
      // CHECK-NEXT: aiex.npu.write32 {address = 208 : ui32, value = 20 : ui32}
      aiex.npu.write32 {address = 200 : ui32, value = 10 : ui32}
      aiex.npu.write32 {address = 208 : ui32, value = 20 : ui32}
    }
  }
}
