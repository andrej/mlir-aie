// Test demonstrating reordering across various operations
// All these writes are to non-special registers and should be coalesced

module {
  aie.device(npu2) {
    memref.global "private" constant @data : memref<2xi32> = dense<[1, 2]>
    
    aiex.runtime_sequence @test_complex_reorder() {
      // Writes scattered with get_global operations interspersed
      // Addresses: 1000, 1008, 1004, 1012
      // Should all be reordered and coalesced into one blockwrite
      aiex.npu.write32 {address = 1000 : ui32, value = 10 : ui32}
      %0 = memref.get_global @data : memref<2xi32>
      aiex.npu.write32 {address = 1008 : ui32, value = 30 : ui32}
      %1 = memref.get_global @data : memref<2xi32>
      aiex.npu.write32 {address = 1004 : ui32, value = 20 : ui32}
      %2 = memref.get_global @data : memref<2xi32>
      aiex.npu.write32 {address = 1012 : ui32, value = 40 : ui32}
      
      // Another non-write operation (blockwrite to different address range)
      // This should still not break the slice since it's not a special register
      %3 = memref.get_global @data : memref<2xi32>
      aiex.npu.blockwrite(%3) {address = 5000 : ui32} : memref<2xi32>
      
      // More writes that should be coalesced separately
      aiex.npu.write32 {address = 2000 : ui32, value = 100 : ui32}
      aiex.npu.write32 {address = 2004 : ui32, value = 200 : ui32}
    }
  }
}
