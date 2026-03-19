// Test demonstrating that writes can be coalesced across get_global operations
// but NOT across special register writes

module {
  aie.device(npu2) {
    memref.global "private" constant @some_data : memref<2xi32> = dense<[1, 2]>
    
    aiex.runtime_sequence @test_reordering() {
      // Scenario 1: Writes with get_global in between (should coalesce)
      aiex.npu.write32 {address = 1000 : ui32, value = 1 : ui32}
      %0 = memref.get_global @some_data : memref<2xi32>
      aiex.npu.write32 {address = 1004 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 1008 : ui32, value = 3 : ui32}
      
      // Scenario 2: Writes with special register in between (should NOT coalesce across it)
      aiex.npu.write32 {address = 2000 : ui32, value = 10 : ui32}
      aiex.npu.write32 {address = 2004 : ui32, value = 20 : ui32}
      // Special register for memtile (0,1): (0 << 25) | (1 << 20) | 0x94008 = 1048576 + 606216 = 1654792
      aiex.npu.write32 {address = 1654792 : ui32, value = 99 : ui32}
      aiex.npu.write32 {address = 3000 : ui32, value = 30 : ui32}
      aiex.npu.write32 {address = 3004 : ui32, value = 40 : ui32}
    }
  }
}
