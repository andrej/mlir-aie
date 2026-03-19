// Test demonstrating reordering of non-consecutive writes
// Writes A C B should be reordered to A B C and then coalesced

module {
  aie.device(npu2) {
    aiex.runtime_sequence @test_reorder() {
      // Write pattern: address 1000, 1008, 1004
      // Should be reordered to: 1000, 1004, 1008 and then coalesced
      aiex.npu.write32 {address = 1000 : ui32, value = 10 : ui32}
      aiex.npu.write32 {address = 1008 : ui32, value = 30 : ui32}
      aiex.npu.write32 {address = 1004 : ui32, value = 20 : ui32}
      
      // Special register barrier
      aiex.npu.write32 {address = 262144 : ui32, value = 99 : ui32}  // 0x40000 at shim tile (0,0)
      
      // After special register: addresses 2000, 2012, 2004, 2008
      // Should be reordered to: 2000, 2004, 2008, 2012 and coalesced
      aiex.npu.write32 {address = 2000 : ui32, value = 100 : ui32}
      aiex.npu.write32 {address = 2012 : ui32, value = 400 : ui32}
      aiex.npu.write32 {address = 2004 : ui32, value = 200 : ui32}
      aiex.npu.write32 {address = 2008 : ui32, value = 300 : ui32}
    }
  }
}
