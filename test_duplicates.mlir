// Test duplicate write elimination
// Multiple writes to the same register should keep only the last value

module {
  aie.device(npu2) {
    aiex.runtime_sequence @test_duplicates() {
      // Write to address 1000 three times - should keep only the last value (333)
      aiex.npu.write32 {address = 1000 : ui32, value = 111 : ui32}
      aiex.npu.write32 {address = 1000 : ui32, value = 222 : ui32}
      aiex.npu.write32 {address = 1000 : ui32, value = 333 : ui32}
      
      // Write to address 1004 twice - should keep only the last value (555)
      aiex.npu.write32 {address = 1004 : ui32, value = 444 : ui32}
      aiex.npu.write32 {address = 1004 : ui32, value = 555 : ui32}
      
      // Write to address 1008 once - should be kept (666)
      aiex.npu.write32 {address = 1008 : ui32, value = 666 : ui32}
      
      // Expected result: blockwrite with [333, 555, 666] at address 1000
      
      // After special register barrier
      aiex.npu.write32 {address = 262144 : ui32, value = 99 : ui32}  // Special register
      
      // More duplicates after barrier
      aiex.npu.write32 {address = 2000 : ui32, value = 10 : ui32}
      aiex.npu.write32 {address = 2004 : ui32, value = 20 : ui32}
      aiex.npu.write32 {address = 2000 : ui32, value = 11 : ui32}  // Overwrite 2000
      aiex.npu.write32 {address = 2008 : ui32, value = 30 : ui32}
      
      // Expected result: blockwrite with [11, 20, 30] at address 2000
    }
  }
}
