// Test maskwrite operation handling
// Maskwrites should act as barriers for coalescing and handle superseding correctly

module {
  aie.device(npu2) {
    aiex.runtime_sequence @test_maskwrite() {
      // Case 1: write32 followed by maskwrite to same address
      // The maskwrite should supersede the write32
      aiex.npu.write32 {address = 1000 : ui32, value = 100 : ui32}
      aiex.npu.maskwrite32 {address = 1000 : ui32, value = 200 : ui32, mask = 255 : ui32}
      // Expected: only maskwrite remains
      
      // Case 2: maskwrite followed by write32 to same address
      // The write32 should supersede the maskwrite
      aiex.npu.maskwrite32 {address = 2000 : ui32, value = 300 : ui32, mask = 255 : ui32}
      aiex.npu.write32 {address = 2000 : ui32, value = 400 : ui32}
      // Expected: only write32 remains
      
      // Case 3: maskwrite acts as barrier for coalescing
      // Writes before maskwrite should coalesce, writes after should start new sequence
      aiex.npu.write32 {address = 3000 : ui32, value = 10 : ui32}
      aiex.npu.write32 {address = 3004 : ui32, value = 20 : ui32}
      aiex.npu.write32 {address = 3008 : ui32, value = 30 : ui32}
      aiex.npu.maskwrite32 {address = 3012 : ui32, value = 40 : ui32, mask = 255 : ui32}
      aiex.npu.write32 {address = 3016 : ui32, value = 50 : ui32}
      aiex.npu.write32 {address = 3020 : ui32, value = 60 : ui32}
      aiex.npu.write32 {address = 3024 : ui32, value = 70 : ui32}
      // Expected: blockwrite[10,20,30] at 3000, maskwrite at 3012, blockwrite[50,60,70] at 3016
      
      // Case 4: isolated maskwrite should remain unchanged
      aiex.npu.maskwrite32 {address = 4000 : ui32, value = 500 : ui32, mask = 65535 : ui32}
      // Expected: maskwrite remains as-is
      
      // Case 5: multiple maskwrites to same address, keep last
      aiex.npu.maskwrite32 {address = 5000 : ui32, value = 600 : ui32, mask = 255 : ui32}
      aiex.npu.maskwrite32 {address = 5000 : ui32, value = 700 : ui32, mask = 65280 : ui32}
      // Expected: only second maskwrite remains
    }
  }
}
