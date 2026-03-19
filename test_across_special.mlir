// Test coalescing behavior around special registers

module {
  aie.device(npu2) {
    
    aiex.runtime_sequence @test_across_special() {
      // Write sequence that would be contiguous with a special register
      // Shim tile (0,0) special register at 0x40000
      // Absolute address: (0 << 25) | (0 << 20) | 0x40000 = 0x40000 = 262144
      
      // These should be coalesced (before special register)
      aiex.npu.write32 {address = 262136 : ui32, value = 1 : ui32}  // 0x3FFF8
      aiex.npu.write32 {address = 262140 : ui32, value = 2 : ui32}  // 0x3FFFC
      
      // Special register - should NOT be coalesced with neighbors
      aiex.npu.write32 {address = 262144 : ui32, value = 99 : ui32}  // 0x40000 (special)
      
      // These should be coalesced (after special register) 
      aiex.npu.write32 {address = 262148 : ui32, value = 3 : ui32}  // 0x40004
      aiex.npu.write32 {address = 262152 : ui32, value = 4 : ui32}  // 0x40008
    }
  }
}
