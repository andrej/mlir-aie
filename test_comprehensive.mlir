// Comprehensive test showing all coalescing capabilities:
// 1. Reordering of non-consecutive writes
// 2. Coalescing across blockwrites
// 3. Special registers as barriers
// 4. get_global operations don't break coalescing

module {
  aie.device(npu2) {
    memref.global "private" constant @data1 : memref<2xi32> = dense<[111, 222]>
    memref.global "private" constant @data2 : memref<3xi32> = dense<[333, 444, 555]>
    
    aiex.runtime_sequence @comprehensive_test() {
      // === SLICE 1: Before first special register ===
      // Scattered writes: 1000, 1012, 1004, 1008
      // Should be reordered to: 1000, 1004, 1008, 1012 and coalesced
      aiex.npu.write32 {address = 1000 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 1012 : ui32, value = 4 : ui32}
      %0 = memref.get_global @data1 : memref<2xi32>
      aiex.npu.write32 {address = 1004 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 1008 : ui32, value = 3 : ui32}
      
      // Blockwrite in the middle - should be decomposed and merged
      %1 = memref.get_global @data1 : memref<2xi32>
      aiex.npu.blockwrite(%1) {address = 1016 : ui32} : memref<2xi32>  // Adds 111, 222 at 1016, 1020
      
      // More writes that should merge with above
      aiex.npu.write32 {address = 1024 : ui32, value = 5 : ui32}
      
      // === SPECIAL REGISTER BARRIER ===
      // Core Control at tile (0,2): (0 << 25) | (2 << 20) | 0x32000 = 2301952
      aiex.npu.write32 {address = 2301952 : ui32, value = 999 : ui32}
      
      // === SLICE 2: After special register ===
      // Non-consecutive writes: 2000, 2012, 2004, 2008
      // Should be reordered to: 2000, 2004, 2008, 2012 and coalesced
      aiex.npu.write32 {address = 2000 : ui32, value = 10 : ui32}
      %2 = memref.get_global @data2 : memref<3xi32>
      aiex.npu.write32 {address = 2012 : ui32, value = 40 : ui32}
      aiex.npu.write32 {address = 2004 : ui32, value = 20 : ui32}
      
      // Blockwrite that should merge
      %3 = memref.get_global @data2 : memref<3xi32>
      aiex.npu.blockwrite(%3) {address = 2016 : ui32} : memref<3xi32>  // Adds 333, 444, 555 at 2016, 2020, 2024
      
      aiex.npu.write32 {address = 2008 : ui32, value = 30 : ui32}
    }
  }
}
