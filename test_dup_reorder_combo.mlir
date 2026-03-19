// Test duplicate elimination combined with reordering
// Pattern: A B C B A should eliminate first A and first B, keeping: C (second B) (second A)
// Then reorder to: (second A) (second B) C and coalesce

module {
  aie.device(npu2) {
    aiex.runtime_sequence @test_dup_and_reorder() {
      // Write pattern with duplicates out of order:
      // 1004=10, 1008=20, 1000=30, 1008=40, 1004=50
      // After duplicate elimination: 1000=30, 1008=40, 1004=50
      // After reordering: 1000=30, 1004=50, 1008=40
      // Should coalesce to: blockwrite([30, 50, 40], 1000)
      
      aiex.npu.write32 {address = 1004 : ui32, value = 10 : ui32}  // Will be overwritten
      aiex.npu.write32 {address = 1008 : ui32, value = 20 : ui32}  // Will be overwritten
      aiex.npu.write32 {address = 1000 : ui32, value = 30 : ui32}  // Kept
      aiex.npu.write32 {address = 1008 : ui32, value = 40 : ui32}  // Kept (overwrites 20)
      aiex.npu.write32 {address = 1004 : ui32, value = 50 : ui32}  // Kept (overwrites 10)
      
      // Expected: blockwrite([30, 50, 40], 1000)
      // Verification: 1000->30, 1004->50, 1008->40
    }
  }
}
