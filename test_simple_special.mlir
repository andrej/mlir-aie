module {
  aie.device(npu2) {
    aiex.runtime_sequence @test() {
      // Single write to special register - should not be changed
      aiex.npu.write32 {address = 262144 : ui32, value = 99 : ui32}
    }
  }
}
