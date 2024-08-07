module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aiex.runtime_sequence(%arg0: memref<4xi32>) {
      // Do nothing
    }
  }
}
