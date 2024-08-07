module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aiex.runtime_sequence(%arg0: memref<4xi32>) {
      // Send a single four-byte word and wait for the TCT.
      %0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0: memref<4xi32>, 0, 1) { bd_id = 0 : i32 }
        aie.end
      } {issue_token = true} // this will lower to blockwride + an address patch instruction
    }
  }
}
