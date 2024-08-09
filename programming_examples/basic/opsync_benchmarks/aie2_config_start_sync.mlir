module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) : memref<1024xi8>
    // We need to actually consume the data we send in or backpressure will stop everything after 40 bytes transferred.
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.memtile_dma(%tile_0_1) {
        aie.dma_start(S2MM, 0, ^bb1, ^bb2)
      ^bb1:
        aie.dma_bd(%buf: memref<1024xi8>, 0, 1024)
        aie.next_bd ^bb1
      ^bb2:
        aie.end
    }
    aiex.runtime_sequence(%arg0: memref<1024xi8>) {
      // Do nothing
      // Send a single four-byte word and wait for the TCT.
      %0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0: memref<1024xi8>, 0, 1024) { bd_id = 0 : i32 }
        aie.end
      } {issue_token = true} // this will lower to blockwride + an address patch instruction
      aiex.dma_start_task(%0) // this will lower to a push queue instruction
      aiex.dma_await_task(%0) // this will lower to a sync instruction
    }
  }
}
