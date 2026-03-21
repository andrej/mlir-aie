module {
  aie.device(npu1_1col) {
    %ext_buf_0_0_1 = aie.external_buffer {sym_name = "ext_buf_0_0_1"} : memref<64xi32>
    %ext_buf_0_0_0 = aie.external_buffer {sym_name = "ext_buf_0_0_0"} : memref<64xi32>
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_dma_0_0 = aie.shim_dma(%shim_noc_tile_0_0) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 3 preds: ^bb0, ^bb1, ^bb2
      aie.dma_bd(%ext_buf_0_0_0 : memref<64xi32>, 0, 64)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%ext_buf_0_0_1 : memref<64xi32>, 0, 64)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    %tile_0_2 = aie.tile(0, 2)
    %bd_buf_0_2_3 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_3"} : memref<1xi32> 
    %bd_buf_0_2_2 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_2"} : memref<1xi32> 
    %bd_buf_0_2_1 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_1"} : memref<1xi32> 
    %bd_buf_0_2_0 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_0"} : memref<1xi32> 
    %lock_0_2 = aie.lock(%tile_0_2, 3) {init = 0 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 2) {init = 1 : i32}  // MODIFIED: changed from 2 to 1
    %lock_0_2_1 = aie.lock(%tile_0_2, 1) {init = 0 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 0) {init = 2 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 5 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4
      aie.dma_bd(%bd_buf_0_2_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_0_2_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_0_2_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_0_2_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // pred: ^bb0
      aie.end
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    aie.runtime_sequence @configure() {
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2098336 : ui32, value = 99 : ui32}  // MODIFIED: changed from 0 to 99
      aiex.npu.write32 {address = 2228224 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2229120 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2229152 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2229184 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2229360 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224128 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224144 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224160 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224176 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224192 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224208 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224224 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224240 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224256 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224272 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224288 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224304 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224320 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224336 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224352 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224368 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224128 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224144 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224160 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224176 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2219524 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2219540 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258560 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 126976 : ui32, mask = 3072 : ui32, value = 1024 : ui32}
      aiex.npu.maskwrite32 {address = 126980 : ui32, mask = 48 : ui32, value = 16 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aie.end
    }
  }
}
