module {
  aie.device(npu1_1col) @main {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 4>
      aie.connect<North : 2, South : 2>
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<North : 0, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<North : 2, DMA : 1>
      aie.connect<DMA : 1, South : 2>
    }
    %bd_buf_0_1_27 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_27"} : memref<1xi32> 
    %bd_buf_0_1_26 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_26"} : memref<1xi32> 
    %bd_buf_0_1_25 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_25"} : memref<1xi32> 
    %bd_buf_0_1_24 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_24"} : memref<1xi32> 
    %bd_buf_0_1_3 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_3"} : memref<1xi32> 
    %bd_buf_0_1_2 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_2"} : memref<1xi32> 
    %bd_buf_0_1_1 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_1"} : memref<1xi32> 
    %bd_buf_0_1_0 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_0"} : memref<1xi32> 
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb9)
    ^bb1:  // 9 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8
      aie.dma_bd(%bd_buf_0_1_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_24 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_25 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_26 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb8:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_27 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      aie.end
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, South : 0>
    }
    %bd_buf_0_2_3 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_3"} : memref<1xi32> 
    %bd_buf_0_2_2 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_2"} : memref<1xi32> 
    %bd_buf_0_2_1 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_1"} : memref<1xi32> 
    %bd_buf_0_2_0 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_0"} : memref<1xi32> 
    %lock_0_2 = aie.lock(%tile_0_2, 3) {init = 0 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32}
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
    aie.runtime_sequence @configure() {
      aie.end
    }
  }
}
