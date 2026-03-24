module {
  aie.device(npu1) @xclbin_device {
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 0, North : 7>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<South : 7, North : 5>
      aie.connect<North : 2, South : 2>
    }
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 0, North : 7>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 3>
      aie.connect<South : 7, North : 5>
      aie.connect<North : 2, South : 2>
    }
    %tile_3_3 = aie.tile(3, 3)
    %bd_buf_3_3_5 = aie.buffer(%tile_3_3) {sym_name = "bd_buf_3_3_5"} : memref<1xi32> 
    %bd_buf_3_3_4 = aie.buffer(%tile_3_3) {sym_name = "bd_buf_3_3_4"} : memref<1xi32> 
    %bd_buf_3_3_3 = aie.buffer(%tile_3_3) {sym_name = "bd_buf_3_3_3"} : memref<1xi32> 
    %bd_buf_3_3_2 = aie.buffer(%tile_3_3) {sym_name = "bd_buf_3_3_2"} : memref<1xi32> 
    %bd_buf_3_3_1 = aie.buffer(%tile_3_3) {sym_name = "bd_buf_3_3_1"} : memref<1xi32> 
    %bd_buf_3_3_0 = aie.buffer(%tile_3_3) {sym_name = "bd_buf_3_3_0"} : memref<1xi32> 
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
    ^bb1:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
      aie.dma_bd(%bd_buf_3_3_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_3_3_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_3_3_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_3_3_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_3_3_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_3_3_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // pred: ^bb0
      aie.end
    }
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<North : 1, DMA : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<North : 3, DMA : 2>
      aie.connect<North : 2, DMA : 3>
      aie.connect<DMA : 2, South : 2>
    }
    %bd_buf_1_1_31 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_31"} : memref<1xi32> 
    %bd_buf_1_1_30 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_30"} : memref<1xi32> 
    %bd_buf_1_1_29 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_29"} : memref<1xi32> 
    %bd_buf_1_1_28 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_28"} : memref<1xi32> 
    %bd_buf_1_1_27 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_27"} : memref<1xi32> 
    %bd_buf_1_1_26 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_26"} : memref<1xi32> 
    %bd_buf_1_1_25 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_25"} : memref<1xi32> 
    %bd_buf_1_1_24 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_24"} : memref<1xi32> 
    %bd_buf_1_1_15 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_15"} : memref<1xi32> 
    %bd_buf_1_1_14 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_14"} : memref<1xi32> 
    %bd_buf_1_1_13 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_13"} : memref<1xi32> 
    %bd_buf_1_1_12 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_12"} : memref<1xi32> 
    %bd_buf_1_1_11 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_11"} : memref<1xi32> 
    %bd_buf_1_1_10 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_10"} : memref<1xi32> 
    %bd_buf_1_1_9 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_9"} : memref<1xi32> 
    %bd_buf_1_1_8 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_8"} : memref<1xi32> 
    %bd_buf_1_1_7 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_7"} : memref<1xi32> 
    %bd_buf_1_1_6 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_6"} : memref<1xi32> 
    %bd_buf_1_1_5 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_5"} : memref<1xi32> 
    %bd_buf_1_1_4 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_4"} : memref<1xi32> 
    %bd_buf_1_1_3 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_3"} : memref<1xi32> 
    %bd_buf_1_1_2 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_2"} : memref<1xi32> 
    %bd_buf_1_1_1 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_1"} : memref<1xi32> 
    %bd_buf_1_1_0 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_0"} : memref<1xi32> 
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb25)
    ^bb1:  // 25 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8, ^bb9, ^bb10, ^bb11, ^bb12, ^bb13, ^bb14, ^bb15, ^bb16, ^bb17, ^bb18, ^bb19, ^bb20, ^bb21, ^bb22, ^bb23, ^bb24
      aie.dma_bd(%bd_buf_1_1_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_6 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb8:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_7 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb9:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_8 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb10:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_9 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb11:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_10 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb12:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_11 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb13:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_12 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb14:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_13 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb15:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_14 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb16:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_15 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb17:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_24 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb18:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_25 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb19:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_26 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb20:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_27 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb21:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_28 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb22:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_29 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb23:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_30 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb24:  // no predecessors
      aie.dma_bd(%bd_buf_1_1_31 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb25:  // pred: ^bb0
      aie.end
    }
    %tile_3_2 = aie.tile(3, 2)
    %bd_buf_3_2_5 = aie.buffer(%tile_3_2) {sym_name = "bd_buf_3_2_5"} : memref<1xi32> 
    %bd_buf_3_2_4 = aie.buffer(%tile_3_2) {sym_name = "bd_buf_3_2_4"} : memref<1xi32> 
    %bd_buf_3_2_3 = aie.buffer(%tile_3_2) {sym_name = "bd_buf_3_2_3"} : memref<1xi32> 
    %bd_buf_3_2_2 = aie.buffer(%tile_3_2) {sym_name = "bd_buf_3_2_2"} : memref<1xi32> 
    %bd_buf_3_2_1 = aie.buffer(%tile_3_2) {sym_name = "bd_buf_3_2_1"} : memref<1xi32> 
    %bd_buf_3_2_0 = aie.buffer(%tile_3_2) {sym_name = "bd_buf_3_2_0"} : memref<1xi32> 
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
    ^bb1:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
      aie.dma_bd(%bd_buf_3_2_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_3_2_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_3_2_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_3_2_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_3_2_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_3_2_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // pred: ^bb0
      aie.end
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<North : 1, DMA : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<North : 3, DMA : 2>
      aie.connect<North : 2, DMA : 5>
      aie.connect<DMA : 2, South : 2>
    }
    %bd_buf_0_1_31 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_31"} : memref<1xi32> 
    %bd_buf_0_1_30 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_30"} : memref<1xi32> 
    %bd_buf_0_1_29 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_29"} : memref<1xi32> 
    %bd_buf_0_1_28 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_28"} : memref<1xi32> 
    %bd_buf_0_1_27 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_27"} : memref<1xi32> 
    %bd_buf_0_1_26 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_26"} : memref<1xi32> 
    %bd_buf_0_1_25 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_25"} : memref<1xi32> 
    %bd_buf_0_1_24 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_24"} : memref<1xi32> 
    %bd_buf_0_1_15 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_15"} : memref<1xi32> 
    %bd_buf_0_1_14 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_14"} : memref<1xi32> 
    %bd_buf_0_1_13 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_13"} : memref<1xi32> 
    %bd_buf_0_1_12 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_12"} : memref<1xi32> 
    %bd_buf_0_1_11 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_11"} : memref<1xi32> 
    %bd_buf_0_1_10 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_10"} : memref<1xi32> 
    %bd_buf_0_1_9 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_9"} : memref<1xi32> 
    %bd_buf_0_1_8 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_8"} : memref<1xi32> 
    %bd_buf_0_1_7 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_7"} : memref<1xi32> 
    %bd_buf_0_1_6 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_6"} : memref<1xi32> 
    %bd_buf_0_1_5 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_5"} : memref<1xi32> 
    %bd_buf_0_1_4 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_4"} : memref<1xi32> 
    %bd_buf_0_1_3 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_3"} : memref<1xi32> 
    %bd_buf_0_1_2 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_2"} : memref<1xi32> 
    %bd_buf_0_1_1 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_1"} : memref<1xi32> 
    %bd_buf_0_1_0 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_0"} : memref<1xi32> 
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb25)
    ^bb1:  // 25 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8, ^bb9, ^bb10, ^bb11, ^bb12, ^bb13, ^bb14, ^bb15, ^bb16, ^bb17, ^bb18, ^bb19, ^bb20, ^bb21, ^bb22, ^bb23, ^bb24
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
      aie.dma_bd(%bd_buf_0_1_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_6 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb8:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_7 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb9:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_8 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb10:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_9 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb11:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_10 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb12:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_11 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb13:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_12 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb14:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_13 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb15:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_14 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb16:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_15 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb17:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_24 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb18:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_25 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb19:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_26 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb20:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_27 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb21:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_28 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb22:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_29 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb23:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_30 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb24:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_31 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb25:  // pred: ^bb0
      aie.end
    }
    %tile_1_2 = aie.tile(1, 2)
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<West : 2, East : 1>
      aie.connect<West : 2, DMA : 0>
      aie.connect<South : 1, East : 2>
      aie.connect<South : 1, North : 3>
      aie.connect<South : 1, West : 3>
      aie.connect<East : 3, North : 4>
      aie.connect<South : 5, North : 1>
      aie.connect<South : 5, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 2, South : 3>
    }
    %bd_buf_1_2_5 = aie.buffer(%tile_1_2) {sym_name = "bd_buf_1_2_5"} : memref<1xi32> 
    %bd_buf_1_2_4 = aie.buffer(%tile_1_2) {sym_name = "bd_buf_1_2_4"} : memref<1xi32> 
    %bd_buf_1_2_3 = aie.buffer(%tile_1_2) {sym_name = "bd_buf_1_2_3"} : memref<1xi32> 
    %bd_buf_1_2_2 = aie.buffer(%tile_1_2) {sym_name = "bd_buf_1_2_2"} : memref<1xi32> 
    %bd_buf_1_2_1 = aie.buffer(%tile_1_2) {sym_name = "bd_buf_1_2_1"} : memref<1xi32> 
    %bd_buf_1_2_0 = aie.buffer(%tile_1_2) {sym_name = "bd_buf_1_2_0"} : memref<1xi32> 
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
    ^bb1:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
      aie.dma_bd(%bd_buf_1_2_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_1_2_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_1_2_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_1_2_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_1_2_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_1_2_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // pred: ^bb0
      aie.end
    }
    %tile_2_2 = aie.tile(2, 2)
    %bd_buf_2_2_5 = aie.buffer(%tile_2_2) {sym_name = "bd_buf_2_2_5"} : memref<1xi32> 
    %bd_buf_2_2_4 = aie.buffer(%tile_2_2) {sym_name = "bd_buf_2_2_4"} : memref<1xi32> 
    %bd_buf_2_2_3 = aie.buffer(%tile_2_2) {sym_name = "bd_buf_2_2_3"} : memref<1xi32> 
    %bd_buf_2_2_2 = aie.buffer(%tile_2_2) {sym_name = "bd_buf_2_2_2"} : memref<1xi32> 
    %bd_buf_2_2_1 = aie.buffer(%tile_2_2) {sym_name = "bd_buf_2_2_1"} : memref<1xi32> 
    %bd_buf_2_2_0 = aie.buffer(%tile_2_2) {sym_name = "bd_buf_2_2_0"} : memref<1xi32> 
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
    ^bb1:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
      aie.dma_bd(%bd_buf_2_2_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_2_2_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_2_2_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_2_2_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_2_2_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_2_2_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // pred: ^bb0
      aie.end
    }
    %tile_1_3 = aie.tile(1, 3)
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<East : 3, West : 3>
      aie.connect<South : 4, West : 2>
      aie.connect<South : 1, North : 3>
      aie.connect<South : 1, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 3, South : 2>
    }
    %bd_buf_1_3_5 = aie.buffer(%tile_1_3) {sym_name = "bd_buf_1_3_5"} : memref<1xi32> 
    %bd_buf_1_3_4 = aie.buffer(%tile_1_3) {sym_name = "bd_buf_1_3_4"} : memref<1xi32> 
    %bd_buf_1_3_3 = aie.buffer(%tile_1_3) {sym_name = "bd_buf_1_3_3"} : memref<1xi32> 
    %bd_buf_1_3_2 = aie.buffer(%tile_1_3) {sym_name = "bd_buf_1_3_2"} : memref<1xi32> 
    %bd_buf_1_3_1 = aie.buffer(%tile_1_3) {sym_name = "bd_buf_1_3_1"} : memref<1xi32> 
    %bd_buf_1_3_0 = aie.buffer(%tile_1_3) {sym_name = "bd_buf_1_3_0"} : memref<1xi32> 
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
    ^bb1:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
      aie.dma_bd(%bd_buf_1_3_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_1_3_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_1_3_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_1_3_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_1_3_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_1_3_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // pred: ^bb0
      aie.end
    }
    %tile_2_4 = aie.tile(2, 4)
    %bd_buf_2_4_5 = aie.buffer(%tile_2_4) {sym_name = "bd_buf_2_4_5"} : memref<1xi32> 
    %bd_buf_2_4_4 = aie.buffer(%tile_2_4) {sym_name = "bd_buf_2_4_4"} : memref<1xi32> 
    %bd_buf_2_4_3 = aie.buffer(%tile_2_4) {sym_name = "bd_buf_2_4_3"} : memref<1xi32> 
    %bd_buf_2_4_2 = aie.buffer(%tile_2_4) {sym_name = "bd_buf_2_4_2"} : memref<1xi32> 
    %bd_buf_2_4_1 = aie.buffer(%tile_2_4) {sym_name = "bd_buf_2_4_1"} : memref<1xi32> 
    %bd_buf_2_4_0 = aie.buffer(%tile_2_4) {sym_name = "bd_buf_2_4_0"} : memref<1xi32> 
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
    ^bb1:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
      aie.dma_bd(%bd_buf_2_4_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_2_4_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_2_4_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_2_4_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_2_4_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_2_4_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // pred: ^bb0
      aie.end
    }
    %tile_2_3 = aie.tile(2, 3)
    %bd_buf_2_3_5 = aie.buffer(%tile_2_3) {sym_name = "bd_buf_2_3_5"} : memref<1xi32> 
    %bd_buf_2_3_4 = aie.buffer(%tile_2_3) {sym_name = "bd_buf_2_3_4"} : memref<1xi32> 
    %bd_buf_2_3_3 = aie.buffer(%tile_2_3) {sym_name = "bd_buf_2_3_3"} : memref<1xi32> 
    %bd_buf_2_3_2 = aie.buffer(%tile_2_3) {sym_name = "bd_buf_2_3_2"} : memref<1xi32> 
    %bd_buf_2_3_1 = aie.buffer(%tile_2_3) {sym_name = "bd_buf_2_3_1"} : memref<1xi32> 
    %bd_buf_2_3_0 = aie.buffer(%tile_2_3) {sym_name = "bd_buf_2_3_0"} : memref<1xi32> 
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
    ^bb1:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
      aie.dma_bd(%bd_buf_2_3_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_2_3_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_2_3_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_2_3_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_2_3_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_2_3_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // pred: ^bb0
      aie.end
    }
    %tile_3_4 = aie.tile(3, 4)
    %bd_buf_3_4_5 = aie.buffer(%tile_3_4) {sym_name = "bd_buf_3_4_5"} : memref<1xi32> 
    %bd_buf_3_4_4 = aie.buffer(%tile_3_4) {sym_name = "bd_buf_3_4_4"} : memref<1xi32> 
    %bd_buf_3_4_3 = aie.buffer(%tile_3_4) {sym_name = "bd_buf_3_4_3"} : memref<1xi32> 
    %bd_buf_3_4_2 = aie.buffer(%tile_3_4) {sym_name = "bd_buf_3_4_2"} : memref<1xi32> 
    %bd_buf_3_4_1 = aie.buffer(%tile_3_4) {sym_name = "bd_buf_3_4_1"} : memref<1xi32> 
    %bd_buf_3_4_0 = aie.buffer(%tile_3_4) {sym_name = "bd_buf_3_4_0"} : memref<1xi32> 
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
    ^bb1:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
      aie.dma_bd(%bd_buf_3_4_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_3_4_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_3_4_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_3_4_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_3_4_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_3_4_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // pred: ^bb0
      aie.end
    }
    %tile_1_5 = aie.tile(1, 5)
    %switchbox_1_5 = aie.switchbox(%tile_1_5) {
      aie.connect<East : 2, DMA : 0>
      aie.connect<South : 5, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %bd_buf_1_5_5 = aie.buffer(%tile_1_5) {sym_name = "bd_buf_1_5_5"} : memref<1xi32> 
    %bd_buf_1_5_4 = aie.buffer(%tile_1_5) {sym_name = "bd_buf_1_5_4"} : memref<1xi32> 
    %bd_buf_1_5_3 = aie.buffer(%tile_1_5) {sym_name = "bd_buf_1_5_3"} : memref<1xi32> 
    %bd_buf_1_5_2 = aie.buffer(%tile_1_5) {sym_name = "bd_buf_1_5_2"} : memref<1xi32> 
    %bd_buf_1_5_1 = aie.buffer(%tile_1_5) {sym_name = "bd_buf_1_5_1"} : memref<1xi32> 
    %bd_buf_1_5_0 = aie.buffer(%tile_1_5) {sym_name = "bd_buf_1_5_0"} : memref<1xi32> 
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
    ^bb1:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
      aie.dma_bd(%bd_buf_1_5_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_1_5_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_1_5_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_1_5_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_1_5_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_1_5_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // pred: ^bb0
      aie.end
    }
    %tile_1_4 = aie.tile(1, 4)
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<East : 0, DMA : 0>
      aie.connect<South : 3, North : 5>
      aie.connect<South : 3, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 3>
    }
    %bd_buf_1_4_5 = aie.buffer(%tile_1_4) {sym_name = "bd_buf_1_4_5"} : memref<1xi32> 
    %bd_buf_1_4_4 = aie.buffer(%tile_1_4) {sym_name = "bd_buf_1_4_4"} : memref<1xi32> 
    %bd_buf_1_4_3 = aie.buffer(%tile_1_4) {sym_name = "bd_buf_1_4_3"} : memref<1xi32> 
    %bd_buf_1_4_2 = aie.buffer(%tile_1_4) {sym_name = "bd_buf_1_4_2"} : memref<1xi32> 
    %bd_buf_1_4_1 = aie.buffer(%tile_1_4) {sym_name = "bd_buf_1_4_1"} : memref<1xi32> 
    %bd_buf_1_4_0 = aie.buffer(%tile_1_4) {sym_name = "bd_buf_1_4_0"} : memref<1xi32> 
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
    ^bb1:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
      aie.dma_bd(%bd_buf_1_4_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_1_4_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_1_4_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_1_4_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_1_4_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_1_4_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // pred: ^bb0
      aie.end
    }
    %tile_2_5 = aie.tile(2, 5)
    %bd_buf_2_5_5 = aie.buffer(%tile_2_5) {sym_name = "bd_buf_2_5_5"} : memref<1xi32> 
    %bd_buf_2_5_4 = aie.buffer(%tile_2_5) {sym_name = "bd_buf_2_5_4"} : memref<1xi32> 
    %bd_buf_2_5_3 = aie.buffer(%tile_2_5) {sym_name = "bd_buf_2_5_3"} : memref<1xi32> 
    %bd_buf_2_5_2 = aie.buffer(%tile_2_5) {sym_name = "bd_buf_2_5_2"} : memref<1xi32> 
    %bd_buf_2_5_1 = aie.buffer(%tile_2_5) {sym_name = "bd_buf_2_5_1"} : memref<1xi32> 
    %bd_buf_2_5_0 = aie.buffer(%tile_2_5) {sym_name = "bd_buf_2_5_0"} : memref<1xi32> 
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
    ^bb1:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
      aie.dma_bd(%bd_buf_2_5_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_2_5_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_2_5_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_2_5_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_2_5_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_2_5_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // pred: ^bb0
      aie.end
    }
    %tile_3_5 = aie.tile(3, 5)
    %bd_buf_3_5_5 = aie.buffer(%tile_3_5) {sym_name = "bd_buf_3_5_5"} : memref<1xi32> 
    %bd_buf_3_5_4 = aie.buffer(%tile_3_5) {sym_name = "bd_buf_3_5_4"} : memref<1xi32> 
    %bd_buf_3_5_3 = aie.buffer(%tile_3_5) {sym_name = "bd_buf_3_5_3"} : memref<1xi32> 
    %bd_buf_3_5_2 = aie.buffer(%tile_3_5) {sym_name = "bd_buf_3_5_2"} : memref<1xi32> 
    %bd_buf_3_5_1 = aie.buffer(%tile_3_5) {sym_name = "bd_buf_3_5_1"} : memref<1xi32> 
    %bd_buf_3_5_0 = aie.buffer(%tile_3_5) {sym_name = "bd_buf_3_5_0"} : memref<1xi32> 
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
    ^bb1:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
      aie.dma_bd(%bd_buf_3_5_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_3_5_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_3_5_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_3_5_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_3_5_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_3_5_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // pred: ^bb0
      aie.end
    }
    %mem_tile_3_1 = aie.tile(3, 1)
    %bd_buf_3_1_31 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_31"} : memref<1xi32> 
    %bd_buf_3_1_30 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_30"} : memref<1xi32> 
    %bd_buf_3_1_29 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_29"} : memref<1xi32> 
    %bd_buf_3_1_28 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_28"} : memref<1xi32> 
    %bd_buf_3_1_27 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_27"} : memref<1xi32> 
    %bd_buf_3_1_26 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_26"} : memref<1xi32> 
    %bd_buf_3_1_25 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_25"} : memref<1xi32> 
    %bd_buf_3_1_24 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_24"} : memref<1xi32> 
    %bd_buf_3_1_15 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_15"} : memref<1xi32> 
    %bd_buf_3_1_14 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_14"} : memref<1xi32> 
    %bd_buf_3_1_13 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_13"} : memref<1xi32> 
    %bd_buf_3_1_12 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_12"} : memref<1xi32> 
    %bd_buf_3_1_11 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_11"} : memref<1xi32> 
    %bd_buf_3_1_10 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_10"} : memref<1xi32> 
    %bd_buf_3_1_9 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_9"} : memref<1xi32> 
    %bd_buf_3_1_8 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_8"} : memref<1xi32> 
    %bd_buf_3_1_7 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_7"} : memref<1xi32> 
    %bd_buf_3_1_6 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_6"} : memref<1xi32> 
    %bd_buf_3_1_5 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_5"} : memref<1xi32> 
    %bd_buf_3_1_4 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_4"} : memref<1xi32> 
    %bd_buf_3_1_3 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_3"} : memref<1xi32> 
    %bd_buf_3_1_2 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_2"} : memref<1xi32> 
    %bd_buf_3_1_1 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_1"} : memref<1xi32> 
    %bd_buf_3_1_0 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_0"} : memref<1xi32> 
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb25)
    ^bb1:  // 25 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8, ^bb9, ^bb10, ^bb11, ^bb12, ^bb13, ^bb14, ^bb15, ^bb16, ^bb17, ^bb18, ^bb19, ^bb20, ^bb21, ^bb22, ^bb23, ^bb24
      aie.dma_bd(%bd_buf_3_1_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_6 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb8:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_7 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb9:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_8 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb10:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_9 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb11:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_10 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb12:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_11 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb13:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_12 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb14:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_13 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb15:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_14 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb16:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_15 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb17:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_24 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb18:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_25 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb19:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_26 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb20:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_27 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb21:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_28 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb22:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_29 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb23:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_30 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb24:  // no predecessors
      aie.dma_bd(%bd_buf_3_1_31 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb25:  // pred: ^bb0
      aie.end
    }
    %mem_tile_2_1 = aie.tile(2, 1)
    %bd_buf_2_1_31 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_31"} : memref<1xi32> 
    %bd_buf_2_1_30 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_30"} : memref<1xi32> 
    %bd_buf_2_1_29 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_29"} : memref<1xi32> 
    %bd_buf_2_1_28 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_28"} : memref<1xi32> 
    %bd_buf_2_1_27 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_27"} : memref<1xi32> 
    %bd_buf_2_1_26 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_26"} : memref<1xi32> 
    %bd_buf_2_1_25 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_25"} : memref<1xi32> 
    %bd_buf_2_1_24 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_24"} : memref<1xi32> 
    %bd_buf_2_1_15 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_15"} : memref<1xi32> 
    %bd_buf_2_1_14 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_14"} : memref<1xi32> 
    %bd_buf_2_1_13 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_13"} : memref<1xi32> 
    %bd_buf_2_1_12 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_12"} : memref<1xi32> 
    %bd_buf_2_1_11 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_11"} : memref<1xi32> 
    %bd_buf_2_1_10 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_10"} : memref<1xi32> 
    %bd_buf_2_1_9 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_9"} : memref<1xi32> 
    %bd_buf_2_1_8 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_8"} : memref<1xi32> 
    %bd_buf_2_1_7 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_7"} : memref<1xi32> 
    %bd_buf_2_1_6 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_6"} : memref<1xi32> 
    %bd_buf_2_1_5 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_5"} : memref<1xi32> 
    %bd_buf_2_1_4 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_4"} : memref<1xi32> 
    %bd_buf_2_1_3 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_3"} : memref<1xi32> 
    %bd_buf_2_1_2 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_2"} : memref<1xi32> 
    %bd_buf_2_1_1 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_1"} : memref<1xi32> 
    %bd_buf_2_1_0 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_0"} : memref<1xi32> 
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb25)
    ^bb1:  // 25 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8, ^bb9, ^bb10, ^bb11, ^bb12, ^bb13, ^bb14, ^bb15, ^bb16, ^bb17, ^bb18, ^bb19, ^bb20, ^bb21, ^bb22, ^bb23, ^bb24
      aie.dma_bd(%bd_buf_2_1_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_6 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb8:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_7 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb9:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_8 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb10:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_9 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb11:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_10 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb12:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_11 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb13:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_12 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb14:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_13 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb15:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_14 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb16:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_15 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb17:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_24 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb18:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_25 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb19:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_26 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb20:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_27 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb21:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_28 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb22:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_29 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb23:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_30 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb24:  // no predecessors
      aie.dma_bd(%bd_buf_2_1_31 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb25:  // pred: ^bb0
      aie.end
    }
    %tile_0_5 = aie.tile(0, 5)
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<South : 5, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %bd_buf_0_5_5 = aie.buffer(%tile_0_5) {sym_name = "bd_buf_0_5_5"} : memref<1xi32> 
    %bd_buf_0_5_4 = aie.buffer(%tile_0_5) {sym_name = "bd_buf_0_5_4"} : memref<1xi32> 
    %bd_buf_0_5_3 = aie.buffer(%tile_0_5) {sym_name = "bd_buf_0_5_3"} : memref<1xi32> 
    %bd_buf_0_5_2 = aie.buffer(%tile_0_5) {sym_name = "bd_buf_0_5_2"} : memref<1xi32> 
    %bd_buf_0_5_1 = aie.buffer(%tile_0_5) {sym_name = "bd_buf_0_5_1"} : memref<1xi32> 
    %bd_buf_0_5_0 = aie.buffer(%tile_0_5) {sym_name = "bd_buf_0_5_0"} : memref<1xi32> 
    %lock_0_5 = aie.lock(%tile_0_5, 3) {init = 0 : i32}
    %lock_0_5_0 = aie.lock(%tile_0_5, 2) {init = 2 : i32}
    %lock_0_5_1 = aie.lock(%tile_0_5, 1) {init = 2 : i32}
    %lock_0_5_2 = aie.lock(%tile_0_5, 0) {init = 2 : i32}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
    ^bb1:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
      aie.dma_bd(%bd_buf_0_5_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_0_5_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_0_5_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_0_5_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_0_5_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_0_5_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // pred: ^bb0
      aie.end
    }
    %tile_0_4 = aie.tile(0, 4)
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, North : 3>
      aie.connect<South : 2, North : 5>
      aie.connect<South : 2, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 0>
    }
    %bd_buf_0_4_5 = aie.buffer(%tile_0_4) {sym_name = "bd_buf_0_4_5"} : memref<1xi32> 
    %bd_buf_0_4_4 = aie.buffer(%tile_0_4) {sym_name = "bd_buf_0_4_4"} : memref<1xi32> 
    %bd_buf_0_4_3 = aie.buffer(%tile_0_4) {sym_name = "bd_buf_0_4_3"} : memref<1xi32> 
    %bd_buf_0_4_2 = aie.buffer(%tile_0_4) {sym_name = "bd_buf_0_4_2"} : memref<1xi32> 
    %bd_buf_0_4_1 = aie.buffer(%tile_0_4) {sym_name = "bd_buf_0_4_1"} : memref<1xi32> 
    %bd_buf_0_4_0 = aie.buffer(%tile_0_4) {sym_name = "bd_buf_0_4_0"} : memref<1xi32> 
    %lock_0_4 = aie.lock(%tile_0_4, 3) {init = 0 : i32}
    %lock_0_4_3 = aie.lock(%tile_0_4, 2) {init = 2 : i32}
    %lock_0_4_4 = aie.lock(%tile_0_4, 1) {init = 2 : i32}
    %lock_0_4_5 = aie.lock(%tile_0_4, 0) {init = 2 : i32}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
    ^bb1:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
      aie.dma_bd(%bd_buf_0_4_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_0_4_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_0_4_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_0_4_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_0_4_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_0_4_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // pred: ^bb0
      aie.end
    }
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 5, DMA : 0>
      aie.connect<East : 3, North : 0>
      aie.connect<East : 2, North : 1>
      aie.connect<South : 3, North : 2>
      aie.connect<South : 3, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 0, South : 3>
    }
    %bd_buf_0_3_5 = aie.buffer(%tile_0_3) {sym_name = "bd_buf_0_3_5"} : memref<1xi32> 
    %bd_buf_0_3_4 = aie.buffer(%tile_0_3) {sym_name = "bd_buf_0_3_4"} : memref<1xi32> 
    %bd_buf_0_3_3 = aie.buffer(%tile_0_3) {sym_name = "bd_buf_0_3_3"} : memref<1xi32> 
    %bd_buf_0_3_2 = aie.buffer(%tile_0_3) {sym_name = "bd_buf_0_3_2"} : memref<1xi32> 
    %bd_buf_0_3_1 = aie.buffer(%tile_0_3) {sym_name = "bd_buf_0_3_1"} : memref<1xi32> 
    %bd_buf_0_3_0 = aie.buffer(%tile_0_3) {sym_name = "bd_buf_0_3_0"} : memref<1xi32> 
    %lock_0_3 = aie.lock(%tile_0_3, 3) {init = 0 : i32}
    %lock_0_3_6 = aie.lock(%tile_0_3, 2) {init = 2 : i32}
    %lock_0_3_7 = aie.lock(%tile_0_3, 1) {init = 2 : i32}
    %lock_0_3_8 = aie.lock(%tile_0_3, 0) {init = 2 : i32}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
    ^bb1:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
      aie.dma_bd(%bd_buf_0_3_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_0_3_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_0_3_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_0_3_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_0_3_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_0_3_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // pred: ^bb0
      aie.end
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, East : 2>
      aie.connect<South : 1, DMA : 0>
      aie.connect<East : 3, North : 5>
      aie.connect<South : 5, North : 3>
      aie.connect<South : 5, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 3>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 3, South : 0>
    }
    %bd_buf_0_2_5 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_5"} : memref<1xi32> 
    %bd_buf_0_2_4 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_4"} : memref<1xi32> 
    %bd_buf_0_2_3 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_3"} : memref<1xi32> 
    %bd_buf_0_2_2 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_2"} : memref<1xi32> 
    %bd_buf_0_2_1 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_1"} : memref<1xi32> 
    %bd_buf_0_2_0 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_0"} : memref<1xi32> 
    %lock_0_2 = aie.lock(%tile_0_2, 3) {init = 0 : i32}
    %lock_0_2_9 = aie.lock(%tile_0_2, 2) {init = 2 : i32}
    %lock_0_2_10 = aie.lock(%tile_0_2, 1) {init = 2 : i32}
    %lock_0_2_11 = aie.lock(%tile_0_2, 0) {init = 2 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
    ^bb1:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
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
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_0_2_4 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_0_2_5 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // pred: ^bb0
      aie.end
    }
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_mux_2_0 = aie.shim_mux(%shim_noc_tile_2_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 0, North : 7>
      aie.connect<North : 2, DMA : 0>
    }
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_mux_3_0 = aie.shim_mux(%shim_noc_tile_3_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 0, North : 7>
      aie.connect<North : 2, DMA : 0>
    }
    memref.global "private" constant @config_blockwrite_data_39 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33562623, 3145791, 33554432]>
    memref.global "private" constant @config_blockwrite_data_38 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33554447, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_37 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33562623, 3145791, 33554432]>
    memref.global "private" constant @config_blockwrite_data_36 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33554447, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_35 : memref<8xi32> = dense<[16384, 0, 0, 33554432, -939523585, 33554559, 1114111, 33554432]>
    memref.global "private" constant @config_blockwrite_data_34 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33562623, 3145791, 33554432]>
    memref.global "private" constant @config_blockwrite_data_33 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33554447, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_32 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33562623, 3145791, 33554432]>
    memref.global "private" constant @config_blockwrite_data_31 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33554447, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_30 : memref<8xi32> = dense<[16384, 0, 0, 33554432, -939523585, 33554559, 1114111, 33554432]>
    memref.global "private" constant @config_blockwrite_data_29 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33562623, 3145791, 33554432]>
    memref.global "private" constant @config_blockwrite_data_28 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33554447, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_27 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33562623, 3145791, 33554432]>
    memref.global "private" constant @config_blockwrite_data_26 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33554447, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_25 : memref<8xi32> = dense<[16384, 0, 0, 33554432, -939523585, 33554559, 1114111, 33554432]>
    memref.global "private" constant @config_blockwrite_data_24 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33562623, 3145791, 33554432]>
    memref.global "private" constant @config_blockwrite_data_23 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33554447, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_22 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33562623, 3145791, 33554432]>
    memref.global "private" constant @config_blockwrite_data_21 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33554447, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_20 : memref<8xi32> = dense<[16384, 0, 0, 33554432, -939523585, 33554559, 1114111, 33554432]>
    memref.global "private" constant @config_blockwrite_data_19 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33562623, 3145791, 33554432]>
    memref.global "private" constant @config_blockwrite_data_18 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33554447, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_17 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33562623, 3145791, 33554432]>
    memref.global "private" constant @config_blockwrite_data_16 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33554447, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_15 : memref<8xi32> = dense<[16384, 0, 0, 33554432, -939523585, 33554559, 1114111, 33554432]>
    memref.global "private" constant @config_blockwrite_data_14 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33562623, 3145791, 33554432]>
    memref.global "private" constant @config_blockwrite_data_13 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33554447, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_12 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33562623, 3145791, 33554432]>
    memref.global "private" constant @config_blockwrite_data_11 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33554447, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_10 : memref<8xi32> = dense<[16384, 0, 0, 33554432, -939523585, 33554559, 1114111, 33554432]>
    memref.global "private" constant @config_blockwrite_data_9 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33562623, 3145791, 33554432]>
    memref.global "private" constant @config_blockwrite_data_8 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33554447, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_7 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33562623, 3145791, 33554432]>
    memref.global "private" constant @config_blockwrite_data_6 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33554447, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_5 : memref<8xi32> = dense<[16384, 0, 0, 33554432, -939523585, 33554559, 1114111, 33554432]>
    memref.global "private" constant @config_blockwrite_data_4 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33562623, 3145791, 33554432]>
    memref.global "private" constant @config_blockwrite_data_3 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33554447, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_2 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33562623, 3145791, 33554432]>
    memref.global "private" constant @config_blockwrite_data_1 : memref<8xi32> = dense<[8192, 0, 0, 16777216, -1040187137, 33554447, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_0 : memref<8xi32> = dense<[16384, 0, 0, 33554432, -939523585, 33554559, 1114111, 33554432]>
    aie.runtime_sequence @configure() {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 16384 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 32 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 128 : i32, d1_stride = 511 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 127 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 1 : i32, iteration_stride = 65535 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(0, 0, S2MM : 0) {bd_id = 0 : i32, issue_token = true, repeat_count = 1 : i32}
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 15 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(0, 0, MM2S : 0) {bd_id = 1 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 2 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 8191 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 118852 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(0, 0, MM2S : 1) {bd_id = 2 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 3 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 15 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 118884 : ui32, arg_idx = 0 : i32, arg_plus = 131072 : i32}
      aiex.npu.push_queue(0, 0, MM2S : 0) {bd_id = 3 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 4 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 8191 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 118916 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(0, 0, MM2S : 1) {bd_id = 4 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 16384 : i32, buffer_offset = 0 : i32, column = 1 : i32, d0_size = 32 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 128 : i32, d1_stride = 511 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 127 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 1 : i32, iteration_stride = 65535 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 33673220 : ui32, arg_idx = 2 : i32, arg_plus = 128 : i32}
      aiex.npu.push_queue(1, 0, S2MM : 0) {bd_id = 0 : i32, issue_token = true, repeat_count = 1 : i32}
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 1 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 15 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 33673252 : ui32, arg_idx = 0 : i32, arg_plus = 32768 : i32}
      aiex.npu.push_queue(1, 0, MM2S : 0) {bd_id = 1 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 2 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 1 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 8191 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 33673284 : ui32, arg_idx = 1 : i32, arg_plus = 64 : i32}
      aiex.npu.push_queue(1, 0, MM2S : 1) {bd_id = 2 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 3 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 1 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 15 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 33673316 : ui32, arg_idx = 0 : i32, arg_plus = 163840 : i32}
      aiex.npu.push_queue(1, 0, MM2S : 0) {bd_id = 3 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 4 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 1 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 8191 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 33673348 : ui32, arg_idx = 1 : i32, arg_plus = 64 : i32}
      aiex.npu.push_queue(1, 0, MM2S : 1) {bd_id = 4 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 16384 : i32, buffer_offset = 0 : i32, column = 2 : i32, d0_size = 32 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 128 : i32, d1_stride = 511 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 127 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 1 : i32, iteration_stride = 65535 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67227652 : ui32, arg_idx = 2 : i32, arg_plus = 256 : i32}
      aiex.npu.push_queue(2, 0, S2MM : 0) {bd_id = 0 : i32, issue_token = true, repeat_count = 1 : i32}
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 2 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 15 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67227684 : ui32, arg_idx = 0 : i32, arg_plus = 65536 : i32}
      aiex.npu.push_queue(2, 0, MM2S : 0) {bd_id = 1 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 2 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 2 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 8191 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67227716 : ui32, arg_idx = 1 : i32, arg_plus = 128 : i32}
      aiex.npu.push_queue(2, 0, MM2S : 1) {bd_id = 2 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 3 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 2 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 15 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67227748 : ui32, arg_idx = 0 : i32, arg_plus = 196608 : i32}
      aiex.npu.push_queue(2, 0, MM2S : 0) {bd_id = 3 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 4 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 2 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 8191 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67227780 : ui32, arg_idx = 1 : i32, arg_plus = 128 : i32}
      aiex.npu.push_queue(2, 0, MM2S : 1) {bd_id = 4 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 16384 : i32, buffer_offset = 0 : i32, column = 3 : i32, d0_size = 32 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 128 : i32, d1_stride = 511 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 127 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 1 : i32, iteration_stride = 65535 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 100782084 : ui32, arg_idx = 2 : i32, arg_plus = 384 : i32}
      aiex.npu.push_queue(3, 0, S2MM : 0) {bd_id = 0 : i32, issue_token = true, repeat_count = 1 : i32}
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 3 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 15 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 100782116 : ui32, arg_idx = 0 : i32, arg_plus = 98304 : i32}
      aiex.npu.push_queue(3, 0, MM2S : 0) {bd_id = 1 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 2 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 3 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 8191 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 100782148 : ui32, arg_idx = 1 : i32, arg_plus = 192 : i32}
      aiex.npu.push_queue(3, 0, MM2S : 1) {bd_id = 2 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 3 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 3 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 15 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 100782180 : ui32, arg_idx = 0 : i32, arg_plus = 229376 : i32}
      aiex.npu.push_queue(3, 0, MM2S : 0) {bd_id = 3 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 4 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 3 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 8191 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 100782212 : ui32, arg_idx = 1 : i32, arg_plus = 192 : i32}
      aiex.npu.push_queue(3, 0, MM2S : 1) {bd_id = 4 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 8 : i32, buffer_length = 16384 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 32 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 128 : i32, d1_stride = 511 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 127 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 1 : i32, iteration_stride = 65535 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 119044 : ui32, arg_idx = 2 : i32, arg_plus = 524288 : i32}
      aiex.npu.push_queue(0, 0, S2MM : 0) {bd_id = 8 : i32, issue_token = true, repeat_count = 1 : i32}
      aiex.npu.writebd {bd_id = 9 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 15 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 119076 : ui32, arg_idx = 0 : i32, arg_plus = 262144 : i32}
      aiex.npu.push_queue(0, 0, MM2S : 0) {bd_id = 9 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 10 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 8191 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 119108 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(0, 0, MM2S : 1) {bd_id = 10 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 11 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 15 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 119140 : ui32, arg_idx = 0 : i32, arg_plus = 393216 : i32}
      aiex.npu.push_queue(0, 0, MM2S : 0) {bd_id = 11 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 12 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 8191 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 119172 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(0, 0, MM2S : 1) {bd_id = 12 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 8 : i32, buffer_length = 16384 : i32, buffer_offset = 0 : i32, column = 1 : i32, d0_size = 32 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 128 : i32, d1_stride = 511 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 127 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 1 : i32, iteration_stride = 65535 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 33673476 : ui32, arg_idx = 2 : i32, arg_plus = 524416 : i32}
      aiex.npu.push_queue(1, 0, S2MM : 0) {bd_id = 8 : i32, issue_token = true, repeat_count = 1 : i32}
      aiex.npu.writebd {bd_id = 9 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 1 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 15 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 33673508 : ui32, arg_idx = 0 : i32, arg_plus = 294912 : i32}
      aiex.npu.push_queue(1, 0, MM2S : 0) {bd_id = 9 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 10 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 1 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 8191 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 33673540 : ui32, arg_idx = 1 : i32, arg_plus = 64 : i32}
      aiex.npu.push_queue(1, 0, MM2S : 1) {bd_id = 10 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 11 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 1 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 15 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 33673572 : ui32, arg_idx = 0 : i32, arg_plus = 425984 : i32}
      aiex.npu.push_queue(1, 0, MM2S : 0) {bd_id = 11 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 12 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 1 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 8191 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 33673604 : ui32, arg_idx = 1 : i32, arg_plus = 64 : i32}
      aiex.npu.push_queue(1, 0, MM2S : 1) {bd_id = 12 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 8 : i32, buffer_length = 16384 : i32, buffer_offset = 0 : i32, column = 2 : i32, d0_size = 32 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 128 : i32, d1_stride = 511 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 127 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 1 : i32, iteration_stride = 65535 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67227908 : ui32, arg_idx = 2 : i32, arg_plus = 524544 : i32}
      aiex.npu.push_queue(2, 0, S2MM : 0) {bd_id = 8 : i32, issue_token = true, repeat_count = 1 : i32}
      aiex.npu.writebd {bd_id = 9 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 2 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 15 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67227940 : ui32, arg_idx = 0 : i32, arg_plus = 327680 : i32}
      aiex.npu.push_queue(2, 0, MM2S : 0) {bd_id = 9 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 10 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 2 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 8191 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67227972 : ui32, arg_idx = 1 : i32, arg_plus = 128 : i32}
      aiex.npu.push_queue(2, 0, MM2S : 1) {bd_id = 10 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 11 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 2 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 15 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67228004 : ui32, arg_idx = 0 : i32, arg_plus = 458752 : i32}
      aiex.npu.push_queue(2, 0, MM2S : 0) {bd_id = 11 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 12 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 2 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 8191 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67228036 : ui32, arg_idx = 1 : i32, arg_plus = 128 : i32}
      aiex.npu.push_queue(2, 0, MM2S : 1) {bd_id = 12 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 8 : i32, buffer_length = 16384 : i32, buffer_offset = 0 : i32, column = 3 : i32, d0_size = 32 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 128 : i32, d1_stride = 511 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 127 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 1 : i32, iteration_stride = 65535 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 100782340 : ui32, arg_idx = 2 : i32, arg_plus = 524672 : i32}
      aiex.npu.push_queue(3, 0, S2MM : 0) {bd_id = 8 : i32, issue_token = true, repeat_count = 1 : i32}
      aiex.npu.writebd {bd_id = 9 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 3 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 15 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 100782372 : ui32, arg_idx = 0 : i32, arg_plus = 360448 : i32}
      aiex.npu.push_queue(3, 0, MM2S : 0) {bd_id = 9 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 10 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 3 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 8191 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 100782404 : ui32, arg_idx = 1 : i32, arg_plus = 192 : i32}
      aiex.npu.push_queue(3, 0, MM2S : 1) {bd_id = 10 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 11 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 3 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 15 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 100782436 : ui32, arg_idx = 0 : i32, arg_plus = 491520 : i32}
      aiex.npu.push_queue(3, 0, MM2S : 0) {bd_id = 11 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.writebd {bd_id = 12 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 3 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 255 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 8191 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 100782468 : ui32, arg_idx = 1 : i32, arg_plus = 192 : i32}
      aiex.npu.push_queue(3, 0, MM2S : 1) {bd_id = 12 : i32, issue_token = false, repeat_count = 3 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 2 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 3 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 2 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 3 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aie.end
    }
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {elf_file = "core_0_2.elf"}
    %core_0_3 = aie.core(%tile_0_3) {
      aie.end
    } {elf_file = "core_0_3.elf"}
    %core_0_4 = aie.core(%tile_0_4) {
      aie.end
    } {elf_file = "core_0_4.elf"}
    %core_0_5 = aie.core(%tile_0_5) {
      aie.end
    } {elf_file = "core_0_5.elf"}
    %core_1_2 = aie.core(%tile_1_2) {
      aie.end
    } {elf_file = "core_1_2.elf"}
    %core_1_3 = aie.core(%tile_1_3) {
      aie.end
    } {elf_file = "core_1_3.elf"}
    %core_1_4 = aie.core(%tile_1_4) {
      aie.end
    } {elf_file = "core_1_4.elf"}
    %core_1_5 = aie.core(%tile_1_5) {
      aie.end
    } {elf_file = "core_1_5.elf"}
    %core_2_2 = aie.core(%tile_2_2) {
      aie.end
    } {elf_file = "core_2_2.elf"}
    %core_2_3 = aie.core(%tile_2_3) {
      aie.end
    } {elf_file = "core_2_3.elf"}
    %core_2_4 = aie.core(%tile_2_4) {
      aie.end
    } {elf_file = "core_2_4.elf"}
    %core_2_5 = aie.core(%tile_2_5) {
      aie.end
    } {elf_file = "core_2_5.elf"}
    %core_3_2 = aie.core(%tile_3_2) {
      aie.end
    } {elf_file = "core_3_2.elf"}
    %core_3_3 = aie.core(%tile_3_3) {
      aie.end
    } {elf_file = "core_3_3.elf"}
    %core_3_4 = aie.core(%tile_3_4) {
      aie.end
    } {elf_file = "core_3_4.elf"}
    %core_3_5 = aie.core(%tile_3_5) {
      aie.end
    } {elf_file = "core_3_5.elf"}
  }
}
