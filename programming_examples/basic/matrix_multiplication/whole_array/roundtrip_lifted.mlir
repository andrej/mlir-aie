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
    %bd_buf_3_3_0 = aie.buffer(%tile_3_3) {sym_name = "bd_buf_3_3_0"} : memref<1xi32> 
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_3_3_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<North : 3, DMA : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<DMA : 2, South : 2>
    }
    %bd_buf_1_1_0 = aie.buffer(%mem_tile_1_1) {sym_name = "bd_buf_1_1_0"} : memref<1xi32> 
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_1_1_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_3_2 = aie.tile(3, 2)
    %bd_buf_3_2_0 = aie.buffer(%tile_3_2) {sym_name = "bd_buf_3_2_0"} : memref<1xi32> 
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_3_2_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<North : 3, DMA : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<DMA : 2, South : 2>
    }
    %bd_buf_0_1_0 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_0"} : memref<1xi32> 
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_0_1_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_1_2 = aie.tile(1, 2)
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<West : 2, East : 1>
      aie.connect<West : 2, DMA : 0>
      aie.connect<South : 1, East : 2>
      aie.connect<South : 1, North : 3>
      aie.connect<South : 1, West : 3>
      aie.connect<East : 3, West : 1>
      aie.connect<East : 2, West : 2>
      aie.connect<South : 5, North : 1>
      aie.connect<South : 5, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 3, South : 3>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 2, South : 0>
    }
    %bd_buf_1_2_0 = aie.buffer(%tile_1_2) {sym_name = "bd_buf_1_2_0"} : memref<1xi32> 
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_1_2_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_2_2 = aie.tile(2, 2)
    %bd_buf_2_2_0 = aie.buffer(%tile_2_2) {sym_name = "bd_buf_2_2_0"} : memref<1xi32> 
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_2_2_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_1_3 = aie.tile(1, 3)
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<East : 0, North : 4>
      aie.connect<East : 3, North : 5>
      aie.connect<South : 1, North : 3>
      aie.connect<South : 1, DMA : 1>
      aie.connect<DMA : 0, South : 3>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 3, South : 2>
    }
    %bd_buf_1_3_0 = aie.buffer(%tile_1_3) {sym_name = "bd_buf_1_3_0"} : memref<1xi32> 
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_1_3_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_2_4 = aie.tile(2, 4)
    %bd_buf_2_4_0 = aie.buffer(%tile_2_4) {sym_name = "bd_buf_2_4_0"} : memref<1xi32> 
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_2_4_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_2_3 = aie.tile(2, 3)
    %bd_buf_2_3_0 = aie.buffer(%tile_2_3) {sym_name = "bd_buf_2_3_0"} : memref<1xi32> 
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_2_3_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_3_4 = aie.tile(3, 4)
    %bd_buf_3_4_0 = aie.buffer(%tile_3_4) {sym_name = "bd_buf_3_4_0"} : memref<1xi32> 
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_3_4_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_1_5 = aie.tile(1, 5)
    %switchbox_1_5 = aie.switchbox(%tile_1_5) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<South : 5, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %bd_buf_1_5_0 = aie.buffer(%tile_1_5) {sym_name = "bd_buf_1_5_0"} : memref<1xi32> 
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_1_5_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_1_4 = aie.tile(1, 4)
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<South : 4, DMA : 0>
      aie.connect<South : 5, North : 3>
      aie.connect<South : 3, North : 5>
      aie.connect<South : 3, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 3>
    }
    %bd_buf_1_4_0 = aie.buffer(%tile_1_4) {sym_name = "bd_buf_1_4_0"} : memref<1xi32> 
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_1_4_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_2_5 = aie.tile(2, 5)
    %bd_buf_2_5_0 = aie.buffer(%tile_2_5) {sym_name = "bd_buf_2_5_0"} : memref<1xi32> 
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_2_5_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_3_5 = aie.tile(3, 5)
    %bd_buf_3_5_0 = aie.buffer(%tile_3_5) {sym_name = "bd_buf_3_5_0"} : memref<1xi32> 
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_3_5_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %mem_tile_3_1 = aie.tile(3, 1)
    %bd_buf_3_1_0 = aie.buffer(%mem_tile_3_1) {sym_name = "bd_buf_3_1_0"} : memref<1xi32> 
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_3_1_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %mem_tile_2_1 = aie.tile(2, 1)
    %bd_buf_2_1_0 = aie.buffer(%mem_tile_2_1) {sym_name = "bd_buf_2_1_0"} : memref<1xi32> 
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_2_1_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_0_5 = aie.tile(0, 5)
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<South : 0, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %bd_buf_0_5_0 = aie.buffer(%tile_0_5) {sym_name = "bd_buf_0_5_0"} : memref<1xi32> 
    %lock_0_5 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_0_5_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_0_4 = aie.tile(0, 4)
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<South : 5, DMA : 0>
      aie.connect<South : 2, North : 3>
      aie.connect<South : 1, North : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 0>
    }
    %bd_buf_0_4_0 = aie.buffer(%tile_0_4) {sym_name = "bd_buf_0_4_0"} : memref<1xi32> 
    %lock_0_4 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_0_4_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 5, DMA : 0>
      aie.connect<South : 0, North : 5>
      aie.connect<South : 3, North : 2>
      aie.connect<South : 2, North : 1>
      aie.connect<South : 2, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 0, South : 3>
    }
    %bd_buf_0_3_0 = aie.buffer(%tile_0_3) {sym_name = "bd_buf_0_3_0"} : memref<1xi32> 
    %lock_0_3 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_0_3_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, East : 2>
      aie.connect<South : 1, DMA : 0>
      aie.connect<East : 3, North : 5>
      aie.connect<East : 1, North : 0>
      aie.connect<East : 2, North : 3>
      aie.connect<South : 5, North : 2>
      aie.connect<South : 5, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 0, South : 3>
      aie.connect<North : 3, South : 2>
    }
    %bd_buf_0_2_0 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_0"} : memref<1xi32> 
    %lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_0_2_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
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
    aie.runtime_sequence @configure() {
      aiex.npu.write32 {address = 103931396 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 103931392 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 102882820 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102882816 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 35773956 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35773952 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 69328388 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69328384 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 36822532 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 36822528 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 71425540 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 71425536 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 70376964 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 70376960 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 104979972 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 104979968 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 38919684 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 38919680 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 37871108 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 37871104 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 72474116 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 72474112 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 106028548 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 106028544 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 35259908 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35259904 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 102368772 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102368768 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 68814340 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 68814336 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 33812992 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 67367424 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 100921856 : ui32, value = 0 : ui32}
      aie.end
    }
  }
}
