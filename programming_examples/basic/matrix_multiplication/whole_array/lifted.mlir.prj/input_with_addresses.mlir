module {
  aie.device(npu1) @xclbin_device {
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
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
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
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
    %tile_3_3 = aie.tile(3, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %bd_buf_3_3_5 = aie.buffer(%tile_3_3) {address = 0 : i32, sym_name = "bd_buf_3_3_5"} : memref<1xi32> 
    %bd_buf_3_3_4 = aie.buffer(%tile_3_3) {address = 4 : i32, sym_name = "bd_buf_3_3_4"} : memref<1xi32> 
    %bd_buf_3_3_3 = aie.buffer(%tile_3_3) {address = 8 : i32, sym_name = "bd_buf_3_3_3"} : memref<1xi32> 
    %bd_buf_3_3_2 = aie.buffer(%tile_3_3) {address = 12 : i32, sym_name = "bd_buf_3_3_2"} : memref<1xi32> 
    %bd_buf_3_3_1 = aie.buffer(%tile_3_3) {address = 16 : i32, sym_name = "bd_buf_3_3_1"} : memref<1xi32> 
    %bd_buf_3_3_0 = aie.buffer(%tile_3_3) {address = 20 : i32, sym_name = "bd_buf_3_3_0"} : memref<1xi32> 
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_3_3_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<North : 1, DMA : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<North : 3, DMA : 2>
      aie.connect<North : 2, DMA : 5>
      aie.connect<DMA : 2, South : 2>
    }
    %bd_buf_1_1_31 = aie.buffer(%mem_tile_1_1) {address = 48 : i32, sym_name = "bd_buf_1_1_31"} : memref<1xi32> 
    %bd_buf_1_1_30 = aie.buffer(%mem_tile_1_1) {address = 92 : i32, sym_name = "bd_buf_1_1_30"} : memref<1xi32> 
    %bd_buf_1_1_29 = aie.buffer(%mem_tile_1_1) {address = 88 : i32, sym_name = "bd_buf_1_1_29"} : memref<1xi32> 
    %bd_buf_1_1_28 = aie.buffer(%mem_tile_1_1) {address = 84 : i32, sym_name = "bd_buf_1_1_28"} : memref<1xi32> 
    %bd_buf_1_1_27 = aie.buffer(%mem_tile_1_1) {address = 80 : i32, sym_name = "bd_buf_1_1_27"} : memref<1xi32> 
    %bd_buf_1_1_26 = aie.buffer(%mem_tile_1_1) {address = 76 : i32, sym_name = "bd_buf_1_1_26"} : memref<1xi32> 
    %bd_buf_1_1_25 = aie.buffer(%mem_tile_1_1) {address = 72 : i32, sym_name = "bd_buf_1_1_25"} : memref<1xi32> 
    %bd_buf_1_1_24 = aie.buffer(%mem_tile_1_1) {address = 68 : i32, sym_name = "bd_buf_1_1_24"} : memref<1xi32> 
    %bd_buf_1_1_15 = aie.buffer(%mem_tile_1_1) {address = 64 : i32, sym_name = "bd_buf_1_1_15"} : memref<1xi32> 
    %bd_buf_1_1_14 = aie.buffer(%mem_tile_1_1) {address = 60 : i32, sym_name = "bd_buf_1_1_14"} : memref<1xi32> 
    %bd_buf_1_1_13 = aie.buffer(%mem_tile_1_1) {address = 56 : i32, sym_name = "bd_buf_1_1_13"} : memref<1xi32> 
    %bd_buf_1_1_12 = aie.buffer(%mem_tile_1_1) {address = 52 : i32, sym_name = "bd_buf_1_1_12"} : memref<1xi32> 
    %bd_buf_1_1_11 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, sym_name = "bd_buf_1_1_11"} : memref<1xi32> 
    %bd_buf_1_1_10 = aie.buffer(%mem_tile_1_1) {address = 44 : i32, sym_name = "bd_buf_1_1_10"} : memref<1xi32> 
    %bd_buf_1_1_9 = aie.buffer(%mem_tile_1_1) {address = 40 : i32, sym_name = "bd_buf_1_1_9"} : memref<1xi32> 
    %bd_buf_1_1_8 = aie.buffer(%mem_tile_1_1) {address = 36 : i32, sym_name = "bd_buf_1_1_8"} : memref<1xi32> 
    %bd_buf_1_1_7 = aie.buffer(%mem_tile_1_1) {address = 32 : i32, sym_name = "bd_buf_1_1_7"} : memref<1xi32> 
    %bd_buf_1_1_6 = aie.buffer(%mem_tile_1_1) {address = 28 : i32, sym_name = "bd_buf_1_1_6"} : memref<1xi32> 
    %bd_buf_1_1_5 = aie.buffer(%mem_tile_1_1) {address = 24 : i32, sym_name = "bd_buf_1_1_5"} : memref<1xi32> 
    %bd_buf_1_1_4 = aie.buffer(%mem_tile_1_1) {address = 20 : i32, sym_name = "bd_buf_1_1_4"} : memref<1xi32> 
    %bd_buf_1_1_3 = aie.buffer(%mem_tile_1_1) {address = 16 : i32, sym_name = "bd_buf_1_1_3"} : memref<1xi32> 
    %bd_buf_1_1_2 = aie.buffer(%mem_tile_1_1) {address = 12 : i32, sym_name = "bd_buf_1_1_2"} : memref<1xi32> 
    %bd_buf_1_1_1 = aie.buffer(%mem_tile_1_1) {address = 8 : i32, sym_name = "bd_buf_1_1_1"} : memref<1xi32> 
    %bd_buf_1_1_0 = aie.buffer(%mem_tile_1_1) {address = 4 : i32, sym_name = "bd_buf_1_1_0"} : memref<1xi32> 
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_1_1_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_3_2 = aie.tile(3, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %bd_buf_3_2_5 = aie.buffer(%tile_3_2) {address = 0 : i32, sym_name = "bd_buf_3_2_5"} : memref<1xi32> 
    %bd_buf_3_2_4 = aie.buffer(%tile_3_2) {address = 4 : i32, sym_name = "bd_buf_3_2_4"} : memref<1xi32> 
    %bd_buf_3_2_3 = aie.buffer(%tile_3_2) {address = 8 : i32, sym_name = "bd_buf_3_2_3"} : memref<1xi32> 
    %bd_buf_3_2_2 = aie.buffer(%tile_3_2) {address = 12 : i32, sym_name = "bd_buf_3_2_2"} : memref<1xi32> 
    %bd_buf_3_2_1 = aie.buffer(%tile_3_2) {address = 16 : i32, sym_name = "bd_buf_3_2_1"} : memref<1xi32> 
    %bd_buf_3_2_0 = aie.buffer(%tile_3_2) {address = 20 : i32, sym_name = "bd_buf_3_2_0"} : memref<1xi32> 
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_3_2_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<North : 1, DMA : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<North : 3, DMA : 2>
      aie.connect<North : 2, DMA : 3>
      aie.connect<DMA : 2, South : 2>
    }
    %bd_buf_0_1_31 = aie.buffer(%mem_tile_0_1) {address = 48 : i32, sym_name = "bd_buf_0_1_31"} : memref<1xi32> 
    %bd_buf_0_1_30 = aie.buffer(%mem_tile_0_1) {address = 92 : i32, sym_name = "bd_buf_0_1_30"} : memref<1xi32> 
    %bd_buf_0_1_29 = aie.buffer(%mem_tile_0_1) {address = 88 : i32, sym_name = "bd_buf_0_1_29"} : memref<1xi32> 
    %bd_buf_0_1_28 = aie.buffer(%mem_tile_0_1) {address = 84 : i32, sym_name = "bd_buf_0_1_28"} : memref<1xi32> 
    %bd_buf_0_1_27 = aie.buffer(%mem_tile_0_1) {address = 80 : i32, sym_name = "bd_buf_0_1_27"} : memref<1xi32> 
    %bd_buf_0_1_26 = aie.buffer(%mem_tile_0_1) {address = 76 : i32, sym_name = "bd_buf_0_1_26"} : memref<1xi32> 
    %bd_buf_0_1_25 = aie.buffer(%mem_tile_0_1) {address = 72 : i32, sym_name = "bd_buf_0_1_25"} : memref<1xi32> 
    %bd_buf_0_1_24 = aie.buffer(%mem_tile_0_1) {address = 68 : i32, sym_name = "bd_buf_0_1_24"} : memref<1xi32> 
    %bd_buf_0_1_15 = aie.buffer(%mem_tile_0_1) {address = 64 : i32, sym_name = "bd_buf_0_1_15"} : memref<1xi32> 
    %bd_buf_0_1_14 = aie.buffer(%mem_tile_0_1) {address = 60 : i32, sym_name = "bd_buf_0_1_14"} : memref<1xi32> 
    %bd_buf_0_1_13 = aie.buffer(%mem_tile_0_1) {address = 56 : i32, sym_name = "bd_buf_0_1_13"} : memref<1xi32> 
    %bd_buf_0_1_12 = aie.buffer(%mem_tile_0_1) {address = 52 : i32, sym_name = "bd_buf_0_1_12"} : memref<1xi32> 
    %bd_buf_0_1_11 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "bd_buf_0_1_11"} : memref<1xi32> 
    %bd_buf_0_1_10 = aie.buffer(%mem_tile_0_1) {address = 44 : i32, sym_name = "bd_buf_0_1_10"} : memref<1xi32> 
    %bd_buf_0_1_9 = aie.buffer(%mem_tile_0_1) {address = 40 : i32, sym_name = "bd_buf_0_1_9"} : memref<1xi32> 
    %bd_buf_0_1_8 = aie.buffer(%mem_tile_0_1) {address = 36 : i32, sym_name = "bd_buf_0_1_8"} : memref<1xi32> 
    %bd_buf_0_1_7 = aie.buffer(%mem_tile_0_1) {address = 32 : i32, sym_name = "bd_buf_0_1_7"} : memref<1xi32> 
    %bd_buf_0_1_6 = aie.buffer(%mem_tile_0_1) {address = 28 : i32, sym_name = "bd_buf_0_1_6"} : memref<1xi32> 
    %bd_buf_0_1_5 = aie.buffer(%mem_tile_0_1) {address = 24 : i32, sym_name = "bd_buf_0_1_5"} : memref<1xi32> 
    %bd_buf_0_1_4 = aie.buffer(%mem_tile_0_1) {address = 20 : i32, sym_name = "bd_buf_0_1_4"} : memref<1xi32> 
    %bd_buf_0_1_3 = aie.buffer(%mem_tile_0_1) {address = 16 : i32, sym_name = "bd_buf_0_1_3"} : memref<1xi32> 
    %bd_buf_0_1_2 = aie.buffer(%mem_tile_0_1) {address = 12 : i32, sym_name = "bd_buf_0_1_2"} : memref<1xi32> 
    %bd_buf_0_1_1 = aie.buffer(%mem_tile_0_1) {address = 8 : i32, sym_name = "bd_buf_0_1_1"} : memref<1xi32> 
    %bd_buf_0_1_0 = aie.buffer(%mem_tile_0_1) {address = 4 : i32, sym_name = "bd_buf_0_1_0"} : memref<1xi32> 
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_0_1_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
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
    %bd_buf_1_2_5 = aie.buffer(%tile_1_2) {address = 0 : i32, sym_name = "bd_buf_1_2_5"} : memref<1xi32> 
    %bd_buf_1_2_4 = aie.buffer(%tile_1_2) {address = 4 : i32, sym_name = "bd_buf_1_2_4"} : memref<1xi32> 
    %bd_buf_1_2_3 = aie.buffer(%tile_1_2) {address = 8 : i32, sym_name = "bd_buf_1_2_3"} : memref<1xi32> 
    %bd_buf_1_2_2 = aie.buffer(%tile_1_2) {address = 12 : i32, sym_name = "bd_buf_1_2_2"} : memref<1xi32> 
    %bd_buf_1_2_1 = aie.buffer(%tile_1_2) {address = 16 : i32, sym_name = "bd_buf_1_2_1"} : memref<1xi32> 
    %bd_buf_1_2_0 = aie.buffer(%tile_1_2) {address = 20 : i32, sym_name = "bd_buf_1_2_0"} : memref<1xi32> 
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_1_2_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_2_2 = aie.tile(2, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %bd_buf_2_2_5 = aie.buffer(%tile_2_2) {address = 0 : i32, sym_name = "bd_buf_2_2_5"} : memref<1xi32> 
    %bd_buf_2_2_4 = aie.buffer(%tile_2_2) {address = 4 : i32, sym_name = "bd_buf_2_2_4"} : memref<1xi32> 
    %bd_buf_2_2_3 = aie.buffer(%tile_2_2) {address = 8 : i32, sym_name = "bd_buf_2_2_3"} : memref<1xi32> 
    %bd_buf_2_2_2 = aie.buffer(%tile_2_2) {address = 12 : i32, sym_name = "bd_buf_2_2_2"} : memref<1xi32> 
    %bd_buf_2_2_1 = aie.buffer(%tile_2_2) {address = 16 : i32, sym_name = "bd_buf_2_2_1"} : memref<1xi32> 
    %bd_buf_2_2_0 = aie.buffer(%tile_2_2) {address = 20 : i32, sym_name = "bd_buf_2_2_0"} : memref<1xi32> 
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_2_2_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
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
    %bd_buf_1_3_5 = aie.buffer(%tile_1_3) {address = 0 : i32, sym_name = "bd_buf_1_3_5"} : memref<1xi32> 
    %bd_buf_1_3_4 = aie.buffer(%tile_1_3) {address = 4 : i32, sym_name = "bd_buf_1_3_4"} : memref<1xi32> 
    %bd_buf_1_3_3 = aie.buffer(%tile_1_3) {address = 8 : i32, sym_name = "bd_buf_1_3_3"} : memref<1xi32> 
    %bd_buf_1_3_2 = aie.buffer(%tile_1_3) {address = 12 : i32, sym_name = "bd_buf_1_3_2"} : memref<1xi32> 
    %bd_buf_1_3_1 = aie.buffer(%tile_1_3) {address = 16 : i32, sym_name = "bd_buf_1_3_1"} : memref<1xi32> 
    %bd_buf_1_3_0 = aie.buffer(%tile_1_3) {address = 20 : i32, sym_name = "bd_buf_1_3_0"} : memref<1xi32> 
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_1_3_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_2_4 = aie.tile(2, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %bd_buf_2_4_5 = aie.buffer(%tile_2_4) {address = 0 : i32, sym_name = "bd_buf_2_4_5"} : memref<1xi32> 
    %bd_buf_2_4_4 = aie.buffer(%tile_2_4) {address = 4 : i32, sym_name = "bd_buf_2_4_4"} : memref<1xi32> 
    %bd_buf_2_4_3 = aie.buffer(%tile_2_4) {address = 8 : i32, sym_name = "bd_buf_2_4_3"} : memref<1xi32> 
    %bd_buf_2_4_2 = aie.buffer(%tile_2_4) {address = 12 : i32, sym_name = "bd_buf_2_4_2"} : memref<1xi32> 
    %bd_buf_2_4_1 = aie.buffer(%tile_2_4) {address = 16 : i32, sym_name = "bd_buf_2_4_1"} : memref<1xi32> 
    %bd_buf_2_4_0 = aie.buffer(%tile_2_4) {address = 20 : i32, sym_name = "bd_buf_2_4_0"} : memref<1xi32> 
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_2_4_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_2_3 = aie.tile(2, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %bd_buf_2_3_5 = aie.buffer(%tile_2_3) {address = 0 : i32, sym_name = "bd_buf_2_3_5"} : memref<1xi32> 
    %bd_buf_2_3_4 = aie.buffer(%tile_2_3) {address = 4 : i32, sym_name = "bd_buf_2_3_4"} : memref<1xi32> 
    %bd_buf_2_3_3 = aie.buffer(%tile_2_3) {address = 8 : i32, sym_name = "bd_buf_2_3_3"} : memref<1xi32> 
    %bd_buf_2_3_2 = aie.buffer(%tile_2_3) {address = 12 : i32, sym_name = "bd_buf_2_3_2"} : memref<1xi32> 
    %bd_buf_2_3_1 = aie.buffer(%tile_2_3) {address = 16 : i32, sym_name = "bd_buf_2_3_1"} : memref<1xi32> 
    %bd_buf_2_3_0 = aie.buffer(%tile_2_3) {address = 20 : i32, sym_name = "bd_buf_2_3_0"} : memref<1xi32> 
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_2_3_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_3_4 = aie.tile(3, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %bd_buf_3_4_5 = aie.buffer(%tile_3_4) {address = 0 : i32, sym_name = "bd_buf_3_4_5"} : memref<1xi32> 
    %bd_buf_3_4_4 = aie.buffer(%tile_3_4) {address = 4 : i32, sym_name = "bd_buf_3_4_4"} : memref<1xi32> 
    %bd_buf_3_4_3 = aie.buffer(%tile_3_4) {address = 8 : i32, sym_name = "bd_buf_3_4_3"} : memref<1xi32> 
    %bd_buf_3_4_2 = aie.buffer(%tile_3_4) {address = 12 : i32, sym_name = "bd_buf_3_4_2"} : memref<1xi32> 
    %bd_buf_3_4_1 = aie.buffer(%tile_3_4) {address = 16 : i32, sym_name = "bd_buf_3_4_1"} : memref<1xi32> 
    %bd_buf_3_4_0 = aie.buffer(%tile_3_4) {address = 20 : i32, sym_name = "bd_buf_3_4_0"} : memref<1xi32> 
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_3_4_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %switchbox_1_5 = aie.switchbox(%tile_1_5) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<South : 5, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %bd_buf_1_5_5 = aie.buffer(%tile_1_5) {address = 0 : i32, sym_name = "bd_buf_1_5_5"} : memref<1xi32> 
    %bd_buf_1_5_4 = aie.buffer(%tile_1_5) {address = 4 : i32, sym_name = "bd_buf_1_5_4"} : memref<1xi32> 
    %bd_buf_1_5_3 = aie.buffer(%tile_1_5) {address = 8 : i32, sym_name = "bd_buf_1_5_3"} : memref<1xi32> 
    %bd_buf_1_5_2 = aie.buffer(%tile_1_5) {address = 12 : i32, sym_name = "bd_buf_1_5_2"} : memref<1xi32> 
    %bd_buf_1_5_1 = aie.buffer(%tile_1_5) {address = 16 : i32, sym_name = "bd_buf_1_5_1"} : memref<1xi32> 
    %bd_buf_1_5_0 = aie.buffer(%tile_1_5) {address = 20 : i32, sym_name = "bd_buf_1_5_0"} : memref<1xi32> 
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_1_5_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<South : 4, DMA : 0>
      aie.connect<South : 5, North : 3>
      aie.connect<South : 3, North : 5>
      aie.connect<South : 3, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 3>
    }
    %bd_buf_1_4_5 = aie.buffer(%tile_1_4) {address = 0 : i32, sym_name = "bd_buf_1_4_5"} : memref<1xi32> 
    %bd_buf_1_4_4 = aie.buffer(%tile_1_4) {address = 4 : i32, sym_name = "bd_buf_1_4_4"} : memref<1xi32> 
    %bd_buf_1_4_3 = aie.buffer(%tile_1_4) {address = 8 : i32, sym_name = "bd_buf_1_4_3"} : memref<1xi32> 
    %bd_buf_1_4_2 = aie.buffer(%tile_1_4) {address = 12 : i32, sym_name = "bd_buf_1_4_2"} : memref<1xi32> 
    %bd_buf_1_4_1 = aie.buffer(%tile_1_4) {address = 16 : i32, sym_name = "bd_buf_1_4_1"} : memref<1xi32> 
    %bd_buf_1_4_0 = aie.buffer(%tile_1_4) {address = 20 : i32, sym_name = "bd_buf_1_4_0"} : memref<1xi32> 
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_1_4_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_2_5 = aie.tile(2, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %bd_buf_2_5_5 = aie.buffer(%tile_2_5) {address = 0 : i32, sym_name = "bd_buf_2_5_5"} : memref<1xi32> 
    %bd_buf_2_5_4 = aie.buffer(%tile_2_5) {address = 4 : i32, sym_name = "bd_buf_2_5_4"} : memref<1xi32> 
    %bd_buf_2_5_3 = aie.buffer(%tile_2_5) {address = 8 : i32, sym_name = "bd_buf_2_5_3"} : memref<1xi32> 
    %bd_buf_2_5_2 = aie.buffer(%tile_2_5) {address = 12 : i32, sym_name = "bd_buf_2_5_2"} : memref<1xi32> 
    %bd_buf_2_5_1 = aie.buffer(%tile_2_5) {address = 16 : i32, sym_name = "bd_buf_2_5_1"} : memref<1xi32> 
    %bd_buf_2_5_0 = aie.buffer(%tile_2_5) {address = 20 : i32, sym_name = "bd_buf_2_5_0"} : memref<1xi32> 
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_2_5_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_3_5 = aie.tile(3, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %bd_buf_3_5_5 = aie.buffer(%tile_3_5) {address = 0 : i32, sym_name = "bd_buf_3_5_5"} : memref<1xi32> 
    %bd_buf_3_5_4 = aie.buffer(%tile_3_5) {address = 4 : i32, sym_name = "bd_buf_3_5_4"} : memref<1xi32> 
    %bd_buf_3_5_3 = aie.buffer(%tile_3_5) {address = 8 : i32, sym_name = "bd_buf_3_5_3"} : memref<1xi32> 
    %bd_buf_3_5_2 = aie.buffer(%tile_3_5) {address = 12 : i32, sym_name = "bd_buf_3_5_2"} : memref<1xi32> 
    %bd_buf_3_5_1 = aie.buffer(%tile_3_5) {address = 16 : i32, sym_name = "bd_buf_3_5_1"} : memref<1xi32> 
    %bd_buf_3_5_0 = aie.buffer(%tile_3_5) {address = 20 : i32, sym_name = "bd_buf_3_5_0"} : memref<1xi32> 
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_3_5_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %mem_tile_3_1 = aie.tile(3, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %bd_buf_3_1_31 = aie.buffer(%mem_tile_3_1) {address = 48 : i32, sym_name = "bd_buf_3_1_31"} : memref<1xi32> 
    %bd_buf_3_1_30 = aie.buffer(%mem_tile_3_1) {address = 92 : i32, sym_name = "bd_buf_3_1_30"} : memref<1xi32> 
    %bd_buf_3_1_29 = aie.buffer(%mem_tile_3_1) {address = 88 : i32, sym_name = "bd_buf_3_1_29"} : memref<1xi32> 
    %bd_buf_3_1_28 = aie.buffer(%mem_tile_3_1) {address = 84 : i32, sym_name = "bd_buf_3_1_28"} : memref<1xi32> 
    %bd_buf_3_1_27 = aie.buffer(%mem_tile_3_1) {address = 80 : i32, sym_name = "bd_buf_3_1_27"} : memref<1xi32> 
    %bd_buf_3_1_26 = aie.buffer(%mem_tile_3_1) {address = 76 : i32, sym_name = "bd_buf_3_1_26"} : memref<1xi32> 
    %bd_buf_3_1_25 = aie.buffer(%mem_tile_3_1) {address = 72 : i32, sym_name = "bd_buf_3_1_25"} : memref<1xi32> 
    %bd_buf_3_1_24 = aie.buffer(%mem_tile_3_1) {address = 68 : i32, sym_name = "bd_buf_3_1_24"} : memref<1xi32> 
    %bd_buf_3_1_15 = aie.buffer(%mem_tile_3_1) {address = 64 : i32, sym_name = "bd_buf_3_1_15"} : memref<1xi32> 
    %bd_buf_3_1_14 = aie.buffer(%mem_tile_3_1) {address = 60 : i32, sym_name = "bd_buf_3_1_14"} : memref<1xi32> 
    %bd_buf_3_1_13 = aie.buffer(%mem_tile_3_1) {address = 56 : i32, sym_name = "bd_buf_3_1_13"} : memref<1xi32> 
    %bd_buf_3_1_12 = aie.buffer(%mem_tile_3_1) {address = 52 : i32, sym_name = "bd_buf_3_1_12"} : memref<1xi32> 
    %bd_buf_3_1_11 = aie.buffer(%mem_tile_3_1) {address = 0 : i32, sym_name = "bd_buf_3_1_11"} : memref<1xi32> 
    %bd_buf_3_1_10 = aie.buffer(%mem_tile_3_1) {address = 44 : i32, sym_name = "bd_buf_3_1_10"} : memref<1xi32> 
    %bd_buf_3_1_9 = aie.buffer(%mem_tile_3_1) {address = 40 : i32, sym_name = "bd_buf_3_1_9"} : memref<1xi32> 
    %bd_buf_3_1_8 = aie.buffer(%mem_tile_3_1) {address = 36 : i32, sym_name = "bd_buf_3_1_8"} : memref<1xi32> 
    %bd_buf_3_1_7 = aie.buffer(%mem_tile_3_1) {address = 32 : i32, sym_name = "bd_buf_3_1_7"} : memref<1xi32> 
    %bd_buf_3_1_6 = aie.buffer(%mem_tile_3_1) {address = 28 : i32, sym_name = "bd_buf_3_1_6"} : memref<1xi32> 
    %bd_buf_3_1_5 = aie.buffer(%mem_tile_3_1) {address = 24 : i32, sym_name = "bd_buf_3_1_5"} : memref<1xi32> 
    %bd_buf_3_1_4 = aie.buffer(%mem_tile_3_1) {address = 20 : i32, sym_name = "bd_buf_3_1_4"} : memref<1xi32> 
    %bd_buf_3_1_3 = aie.buffer(%mem_tile_3_1) {address = 16 : i32, sym_name = "bd_buf_3_1_3"} : memref<1xi32> 
    %bd_buf_3_1_2 = aie.buffer(%mem_tile_3_1) {address = 12 : i32, sym_name = "bd_buf_3_1_2"} : memref<1xi32> 
    %bd_buf_3_1_1 = aie.buffer(%mem_tile_3_1) {address = 8 : i32, sym_name = "bd_buf_3_1_1"} : memref<1xi32> 
    %bd_buf_3_1_0 = aie.buffer(%mem_tile_3_1) {address = 4 : i32, sym_name = "bd_buf_3_1_0"} : memref<1xi32> 
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_3_1_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %mem_tile_2_1 = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %bd_buf_2_1_31 = aie.buffer(%mem_tile_2_1) {address = 48 : i32, sym_name = "bd_buf_2_1_31"} : memref<1xi32> 
    %bd_buf_2_1_30 = aie.buffer(%mem_tile_2_1) {address = 92 : i32, sym_name = "bd_buf_2_1_30"} : memref<1xi32> 
    %bd_buf_2_1_29 = aie.buffer(%mem_tile_2_1) {address = 88 : i32, sym_name = "bd_buf_2_1_29"} : memref<1xi32> 
    %bd_buf_2_1_28 = aie.buffer(%mem_tile_2_1) {address = 84 : i32, sym_name = "bd_buf_2_1_28"} : memref<1xi32> 
    %bd_buf_2_1_27 = aie.buffer(%mem_tile_2_1) {address = 80 : i32, sym_name = "bd_buf_2_1_27"} : memref<1xi32> 
    %bd_buf_2_1_26 = aie.buffer(%mem_tile_2_1) {address = 76 : i32, sym_name = "bd_buf_2_1_26"} : memref<1xi32> 
    %bd_buf_2_1_25 = aie.buffer(%mem_tile_2_1) {address = 72 : i32, sym_name = "bd_buf_2_1_25"} : memref<1xi32> 
    %bd_buf_2_1_24 = aie.buffer(%mem_tile_2_1) {address = 68 : i32, sym_name = "bd_buf_2_1_24"} : memref<1xi32> 
    %bd_buf_2_1_15 = aie.buffer(%mem_tile_2_1) {address = 64 : i32, sym_name = "bd_buf_2_1_15"} : memref<1xi32> 
    %bd_buf_2_1_14 = aie.buffer(%mem_tile_2_1) {address = 60 : i32, sym_name = "bd_buf_2_1_14"} : memref<1xi32> 
    %bd_buf_2_1_13 = aie.buffer(%mem_tile_2_1) {address = 56 : i32, sym_name = "bd_buf_2_1_13"} : memref<1xi32> 
    %bd_buf_2_1_12 = aie.buffer(%mem_tile_2_1) {address = 52 : i32, sym_name = "bd_buf_2_1_12"} : memref<1xi32> 
    %bd_buf_2_1_11 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, sym_name = "bd_buf_2_1_11"} : memref<1xi32> 
    %bd_buf_2_1_10 = aie.buffer(%mem_tile_2_1) {address = 44 : i32, sym_name = "bd_buf_2_1_10"} : memref<1xi32> 
    %bd_buf_2_1_9 = aie.buffer(%mem_tile_2_1) {address = 40 : i32, sym_name = "bd_buf_2_1_9"} : memref<1xi32> 
    %bd_buf_2_1_8 = aie.buffer(%mem_tile_2_1) {address = 36 : i32, sym_name = "bd_buf_2_1_8"} : memref<1xi32> 
    %bd_buf_2_1_7 = aie.buffer(%mem_tile_2_1) {address = 32 : i32, sym_name = "bd_buf_2_1_7"} : memref<1xi32> 
    %bd_buf_2_1_6 = aie.buffer(%mem_tile_2_1) {address = 28 : i32, sym_name = "bd_buf_2_1_6"} : memref<1xi32> 
    %bd_buf_2_1_5 = aie.buffer(%mem_tile_2_1) {address = 24 : i32, sym_name = "bd_buf_2_1_5"} : memref<1xi32> 
    %bd_buf_2_1_4 = aie.buffer(%mem_tile_2_1) {address = 20 : i32, sym_name = "bd_buf_2_1_4"} : memref<1xi32> 
    %bd_buf_2_1_3 = aie.buffer(%mem_tile_2_1) {address = 16 : i32, sym_name = "bd_buf_2_1_3"} : memref<1xi32> 
    %bd_buf_2_1_2 = aie.buffer(%mem_tile_2_1) {address = 12 : i32, sym_name = "bd_buf_2_1_2"} : memref<1xi32> 
    %bd_buf_2_1_1 = aie.buffer(%mem_tile_2_1) {address = 8 : i32, sym_name = "bd_buf_2_1_1"} : memref<1xi32> 
    %bd_buf_2_1_0 = aie.buffer(%mem_tile_2_1) {address = 4 : i32, sym_name = "bd_buf_2_1_0"} : memref<1xi32> 
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_2_1_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<South : 0, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %bd_buf_0_5_5 = aie.buffer(%tile_0_5) {address = 0 : i32, sym_name = "bd_buf_0_5_5"} : memref<1xi32> 
    %bd_buf_0_5_4 = aie.buffer(%tile_0_5) {address = 4 : i32, sym_name = "bd_buf_0_5_4"} : memref<1xi32> 
    %bd_buf_0_5_3 = aie.buffer(%tile_0_5) {address = 8 : i32, sym_name = "bd_buf_0_5_3"} : memref<1xi32> 
    %bd_buf_0_5_2 = aie.buffer(%tile_0_5) {address = 12 : i32, sym_name = "bd_buf_0_5_2"} : memref<1xi32> 
    %bd_buf_0_5_1 = aie.buffer(%tile_0_5) {address = 16 : i32, sym_name = "bd_buf_0_5_1"} : memref<1xi32> 
    %bd_buf_0_5_0 = aie.buffer(%tile_0_5) {address = 20 : i32, sym_name = "bd_buf_0_5_0"} : memref<1xi32> 
    %lock_0_5 = aie.lock(%tile_0_5, 3) {init = 0 : i32}
    %lock_0_5_0 = aie.lock(%tile_0_5, 2) {init = 2 : i32}
    %lock_0_5_1 = aie.lock(%tile_0_5, 1) {init = 2 : i32}
    %lock_0_5_2 = aie.lock(%tile_0_5, 0) {init = 2 : i32}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_0_5_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<South : 5, DMA : 0>
      aie.connect<South : 2, North : 3>
      aie.connect<South : 1, North : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 0>
    }
    %bd_buf_0_4_5 = aie.buffer(%tile_0_4) {address = 0 : i32, sym_name = "bd_buf_0_4_5"} : memref<1xi32> 
    %bd_buf_0_4_4 = aie.buffer(%tile_0_4) {address = 4 : i32, sym_name = "bd_buf_0_4_4"} : memref<1xi32> 
    %bd_buf_0_4_3 = aie.buffer(%tile_0_4) {address = 8 : i32, sym_name = "bd_buf_0_4_3"} : memref<1xi32> 
    %bd_buf_0_4_2 = aie.buffer(%tile_0_4) {address = 12 : i32, sym_name = "bd_buf_0_4_2"} : memref<1xi32> 
    %bd_buf_0_4_1 = aie.buffer(%tile_0_4) {address = 16 : i32, sym_name = "bd_buf_0_4_1"} : memref<1xi32> 
    %bd_buf_0_4_0 = aie.buffer(%tile_0_4) {address = 20 : i32, sym_name = "bd_buf_0_4_0"} : memref<1xi32> 
    %lock_0_4 = aie.lock(%tile_0_4, 3) {init = 0 : i32}
    %lock_0_4_3 = aie.lock(%tile_0_4, 2) {init = 2 : i32}
    %lock_0_4_4 = aie.lock(%tile_0_4, 1) {init = 2 : i32}
    %lock_0_4_5 = aie.lock(%tile_0_4, 0) {init = 2 : i32}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_0_4_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
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
    %bd_buf_0_3_5 = aie.buffer(%tile_0_3) {address = 0 : i32, sym_name = "bd_buf_0_3_5"} : memref<1xi32> 
    %bd_buf_0_3_4 = aie.buffer(%tile_0_3) {address = 4 : i32, sym_name = "bd_buf_0_3_4"} : memref<1xi32> 
    %bd_buf_0_3_3 = aie.buffer(%tile_0_3) {address = 8 : i32, sym_name = "bd_buf_0_3_3"} : memref<1xi32> 
    %bd_buf_0_3_2 = aie.buffer(%tile_0_3) {address = 12 : i32, sym_name = "bd_buf_0_3_2"} : memref<1xi32> 
    %bd_buf_0_3_1 = aie.buffer(%tile_0_3) {address = 16 : i32, sym_name = "bd_buf_0_3_1"} : memref<1xi32> 
    %bd_buf_0_3_0 = aie.buffer(%tile_0_3) {address = 20 : i32, sym_name = "bd_buf_0_3_0"} : memref<1xi32> 
    %lock_0_3 = aie.lock(%tile_0_3, 3) {init = 0 : i32}
    %lock_0_3_6 = aie.lock(%tile_0_3, 2) {init = 2 : i32}
    %lock_0_3_7 = aie.lock(%tile_0_3, 1) {init = 2 : i32}
    %lock_0_3_8 = aie.lock(%tile_0_3, 0) {init = 2 : i32}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_0_3_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
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
    %bd_buf_0_2_5 = aie.buffer(%tile_0_2) {address = 0 : i32, sym_name = "bd_buf_0_2_5"} : memref<1xi32> 
    %bd_buf_0_2_4 = aie.buffer(%tile_0_2) {address = 4 : i32, sym_name = "bd_buf_0_2_4"} : memref<1xi32> 
    %bd_buf_0_2_3 = aie.buffer(%tile_0_2) {address = 8 : i32, sym_name = "bd_buf_0_2_3"} : memref<1xi32> 
    %bd_buf_0_2_2 = aie.buffer(%tile_0_2) {address = 12 : i32, sym_name = "bd_buf_0_2_2"} : memref<1xi32> 
    %bd_buf_0_2_1 = aie.buffer(%tile_0_2) {address = 16 : i32, sym_name = "bd_buf_0_2_1"} : memref<1xi32> 
    %bd_buf_0_2_0 = aie.buffer(%tile_0_2) {address = 20 : i32, sym_name = "bd_buf_0_2_0"} : memref<1xi32> 
    %lock_0_2 = aie.lock(%tile_0_2, 3) {init = 0 : i32}
    %lock_0_2_9 = aie.lock(%tile_0_2, 2) {init = 2 : i32}
    %lock_0_2_10 = aie.lock(%tile_0_2, 1) {init = 2 : i32}
    %lock_0_2_11 = aie.lock(%tile_0_2, 0) {init = 2 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_0_2_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_mux_2_0 = aie.shim_mux(%shim_noc_tile_2_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 0, North : 7>
      aie.connect<North : 2, DMA : 0>
    }
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_mux_3_0 = aie.shim_mux(%shim_noc_tile_3_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 0, North : 7>
      aie.connect<North : 2, DMA : 0>
    }
    aie.runtime_sequence @configure() {
      aiex.npu.maskwrite32 {address = 35856384 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35773968 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 35773976 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 35773952 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 35773960 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 35782656 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35773968 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35773976 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35773952 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35773960 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69410816 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69328400 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 69328408 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 69328384 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 69328392 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 69337088 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69328400 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69328408 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69328384 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69328392 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102965248 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102882832 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 102882840 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 102882816 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 102882824 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 102891520 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102882832 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102882840 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102882816 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102882824 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 36904960 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 36822544 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 36822552 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 36822528 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 36822536 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 36831232 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 36822544 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 36822552 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 36822528 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 36822536 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 70459392 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 70376976 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 70376984 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 70376960 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 70376968 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 70385664 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 70376976 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 70376984 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 70376960 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 70376968 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 104013824 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 103931408 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 103931416 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 103931392 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 103931400 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 103940096 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 103931408 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 103931416 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 103931392 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 103931400 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 37953536 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 37871120 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 37871128 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 37871104 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 37871112 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 37879808 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 37871120 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 37871128 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 37871104 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 37871112 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 71507968 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 71425552 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 71425560 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 71425536 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 71425544 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 71434240 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 71425552 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 71425560 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 71425536 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 71425544 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 105062400 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 104979984 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 104979992 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 104979968 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 104979976 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 104988672 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 104979984 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 104979992 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 104979968 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 104979976 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 39002112 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 38919696 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 38919704 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 38919680 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 38919688 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 38928384 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 38919696 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 38919704 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 38919680 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 38919688 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 72556544 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 72474128 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 72474136 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 72474112 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 72474120 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 72482816 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 72474128 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 72474136 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 72474112 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 72474120 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 106110976 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 106028560 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 106028568 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 106028544 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 106028552 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 106037248 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 106028560 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 106028568 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 106028544 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 106028552 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35856384 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 35856384 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778560 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778576 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778592 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778608 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778624 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778640 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778656 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778672 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778688 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778704 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778720 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778736 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778752 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778768 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778784 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778800 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69410816 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 69410816 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69332992 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333008 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333024 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333040 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333056 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333072 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333088 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333104 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333120 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333136 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333152 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333168 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333184 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333200 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333216 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333232 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102965248 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 102965248 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887424 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887440 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887456 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887472 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887488 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887504 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887520 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887536 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887552 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887568 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887584 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887600 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887616 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887632 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887648 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887664 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 36904960 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 36904960 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827136 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827152 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827168 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827184 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827200 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827216 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827232 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827248 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827264 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827280 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827296 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827312 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827328 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827344 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827360 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827376 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 70459392 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 70459392 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381568 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381584 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381600 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381616 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381632 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381648 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381664 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381680 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381696 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381712 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381728 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381744 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381760 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381776 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381792 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381808 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 104013824 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 104013824 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936000 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936016 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936032 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936048 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936064 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936080 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936096 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936112 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936128 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936144 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936160 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936176 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936192 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936208 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936224 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936240 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 37953536 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 37953536 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875712 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875728 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875744 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875760 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875776 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875792 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875808 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875824 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875840 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875856 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875872 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875888 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875904 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875920 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875936 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875952 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 71507968 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 71507968 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430144 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430160 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430176 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430192 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430208 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430224 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430240 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430256 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430272 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430288 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430304 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430320 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430336 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430352 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430368 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430384 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 105062400 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 105062400 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984576 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984592 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984608 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984624 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984640 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984656 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984672 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984688 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984704 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984720 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984736 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984752 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984768 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984784 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984800 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984816 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 39002112 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 39002112 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924288 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924304 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924320 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924336 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924352 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924368 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924384 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924400 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924416 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924432 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924448 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924464 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924480 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924496 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924512 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924528 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 72556544 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 72556544 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478720 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478736 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478752 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478768 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478784 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478800 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478816 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478832 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478848 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478864 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478880 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478896 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478912 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478928 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478944 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478960 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 106110976 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 106110976 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033152 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033168 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033184 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033200 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033216 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033232 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033248 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033264 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033280 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033296 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033312 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033328 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033344 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033360 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033376 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033392 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 100745280 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 100745296 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102498368 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102498384 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102498400 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102498416 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102498432 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102498448 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102498464 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102498480 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033216 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033232 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984640 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984656 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936064 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936080 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887488 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887504 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 67190848 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 67190864 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 68943936 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 68943952 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 68943968 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 68943984 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 68944000 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 68944016 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 68944032 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 68944048 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478784 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478800 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430208 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430224 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381632 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381648 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333056 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333072 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 33636416 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 33636432 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35389504 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35389520 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35389536 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35389552 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35389568 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35389584 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35389600 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35389616 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924352 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924368 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875776 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875792 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827200 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827216 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778624 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778640 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887456 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887472 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936032 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936048 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984608 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984624 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033184 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033200 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102498336 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102498352 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 100745248 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 100745264 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333024 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333040 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381600 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381616 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430176 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430192 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478752 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478768 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 68943904 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 68943920 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 67190816 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 67190832 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778592 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778608 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827168 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827184 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875744 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875760 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924320 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924336 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35389472 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35389488 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 33636384 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 33636400 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924288 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 38924304 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478720 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 72478736 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033152 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 106033168 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875712 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 37875728 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430144 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 71430160 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984576 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 104984592 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827136 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 36827152 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381568 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 70381584 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936000 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 103936016 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778560 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35778576 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69332992 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 69333008 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887424 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102887440 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102498304 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 102498320 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 100745216 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 100745232 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 68943872 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 68943888 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 67190784 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 67190800 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35389440 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35389456 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 33636352 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 33636368 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 35773956 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35773952 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 35773964 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35773960 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 35773972 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35773968 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 69328388 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69328384 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 69328396 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69328392 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 69328404 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 69328400 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 102882820 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102882816 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 102882828 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102882824 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 102882836 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102882832 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 36822532 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 36822528 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 36822540 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 36822536 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 36822548 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 36822544 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 70376964 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 70376960 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 70376972 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 70376968 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 70376980 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 70376976 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 103931396 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 103931392 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 103931404 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 103931400 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 103931412 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 103931408 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 37871108 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 37871104 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 37871116 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 37871112 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 37871124 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 37871120 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 71425540 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 71425536 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 71425548 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 71425544 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 71425556 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 71425552 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 104979972 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 104979968 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 104979980 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 104979976 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 104979988 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 104979984 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 38919684 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 38919680 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 38919692 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 38919688 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 38919700 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 38919696 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 72474116 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 72474112 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 72474124 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 72474120 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 72474132 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 72474128 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 106028548 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 106028544 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 106028556 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 106028552 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 106028564 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 106028560 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 1705488 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 1705496 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 1705504 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 1705512 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 35259908 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35259904 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 35259956 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35259952 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 35259916 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35259912 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 35259964 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35259960 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 35259924 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35259920 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 35259932 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35259928 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 35259940 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35259936 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 35259948 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35259944 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 35259972 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35259968 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 68814340 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 68814336 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 68814388 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 68814384 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 68814348 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 68814344 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 68814396 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 68814392 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 68814356 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 68814352 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 68814364 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 68814360 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 68814372 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 68814368 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 68814380 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 68814376 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 68814404 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 68814400 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 102368772 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102368768 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 102368820 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102368816 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 102368780 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102368776 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 102368828 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102368824 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 102368788 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102368784 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 102368796 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102368792 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 102368804 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102368800 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 102368812 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102368808 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 102368836 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 102368832 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 33812992 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 67367424 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 100921856 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 35856384 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 69410816 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 102965248 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 36904960 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 70459392 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 104013824 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 37953536 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 71507968 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 105062400 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 39002112 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 72556544 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 106110976 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aie.end
    }
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
