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
    %bd_buf_3_3_5 = aie.buffer(%tile_3_3) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_3_3_5"} : memref<1xi32> 
    %bd_buf_3_3_4 = aie.buffer(%tile_3_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_3_3_4"} : memref<1xi32> 
    %bd_buf_3_3_3 = aie.buffer(%tile_3_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_3_3_3"} : memref<1xi32> 
    %bd_buf_3_3_2 = aie.buffer(%tile_3_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_3_3_2"} : memref<1xi32> 
    %bd_buf_3_3_1 = aie.buffer(%tile_3_3) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_3_3_1"} : memref<1xi32> 
    %bd_buf_3_3_0 = aie.buffer(%tile_3_3) {address = 16388 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_3_3_0"} : memref<1xi32> 
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
    %bd_buf_1_1_31 = aie.buffer(%mem_tile_1_1) {address = 262148 : i32, mem_bank = 4 : i32, sym_name = "bd_buf_1_1_31"} : memref<1xi32> 
    %bd_buf_1_1_30 = aie.buffer(%mem_tile_1_1) {address = 458760 : i32, mem_bank = 7 : i32, sym_name = "bd_buf_1_1_30"} : memref<1xi32> 
    %bd_buf_1_1_29 = aie.buffer(%mem_tile_1_1) {address = 393224 : i32, mem_bank = 6 : i32, sym_name = "bd_buf_1_1_29"} : memref<1xi32> 
    %bd_buf_1_1_28 = aie.buffer(%mem_tile_1_1) {address = 327688 : i32, mem_bank = 5 : i32, sym_name = "bd_buf_1_1_28"} : memref<1xi32> 
    %bd_buf_1_1_27 = aie.buffer(%mem_tile_1_1) {address = 262152 : i32, mem_bank = 4 : i32, sym_name = "bd_buf_1_1_27"} : memref<1xi32> 
    %bd_buf_1_1_26 = aie.buffer(%mem_tile_1_1) {address = 196616 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_1_1_26"} : memref<1xi32> 
    %bd_buf_1_1_25 = aie.buffer(%mem_tile_1_1) {address = 131080 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_1_1_25"} : memref<1xi32> 
    %bd_buf_1_1_24 = aie.buffer(%mem_tile_1_1) {address = 65544 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_1_1_24"} : memref<1xi32> 
    %bd_buf_1_1_15 = aie.buffer(%mem_tile_1_1) {address = 8 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_1_1_15"} : memref<1xi32> 
    %bd_buf_1_1_14 = aie.buffer(%mem_tile_1_1) {address = 458756 : i32, mem_bank = 7 : i32, sym_name = "bd_buf_1_1_14"} : memref<1xi32> 
    %bd_buf_1_1_13 = aie.buffer(%mem_tile_1_1) {address = 393220 : i32, mem_bank = 6 : i32, sym_name = "bd_buf_1_1_13"} : memref<1xi32> 
    %bd_buf_1_1_12 = aie.buffer(%mem_tile_1_1) {address = 327684 : i32, mem_bank = 5 : i32, sym_name = "bd_buf_1_1_12"} : memref<1xi32> 
    %bd_buf_1_1_11 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_1_1_11"} : memref<1xi32> 
    %bd_buf_1_1_10 = aie.buffer(%mem_tile_1_1) {address = 196612 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_1_1_10"} : memref<1xi32> 
    %bd_buf_1_1_9 = aie.buffer(%mem_tile_1_1) {address = 131076 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_1_1_9"} : memref<1xi32> 
    %bd_buf_1_1_8 = aie.buffer(%mem_tile_1_1) {address = 65540 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_1_1_8"} : memref<1xi32> 
    %bd_buf_1_1_7 = aie.buffer(%mem_tile_1_1) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_1_1_7"} : memref<1xi32> 
    %bd_buf_1_1_6 = aie.buffer(%mem_tile_1_1) {address = 458752 : i32, mem_bank = 7 : i32, sym_name = "bd_buf_1_1_6"} : memref<1xi32> 
    %bd_buf_1_1_5 = aie.buffer(%mem_tile_1_1) {address = 393216 : i32, mem_bank = 6 : i32, sym_name = "bd_buf_1_1_5"} : memref<1xi32> 
    %bd_buf_1_1_4 = aie.buffer(%mem_tile_1_1) {address = 327680 : i32, mem_bank = 5 : i32, sym_name = "bd_buf_1_1_4"} : memref<1xi32> 
    %bd_buf_1_1_3 = aie.buffer(%mem_tile_1_1) {address = 262144 : i32, mem_bank = 4 : i32, sym_name = "bd_buf_1_1_3"} : memref<1xi32> 
    %bd_buf_1_1_2 = aie.buffer(%mem_tile_1_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_1_1_2"} : memref<1xi32> 
    %bd_buf_1_1_1 = aie.buffer(%mem_tile_1_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_1_1_1"} : memref<1xi32> 
    %bd_buf_1_1_0 = aie.buffer(%mem_tile_1_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_1_1_0"} : memref<1xi32> 
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_1_1_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_3_2 = aie.tile(3, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %bd_buf_3_2_5 = aie.buffer(%tile_3_2) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_3_2_5"} : memref<1xi32> 
    %bd_buf_3_2_4 = aie.buffer(%tile_3_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_3_2_4"} : memref<1xi32> 
    %bd_buf_3_2_3 = aie.buffer(%tile_3_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_3_2_3"} : memref<1xi32> 
    %bd_buf_3_2_2 = aie.buffer(%tile_3_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_3_2_2"} : memref<1xi32> 
    %bd_buf_3_2_1 = aie.buffer(%tile_3_2) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_3_2_1"} : memref<1xi32> 
    %bd_buf_3_2_0 = aie.buffer(%tile_3_2) {address = 16388 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_3_2_0"} : memref<1xi32> 
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
    %bd_buf_0_1_31 = aie.buffer(%mem_tile_0_1) {address = 262148 : i32, mem_bank = 4 : i32, sym_name = "bd_buf_0_1_31"} : memref<1xi32> 
    %bd_buf_0_1_30 = aie.buffer(%mem_tile_0_1) {address = 458760 : i32, mem_bank = 7 : i32, sym_name = "bd_buf_0_1_30"} : memref<1xi32> 
    %bd_buf_0_1_29 = aie.buffer(%mem_tile_0_1) {address = 393224 : i32, mem_bank = 6 : i32, sym_name = "bd_buf_0_1_29"} : memref<1xi32> 
    %bd_buf_0_1_28 = aie.buffer(%mem_tile_0_1) {address = 327688 : i32, mem_bank = 5 : i32, sym_name = "bd_buf_0_1_28"} : memref<1xi32> 
    %bd_buf_0_1_27 = aie.buffer(%mem_tile_0_1) {address = 262152 : i32, mem_bank = 4 : i32, sym_name = "bd_buf_0_1_27"} : memref<1xi32> 
    %bd_buf_0_1_26 = aie.buffer(%mem_tile_0_1) {address = 196616 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_0_1_26"} : memref<1xi32> 
    %bd_buf_0_1_25 = aie.buffer(%mem_tile_0_1) {address = 131080 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_0_1_25"} : memref<1xi32> 
    %bd_buf_0_1_24 = aie.buffer(%mem_tile_0_1) {address = 65544 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_0_1_24"} : memref<1xi32> 
    %bd_buf_0_1_15 = aie.buffer(%mem_tile_0_1) {address = 8 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_0_1_15"} : memref<1xi32> 
    %bd_buf_0_1_14 = aie.buffer(%mem_tile_0_1) {address = 458756 : i32, mem_bank = 7 : i32, sym_name = "bd_buf_0_1_14"} : memref<1xi32> 
    %bd_buf_0_1_13 = aie.buffer(%mem_tile_0_1) {address = 393220 : i32, mem_bank = 6 : i32, sym_name = "bd_buf_0_1_13"} : memref<1xi32> 
    %bd_buf_0_1_12 = aie.buffer(%mem_tile_0_1) {address = 327684 : i32, mem_bank = 5 : i32, sym_name = "bd_buf_0_1_12"} : memref<1xi32> 
    %bd_buf_0_1_11 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_0_1_11"} : memref<1xi32> 
    %bd_buf_0_1_10 = aie.buffer(%mem_tile_0_1) {address = 196612 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_0_1_10"} : memref<1xi32> 
    %bd_buf_0_1_9 = aie.buffer(%mem_tile_0_1) {address = 131076 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_0_1_9"} : memref<1xi32> 
    %bd_buf_0_1_8 = aie.buffer(%mem_tile_0_1) {address = 65540 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_0_1_8"} : memref<1xi32> 
    %bd_buf_0_1_7 = aie.buffer(%mem_tile_0_1) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_0_1_7"} : memref<1xi32> 
    %bd_buf_0_1_6 = aie.buffer(%mem_tile_0_1) {address = 458752 : i32, mem_bank = 7 : i32, sym_name = "bd_buf_0_1_6"} : memref<1xi32> 
    %bd_buf_0_1_5 = aie.buffer(%mem_tile_0_1) {address = 393216 : i32, mem_bank = 6 : i32, sym_name = "bd_buf_0_1_5"} : memref<1xi32> 
    %bd_buf_0_1_4 = aie.buffer(%mem_tile_0_1) {address = 327680 : i32, mem_bank = 5 : i32, sym_name = "bd_buf_0_1_4"} : memref<1xi32> 
    %bd_buf_0_1_3 = aie.buffer(%mem_tile_0_1) {address = 262144 : i32, mem_bank = 4 : i32, sym_name = "bd_buf_0_1_3"} : memref<1xi32> 
    %bd_buf_0_1_2 = aie.buffer(%mem_tile_0_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_0_1_2"} : memref<1xi32> 
    %bd_buf_0_1_1 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_0_1_1"} : memref<1xi32> 
    %bd_buf_0_1_0 = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_0_1_0"} : memref<1xi32> 
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
    %bd_buf_1_2_5 = aie.buffer(%tile_1_2) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_1_2_5"} : memref<1xi32> 
    %bd_buf_1_2_4 = aie.buffer(%tile_1_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_1_2_4"} : memref<1xi32> 
    %bd_buf_1_2_3 = aie.buffer(%tile_1_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_1_2_3"} : memref<1xi32> 
    %bd_buf_1_2_2 = aie.buffer(%tile_1_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_1_2_2"} : memref<1xi32> 
    %bd_buf_1_2_1 = aie.buffer(%tile_1_2) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_1_2_1"} : memref<1xi32> 
    %bd_buf_1_2_0 = aie.buffer(%tile_1_2) {address = 16388 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_1_2_0"} : memref<1xi32> 
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_1_2_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_2_2 = aie.tile(2, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %bd_buf_2_2_5 = aie.buffer(%tile_2_2) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_2_2_5"} : memref<1xi32> 
    %bd_buf_2_2_4 = aie.buffer(%tile_2_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_2_2_4"} : memref<1xi32> 
    %bd_buf_2_2_3 = aie.buffer(%tile_2_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_2_2_3"} : memref<1xi32> 
    %bd_buf_2_2_2 = aie.buffer(%tile_2_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_2_2_2"} : memref<1xi32> 
    %bd_buf_2_2_1 = aie.buffer(%tile_2_2) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_2_2_1"} : memref<1xi32> 
    %bd_buf_2_2_0 = aie.buffer(%tile_2_2) {address = 16388 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_2_2_0"} : memref<1xi32> 
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
    %bd_buf_1_3_5 = aie.buffer(%tile_1_3) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_1_3_5"} : memref<1xi32> 
    %bd_buf_1_3_4 = aie.buffer(%tile_1_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_1_3_4"} : memref<1xi32> 
    %bd_buf_1_3_3 = aie.buffer(%tile_1_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_1_3_3"} : memref<1xi32> 
    %bd_buf_1_3_2 = aie.buffer(%tile_1_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_1_3_2"} : memref<1xi32> 
    %bd_buf_1_3_1 = aie.buffer(%tile_1_3) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_1_3_1"} : memref<1xi32> 
    %bd_buf_1_3_0 = aie.buffer(%tile_1_3) {address = 16388 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_1_3_0"} : memref<1xi32> 
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_1_3_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_2_4 = aie.tile(2, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %bd_buf_2_4_5 = aie.buffer(%tile_2_4) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_2_4_5"} : memref<1xi32> 
    %bd_buf_2_4_4 = aie.buffer(%tile_2_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_2_4_4"} : memref<1xi32> 
    %bd_buf_2_4_3 = aie.buffer(%tile_2_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_2_4_3"} : memref<1xi32> 
    %bd_buf_2_4_2 = aie.buffer(%tile_2_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_2_4_2"} : memref<1xi32> 
    %bd_buf_2_4_1 = aie.buffer(%tile_2_4) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_2_4_1"} : memref<1xi32> 
    %bd_buf_2_4_0 = aie.buffer(%tile_2_4) {address = 16388 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_2_4_0"} : memref<1xi32> 
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_2_4_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_2_3 = aie.tile(2, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %bd_buf_2_3_5 = aie.buffer(%tile_2_3) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_2_3_5"} : memref<1xi32> 
    %bd_buf_2_3_4 = aie.buffer(%tile_2_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_2_3_4"} : memref<1xi32> 
    %bd_buf_2_3_3 = aie.buffer(%tile_2_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_2_3_3"} : memref<1xi32> 
    %bd_buf_2_3_2 = aie.buffer(%tile_2_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_2_3_2"} : memref<1xi32> 
    %bd_buf_2_3_1 = aie.buffer(%tile_2_3) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_2_3_1"} : memref<1xi32> 
    %bd_buf_2_3_0 = aie.buffer(%tile_2_3) {address = 16388 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_2_3_0"} : memref<1xi32> 
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_2_3_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_3_4 = aie.tile(3, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %bd_buf_3_4_5 = aie.buffer(%tile_3_4) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_3_4_5"} : memref<1xi32> 
    %bd_buf_3_4_4 = aie.buffer(%tile_3_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_3_4_4"} : memref<1xi32> 
    %bd_buf_3_4_3 = aie.buffer(%tile_3_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_3_4_3"} : memref<1xi32> 
    %bd_buf_3_4_2 = aie.buffer(%tile_3_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_3_4_2"} : memref<1xi32> 
    %bd_buf_3_4_1 = aie.buffer(%tile_3_4) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_3_4_1"} : memref<1xi32> 
    %bd_buf_3_4_0 = aie.buffer(%tile_3_4) {address = 16388 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_3_4_0"} : memref<1xi32> 
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
    %bd_buf_1_5_5 = aie.buffer(%tile_1_5) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_1_5_5"} : memref<1xi32> 
    %bd_buf_1_5_4 = aie.buffer(%tile_1_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_1_5_4"} : memref<1xi32> 
    %bd_buf_1_5_3 = aie.buffer(%tile_1_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_1_5_3"} : memref<1xi32> 
    %bd_buf_1_5_2 = aie.buffer(%tile_1_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_1_5_2"} : memref<1xi32> 
    %bd_buf_1_5_1 = aie.buffer(%tile_1_5) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_1_5_1"} : memref<1xi32> 
    %bd_buf_1_5_0 = aie.buffer(%tile_1_5) {address = 16388 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_1_5_0"} : memref<1xi32> 
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
    %bd_buf_1_4_5 = aie.buffer(%tile_1_4) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_1_4_5"} : memref<1xi32> 
    %bd_buf_1_4_4 = aie.buffer(%tile_1_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_1_4_4"} : memref<1xi32> 
    %bd_buf_1_4_3 = aie.buffer(%tile_1_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_1_4_3"} : memref<1xi32> 
    %bd_buf_1_4_2 = aie.buffer(%tile_1_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_1_4_2"} : memref<1xi32> 
    %bd_buf_1_4_1 = aie.buffer(%tile_1_4) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_1_4_1"} : memref<1xi32> 
    %bd_buf_1_4_0 = aie.buffer(%tile_1_4) {address = 16388 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_1_4_0"} : memref<1xi32> 
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_1_4_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_2_5 = aie.tile(2, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %bd_buf_2_5_5 = aie.buffer(%tile_2_5) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_2_5_5"} : memref<1xi32> 
    %bd_buf_2_5_4 = aie.buffer(%tile_2_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_2_5_4"} : memref<1xi32> 
    %bd_buf_2_5_3 = aie.buffer(%tile_2_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_2_5_3"} : memref<1xi32> 
    %bd_buf_2_5_2 = aie.buffer(%tile_2_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_2_5_2"} : memref<1xi32> 
    %bd_buf_2_5_1 = aie.buffer(%tile_2_5) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_2_5_1"} : memref<1xi32> 
    %bd_buf_2_5_0 = aie.buffer(%tile_2_5) {address = 16388 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_2_5_0"} : memref<1xi32> 
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_2_5_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %tile_3_5 = aie.tile(3, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %bd_buf_3_5_5 = aie.buffer(%tile_3_5) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_3_5_5"} : memref<1xi32> 
    %bd_buf_3_5_4 = aie.buffer(%tile_3_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_3_5_4"} : memref<1xi32> 
    %bd_buf_3_5_3 = aie.buffer(%tile_3_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_3_5_3"} : memref<1xi32> 
    %bd_buf_3_5_2 = aie.buffer(%tile_3_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_3_5_2"} : memref<1xi32> 
    %bd_buf_3_5_1 = aie.buffer(%tile_3_5) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_3_5_1"} : memref<1xi32> 
    %bd_buf_3_5_0 = aie.buffer(%tile_3_5) {address = 16388 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_3_5_0"} : memref<1xi32> 
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_3_5_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %mem_tile_3_1 = aie.tile(3, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %bd_buf_3_1_31 = aie.buffer(%mem_tile_3_1) {address = 262148 : i32, mem_bank = 4 : i32, sym_name = "bd_buf_3_1_31"} : memref<1xi32> 
    %bd_buf_3_1_30 = aie.buffer(%mem_tile_3_1) {address = 458760 : i32, mem_bank = 7 : i32, sym_name = "bd_buf_3_1_30"} : memref<1xi32> 
    %bd_buf_3_1_29 = aie.buffer(%mem_tile_3_1) {address = 393224 : i32, mem_bank = 6 : i32, sym_name = "bd_buf_3_1_29"} : memref<1xi32> 
    %bd_buf_3_1_28 = aie.buffer(%mem_tile_3_1) {address = 327688 : i32, mem_bank = 5 : i32, sym_name = "bd_buf_3_1_28"} : memref<1xi32> 
    %bd_buf_3_1_27 = aie.buffer(%mem_tile_3_1) {address = 262152 : i32, mem_bank = 4 : i32, sym_name = "bd_buf_3_1_27"} : memref<1xi32> 
    %bd_buf_3_1_26 = aie.buffer(%mem_tile_3_1) {address = 196616 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_3_1_26"} : memref<1xi32> 
    %bd_buf_3_1_25 = aie.buffer(%mem_tile_3_1) {address = 131080 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_3_1_25"} : memref<1xi32> 
    %bd_buf_3_1_24 = aie.buffer(%mem_tile_3_1) {address = 65544 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_3_1_24"} : memref<1xi32> 
    %bd_buf_3_1_15 = aie.buffer(%mem_tile_3_1) {address = 8 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_3_1_15"} : memref<1xi32> 
    %bd_buf_3_1_14 = aie.buffer(%mem_tile_3_1) {address = 458756 : i32, mem_bank = 7 : i32, sym_name = "bd_buf_3_1_14"} : memref<1xi32> 
    %bd_buf_3_1_13 = aie.buffer(%mem_tile_3_1) {address = 393220 : i32, mem_bank = 6 : i32, sym_name = "bd_buf_3_1_13"} : memref<1xi32> 
    %bd_buf_3_1_12 = aie.buffer(%mem_tile_3_1) {address = 327684 : i32, mem_bank = 5 : i32, sym_name = "bd_buf_3_1_12"} : memref<1xi32> 
    %bd_buf_3_1_11 = aie.buffer(%mem_tile_3_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_3_1_11"} : memref<1xi32> 
    %bd_buf_3_1_10 = aie.buffer(%mem_tile_3_1) {address = 196612 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_3_1_10"} : memref<1xi32> 
    %bd_buf_3_1_9 = aie.buffer(%mem_tile_3_1) {address = 131076 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_3_1_9"} : memref<1xi32> 
    %bd_buf_3_1_8 = aie.buffer(%mem_tile_3_1) {address = 65540 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_3_1_8"} : memref<1xi32> 
    %bd_buf_3_1_7 = aie.buffer(%mem_tile_3_1) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_3_1_7"} : memref<1xi32> 
    %bd_buf_3_1_6 = aie.buffer(%mem_tile_3_1) {address = 458752 : i32, mem_bank = 7 : i32, sym_name = "bd_buf_3_1_6"} : memref<1xi32> 
    %bd_buf_3_1_5 = aie.buffer(%mem_tile_3_1) {address = 393216 : i32, mem_bank = 6 : i32, sym_name = "bd_buf_3_1_5"} : memref<1xi32> 
    %bd_buf_3_1_4 = aie.buffer(%mem_tile_3_1) {address = 327680 : i32, mem_bank = 5 : i32, sym_name = "bd_buf_3_1_4"} : memref<1xi32> 
    %bd_buf_3_1_3 = aie.buffer(%mem_tile_3_1) {address = 262144 : i32, mem_bank = 4 : i32, sym_name = "bd_buf_3_1_3"} : memref<1xi32> 
    %bd_buf_3_1_2 = aie.buffer(%mem_tile_3_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_3_1_2"} : memref<1xi32> 
    %bd_buf_3_1_1 = aie.buffer(%mem_tile_3_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_3_1_1"} : memref<1xi32> 
    %bd_buf_3_1_0 = aie.buffer(%mem_tile_3_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_3_1_0"} : memref<1xi32> 
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%bd_buf_3_1_0 : memref<1xi32>, 0, 0) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %mem_tile_2_1 = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %bd_buf_2_1_31 = aie.buffer(%mem_tile_2_1) {address = 262148 : i32, mem_bank = 4 : i32, sym_name = "bd_buf_2_1_31"} : memref<1xi32> 
    %bd_buf_2_1_30 = aie.buffer(%mem_tile_2_1) {address = 458760 : i32, mem_bank = 7 : i32, sym_name = "bd_buf_2_1_30"} : memref<1xi32> 
    %bd_buf_2_1_29 = aie.buffer(%mem_tile_2_1) {address = 393224 : i32, mem_bank = 6 : i32, sym_name = "bd_buf_2_1_29"} : memref<1xi32> 
    %bd_buf_2_1_28 = aie.buffer(%mem_tile_2_1) {address = 327688 : i32, mem_bank = 5 : i32, sym_name = "bd_buf_2_1_28"} : memref<1xi32> 
    %bd_buf_2_1_27 = aie.buffer(%mem_tile_2_1) {address = 262152 : i32, mem_bank = 4 : i32, sym_name = "bd_buf_2_1_27"} : memref<1xi32> 
    %bd_buf_2_1_26 = aie.buffer(%mem_tile_2_1) {address = 196616 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_2_1_26"} : memref<1xi32> 
    %bd_buf_2_1_25 = aie.buffer(%mem_tile_2_1) {address = 131080 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_2_1_25"} : memref<1xi32> 
    %bd_buf_2_1_24 = aie.buffer(%mem_tile_2_1) {address = 65544 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_2_1_24"} : memref<1xi32> 
    %bd_buf_2_1_15 = aie.buffer(%mem_tile_2_1) {address = 8 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_2_1_15"} : memref<1xi32> 
    %bd_buf_2_1_14 = aie.buffer(%mem_tile_2_1) {address = 458756 : i32, mem_bank = 7 : i32, sym_name = "bd_buf_2_1_14"} : memref<1xi32> 
    %bd_buf_2_1_13 = aie.buffer(%mem_tile_2_1) {address = 393220 : i32, mem_bank = 6 : i32, sym_name = "bd_buf_2_1_13"} : memref<1xi32> 
    %bd_buf_2_1_12 = aie.buffer(%mem_tile_2_1) {address = 327684 : i32, mem_bank = 5 : i32, sym_name = "bd_buf_2_1_12"} : memref<1xi32> 
    %bd_buf_2_1_11 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_2_1_11"} : memref<1xi32> 
    %bd_buf_2_1_10 = aie.buffer(%mem_tile_2_1) {address = 196612 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_2_1_10"} : memref<1xi32> 
    %bd_buf_2_1_9 = aie.buffer(%mem_tile_2_1) {address = 131076 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_2_1_9"} : memref<1xi32> 
    %bd_buf_2_1_8 = aie.buffer(%mem_tile_2_1) {address = 65540 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_2_1_8"} : memref<1xi32> 
    %bd_buf_2_1_7 = aie.buffer(%mem_tile_2_1) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_2_1_7"} : memref<1xi32> 
    %bd_buf_2_1_6 = aie.buffer(%mem_tile_2_1) {address = 458752 : i32, mem_bank = 7 : i32, sym_name = "bd_buf_2_1_6"} : memref<1xi32> 
    %bd_buf_2_1_5 = aie.buffer(%mem_tile_2_1) {address = 393216 : i32, mem_bank = 6 : i32, sym_name = "bd_buf_2_1_5"} : memref<1xi32> 
    %bd_buf_2_1_4 = aie.buffer(%mem_tile_2_1) {address = 327680 : i32, mem_bank = 5 : i32, sym_name = "bd_buf_2_1_4"} : memref<1xi32> 
    %bd_buf_2_1_3 = aie.buffer(%mem_tile_2_1) {address = 262144 : i32, mem_bank = 4 : i32, sym_name = "bd_buf_2_1_3"} : memref<1xi32> 
    %bd_buf_2_1_2 = aie.buffer(%mem_tile_2_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_2_1_2"} : memref<1xi32> 
    %bd_buf_2_1_1 = aie.buffer(%mem_tile_2_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_2_1_1"} : memref<1xi32> 
    %bd_buf_2_1_0 = aie.buffer(%mem_tile_2_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_2_1_0"} : memref<1xi32> 
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
    %bd_buf_0_5_5 = aie.buffer(%tile_0_5) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_0_5_5"} : memref<1xi32> 
    %bd_buf_0_5_4 = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_0_5_4"} : memref<1xi32> 
    %bd_buf_0_5_3 = aie.buffer(%tile_0_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_0_5_3"} : memref<1xi32> 
    %bd_buf_0_5_2 = aie.buffer(%tile_0_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_0_5_2"} : memref<1xi32> 
    %bd_buf_0_5_1 = aie.buffer(%tile_0_5) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_0_5_1"} : memref<1xi32> 
    %bd_buf_0_5_0 = aie.buffer(%tile_0_5) {address = 16388 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_0_5_0"} : memref<1xi32> 
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
    %bd_buf_0_4_5 = aie.buffer(%tile_0_4) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_0_4_5"} : memref<1xi32> 
    %bd_buf_0_4_4 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_0_4_4"} : memref<1xi32> 
    %bd_buf_0_4_3 = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_0_4_3"} : memref<1xi32> 
    %bd_buf_0_4_2 = aie.buffer(%tile_0_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_0_4_2"} : memref<1xi32> 
    %bd_buf_0_4_1 = aie.buffer(%tile_0_4) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_0_4_1"} : memref<1xi32> 
    %bd_buf_0_4_0 = aie.buffer(%tile_0_4) {address = 16388 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_0_4_0"} : memref<1xi32> 
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
    %bd_buf_0_3_5 = aie.buffer(%tile_0_3) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_0_3_5"} : memref<1xi32> 
    %bd_buf_0_3_4 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_0_3_4"} : memref<1xi32> 
    %bd_buf_0_3_3 = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_0_3_3"} : memref<1xi32> 
    %bd_buf_0_3_2 = aie.buffer(%tile_0_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_0_3_2"} : memref<1xi32> 
    %bd_buf_0_3_1 = aie.buffer(%tile_0_3) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_0_3_1"} : memref<1xi32> 
    %bd_buf_0_3_0 = aie.buffer(%tile_0_3) {address = 16388 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_0_3_0"} : memref<1xi32> 
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
    %bd_buf_0_2_5 = aie.buffer(%tile_0_2) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_0_2_5"} : memref<1xi32> 
    %bd_buf_0_2_4 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_0_2_4"} : memref<1xi32> 
    %bd_buf_0_2_3 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "bd_buf_0_2_3"} : memref<1xi32> 
    %bd_buf_0_2_2 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_buf_0_2_2"} : memref<1xi32> 
    %bd_buf_0_2_1 = aie.buffer(%tile_0_2) {address = 4 : i32, mem_bank = 0 : i32, sym_name = "bd_buf_0_2_1"} : memref<1xi32> 
    %bd_buf_0_2_0 = aie.buffer(%tile_0_2) {address = 16388 : i32, mem_bank = 1 : i32, sym_name = "bd_buf_0_2_0"} : memref<1xi32> 
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
