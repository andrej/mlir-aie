module {
  aie.device(npu2_1col) @base {
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<2> (3)
      %1 = aie.amsel<3> (3)
      %2 = aie.amsel<4> (3)
      %3 = aie.amsel<5> (3)
      %4 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      %5 = aie.masterset(North : 1, %1)
      %6 = aie.masterset(North : 4, %2)
      %7 = aie.masterset(TileControl : 0, %3) {keep_pkt_header = true}
      aie.packet_rules(South : 3) {
        aie.rule(31, 27, %1)
        aie.rule(31, 26, %2)
        aie.rule(31, 15, %3)
      }
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<4> (3)
      %1 = aie.amsel<5> (3)
      %2 = aie.masterset(North : 1, %1)
      %3 = aie.masterset(TileControl : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(South : 1) {
        aie.rule(31, 27, %1)
      }
      aie.packet_rules(South : 4) {
        aie.rule(31, 26, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(TileControl : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(South : 1) {
        aie.rule(31, 27, %0)
      }
    }
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%shim_noc_tile_0_0, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%shim_noc_tile_0_0, MM2S, 0)
    aie.packet_flow(26) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(27) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
  }
  aie.device(npu2_1col) {
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<2> (3)
      %2 = aie.amsel<3> (3)
      %3 = aie.amsel<4> (3)
      %4 = aie.amsel<5> (3)
      %5 = aie.masterset(South : 2, %0)
      %6 = aie.masterset(South : 0, %1) {keep_pkt_header = true}
      %7 = aie.masterset(North : 1, %2)
      %8 = aie.masterset(North : 4, %3)
      %9 = aie.masterset(TileControl : 0, %4) {keep_pkt_header = true}
      aie.packet_rules(South : 3) {
        aie.rule(31, 27, %2)
        aie.rule(6, 2, %3)
        aie.rule(31, 15, %4)
      }
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %1)
      }
      aie.packet_rules(North : 2) {
        aie.rule(31, 2, %0)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.amsel<3> (0)
      %4 = aie.amsel<4> (3)
      %5 = aie.amsel<5> (3)
      %6 = aie.masterset(DMA : 0, %2)
      %7 = aie.masterset(DMA : 1, %3)
      %8 = aie.masterset(South : 2, %1)
      %9 = aie.masterset(North : 1, %5)
      %10 = aie.masterset(North : 5, %0)
      %11 = aie.masterset(TileControl : 0, %4) {keep_pkt_header = true}
      aie.packet_rules(South : 1) {
        aie.rule(31, 27, %5)
      }
      aie.packet_rules(South : 4) {
        aie.rule(31, 26, %4)
        aie.rule(31, 3, %2)
      }
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 2, %1)
      }
      aie.packet_rules(North : 0) {
        aie.rule(31, 1, %3)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<5> (3)
      %3 = aie.masterset(DMA : 0, %1)
      %4 = aie.masterset(South : 0, %0)
      %5 = aie.masterset(TileControl : 0, %2) {keep_pkt_header = true}
      aie.packet_rules(South : 1) {
        aie.rule(31, 27, %2)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 1, %0)
      }
      aie.packet_rules(South : 5) {
        aie.rule(31, 0, %1)
      }
    }
    %objFifo_in1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "objFifo_in1_cons_buff_0"} : memref<64x64xi8> 
    %objFifo_in1_cons_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "objFifo_in1_cons_buff_1"} : memref<64x64xi8> 
    %objFifo_out1_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "objFifo_out1_buff_0"} : memref<64x64xi8> 
    %objFifo_out1_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "objFifo_out1_buff_1"} : memref<64x64xi8> 
    %objFifo_in1_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "objFifo_in1_cons_prod_lock"}
    %objFifo_in1_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "objFifo_in1_cons_cons_lock"}
    %objFifo_out1_prod_lock = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "objFifo_out1_prod_lock"}
    %objFifo_out1_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "objFifo_out1_cons_lock"}
    aie.packet_flow(0) {
      aie.packet_source<%mem_tile_0_1, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 1>
    }
    aie.packet_flow(2) {
      aie.packet_source<%mem_tile_0_1, DMA : 1>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 0>
    }
    aie.packet_flow(3) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 0>
    }
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {elf_file = "/scratch/roesti/mlir-aie/test/npu-xrt/ctrl_packet_reconfig/aie_overlay.mlir.prj/main_core_0_2.elf"}
    aie.shim_dma_allocation @objFifo_in0(%shim_noc_tile_0_0, MM2S, 0)
    memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[1024, 0, 1075314688, 16777216, -1006632945, 33554432, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_1 : memref<8xi32> = dense<[1024, 0, 0, 16777216, -1006632945, 33554432, 0, 33554432]>
    aie.runtime_sequence @run(%arg0: memref<64x64xi8>, %arg1: memref<64x64xi8>) {
      aiex.npu.load_pdi {device_ref = @main}
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119316 : ui32, value = 0 : ui32}
      %1 = memref.get_global @blockwrite_data_1 : memref<8xi32>
      aiex.npu.blockwrite(%1) {address = 118816 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.maskwrite32 {address = 119296 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119300 : ui32, value = 2147483649 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %objFifo_in0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "objFifo_in0_cons_buff_0"} : memref<64x64xi8> 
      %objFifo_in0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "objFifo_in0_cons_buff_1"} : memref<64x64xi8> 
      %objFifo_out0_buff_0 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "objFifo_out0_buff_0"} : memref<64x64xi8> 
      %objFifo_out0_buff_1 = aie.buffer(%mem_tile_0_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "objFifo_out0_buff_1"} : memref<64x64xi8> 
      %objFifo_in0_cons_prod_lock = aie.lock(%mem_tile_0_1, 0) {init = 1 : i32, sym_name = "objFifo_in0_cons_prod_lock"}
      %objFifo_in0_cons_cons_lock = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_cons_lock"}
      %objFifo_out0_prod_lock = aie.lock(%mem_tile_0_1, 2) {init = 1 : i32, sym_name = "objFifo_out0_prod_lock"}
      %objFifo_out0_cons_lock = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "objFifo_out0_cons_lock"}
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<64x64xi8>) {bd_id = 0 : i32, next_bd_id = 0 : i32}
        aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<64x64xi8>) {bd_id = 1 : i32, next_bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      }]
      %2 = aie.dma(MM2S, 1) [{
        aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out0_buff_0 : memref<64x64xi8>) {bd_id = 24 : i32, next_bd_id = 24 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
      }]
      %3 = aie.dma(S2MM, 1) [{
        aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out0_buff_0 : memref<64x64xi8>) {bd_id = 25 : i32, next_bd_id = 25 : i32}
        aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
      }]
      aie.end
    }
    aie.shim_dma_allocation @objFifo_out0(%shim_noc_tile_0_0, S2MM, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in1_cons_buff_0 : memref<64x64xi8>) {bd_id = 0 : i32, next_bd_id = 0 : i32}
        aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out1_buff_0 : memref<64x64xi8>) {bd_id = 1 : i32, next_bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
      }]
      aie.end
    }
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%shim_noc_tile_0_0, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%shim_noc_tile_0_0, MM2S, 0)
    aie.packet_flow(26) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(27) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
  }
}
