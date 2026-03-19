module {
  aie.device(npu2_1col) @base {
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
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
  }
  aie.device(npu2_1col) {
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
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
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12_i8 = arith.constant 12 : i8
      %c64 = arith.constant 64 : index
      aie.use_lock(%objFifo_in1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%objFifo_out1_prod_lock, AcquireGreaterEqual, 1)
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c64 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c64 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %4 = memref.load %objFifo_in1_cons_buff_0[%0, %2] : memref<64x64xi8>
      %5 = arith.addi %4, %c12_i8 : i8
      memref.store %5, %objFifo_out1_buff_0[%0, %2] : memref<64x64xi8>
      %6 = arith.addi %2, %c1 : index
      cf.br ^bb3(%6 : index)
    ^bb5:  // pred: ^bb3
      %7 = arith.addi %0, %c1 : index
      cf.br ^bb1(%7 : index)
    ^bb6:  // pred: ^bb1
      aie.use_lock(%objFifo_in1_cons_prod_lock, Release, 1)
      aie.use_lock(%objFifo_out1_cons_lock, Release, 1)
      aie.end
    }
    aie.shim_dma_allocation @objFifo_in0(%shim_noc_tile_0_0, MM2S, 0)
    aie.runtime_sequence @run(%arg0: memref<64x64xi8>, %arg1: memref<64x64xi8>) {
      %0 = aiex.configure @main {
        aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c0_i64, %c64_i64, %c1_i64], packet = <pkt_type = 0, pkt_id = 3>) {id = 0 : i64, metadata = @objFifo_in0} : memref<64x64xi8>
        aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c0_i64, %c64_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @objFifo_out0} : memref<64x64xi8>
        aiex.npu.dma_wait {symbol = @objFifo_out0}
      }
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
  }
}
