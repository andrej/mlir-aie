module {
  aie.device(npu2_1col) @base {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
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
  }
  aie.device(npu2_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %objFifo_in1_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "objFifo_in1_cons_buff_0"} : memref<64x64xi8> 
    %objFifo_in1_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "objFifo_in1_cons_buff_1"} : memref<64x64xi8> 
    %objFifo_out1_buff_0 = aie.buffer(%tile_0_2) {sym_name = "objFifo_out1_buff_0"} : memref<64x64xi8> 
    %objFifo_out1_buff_1 = aie.buffer(%tile_0_2) {sym_name = "objFifo_out1_buff_1"} : memref<64x64xi8> 
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
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12_i8 = arith.constant 12 : i8
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      aie.use_lock(%objFifo_in1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%objFifo_out1_prod_lock, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c64 step %c1 {
        scf.for %arg1 = %c0 to %c64 step %c1 {
          %0 = memref.load %objFifo_in1_cons_buff_0[%arg0, %arg1] : memref<64x64xi8>
          %1 = arith.addi %0, %c12_i8 : i8
          memref.store %1, %objFifo_out1_buff_0[%arg0, %arg1] : memref<64x64xi8>
        }
      }
      aie.use_lock(%objFifo_in1_cons_prod_lock, Release, 1)
      aie.use_lock(%objFifo_out1_cons_lock, Release, 1)
      aie.end
    }
    aie.shim_dma_allocation @objFifo_in0(%shim_noc_tile_0_0, MM2S, 0)
    aie.runtime_sequence @run(%arg0: memref<64x64xi8>, %arg1: memref<64x64xi8>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c56_i64 = arith.constant 56 : i64
      %c61_i64 = arith.constant 61 : i64
      %c64_i64 = arith.constant 64 : i64
      aiex.configure @main {
        aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c0_i64, %c64_i64, %c1_i64], packet = <pkt_type = 0, pkt_id = 3>) {id = 0 : i64, metadata = @objFifo_in0} : memref<64x64xi8>
        aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c64_i64, %c64_i64][%c0_i64, %c0_i64, %c64_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @objFifo_out0} : memref<64x64xi8>
        aiex.npu.dma_wait {symbol = @objFifo_out0}
      }
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %objFifo_in0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {sym_name = "objFifo_in0_cons_buff_0"} : memref<64x64xi8> 
      %objFifo_in0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {sym_name = "objFifo_in0_cons_buff_1"} : memref<64x64xi8> 
      %objFifo_out0_buff_0 = aie.buffer(%mem_tile_0_1) {sym_name = "objFifo_out0_buff_0"} : memref<64x64xi8> 
      %objFifo_out0_buff_1 = aie.buffer(%mem_tile_0_1) {sym_name = "objFifo_out0_buff_1"} : memref<64x64xi8> 
      %objFifo_in0_cons_prod_lock = aie.lock(%mem_tile_0_1, 0) {init = 1 : i32, sym_name = "objFifo_in0_cons_prod_lock"}
      %objFifo_in0_cons_cons_lock = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_cons_lock"}
      %objFifo_out0_prod_lock = aie.lock(%mem_tile_0_1, 2) {init = 1 : i32, sym_name = "objFifo_out0_prod_lock"}
      %objFifo_out0_cons_lock = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "objFifo_out0_cons_lock"}
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<64x64xi8>)
        aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<64x64xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      }]
      %2 = aie.dma(MM2S, 1) [{
        aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out0_buff_0 : memref<64x64xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
      }]
      %3 = aie.dma(S2MM, 1) [{
        aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out0_buff_0 : memref<64x64xi8>)
        aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
      }]
      aie.end
    }
    aie.shim_dma_allocation @objFifo_out0(%shim_noc_tile_0_0, S2MM, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in1_cons_buff_0 : memref<64x64xi8>)
        aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out1_buff_0 : memref<64x64xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
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
  }
}

