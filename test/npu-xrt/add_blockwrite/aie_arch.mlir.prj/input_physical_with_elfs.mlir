module {
  aie.device(npu2_1col) {
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %objFifo_in1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "objFifo_in1_cons_buff_0"} : memref<8xi32> 
    %objFifo_in1_cons_buff_1 = aie.buffer(%tile_0_2) {address = 1056 : i32, sym_name = "objFifo_in1_cons_buff_1"} : memref<8xi32> 
    %objFifo_out1_buff_0 = aie.buffer(%tile_0_2) {address = 1088 : i32, sym_name = "objFifo_out1_buff_0"} : memref<8xi32> 
    %objFifo_out1_buff_1 = aie.buffer(%tile_0_2) {address = 1120 : i32, sym_name = "objFifo_out1_buff_1"} : memref<8xi32> 
    %constant_buffer = aie.buffer(%tile_0_2) {address = 1152 : i32, sym_name = "constant_buffer"} : memref<8xi32> 
    %objFifo_in1_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "objFifo_in1_cons_prod_lock"}
    %objFifo_in1_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "objFifo_in1_cons_cons_lock"}
    %objFifo_out1_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "objFifo_out1_prod_lock"}
    %objFifo_out1_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "objFifo_out1_cons_lock"}
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {elf_file = "/scratch/roesti/mlir-aie/test/npu-xrt/add_blockwrite/aie_arch.mlir.prj/main_core_0_2.elf"}
    aie.shim_dma_allocation @objFifo_in0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @objFifo_out0(%shim_noc_tile_0_0, S2MM, 0)
    memref.global "private" @myData : memref<8xi32> = dense<[1, 2, 3, 4, 5, 6, 7, 8]>
    aie.runtime_sequence(%arg0: memref<64xi32>, %arg1: memref<32xi32>, %arg2: memref<64xi32>) {
      %0 = memref.get_global @myData : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 0 : ui32, buffer = @constant_buffer} : memref<8xi32>
      aiex.npu.write32 {address = 4 : ui32, buffer = @constant_buffer, value = 42 : ui32}
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c64_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, issue_token = true, metadata = @objFifo_in0} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c64_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @objFifo_out0} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @objFifo_in0}
      aiex.npu.dma_wait {symbol = @objFifo_out0}
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in1_cons_buff_0 : memref<8xi32>, 0, 8) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in1_cons_buff_1 : memref<8xi32>, 0, 8) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_out1_buff_0 : memref<8xi32>, 0, 8) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_out1_buff_1 : memref<8xi32>, 0, 8) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 1, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<North : 1, South : 1>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, South : 1>
    }
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
