module {
  aie.device(npu1_1col) {
    %c16_i64 = arith.constant 16 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %buf_in = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "buf_in"} : memref<16xi32> 
    %buf_out = aie.buffer(%tile_0_2) {address = 1088 : i32, sym_name = "buf_out"} : memref<16xi32> 
    %lock_in_prod = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "lock_in_prod"}
    %lock_in_cons = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "lock_in_cons"}
    %lock_out_prod = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "lock_out_prod"}
    %lock_out_cons = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "lock_out_cons"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      aie.use_lock(%lock_in_cons, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_out_prod, AcquireGreaterEqual, 1)
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c16 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %2 = memref.load %buf_in[%0] : memref<16xi32>
      memref.store %2, %buf_out[%0] : memref<16xi32>
      %3 = arith.addi %0, %c1 : index
      cf.br ^bb1(%3 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%lock_in_prod, Release, 1)
      aie.use_lock(%lock_out_cons, Release, 1)
      aie.end
    }
    aie.shim_dma_allocation @in_fifo(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out_fifo(%shim_noc_tile_0_0, S2MM, 0)
    aie.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c16_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, issue_token = true, metadata = @in_fifo} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c16_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @out_fifo} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @in_fifo}
      aiex.npu.dma_wait {symbol = @out_fifo}
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_in_prod, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_in : memref<16xi32>, 0, 16) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_in_cons, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%lock_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_out : memref<16xi32>, 0, 16) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_out_prod, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
