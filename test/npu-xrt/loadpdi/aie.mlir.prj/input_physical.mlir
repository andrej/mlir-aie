module {
  aie.device(npu2) @add_two {
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %objfifo_out_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "objfifo_out_cons_prod_lock_0"}
    %objfifo_out_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "objfifo_out_cons_cons_lock_0"}
    %objfifo_out_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "objfifo_out_buff_0"} : memref<128xi32> 
    %objfifo_out_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "objfifo_out_prod_lock_0"}
    %objfifo_out_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "objfifo_out_cons_lock_0"}
    %objfifo_in_cons_buff_0 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "objfifo_in_cons_buff_0"} : memref<128xi32> 
    %objfifo_in_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "objfifo_in_cons_prod_lock_0"}
    %objfifo_in_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "objfifo_in_cons_cons_lock_0"}
    %objfifo_in_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "objfifo_in_prod_lock_0"}
    %objfifo_in_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "objfifo_in_cons_lock_0"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2_i32 = arith.constant 2 : i32
      %c128 = arith.constant 128 : index
      %c16777214 = arith.constant 16777214 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c16777214 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      aie.use_lock(%objfifo_in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%objfifo_out_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c128 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %4 = memref.load %objfifo_in_cons_buff_0[%2] : memref<128xi32>
      %5 = arith.addi %4, %c2_i32 : i32
      memref.store %5, %objfifo_out_buff_0[%2] : memref<128xi32>
      %6 = arith.addi %2, %c1 : index
      cf.br ^bb3(%6 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%objfifo_in_cons_prod_lock_0, Release, 1)
      aie.use_lock(%objfifo_out_cons_lock_0, Release, 1)
      %7 = arith.addi %0, %c1 : index
      cf.br ^bb1(%7 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    }
    aiex.runtime_sequence(%arg0: memref<512xi32>) {
      aiex.npu.load_pdi {device_ref = @add_two, id = 1 : i32}
      %0 = aiex.dma_configure_task_for @objfifo_in_shim_alloc {
        aie.dma_bd(%arg0 : memref<512xi32>, 0, 512)
        aie.end
      }
      %1 = aiex.dma_configure_task_for @objfifo_out_shim_alloc {
        aie.dma_bd(%arg0 : memref<512xi32>, 0, 512)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
    }
    aie.shim_dma_allocation @objfifo_in_shim_alloc (%tile_0_0, MM2S, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%objfifo_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%objfifo_in_cons_buff_0 : memref<128xi32>, 0, 128) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%objfifo_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%objfifo_out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%objfifo_out_buff_0 : memref<128xi32>, 0, 128) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%objfifo_out_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    aie.shim_dma_allocation @objfifo_out_shim_alloc (%tile_0_0, S2MM, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 0, South : 2>
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
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, South : 0>
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

