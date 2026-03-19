module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    aie.shim_dma_allocation @objfifo_out_shim_alloc_2(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @objfifo_in_shim_alloc_2(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @objfifo_out_shim_alloc_1(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @objfifo_in_shim_alloc_1(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @objfifo_out_shim_alloc_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @objfifo_in_shim_alloc_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @objfifo_in_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @objfifo_out_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[4, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
    aie.runtime_sequence(%arg0: memref<512xi32>) {
      aiex.npu.load_pdi {device_ref = @add_two, id = 2 : i32}
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4], strides: [1] : memref<512xi32> to memref<4xi32>
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      %1 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%1) {address = 118816 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119316 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 119296 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119300 : ui32, value = 2147483649 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      %subview = memref.subview %arg0[12] [4] [1] : memref<512xi32> to memref<4xi32, strided<[1], offset: 12>>
      %reinterpret_cast_0 = memref.reinterpret_cast %subview to offset: [0], sizes: [4], strides: [1] : memref<4xi32, strided<[1], offset: 12>> to memref<4xi32>
      %2 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%2) {address = 118816 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 0 : i32, arg_plus = 48 : i32}
      %3 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%3) {address = 118848 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118852 : ui32, arg_idx = 0 : i32, arg_plus = 48 : i32}
      aiex.npu.write32 {address = 119316 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 119296 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119300 : ui32, value = 2147483650 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.load_pdi {device_ref = @add_three, id = 3 : i32}
      %subview_1 = memref.subview %arg0[4] [4] [1] : memref<512xi32> to memref<4xi32, strided<[1], offset: 4>>
      %reinterpret_cast_2 = memref.reinterpret_cast %subview_1 to offset: [0], sizes: [4], strides: [1] : memref<4xi32, strided<[1], offset: 4>> to memref<4xi32>
      %4 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%4) {address = 118848 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118852 : ui32, arg_idx = 0 : i32, arg_plus = 16 : i32}
      %5 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%5) {address = 118880 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118884 : ui32, arg_idx = 0 : i32, arg_plus = 16 : i32}
      aiex.npu.write32 {address = 119316 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 119296 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119300 : ui32, value = 2147483651 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      %subview_3 = memref.subview %arg0[12] [4] [1] : memref<512xi32> to memref<4xi32, strided<[1], offset: 12>>
      %reinterpret_cast_4 = memref.reinterpret_cast %subview_3 to offset: [0], sizes: [4], strides: [1] : memref<4xi32, strided<[1], offset: 12>> to memref<4xi32>
      %6 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%6) {address = 118880 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118884 : ui32, arg_idx = 0 : i32, arg_plus = 48 : i32}
      %7 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%7) {address = 118912 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118916 : ui32, arg_idx = 0 : i32, arg_plus = 48 : i32}
      aiex.npu.write32 {address = 119316 : ui32, value = 3 : ui32}
      aiex.npu.maskwrite32 {address = 119296 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119300 : ui32, value = 2147483652 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
  aie.device(npu2) @add_two {
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %objfifo_out_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "objfifo_out_cons_prod_lock_0"}
    %objfifo_out_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "objfifo_out_cons_cons_lock_0"}
    %objfifo_out_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "objfifo_out_buff_0"} : memref<4xi32> 
    %objfifo_out_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "objfifo_out_prod_lock_0"}
    %objfifo_out_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "objfifo_out_cons_lock_0"}
    %objfifo_in_cons_buff_0 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "objfifo_in_cons_buff_0"} : memref<4xi32> 
    %objfifo_in_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "objfifo_in_cons_prod_lock_0"}
    %objfifo_in_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "objfifo_in_cons_cons_lock_0"}
    %objfifo_in_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "objfifo_in_prod_lock_0"}
    %objfifo_in_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "objfifo_in_cons_lock_0"}
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {elf_file = "/scratch/roesti/mlir-aie/test/npu-xrt/reconfigure_loadpdi/aie.mlir.prj/add_two_core_0_2.elf"}
    memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[4, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
    aie.runtime_sequence @add_two_sequence(%arg0: memref<4xi32>) {
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      %1 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%1) {address = 118816 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119316 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 119296 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119300 : ui32, value = 2147483649 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    aie.shim_dma_allocation @objfifo_in_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%objfifo_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%objfifo_in_cons_buff_0 : memref<4xi32>, 0, 4) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%objfifo_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%objfifo_out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%objfifo_out_buff_0 : memref<4xi32>, 0, 4) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%objfifo_out_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    aie.shim_dma_allocation @objfifo_out_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
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
  aie.device(npu2) @add_three {
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %objfifo_out_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "objfifo_out_cons_prod_lock_0"}
    %objfifo_out_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "objfifo_out_cons_cons_lock_0"}
    %objfifo_out_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "objfifo_out_buff_0"} : memref<4xi32> 
    %objfifo_out_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "objfifo_out_prod_lock_0"}
    %objfifo_out_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "objfifo_out_cons_lock_0"}
    %objfifo_in_cons_buff_0 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "objfifo_in_cons_buff_0"} : memref<4xi32> 
    %objfifo_in_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "objfifo_in_cons_prod_lock_0"}
    %objfifo_in_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "objfifo_in_cons_cons_lock_0"}
    %objfifo_in_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "objfifo_in_prod_lock_0"}
    %objfifo_in_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "objfifo_in_cons_lock_0"}
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {elf_file = "/scratch/roesti/mlir-aie/test/npu-xrt/reconfigure_loadpdi/aie.mlir.prj/add_three_core_0_2.elf"}
    memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[4, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
    aie.runtime_sequence @add_three_sequence(%arg0: memref<4xi32>) {
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      %1 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%1) {address = 118816 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119316 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 119296 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119300 : ui32, value = 2147483649 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    aie.shim_dma_allocation @objfifo_in_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%objfifo_in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%objfifo_in_cons_buff_0 : memref<4xi32>, 0, 4) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%objfifo_in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%objfifo_out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%objfifo_out_buff_0 : memref<4xi32>, 0, 4) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%objfifo_out_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    aie.shim_dma_allocation @objfifo_out_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
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
