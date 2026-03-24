Note: Suppressing DMA Controller_ID write at tile(0,0) S2MM channel 0 controller_id 15

=== NPU INSTRUCTION LIFTING STARTING ===
Found 7 operations in runtime_sequence
Operation types: blockwrite=0 address_patch=2 write32=0 maskwrite32=0 sync=1
Using write32 sequence pattern matching (transaction binary mode)
Found 0 BDs with write32 sequences
Warning: Emitting 4 BDs with inferred channel S2MM_0 for tile(0,2)
Warning: Emitting 4 BDs with inferred channel S2MM_0 for memtile(0,1)
module {
  aie.device(npu1_1col) @xclbin_device {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 4>
      aie.connect<North : 1, South : 2>
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<DMA : 0, North : 1>
      aie.connect<North : 0, DMA : 0>
    }
    %bd_buf_0_1_3 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_3"} : memref<1xi32> 
    %bd_buf_0_1_2 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_2"} : memref<1xi32> 
    %bd_buf_0_1_1 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_1"} : memref<1xi32> 
    %bd_buf_0_1_0 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_0"} : memref<1xi32> 
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 5 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4
      aie.dma_bd(%bd_buf_0_1_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // pred: ^bb0
      aie.end
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, South : 1>
    }
    %bd_buf_0_2_3 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_3"} : memref<1xi32> 
    %bd_buf_0_2_2 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_2"} : memref<1xi32> 
    %bd_buf_0_2_1 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_1"} : memref<1xi32> 
    %bd_buf_0_2_0 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_0"} : memref<1xi32> 
    %lock_0_2 = aie.lock(%tile_0_2, 3) {init = 0 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 1) {init = 0 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 0) {init = 2 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 5 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4
      aie.dma_bd(%bd_buf_0_2_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_0_2_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_0_2_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_0_2_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // pred: ^bb0
      aie.end
    }
    memref.global "private" constant @config_blockwrite_data_1 : memref<8xi32> = dense<[262144, 0, 0, 67108864, -2113925121, 33685503, 66060351, 33554432]>
    memref.global "private" constant @config_blockwrite_data_0 : memref<8xi32> = dense<[262144, 0, 0, 33554432, -2080370689, 33554463, 66322431, 33554432]>
    aie.runtime_sequence @configure() {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 262144 : i32, buffer_offset = 0 : i32, burst_length = 256 : i32, column = 0 : i32, d0_size = 32 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 64 : i32, d1_stride = 4095 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 31 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 63 : i32, iteration_stride = 262143 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(0, 0, MM2S : 0) {bd_id = 0 : i32, issue_token = false, repeat_count = 63 : i32}
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 262144 : i32, buffer_offset = 0 : i32, burst_length = 256 : i32, column = 0 : i32, d0_size = 64 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 32 : i32, d1_stride = 4095 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 131071 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 63 : i32, iteration_stride = 63 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(0, 0, S2MM : 0) {bd_id = 1 : i32, issue_token = true, repeat_count = 63 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aie.end
    }
  }
}
