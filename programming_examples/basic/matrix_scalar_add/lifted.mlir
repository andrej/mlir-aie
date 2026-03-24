Note: Suppressing DMA Controller_ID write at tile(0,0) S2MM channel 0 controller_id 15

=== NPU INSTRUCTION LIFTING STARTING ===
Found 7 operations in runtime_sequence
Operation types: blockwrite=0 address_patch=2 write32=0 maskwrite32=0 sync=1
Using write32 sequence pattern matching (transaction binary mode)
Found 0 BDs with write32 sequences
Warning: Emitting 4 BDs with inferred channel S2MM_0 for tile(0,2)
DEBUG: emitAllFlows called, switchboxes=3 shimMuxes=1
DEBUG addConnection: tile(0,0) 3:3 -> 5:1
  Adding edge: (0,0) 3:3 isOut=0 -> (0,0) 5:1 isOut=1
  Adding inter-tile edge to neighbor: (0,1) 3:1 isOut=0
DEBUG addConnection: tile(0,0) 5:1 -> 3:2
  Adding edge: (0,0) 5:1 isOut=0 -> (0,0) 3:2 isOut=1
DEBUG addConnection: tile(0,1) 3:1 -> 5:1
  Adding edge: (0,1) 3:1 isOut=0 -> (0,1) 5:1 isOut=1
  Adding inter-tile edge to neighbor: (0,2) 3:1 isOut=0
DEBUG addConnection: tile(0,1) 5:3 -> 3:1
  Adding edge: (0,1) 5:3 isOut=0 -> (0,1) 3:1 isOut=1
  Adding inter-tile edge to neighbor: (0,0) 5:1 isOut=0
DEBUG addConnection: tile(0,2) 3:1 -> 1:0
  Adding edge: (0,2) 3:1 isOut=0 -> (0,2) 1:0 isOut=1
  -> FLOW SINK detected
DEBUG addConnection: tile(0,2) 1:0 -> 3:1
  Adding edge: (0,2) 1:0 isOut=0 -> (0,2) 3:1 isOut=1
  -> FLOW SOURCE detected
  Adding inter-tile edge to neighbor: (0,1) 5:1 isOut=0
DEBUG addShimMuxConfig: column 0
  Shim mux INPUT (mux mode): 1:0 -> North:3
DEBUG addConnection: tile(0,0) 1:0 -> 5:3
  Adding edge: (0,0) 1:0 isOut=0 -> (0,0) 5:3 isOut=1
  -> FLOW SOURCE detected
  Adding inter-tile edge to neighbor: (0,1) 3:3 isOut=0
  Adding explicit edge: shim_mux North:3 (output) -> switchbox South:3 (input)
  Shim mux OUTPUT (demux mode): North:2 -> 1:0
  Adding explicit edge: switchbox South:2 (output) -> shim_mux North:2 (input)
DEBUG addConnection: tile(0,0) 5:2 -> 1:0
  Adding edge: (0,0) 5:2 isOut=0 -> (0,0) 1:0 isOut=1
  -> FLOW SINK detected
DEBUG: Flow reconstruction starting
DEBUG: Number of flow sources: 2
DEBUG: Source at (0,2) 1:0 isOutput=0
DEBUG: Source at (0,0) 1:0 isOutput=0
DEBUG: Number of edges: 14
DEBUG: BFS starting from source (0,2) 1:0 isOut=0
  BFS visiting: (0,2) 1:0 isOut=0 hops=0
    Found 1 successors
      -> successor: (0,2) 3:1 isOut=1
  BFS visiting: (0,2) 3:1 isOut=1 hops=0
    Found 1 successors
      -> successor: (0,1) 5:1 isOut=0
  BFS visiting: (0,1) 5:1 isOut=0 hops=1
    Found 1 successors
      -> successor: (0,1) 3:1 isOut=1
  BFS visiting: (0,1) 3:1 isOut=1 hops=1
    Found 2 successors
      -> successor: (0,0) 5:1 isOut=0
      -> successor: (0,0) 5:1 isOut=0
  BFS visiting: (0,0) 5:1 isOut=0 hops=2
    Found 1 successors
      -> successor: (0,0) 3:2 isOut=1
  BFS visiting: (0,0) 3:2 isOut=1 hops=2
    Found 1 successors
      -> successor: (0,0) 5:2 isOut=0
  BFS visiting: (0,0) 5:2 isOut=0 hops=2
    Found 1 successors
      -> successor: (0,0) 1:0 isOut=1
  BFS visiting: (0,0) 1:0 isOut=1 hops=2
    -> FOUND SINK!
DEBUG: BFS starting from source (0,0) 1:0 isOut=0
  BFS visiting: (0,0) 1:0 isOut=0 hops=0
    Found 1 successors
      -> successor: (0,0) 5:3 isOut=1
  BFS visiting: (0,0) 5:3 isOut=1 hops=0
    Found 2 successors
      -> successor: (0,1) 3:3 isOut=0
      -> successor: (0,0) 3:3 isOut=0
  BFS visiting: (0,1) 3:3 isOut=0 hops=1
    Found 1 successors
      -> successor: (0,1) 5:3 isOut=1
  BFS visiting: (0,0) 3:3 isOut=0 hops=0
    Found 1 successors
      -> successor: (0,0) 5:1 isOut=1
  BFS visiting: (0,1) 5:3 isOut=1 hops=1
    Found 1 successors
      -> successor: (0,2) 3:3 isOut=0
  BFS visiting: (0,0) 5:1 isOut=1 hops=0
    Found 1 successors
      -> successor: (0,1) 3:1 isOut=0
  BFS visiting: (0,2) 3:3 isOut=0 hops=2
    No successors found in synthesizedEdges
  BFS visiting: (0,1) 3:1 isOut=0 hops=1
    Found 1 successors
      -> successor: (0,1) 5:1 isOut=1
  BFS visiting: (0,1) 5:1 isOut=1 hops=1
    Found 1 successors
      -> successor: (0,2) 3:1 isOut=0
  BFS visiting: (0,2) 3:1 isOut=0 hops=2
    Found 1 successors
      -> successor: (0,2) 1:0 isOut=1
  BFS visiting: (0,2) 1:0 isOut=1 hops=2
    -> FOUND SINK!
DEBUG: reconstructFlows returned 2 flows
module {
  aie.device(npu1_1col) @xclbin_device {
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 1, North : 1>
    }
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 1, South : 2>
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, South : 1>
    }
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
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
    memref.global "private" constant @config_blockwrite_data_1 : memref<8xi32> = dense<[128, 0, 0, 16777216, -2139094913, 33554432, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_0 : memref<8xi32> = dense<[128, 0, 0, 16777216, -2139094913, 33554432, 0, 33554432]>
    aie.runtime_sequence @configure() {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 128 : i32, buffer_offset = 0 : i32, burst_length = 256 : i32, column = 0 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 8 : i32, d1_stride = 127 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(0, 0, MM2S : 0) {bd_id = 0 : i32, issue_token = false, repeat_count = 0 : i32}
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 128 : i32, buffer_offset = 0 : i32, burst_length = 256 : i32, column = 0 : i32, d0_size = 16 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 8 : i32, d1_stride = 127 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(0, 0, S2MM : 0) {bd_id = 1 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aie.end
    }
  }
}
