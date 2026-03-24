Warning: Emitting 4 BDs with inferred channel S2MM_0 for tile(0,2)
Warning: Emitting 8 BDs with inferred channel S2MM_0 for memtile(0,1)
DEBUG: emitAllFlows called, switchboxes=3 shimMuxes=1
DEBUG addConnection: tile(0,0) 3:3 -> 5:4
  Adding edge: (0,0) 3:3 isOut=0 -> (0,0) 5:4 isOut=1
  Adding inter-tile edge to neighbor: (0,1) 3:4 isOut=0
DEBUG addConnection: tile(0,0) 5:2 -> 3:2
  Adding edge: (0,0) 5:2 isOut=0 -> (0,0) 3:2 isOut=1
DEBUG addConnection: tile(0,1) 5:0 -> 1:0
  Adding edge: (0,1) 5:0 isOut=0 -> (0,1) 1:0 isOut=1
  -> FLOW SINK detected
DEBUG addConnection: tile(0,1) 1:0 -> 5:1
  Adding edge: (0,1) 1:0 isOut=0 -> (0,1) 5:1 isOut=1
  -> FLOW SOURCE detected
  Adding inter-tile edge to neighbor: (0,2) 3:1 isOut=0
DEBUG addConnection: tile(0,1) 5:2 -> 1:1
  Adding edge: (0,1) 5:2 isOut=0 -> (0,1) 1:1 isOut=1
  -> FLOW SINK detected
DEBUG addConnection: tile(0,1) 1:1 -> 3:2
  Adding edge: (0,1) 1:1 isOut=0 -> (0,1) 3:2 isOut=1
  -> FLOW SOURCE detected
  Adding inter-tile edge to neighbor: (0,0) 5:2 isOut=0
DEBUG addConnection: tile(0,2) 3:1 -> 1:0
  Adding edge: (0,2) 3:1 isOut=0 -> (0,2) 1:0 isOut=1
  -> FLOW SINK detected
DEBUG addConnection: tile(0,2) 1:0 -> 3:0
  Adding edge: (0,2) 1:0 isOut=0 -> (0,2) 3:0 isOut=1
  -> FLOW SOURCE detected
  Adding inter-tile edge to neighbor: (0,1) 5:0 isOut=0
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
DEBUG: Number of flow sources: 4
DEBUG: Source at (0,1) 1:0 isOutput=0
DEBUG: Source at (0,1) 1:1 isOutput=0
DEBUG: Source at (0,2) 1:0 isOutput=0
DEBUG: Source at (0,0) 1:0 isOutput=0
DEBUG: Number of edges: 15
DEBUG: BFS starting from source (0,1) 1:0 isOut=0
  BFS visiting: (0,1) 1:0 isOut=0 hops=0
    Found 1 successors
      -> successor: (0,1) 5:1 isOut=1
  BFS visiting: (0,1) 5:1 isOut=1 hops=0
    Found 1 successors
      -> successor: (0,2) 3:1 isOut=0
  BFS visiting: (0,2) 3:1 isOut=0 hops=1
    Found 1 successors
      -> successor: (0,2) 1:0 isOut=1
  BFS visiting: (0,2) 1:0 isOut=1 hops=1
    -> FOUND SINK!
DEBUG: BFS starting from source (0,1) 1:1 isOut=0
  BFS visiting: (0,1) 1:1 isOut=0 hops=0
    Found 1 successors
      -> successor: (0,1) 3:2 isOut=1
  BFS visiting: (0,1) 3:2 isOut=1 hops=0
    Found 1 successors
      -> successor: (0,0) 5:2 isOut=0
  BFS visiting: (0,0) 5:2 isOut=0 hops=1
    Found 2 successors
      -> successor: (0,0) 3:2 isOut=1
      -> successor: (0,0) 1:0 isOut=1
  BFS visiting: (0,0) 3:2 isOut=1 hops=1
    Found 1 successors
  BFS visiting: (0,0) 1:0 isOut=1 hops=1
    -> FOUND SINK!
DEBUG: BFS starting from source (0,2) 1:0 isOut=0
  BFS visiting: (0,2) 1:0 isOut=0 hops=0
    Found 1 successors
      -> successor: (0,2) 3:0 isOut=1
  BFS visiting: (0,2) 3:0 isOut=1 hops=0
    Found 1 successors
      -> successor: (0,1) 5:0 isOut=0
  BFS visiting: (0,1) 5:0 isOut=0 hops=1
    Found 1 successors
      -> successor: (0,1) 1:0 isOut=1
  BFS visiting: (0,1) 1:0 isOut=1 hops=1
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
      -> successor: (0,0) 5:4 isOut=1
  BFS visiting: (0,1) 5:3 isOut=1 hops=1
    Found 1 successors
      -> successor: (0,2) 3:3 isOut=0
  BFS visiting: (0,0) 5:4 isOut=1 hops=0
    Found 1 successors
      -> successor: (0,1) 3:4 isOut=0
  BFS visiting: (0,2) 3:3 isOut=0 hops=2
    No successors found in synthesizedEdges
  BFS visiting: (0,1) 3:4 isOut=0 hops=1
    Found 1 successors
      -> successor: (0,1) 5:4 isOut=1
  BFS visiting: (0,1) 5:4 isOut=1 hops=1
    Found 1 successors
      -> successor: (0,2) 3:4 isOut=0
  BFS visiting: (0,2) 3:4 isOut=0 hops=2
    No successors found in synthesizedEdges
DEBUG: reconstructFlows returned 3 flows
module {
  aie.device(npu1_1col) @xclbin_device {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 4>
      aie.connect<North : 2, South : 2>
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<North : 0, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<North : 2, DMA : 1>
      aie.connect<DMA : 1, South : 2>
    }
    %bd_buf_0_1_27 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_27"} : memref<1xi32> 
    %bd_buf_0_1_26 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_26"} : memref<1xi32> 
    %bd_buf_0_1_25 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_25"} : memref<1xi32> 
    %bd_buf_0_1_24 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_24"} : memref<1xi32> 
    %bd_buf_0_1_3 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_3"} : memref<1xi32> 
    %bd_buf_0_1_2 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_2"} : memref<1xi32> 
    %bd_buf_0_1_1 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_1"} : memref<1xi32> 
    %bd_buf_0_1_0 = aie.buffer(%mem_tile_0_1) {sym_name = "bd_buf_0_1_0"} : memref<1xi32> 
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb9)
    ^bb1:  // 9 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8
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
    ^bb5:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_24 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb6:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_25 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb7:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_26 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb8:  // no predecessors
      aie.dma_bd(%bd_buf_0_1_27 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb9:  // pred: ^bb0
      aie.end
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, South : 0>
    }
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 0)
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
    aie.runtime_sequence @configure() {
      aie.end
    }
  }
}
