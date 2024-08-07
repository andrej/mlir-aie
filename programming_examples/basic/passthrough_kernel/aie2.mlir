module {
  aie.device(npu1_1col) {
    memref.global "public" @out_cons : memref<4xui8>
    memref.global "public" @out : memref<4xui8>
    memref.global "public" @in_L2L1_cons : memref<4xui8>
    memref.global "public" @in_L2L1 : memref<4xui8>
    memref.global "public" @in_L3L2_cons : memref<4xui8>
    memref.global "public" @in_L3L2 : memref<4xui8>
    func.func private @passThroughLine(memref<4xui8>, memref<4xui8>, i32)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %out_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 0 : i32, sym_name = "out_cons_prod_lock"}
    %out_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "out_cons_cons_lock"}
    %out_buff_0 = aie.buffer(%tile_0_2) {sym_name = "out_buff_0"} : memref<4xui8> 
    %out_buff_1 = aie.buffer(%tile_0_2) {sym_name = "out_buff_1"} : memref<4xui8> 
    %out_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "out_prod_lock"}
    %out_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out_cons_lock"}
    %in_L2L1_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "in_L2L1_cons_buff_0"} : memref<4xui8> 
    %in_L2L1_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "in_L2L1_cons_buff_1"} : memref<4xui8> 
    %in_L2L1_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "in_L2L1_cons_prod_lock"}
    %in_L2L1_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in_L2L1_cons_cons_lock"}
    %in_L3L2_cons_buff_0 = aie.buffer(%tile_0_1) {sym_name = "in_L3L2_cons_buff_0"} : memref<4xui8> 
    %in_L3L2_cons_buff_1 = aie.buffer(%tile_0_1) {sym_name = "in_L3L2_cons_buff_1"} : memref<4xui8> 

    // -- attention --
    // We will repeat from mem tile memory onto the stream REPEAT_COUNT (3) times.
    // In other words: Each time the memtile receives something from the shim (S2MM), it will send that out three times (MM2S).

    // The following locks are acquired by S2MM and released by MM2S.
    // Since S2MM should run once for every three runs of MM2S, the S2MM side must acquire with a value of 3,
    // and the MM2S side releases by 1 each time it runs. This achieves one run of S2MM for three runs of MM2S.
    // We need separate locks for the ping and pong buffers; if we had one lock initialized to value 6,
    // the "pong" consumer side (acquiring only a value of 1) may start running even if only the ping producer side
    // completed (releasing with a value of 3).
    %in_L3L2_cons_prod_ping_lock = aie.lock(%tile_0_1, 0) {init = 3 : i32, sym_name = "in_L3L2_cons_prod_ping_lock"} 
    %in_L3L2_cons_prod_pong_lock = aie.lock(%tile_0_1, 1) {init = 3 : i32, sym_name = "in_L3L2_cons_prod_pong_lock"} 
    // The following locks are acquired by MM2S and released by S2MM.
    // Since any buffer received once should be sent out three times, the receiving side (S2MM) will release this 
    // with a value of 3 for each received buffer. The sending side (MM2S) will acquire with a value of 1 
    // each time it consumes a buffer.
    %in_L3L2_cons_cons_ping_lock = aie.lock(%tile_0_1, 2) {init = 0 : i32, sym_name = "in_L3L2_cons_cons_ping_lock"}
    %in_L3L2_cons_cons_pong_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "in_L3L2_cons_cons_pong_lock"}
    // -- /attention--

    %in_L3L2_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "in_L3L2_prod_lock"}
    %in_L3L2_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "in_L3L2_cons_lock"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c9223372036854775806 step %c2 {
        aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%in_L2L1_cons_cons_lock, AcquireGreaterEqual, 1)
        %c4_i32_0 = arith.constant 4 : i32
        memref.copy %in_L2L1_cons_buff_0, %out_buff_0 : memref<4xui8> to memref<4xui8>
        aie.use_lock(%in_L2L1_cons_prod_lock, Release, 1)
        aie.use_lock(%out_cons_lock, Release, 1)
        aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%in_L2L1_cons_cons_lock, AcquireGreaterEqual, 1)
        %c4_i32_1 = arith.constant 4 : i32
        memref.copy %in_L2L1_cons_buff_1, %out_buff_1 : memref<4xui8> to memref<4xui8>
        aie.use_lock(%in_L2L1_cons_prod_lock, Release, 1)
        aie.use_lock(%out_cons_lock, Release, 1)
      }
      aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%in_L2L1_cons_cons_lock, AcquireGreaterEqual, 1)
      %c4_i32 = arith.constant 4 : i32
      memref.copy %in_L2L1_cons_buff_0, %out_buff_0 : memref<4xui8> to memref<4xui8>
      aie.use_lock(%in_L2L1_cons_prod_lock, Release, 1)
      aie.use_lock(%out_cons_lock, Release, 1)
      aie.end
    }
    aie.shim_dma_allocation @in_L3L2(MM2S, 0, 0)
    aiex.runtime_sequence(%arg0: memref<16xui8>, %arg1: memref<16xui8>, %arg2: memref<16xui8>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 0 : i64, metadata = @in_L3L2} : memref<16xui8>

      // -- attention --
      // Since we are repeating three times, the output size transfer should be 3 times the input size.
      // Note that no repeating occurs on the shim. We are only repeating in the mem tile.
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 48][0, 0, 0, 1]) {id = 1 : i64, metadata = @out} : memref<16xui8>  // transfer len = REPEAT_COUNT * PASSTHROUGH_SIZE
      // -- /attention --

      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {

    // -- attention --
    // This is the receiving side of the DMA.
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:
      // Next lock asks: MM2S side, are you done with the ping buffer? If so, proceed.
      aie.use_lock(%in_L3L2_cons_prod_ping_lock, AcquireGreaterEqual, 3) // lock acquire = REPEAT_COUNT
      aie.dma_bd(%in_L3L2_cons_buff_0 : memref<4xui8>, 0, 4)
      // Next lock says: MM2S side, I have new data for you to consume.
      aie.use_lock(%in_L3L2_cons_cons_ping_lock, Release, 3) // lock release = REPEAT_COUNT
      aie.next_bd ^bb2
    ^bb2:
      // As above, but for the pong buffer.
      aie.use_lock(%in_L3L2_cons_prod_pong_lock, AcquireGreaterEqual, 3) // lock acquire = REPEAT_COUNT
      aie.dma_bd(%in_L3L2_cons_buff_1 : memref<4xui8>, 0, 4)
      aie.use_lock(%in_L3L2_cons_cons_pong_lock, Release, 3)  // lock release = REPEAT_COUNT
      aie.next_bd ^bb1
    
    // This is the sending side of the DMA.
    // This does the actual repetition, since it only acquires the locks with a value of 1, and the S2MM side needs to acqurie a value of 3.
    ^bb3:
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6, repeat_count = 2)  // repeat_count = (REPEAT_COUNT - 1)
    ^bb4:
      aie.use_lock(%in_L3L2_cons_cons_ping_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_L3L2_cons_buff_0 : memref<4xui8>, 0, 4)
      aie.use_lock(%in_L3L2_cons_prod_ping_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:
      aie.use_lock(%in_L3L2_cons_cons_pong_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_L3L2_cons_buff_1 : memref<4xui8>, 0, 4)
      aie.use_lock(%in_L3L2_cons_prod_pong_lock, Release, 1)
      // Since we are looping back to bb4 here, the repeat count really isn't necessary.
      // This chain of BDs will never complete, since it's an infinite cycle.
      // So it will never actually be repeated.
      aie.next_bd ^bb4
    ^bb6:
      aie.end
    // -- /attention --

    }
    aie.shim_dma_allocation @out(S2MM, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:
      aie.use_lock(%in_L2L1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_L2L1_cons_buff_0 : memref<4xui8>, 0, 4)
      aie.use_lock(%in_L2L1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:
      aie.use_lock(%in_L2L1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_L2L1_cons_buff_1 : memref<4xui8>, 0, 4)
      aie.use_lock(%in_L2L1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:
      aie.use_lock(%out_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<4xui8>, 0, 4)
      aie.use_lock(%out_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:
      aie.use_lock(%out_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_1 : memref<4xui8>, 0, 4)
      aie.use_lock(%out_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:
      aie.end
    }
  }
}

