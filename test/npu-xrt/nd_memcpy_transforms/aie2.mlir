module {
  aie.device(npu1) {
    func.func private @concat(memref<8xi16>, memref<12xi16>, memref<20xi16>, i32, i32, i32)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_4 = aie.tile(3, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    aie.objectfifo @fifo_a(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8xi16>> 
    aie.objectfifo @fifo_b(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<12xi16>> 
    aie.objectfifo @fifo_c(%tile_0_2, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<20xi16>> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_c(Produce, 1) : !aie.objectfifosubview<memref<20xi16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<20xi16>> -> memref<20xi16>
        %2 = aie.objectfifo.acquire @fifo_a(Consume, 1) : !aie.objectfifosubview<memref<8xi16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8xi16>> -> memref<8xi16>
        %4 = aie.objectfifo.acquire @fifo_b(Consume, 1) : !aie.objectfifosubview<memref<12xi16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<12xi16>> -> memref<12xi16>
        %c8_i32 = arith.constant 8 : i32
        %c12_i32 = arith.constant 12 : i32
        %c20_i32 = arith.constant 20 : i32
        func.call @concat(%3, %5, %1, %c8_i32, %c12_i32, %c20_i32) : (memref<8xi16>, memref<12xi16>, memref<20xi16>, i32, i32, i32) -> ()
        aie.objectfifo.release @fifo_a(Consume, 1)
        aie.objectfifo.release @fifo_b(Consume, 1)
        aie.objectfifo.release @fifo_c(Produce, 1)
      }
      aie.end
    } {link_with = "kernel.o"}
    aie.runtime_sequence(%arg0: memref<8xi16>, %arg1: memref<12xi16>, %arg2: memref<20xi16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 2, 2, 2][0, 2, 4, 1]) {id = 1 : i64, metadata = @fifo_a} : memref<8xi16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 2, 3, 2][0, 2, 4, 1]) {id = 1 : i64, metadata = @fifo_b} : memref<12xi16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 2][1, 1, 1, 4][1, 1, 1, 4]) {id = 0 : i64, metadata = @fifo_c} : memref<20xi16>
      aiex.npu.dma_wait {symbol = @fifo_c}
    }
  }
}

