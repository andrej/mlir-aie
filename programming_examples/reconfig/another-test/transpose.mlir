module {
  aie.device(npu2_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %mem_tile_0_1 = aie.tile(0, 1)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @in_L2L1_fifo(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @in_L3L2_fifo(%shim_noc_tile_0_0, {%mem_tile_0_1 dimensionsFromStream [<size = 8, stride = 8>, <size = 8, stride = 64>, <size = 4, stride = 512>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo.link [@in_L3L2_fifo] -> [@in_L2L1_fifo]([] [0])
    aie.objectfifo @out_fifo(%tile_0_2, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    func.func private @transpose_8x8(memref<64x32xi32>, memref<64x32xi32>)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %c0_2 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c8 step %c1_3 {
            %0 = aie.objectfifo.acquire @in_L2L1_fifo(Consume, 1) : !aie.objectfifosubview<memref<64x32xi32>>
            %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
            %2 = aie.objectfifo.acquire @out_fifo(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
            func.call @transpose_8x8(%1, %3) : (memref<64x32xi32>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @out_fifo(Produce, 1)
            aie.objectfifo.release @in_L2L1_fifo(Consume, 1)
          }
        }
      }
      aie.end
    } {link_with = "transpose.o"}
    aiex.runtime_sequence @sequence(%arg0: memref<512x512xi32>, %arg1: memref<512x512xi32>) {
      %0 = aiex.dma_configure_task_for @in_L3L2_fifo {
        aie.dma_bd(%arg0 : memref<512x512xi32>, 0, 32768, [<size = 8, stride = 32768>, <size = 16, stride = 32>, <size = 64, stride = 512>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @out_fifo {
        aie.dma_bd(%arg1 : memref<512x512xi32>, 0, 32768, [<size = 8, stride = 64>, <size = 16, stride = 16384>, <size = 32, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }
  }
}

