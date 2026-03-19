module {
  aie.device(npu2) @add_two {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @objfifo_in(%shim_noc_tile_0_0, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<128xi32>> 
    aie.objectfifo @objfifo_out(%tile_0_2, {%shim_noc_tile_0_0}, 1 : i32) : !aie.objectfifo<memref<128xi32>> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c2_i32 = arith.constant 2 : i32
      %c8 = arith.constant 8 : index
      %c128 = arith.constant 128 : index
      %c16777214 = arith.constant 16777214 : index
      scf.for %arg0 = %c0 to %c16777214 step %c1 {
        %0 = aie.objectfifo.acquire @objfifo_in(Consume, 1) : !aie.objectfifosubview<memref<128xi32>>
        %1 = aie.objectfifo.acquire @objfifo_out(Produce, 1) : !aie.objectfifosubview<memref<128xi32>>
        %2 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
        %3 = aie.objectfifo.subview.access %1[0] : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
        scf.for %arg1 = %c0 to %c128 step %c1 {
          %4 = memref.load %2[%arg1] : memref<128xi32>
          %5 = arith.addi %4, %c2_i32 : i32
          memref.store %5, %3[%arg1] : memref<128xi32>
        }
        aie.objectfifo.release @objfifo_in(Consume, 1)
        aie.objectfifo.release @objfifo_out(Produce, 1)
      }
      aie.end
    }
    aiex.runtime_sequence(%arg0: memref<512xi32>) {
      aiex.npu.load_pdi {device_ref = @add_two, id = 1 : i32}
      %0 = aiex.dma_configure_task_for @objfifo_in {
        aie.dma_bd(%arg0 : memref<512xi32>, 0, 512)
        aie.end
      }
      %1 = aiex.dma_configure_task_for @objfifo_out {
        aie.dma_bd(%arg0 : memref<512xi32>, 0, 512)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
    }
  }
}
