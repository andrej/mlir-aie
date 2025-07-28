module {
  aie.device(npu2_1col) @empty {
  }

  aie.device(npu2_1col) @add_two {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
  
    aie.objectfifo @objFifo_in0(%t00, {%t01}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @objFifo_in1(%t01, {%t02}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@objFifo_in0] -> [@objFifo_in1] ([] [])

    aie.objectfifo @objFifo_out1(%t02, {%t01}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @objFifo_out0(%t01, {%t00}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@objFifo_out1] -> [@objFifo_out0] ([] [])
  
    aie.core(%t02) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2_32 = arith.constant 2 : i32

      scf.for %niter = %c0 to %c1 step %c1 {
        scf.for %steps = %c0 to %c8 step %c1 {
          %subview0 = aie.objectfifo.acquire @objFifo_in1(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
          %elem0 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
          %subview1 = aie.objectfifo.acquire @objFifo_out1(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
          %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
          scf.for %arg3 = %c0 to %c8 step %c1 {
              %0 = memref.load %elem0[%arg3] : memref<8xi32>
              %1 = arith.addi %0, %c2_32 : i32
              memref.store %1, %elem1[%arg3] : memref<8xi32>
          }
          aie.objectfifo.release @objFifo_in1(Consume, 1)
          aie.objectfifo.release @objFifo_out1(Produce, 1)
        }
      }
      aie.end
    }

    aiex.runtime_sequence @rt(%in : memref<64xi32>, %out : memref<64xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd (%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0, %c1]) { metadata = @objFifo_out0, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0, %c1]) { metadata = @objFifo_in0, id = 0 : i64 } : memref<64xi32>
      aiex.npu.dma_wait { symbol = @objFifo_out0 }
    }
  }

  aie.device(npu2_1col) @subtract_three {
    %t00 = aie.tile(0, 0)
    %t03 = aie.tile(0, 3)

    aie.objectfifo @in(%t00, {%t03}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out(%t03, {%t00}, 2 : i32) : !aie.objectfifo<memref<64xi32>>

    aie.core(%t03) {
      %c_0 = arith.constant 0 : index
      %c_1 = arith.constant 1 : index
      %c_64 = arith.constant 64 : index
      %c_intmax = arith.constant 0xFFFFFF : index
      %c_3_i32 = arith.constant 3 : i32
      scf.for %run = %c_0 to %c_intmax step %c_1 {
          %subview0 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<64xi32>>
          %in_elem = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
          %subview1 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
          %out_elem = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
          scf.for %i = %c_0 to %c_64 step %c_1 {
              %in_val = memref.load %in_elem[%i] : memref<64xi32>
              %out_val = arith.subi %in_val, %c_3_i32 : i32
              memref.store %out_val, %out_elem[%i] : memref<64xi32>
          }
          aie.objectfifo.release @in(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    }

    aiex.runtime_sequence @rt(%in : memref<64xi32>, %out : memref<64xi32>) {
      %t_in = aiex.dma_configure_task_for @in {
        aie.dma_bd(%in : memref<64xi32>, 0, 64)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @out {
        aie.dma_bd(%out: memref<64xi32>, 0, 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_in)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
    }

  }
}
