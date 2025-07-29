// Proposed syntax

module {

  aie.device(npu2_1col) @add_one {

    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)

    aie.objectfifo @objFifo_in(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @objFifo_out(%t02, {%t00}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c1_32 = arith.constant 1 : i32
      scf.for %steps = %c0 to %c8 step %c1 {
        %subview0 = aie.objectfifo.acquire @objFifo_in(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %elem0 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %subview1 = aie.objectfifo.acquire @objFifo_out(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        scf.for %arg3 = %c0 to %c8 step %c1 {
            %0 = memref.load %elem0[%arg3] : memref<8xi32>
            %1 = arith.addi %0, %c1_32 : i32
            memref.store %1, %elem1[%arg3] : memref<8xi32>
        }
        aie.objectfifo.release @objFifo_in(Consume, 1)
        aie.objectfifo.release @objFifo_out(Produce, 1)
      }
      aie.end
    }

    aiex.runtime_sequence @rt (%in : memref<64xi32>, %out : memref<64xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd (%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0, %c1]) { metadata = @objFifo_out, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_memcpy_nd (%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0, %c1]) { metadata = @objFifo_in, id = 0 : i64 } : memref<64xi32>
      aiex.npu.dma_wait { symbol = @objFifo_out }
    }
  }

  aie.device(npu2_1col) @main {
    aiex.runtime_sequence @rt (%global_in: memref<64xi32>, %global_out: memref<64xi32>) {
      %c1 = aiex.configure @add_one
      aiex.run %c1 -> @rt (%global_in, %global_out) : (memref<64xi32>, memref<64xi32>)
    }
  }

}
