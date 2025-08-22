module {
  // The empty device is used to generate the xclbin, to satisfy XRT. Nothing
  // is configured statically.
  aie.device(npu2) @empty {
  }

  // Our "main" design calls into the two other designs "add_two" and
  // "subtract_three" below. This runtime sequence incorporates the other
  // design's runtime sequences by reference and is the one that will be 
  //executed.
  aie.device(npu2) @main {
    aiex.runtime_sequence @sequence (%buf_inout: memref<1024x1024xi32>) {

      // Set up the add_two design (core program memories, stream switches, etc.)
      %config_add_two = aiex.configure @add_two
      aiex.run %config_add_two -> @sequence (%buf_inout) : (memref<1024x1024xi32>)

      // We can run the runtime sequence of that design twice, which will result
      // in adding four (adding two twice).
      aiex.run %config_add_two -> @sequence (%buf_inout) : (memref<1024x1024xi32>)

      // Reconfiguration. The "patch_marker" and "load_pdi" ops here will be
      // absorbed into the configure op before the PR is merged. They help
      // resetting the NPU before reconfiguration.
      aiex.npu.patch_marker { id = "loadpdi" }
      aiex.npu.load_pdi { id = 0x01 : ui16, size = 1 : ui32, address = 2 : ui64 }
      %config_subtract_three = aiex.configure @subtract_three

      aiex.run %config_subtract_three -> @sequence (%buf_inout) : (memref<1024x1024xi32>)
    }
  }

  aie.device(npu2) @add_two {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
  
    aie.objectfifo @objFifo_in0(%t00, {%t01}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @objFifo_in1(%t01, {%t02}, 1 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@objFifo_in0] -> [@objFifo_in1] ([] [])

    aie.objectfifo @objFifo_out1(%t02, {%t01}, 1 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @objFifo_out0(%t01, {%t00}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@objFifo_out1] -> [@objFifo_out0] ([] [])
  
    aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      %c2_32 = arith.constant 2 : i32

      %c_intmax = arith.constant 0xFFFFFF : index

      scf.for %niter = %c0 to %c_intmax step %c1 {
        scf.for %i = %c0 to %c512 step %c1 {
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

    aiex.runtime_sequence @sequence(%a : memref<1024x1024xi32>) {
      %t_in = aiex.dma_configure_task_for @objFifo_in0 {
        aie.dma_bd(%a : memref<1024x1024xi32>, 0, 1048576)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @objFifo_out0 {
        aie.dma_bd(%a: memref<1024x1024xi32>, 0, 1048576)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_in)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
    }
  }

  aie.device(npu2) @subtract_three {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)

    aie.objectfifo @in(%t00, {%t02}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out(%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<32xi32>>

    aie.core(%t02) {
      %c_0 = arith.constant 0 : index
      %c_1 = arith.constant 1 : index
      %c_2 = arith.constant 2 : index
      %c_32 = arith.constant 32 : index
      %c_64 = arith.constant 64 : index
      %c_128 = arith.constant 128 : index
      %c_intmax = arith.constant 0xFFFFFF : index
      %c_3_i32 = arith.constant 3 : i32
      scf.for %run = %c_0 to %c_intmax step %c_1 {
        scf.for %j = %c_0 to %c_128 step %c_1 {
          %subview0 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<32xi32>>
          %in_elem = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
          %subview1 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<32xi32>>
          %out_elem = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
          scf.for %i = %c_0 to %c_32 step %c_1 {
              %in_val = memref.load %in_elem[%i] : memref<32xi32>
              %out_val = arith.subi %in_val, %c_3_i32 : i32
              memref.store %out_val, %out_elem[%i] : memref<32xi32>
          }
          aie.objectfifo.release @in(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
      }
      aie.end
    }

    aiex.runtime_sequence @sequence(%a : memref<1024x1024xi32>) {
      %t_in = aiex.dma_configure_task_for @in {
        aie.dma_bd(%a : memref<1024x1024xi32>, 0, 1048576)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @out {
        aie.dma_bd(%a: memref<1024x1024xi32>, 0, 1048576)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_in)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
    }

  }

}
