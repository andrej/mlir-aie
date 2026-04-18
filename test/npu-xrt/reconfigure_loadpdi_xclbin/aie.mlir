// (c) Copyright 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This test verifies device reconfiguration via LOAD_PDI using the xclbin
// flow.  Two device configurations (@add_two and @add_three) are loaded
// sequentially within a single runtime sequence.  After each LOAD_PDI the
// test runs a DMA transfer to verify the newly loaded compute kernel is
// active.
//
// Expected results:
//   elements [0..3]  : input + 2  (processed by @add_two)
//   elements [4..7]  : input + 3  (processed by @add_three)
//   elements [8..511]: unchanged

module {

    // Primary device – owns the runtime sequence and objectfifos.
    // LOAD_PDI instructions reconfigure the AIE array mid-stream.
    aie.device(npu2) @add_two {

        %t00 = aie.tile(0, 0)
        %t02 = aie.tile(0, 2)

        aie.objectfifo @objfifo_in (%t00, {%t02}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
        aie.objectfifo @objfifo_out(%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<4xi32>>

        aie.core(%t02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2_i32 = arith.constant 2 : i32
            %c4 = arith.constant 4 : index
            %c_intmax = arith.constant 0xFFFFFE : index

            scf.for %niter = %c0 to %c_intmax step %c1 {
            %subview_in  = aie.objectfifo.acquire @objfifo_in (Consume, 1) : !aie.objectfifosubview<memref<4xi32>>
            %subview_out = aie.objectfifo.acquire @objfifo_out(Produce, 1) : !aie.objectfifosubview<memref<4xi32>>
            %elem_in     = aie.objectfifo.subview.access %subview_in [0] : !aie.objectfifosubview<memref<4xi32>> -> memref<4xi32>
            %elem_out    = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<4xi32>> -> memref<4xi32>
            scf.for %i = %c0 to %c4 step %c1 {
                %0 = memref.load %elem_in[%i] : memref<4xi32>
                %1 = arith.addi %0, %c2_i32 : i32
                memref.store %1, %elem_out[%i] : memref<4xi32>
            }
            aie.objectfifo.release @objfifo_in (Consume, 1)
            aie.objectfifo.release @objfifo_out(Produce, 1)
            }
            aie.end
        }

        aie.runtime_sequence @sequence(%a : memref<512xi32>) {

            // ---- Phase 1: Load add_two configuration ----
            aiex.npu.load_pdi { id = 1 : i32, device_ref = @add_two }

            %t_in = aiex.dma_configure_task_for @objfifo_in {
                aie.dma_bd(%a : memref<512xi32>, 0, 4)
                aie.end
            }
            %t_out = aiex.dma_configure_task_for @objfifo_out {
                aie.dma_bd(%a : memref<512xi32>, 0, 4)
                aie.end
            } {issue_token = true}
            aiex.dma_start_task(%t_in)
            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)

            // ---- Phase 2: Reconfigure to add_three ----
            aiex.npu.load_pdi { id = 2 : i32, device_ref = @add_three }

            %t_in2 = aiex.dma_configure_task_for @objfifo_in {
                aie.dma_bd(%a : memref<512xi32>, 4, 4)
                aie.end
            }
            %t_out2 = aiex.dma_configure_task_for @objfifo_out {
                aie.dma_bd(%a : memref<512xi32>, 4, 4)
                aie.end
            } {issue_token = true}
            aiex.dma_start_task(%t_in2)
            aiex.dma_start_task(%t_out2)
            aiex.dma_await_task(%t_out2)
        }

    }

    // Second device configuration – adds 3 to each element.
    // Compiled to its own PDI; loaded at runtime via LOAD_PDI.
    aie.device(npu2) @add_three {

        %t00 = aie.tile(0, 0)
        %t02 = aie.tile(0, 2)

        aie.objectfifo @objfifo_in (%t00, {%t02}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
        aie.objectfifo @objfifo_out(%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<4xi32>>

        aie.core(%t02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c3_i32 = arith.constant 3 : i32
            %c4 = arith.constant 4 : index
            %c_intmax = arith.constant 0xFFFFFE : index

            scf.for %niter = %c0 to %c_intmax step %c1 {
            %subview_in  = aie.objectfifo.acquire @objfifo_in (Consume, 1) : !aie.objectfifosubview<memref<4xi32>>
            %subview_out = aie.objectfifo.acquire @objfifo_out(Produce, 1) : !aie.objectfifosubview<memref<4xi32>>
            %elem_in     = aie.objectfifo.subview.access %subview_in [0] : !aie.objectfifosubview<memref<4xi32>> -> memref<4xi32>
            %elem_out    = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<4xi32>> -> memref<4xi32>
            scf.for %i = %c0 to %c4 step %c1 {
                %0 = memref.load %elem_in[%i] : memref<4xi32>
                %1 = arith.addi %0, %c3_i32 : i32
                memref.store %1, %elem_out[%i] : memref<4xi32>
            }
            aie.objectfifo.release @objfifo_in (Consume, 1)
            aie.objectfifo.release @objfifo_out(Produce, 1)
            }
            aie.end
        }

    }
}
