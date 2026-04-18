// (c) Copyright 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This test verifies the LOAD_PDI xclbin flow where the host code manually
// patches the LOAD_PDI instruction in the NPU instruction stream with the
// PDI address and size, rather than relying on XRT ELF patching.

module {
    aie.device(npu2) @add_two {

        %t00 = aie.tile(0, 0)
        %t02 = aie.tile(0, 2)

        aie.objectfifo @objfifo_in (%t00, {%t02}, 1 : i32) : !aie.objectfifo<memref<128xi32>>
        aie.objectfifo @objfifo_out(%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<128xi32>>

        aie.core(%t02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2_i32 = arith.constant 2 : i32
            %c128 = arith.constant 128 : index
            %c_intmax = arith.constant 0xFFFFFE : index

            scf.for %niter = %c0 to %c_intmax step %c1 {
            %subview_in  = aie.objectfifo.acquire @objfifo_in (Consume, 1) : !aie.objectfifosubview<memref<128xi32>>
            %subview_out = aie.objectfifo.acquire @objfifo_out(Produce, 1) : !aie.objectfifosubview<memref<128xi32>>
            %elem_in     = aie.objectfifo.subview.access %subview_in [0] : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
            %elem_out    = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
            scf.for %i = %c0 to %c128 step %c1 {
                %0 = memref.load %elem_in[%i] : memref<128xi32>
                %1 = arith.addi %0, %c2_i32 : i32
                memref.store %1, %elem_out[%i] : memref<128xi32>
            }
            aie.objectfifo.release @objfifo_in (Consume, 1)
            aie.objectfifo.release @objfifo_out(Produce, 1)
            }
            aie.end
        }

        aie.runtime_sequence @sequence(%a : memref<512xi32>) {

            aiex.npu.load_pdi { id = 1 : i32, device_ref = @add_two }

            %t_in = aiex.dma_configure_task_for @objfifo_in {
                aie.dma_bd(%a : memref<512xi32>, 0, 512)
                aie.end
            }
            %t_out = aiex.dma_configure_task_for @objfifo_out {
                aie.dma_bd(%a: memref<512xi32>, 0, 512)
                aie.end
            } {issue_token = true}
            aiex.dma_start_task(%t_in)
            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)
        }

    }
}
