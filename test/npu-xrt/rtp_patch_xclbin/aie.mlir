// (c) Copyright 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// End-to-end test for named RTP patching via the xclbin flow.
// The core reads an RTP value from a buffer and adds it to each input element.
// The host patches the RTP value in the instruction stream before execution.

module {
    aie.device(npu2) {

        %t00 = aie.tile(0, 0)
        %t02 = aie.tile(0, 2)

        // RTP buffer: the host will patch this value via aiex.npu.rtp_write
        %add_value = aie.buffer(%t02) {sym_name = "add_value"} : memref<1xi32>

        aie.objectfifo @objfifo_in (%t00, {%t02}, 1 : i32) : !aie.objectfifo<memref<128xi32>>
        aie.objectfifo @objfifo_out(%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<128xi32>>

        aie.core(%t02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c128 = arith.constant 128 : index
            %c_iterations = arith.constant 4 : index

            scf.for %niter = %c0 to %c_iterations step %c1 {
                %subview_in  = aie.objectfifo.acquire @objfifo_in (Consume, 1) : !aie.objectfifosubview<memref<128xi32>>
                %subview_out = aie.objectfifo.acquire @objfifo_out(Produce, 1) : !aie.objectfifosubview<memref<128xi32>>
                %elem_in  = aie.objectfifo.subview.access %subview_in [0] : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
                %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>

                // Read the RTP value
                %rtp_val = memref.load %add_value[%c0] : memref<1xi32>

                scf.for %i = %c0 to %c128 step %c1 {
                    %0 = memref.load %elem_in[%i] : memref<128xi32>
                    %1 = arith.addi %0, %rtp_val : i32
                    memref.store %1, %elem_out[%i] : memref<128xi32>
                }
                aie.objectfifo.release @objfifo_in (Consume, 1)
                aie.objectfifo.release @objfifo_out(Produce, 1)
            }
            aie.end
        }

        aie.runtime_sequence @sequence(%in : memref<512xi32>, %buf : memref<32xi32>, %out : memref<512xi32>) {

            // Write a default RTP value (0); the host will patch this to a
            // different value before submitting the instruction buffer.
            aiex.npu.rtp_write(@add_value, 0, 0)

            %c0 = arith.constant 0 : i64
            %c1 = arith.constant 1 : i64
            %c512 = arith.constant 512 : i64

            aiex.npu.dma_memcpy_nd (%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c512][%c0,%c0,%c0,%c1]) { metadata = @objfifo_out, id = 1 : i64, issue_token = true } : memref<512xi32>
            aiex.npu.dma_memcpy_nd (%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c512][%c0,%c0,%c0,%c1]) { metadata = @objfifo_in, id = 0 : i64 } : memref<512xi32>
            aiex.npu.dma_wait {symbol = @objfifo_out}
        }

    }
}
