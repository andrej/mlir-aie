#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

from collections import OrderedDict

from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *

# tracing, common need to be imported after the above because they may emit
# instructions and need the context
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))
import tracing, common


def main():
    argparser = common.get_default_argparser(1024, 1024, 1)
    argparser.add_argument("--cores", type=int, default=1)
    args = argparser.parse_args()
    assert args.N == 1  # matrix-VECTOR multiplication
    assert 0 < args.cores <= 4
    my_matmul(args.M, args.K, args.cores, args.trace)


def my_matmul(M, K, n_cores, trace_sz):
    m = 32
    k = 32
    word_size_in = 2
    word_size_out = 4

    A_sz = M * K * word_size_in
    B_sz = K * word_size_in
    C_sz = M * word_size_out

    use_A_dummy = False  # Replace object fifo A on core with a constant buffer on core (remove transfer cost)
    use_B_dummy = False
    
    # The trace we set up is on the left-most column and goes out the left-
    # most shim tile. When doing parallel transfers, the pathfinder routes
    # multiple channels in the leftmost column as well, and as a result
    # we run out of channels. To fix this, we move the memory tile that
    # moves the data one column to the right.
    trace_fix_offset = 0
    if trace_sz > 0:
        trace_fix_offset = 1
    assert(n_cores + trace_fix_offset <= 4)

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def device_body():
            memRef_inA_ty = T.memref(m * k, T.bf16())
            memRef_inB_ty = T.memref(k, T.bf16())
            memRef_outC_ty = T.memref(m, T.f32())

            # AIE Core Function declarations
            zero_scalar = external_func("zero_scalar_f32", inputs=[memRef_outC_ty])
            zero = external_func("zero_vectorized_f32", inputs=[memRef_outC_ty])
            matvec = external_func(
                "matvec_vectorized_bf16_f32",
                inputs=[memRef_inA_ty, memRef_inB_ty, memRef_outC_ty],
            )

            # Tile declarations
            ShimTile0 = tile(0, 0)
            ShimTile1 = tile(1, 0)
            ShimTile2 = tile(2, 0)
            ShimTile3 = tile(3, 0)
            ShimTiles = [ShimTile0, ShimTile1, ShimTile2, ShimTile3]
            MemTile0 = tile(0, 1)
            MemTile1 = tile(1, 1)
            MemTile2 = tile(2, 1)
            MemTile3 = tile(3, 1)
            MemTiles = [MemTile0, MemTile1, MemTile2, MemTile3]
            ComputeTile0 = tile(0, 2)
            ComputeTile1 = tile(1, 2)
            ComputeTile2 = tile(2, 2)
            ComputeTile3 = tile(3, 2)
            cores = [ComputeTile0, ComputeTile1, ComputeTile2, ComputeTile3]
            memA_fifos = OrderedDict()
            inA_fifos = OrderedDict()
            inB_fifo_names = ["inB"]
            inB_fifos = {}
            outC_fifo_names = ["outC0", "outC1", "outC2", "outC3"]
            outC_fifos = {}

            # Dummies, used to isolate data transfers during benchmarking by
            # removing fifos
            if use_A_dummy:
                A_dummies = [buffer(cores[i], (m, k), T.bf16(), f"A_dummy_{i}") for i in range(n_cores)]
            if use_B_dummy:
                B_dummies = [buffer(cores[i], (k,), T.bf16(), f"B_dummy_{i}") for i in range(n_cores)]

            # AIE-array data movement with object fifos
            # Input A
            if not use_A_dummy:
                for i in range(n_cores):
                    for j in range(1):
                        core_fifo = f"inA_{i}_{j}"
                        mem_fifo = f"memA_{i}_{j}"

                        inA_fifos[core_fifo] = object_fifo(
                            core_fifo,
                            MemTiles[i*2+trace_fix_offset + j],
                            cores[i],
                            2,
                            memRef_inA_ty,
                            [
                                (m, k),
                                (k, 1),
                            ],
                        )
                        memA_fifos[mem_fifo] = object_fifo(
                            mem_fifo,
                            ShimTiles[i*2 + j],
                            MemTiles[i*2+trace_fix_offset + j],
                            2,
                            memRef_inA_ty,
                        )
                        object_fifo_link(
                            [memA_fifos[mem_fifo]], 
                            inA_fifos[core_fifo]
                        )

            # Input B
            if not use_B_dummy:
                inB_fifos[inB_fifo_names[0]] = object_fifo(
                    inB_fifo_names[0],
                    ShimTiles[1 % n_cores],
                    cores[0:n_cores],
                    2,
                    memRef_inB_ty,
                )

            # Output C
            for i in range(n_cores):
                outC_fifos[outC_fifo_names[i]] = object_fifo(
                    outC_fifo_names[i],
                    cores[i],
                    ShimTiles[i],
                    2,
                    memRef_outC_ty,
                )

            # Set up a circuit-switched flow from core to shim for tracing information
            if trace_sz > 0:
                tracing.trace_flow(ComputeTile0, ShimTile0)

            # Set up compute tiles
            for i in range(n_cores):
                # Compute tile i
                @core(cores[i], "mv.o")
                def core_body():

                    for _ in for_(0xFFFFFFFF):
                        elem_out = outC_fifos[outC_fifo_names[i]].acquire(
                            ObjectFifoPort.Produce,
                            1,
                        )
                        call(zero, [elem_out])

                        for _ in for_(K // k):
                            core_A_fifo = f"inA_{i}_0"
                            elem_in_a = inA_fifos[core_A_fifo].acquire(
                                ObjectFifoPort.Consume,
                                1,
                            ) if not use_A_dummy else A_dummies[i]
                            elem_in_b = inB_fifos[inB_fifo_names[0]].acquire(
                                ObjectFifoPort.Consume,
                                1,
                            ) if not use_B_dummy else B_dummies[i]
                            call(matvec, [elem_in_a, elem_in_b, elem_out])
                            if not use_A_dummy:
                                inA_fifos[core_A_fifo].release(
                                    ObjectFifoPort.Consume,
                                    1,
                                )
                            if not use_B_dummy:
                                inB_fifos[inB_fifo_names[0]].release(
                                    ObjectFifoPort.Consume,
                                    1,
                                )
                            yield_([])

                        outC_fifos[outC_fifo_names[i]].release(
                            ObjectFifoPort.Produce,
                            1,
                        )

                        yield_([])

            # To/from AIE-array data movement

            @FuncOp.from_py_func(
                T.memref(A_sz // 4, T.i32()),
                T.memref(B_sz // 4, T.i32()),
                T.memref(C_sz // 4, T.i32()),
            )
            def sequence(A, B, C):
                # Tracing config
                if trace_sz > 0:
                    tracing.trace_setup(
                        ComputeTile0, ShimTile0, trace_sz, C_sz, 2, common.trace_events
                    )

                # Repeat entire B vector M // m // n_cores times.
                # If M // m // n_cores > 64, we need to split it up into separate transfers.
                transfer_sz = 64
                for transfer_i in range(
                    (M // m // n_cores + transfer_sz - 1) // transfer_sz
                ):
                    B_repeats = min(
                        transfer_sz, M // m // n_cores - transfer_i * transfer_sz
                    )
                    transfer_offset = transfer_i * transfer_sz

                    # B
                    if not use_B_dummy:
                        ipu_dma_memcpy_nd(
                            metadata=inB_fifo_names[0],
                            bd_id=1,
                            mem=B,
                            sizes=[B_repeats, 1, 1, K * word_size_in // 4],
                            strides=[0, 0, 0],
                        )

                    # Each core is responsible for M // n_cores rows of the output C.
                    for i in range(n_cores):
                        # C
                        C_offset = (
                            transfer_offset * m * word_size_out // 4
                            + i * (M // n_cores) * word_size_out // 4
                        )
                        ipu_dma_memcpy_nd(
                            metadata=outC_fifo_names[i],
                            bd_id=0,
                            mem=C,
                            offsets=[0, 0, 0, C_offset],
                            sizes=[1, 1, 1, B_repeats * m * word_size_out // 4],
                            strides=[0, 0, 0],
                        )

                        # A
                        if not use_A_dummy:
                            for j in range(1):
                                A_offset = (
                                    transfer_offset * K * word_size_in // 4
                                    + i * (M // n_cores) * K * word_size_in // 4
                                    + j * (m) * K * word_size_in // 4
                                )
                                ipu_dma_memcpy_nd(
                                    metadata=f"memA_{i}_{j}",
                                    bd_id=3+j,
                                    mem=A,
                                    offsets=[0, 0, 0, A_offset],
                                    sizes=[B_repeats, K // k, m, k * word_size_in // 4],
                                    strides=[
                                        m * K * word_size_in // 4,
                                        k * word_size_in // 4,
                                        K * word_size_in // 4,
                                    ],
                                )

                    for i in range(n_cores):
                        ipu_sync(column=i, row=0, direction=0, channel=0)

    print(ctx.module)


if __name__ == "__main__":
    main()
