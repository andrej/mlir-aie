# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx

import aie.utils.trace as trace_utils


def passthroughKernel(vector_size, trace_size):
    N = vector_size
    lineWidthInBytes = 4
    assert(N % lineWidthInBytes == 0)

    @device(AIEDevice.npu1_1col)
    def device_body():
        # define types
        memRef_ty = T.memref(lineWidthInBytes, T.ui8())

        # AIE Core Function declarations
        passThroughLine = external_func(
            "passThroughLine", inputs=[memRef_ty, memRef_ty, T.i32()]
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile2 = tile(0, 2)

        # Set up a circuit-switched flow from core to shim for tracing information
        if trace_size > 0:
            flow(ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

        # AIE-array data movement with object fifos
        of_in_L3L2 = object_fifo("in_L3L2", ShimTile, MemTile, 2, memRef_ty)
        of_in_L2L1 = object_fifo("in_L2L1", MemTile, ComputeTile2, 2, memRef_ty)
        object_fifo_link(of_in_L3L2, of_in_L2L1)
        of_out = object_fifo("out", ComputeTile2, ShimTile, 2, memRef_ty)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2, "passThrough.cc.o")
        def core_body():
            for _ in for_(sys.maxsize):
                elemOut = of_out.acquire(ObjectFifoPort.Produce, 1)
                elemIn = of_in_L2L1.acquire(ObjectFifoPort.Consume, 1)
                call(passThroughLine, [elemIn, elemOut, lineWidthInBytes])
                of_in_L2L1.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce, 1)
                yield_([])

        #    print(ctx.module.operation.verify())

        tensor_ty = T.memref(N, T.ui8())

        @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
        def sequence(inTensor, outTensor, notUsed):
            if trace_size > 0:
                trace_utils.configure_simple_tracing_aie2(
                    ComputeTile2,
                    ShimTile,
                    ddr_id=1,
                    size=trace_size,
                    offset=N,
                )

            npu_dma_memcpy_nd(
                metadata=of_in_L3L2.sym_name.value,
                bd_id=0,
                mem=inTensor,
                sizes=[1, 1, 1, N],
            )
            npu_dma_memcpy_nd(
                metadata="out",
                bd_id=1,
                mem=outTensor,
                sizes=[1, 1, 1, N],
            )
            npu_sync(column=0, row=0, direction=0, channel=0)


vector_size = 16
trace_size = 0 
with mlir_mod_ctx() as ctx:
    passthroughKernel(vector_size, trace_size)
    print(ctx.module)
