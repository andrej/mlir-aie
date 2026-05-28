# design.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

import argparse
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_

N = 1024
TILE_WIDTH = 256

dev = AIEDevice.xcve3858


def my_vector_scalar_mul(dtype_str):
    if dtype_str == "i16":
        in1_dtype = np.int16
        out_dtype = np.int16
    else:  # default: i32
        in1_dtype = np.int32
        out_dtype = np.int32

    in2_dtype = np.int32
    n_cores = N // TILE_WIDTH

    @device(dev)
    def device_body():
        in1_ty = np.ndarray[(TILE_WIDTH,), np.dtype[in1_dtype]]
        in2_ty = np.ndarray[(1,), np.dtype[in2_dtype]]
        out_ty = np.ndarray[(TILE_WIDTH,), np.dtype[out_dtype]]
        all_in1_ty = np.ndarray[(N,), np.dtype[in1_dtype]]
        all_in2_ty = np.ndarray[(1,), np.dtype[in2_dtype]]
        all_out_ty = np.ndarray[(N,), np.dtype[out_dtype]]

        # Kernel declaration
        kernel = external_func(
            "vector_scalar_mul_vector",
            inputs=[in1_ty, out_ty, in2_ty, np.int32],
            link_with="scale.o",
        )

        # Tiles — row 3 is first core row on xcve3858
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile = tile(0, 3)

        # ObjectFifos
        of_in1 = object_fifo("in1", ShimTile, MemTile, 2, all_in1_ty)
        of_in1_sub = object_fifo("in1_sub", MemTile, ComputeTile, 2, in1_ty)
        object_fifo_link(of_in1, of_in1_sub)

        of_in2 = object_fifo("in2", ShimTile, ComputeTile, 2, in2_ty)

        of_out_sub = object_fifo("out_sub", ComputeTile, MemTile, 2, out_ty)
        of_out = object_fifo("out", MemTile, ShimTile, 2, all_out_ty)
        object_fifo_link(of_out_sub, of_out)

        # Core body
        @core(ComputeTile)
        def core_body():
            for _ in range_(sys.maxsize):
                elem_in2 = of_in2.acquire(ObjectFifoPort.Consume, 1)
                for _ in range_(n_cores):
                    elem_in1 = of_in1_sub.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out_sub.acquire(ObjectFifoPort.Produce, 1)
                    kernel(elem_in1, elem_out, elem_in2, TILE_WIDTH)
                    of_in1_sub.release(ObjectFifoPort.Consume, 1)
                    of_out_sub.release(ObjectFifoPort.Produce, 1)
                of_in2.release(ObjectFifoPort.Consume, 1)

        # Runtime sequence
        @runtime_sequence(all_in1_ty, all_in2_ty, all_out_ty)
        def sequence(in1, in2, out):
            in1_task = shim_dma_single_bd_task(
                of_in1, in1, sizes=[1, 1, 1, N], issue_token=True
            )
            in2_task = shim_dma_single_bd_task(
                of_in2, in2, sizes=[1, 1, 1, 1], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out, out, sizes=[1, 1, 1, N], issue_token=True
            )
            dma_start_task(in1_task, in2_task, out_task)
            dma_await_task(in1_task, in2_task, out_task)

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="i32", choices=["i32", "i16"])
    args = p.parse_args()

    with mlir_mod_ctx() as ctx:
        my_vector_scalar_mul(args.dtype)
