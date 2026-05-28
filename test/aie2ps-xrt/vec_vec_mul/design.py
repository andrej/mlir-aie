# design.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_

tile_size = 1024
n_cores = 3
tiles_per_core = 16
tensor_size = tile_size * n_cores * tiles_per_core


def my_eltwise_mul():
    dtype = np.int32
    buffer_depth = 2

    @device(AIEDevice.xcve3858)
    def device_body():
        A_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
        B_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
        C_ty = np.ndarray[(tile_size,), np.dtype[dtype]]

        # Tiles: 3 shims, 1 core per column (row 3)
        shims = [tile(i, 0) for i in range(n_cores)]
        cores = [tile(i, 3) for i in range(n_cores)]

        inA_fifos = []
        inB_fifos = []
        outC_fifos = []

        for i in range(n_cores):
            inA_fifos.append(
                object_fifo(f"inA{i}", shims[i], cores[i], buffer_depth, A_ty)
            )
            inB_fifos.append(
                object_fifo(f"inB{i}", shims[i], cores[i], buffer_depth, B_ty)
            )
            outC_fifos.append(
                object_fifo(f"outC{i}", cores[i], shims[i], buffer_depth, C_ty)
            )

        for i in range(n_cores):

            @core(cores[i])
            def core_body():
                for _ in range_(sys.maxsize):
                    for _ in range_(tiles_per_core):
                        elem_out = outC_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                        elem_in_a = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        elem_in_b = inB_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        for j in range_(tile_size):
                            elem_out[j] = elem_in_a[j] * elem_in_b[j]
                        inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                        inB_fifos[i].release(ObjectFifoPort.Consume, 1)
                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)

        tensor_ty = np.ndarray[(tensor_size,), np.dtype[dtype]]

        @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
        def sequence(A, B, C):
            a_tasks, b_tasks, c_tasks = [], [], []
            chunk = tensor_size // n_cores
            for i in range(n_cores):
                a_tasks.append(
                    shim_dma_single_bd_task(
                        inA_fifos[i],
                        A,
                        offset=i * chunk,
                        sizes=[1, 1, 1, chunk],
                        issue_token=True,
                    )
                )
                b_tasks.append(
                    shim_dma_single_bd_task(
                        inB_fifos[i],
                        B,
                        offset=i * chunk,
                        sizes=[1, 1, 1, chunk],
                        issue_token=True,
                    )
                )
                c_tasks.append(
                    shim_dma_single_bd_task(
                        outC_fifos[i],
                        C,
                        offset=i * chunk,
                        sizes=[1, 1, 1, chunk],
                        issue_token=True,
                    )
                )
                dma_start_task(a_tasks[i], b_tasks[i], c_tasks[i])

            for i in range(n_cores):
                dma_await_task(a_tasks[i], b_tasks[i], c_tasks[i])

    print(ctx.module)


with mlir_mod_ctx() as ctx:
    my_eltwise_mul()
