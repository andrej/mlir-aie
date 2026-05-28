# design.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import XCVE3858
from aie.iron.controlflow import range_

N = 1024
per_tile_elements = 1024
dtype = bfloat16


def my_eltwise_add():
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]
    tensor_ty = np.ndarray[(N,), np.dtype[dtype]]

    # ObjectFifos
    of_in1 = ObjectFifo(tile_ty, name="in1")
    of_in2 = ObjectFifo(tile_ty, name="in2")
    of_out = ObjectFifo(tile_ty, name="out")

    # External kernel
    eltwise_add_bf16 = Kernel(
        "eltwise_add_bf16_vector", "add.o", [tile_ty, tile_ty, tile_ty]
    )

    # Worker task
    def core_body(of_in1, of_in2, of_out, add_fn):
        for _ in range_(N // per_tile_elements):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out = of_out.acquire(1)
            add_fn(elem_in1, elem_in2, elem_out)
            of_in1.release(1)
            of_in2.release(1)
            of_out.release(1)

    worker = Worker(
        core_body,
        [of_in1.cons(), of_in2.cons(), of_out.prod(), eltwise_add_bf16],
    )

    # Runtime
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(of_in1.prod(), A)
        rt.fill(of_in2.prod(), B)
        rt.drain(of_out.cons(), C, wait=True)

    return Program(XCVE3858(), rt).resolve_program(SequentialPlacer())


module = my_eltwise_add()
print(module)
