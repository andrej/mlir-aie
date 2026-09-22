# test_kernel_arg_collapse.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Feeding a multi-dimensional argument to a kernel that declares a flat one.

The ``aie.iron.kernels`` factories declare 1-D arguments, while a design holds
its L1 data in multi-dimensional ObjectFifo elements and Buffers. The call site
collapses the shape so the two meet.
"""

import numpy as np
import pytest

from aie import ir
from aie.dialects.aie import AIEDevice, Device, buffer, tile
from aie.extras.context import mlir_mod_ctx
from aie.iron.kernel import _maybe_collapse_to_match


def _adapt(shape, dtype, expected):
    """Adapt a buffer op of ``shape`` to ``expected``.

    Returns whether the argument came back untouched, and the type of what came
    back. Both are read inside the context, which owns every type and value here.
    """
    with mlir_mod_ctx():
        device = Device(AIEDevice.npu2)
        with ir.InsertionPoint(device.body_region.blocks.append()):
            buf = buffer(tile=tile(0, 2), datatype=np.ndarray[shape, np.dtype[dtype]])
            adapted = _maybe_collapse_to_match(buf, expected())
            return adapted is buf, str(getattr(adapted, "type", ""))


def test_a_buffer_collapses_to_a_flat_argument():
    """A Buffer hands the call its op, not its value, and must collapse anyway."""
    untouched, adapted_type = _adapt(
        (4, 8), np.float32, lambda: ir.MemRefType.get([32], ir.F32Type.get())
    )

    assert not untouched
    assert adapted_type == "memref<32xf32>"


@pytest.mark.parametrize(
    "shape,dtype,expected",
    [
        pytest.param(
            (4, 8),
            np.float32,
            lambda: ir.MemRefType.get([16], ir.F32Type.get()),
            id="element-count",
        ),
        pytest.param(
            (4, 8),
            np.float32,
            lambda: ir.MemRefType.get([32], ir.IntegerType.get_signless(32)),
            id="element-type",
        ),
        pytest.param(
            (4, 8),
            np.float32,
            lambda: ir.MemRefType.get([4, 8], ir.F32Type.get()),
            id="rank",
        ),
    ],
)
def test_a_buffer_that_does_not_fit_is_left_alone(shape, dtype, expected):
    """A real mismatch reaches MLIR verification rather than a wrong collapse."""
    untouched, _ = _adapt(shape, dtype, expected)

    assert untouched
