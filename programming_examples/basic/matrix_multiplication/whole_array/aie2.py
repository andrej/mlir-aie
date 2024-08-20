#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import sys
import argparse

from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Whole Array)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=512)
    argparser.add_argument("-N", type=int, default=512)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=32)
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4], default=4)
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i16"], default="i16"
    )
    argparser.add_argument(
        "--dtype_out", type=str, choices=["bf16", "i16", "f32", "i32"], default="i16"
    )
    args = argparser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(
            args.M,
            args.K,
            args.N,
            args.m,
            args.k,
            args.n,
            args.n_aie_cols,
            args.dtype_in,
            args.dtype_out,
        )
        # print(ctx.module.operation.verify())
        print(ctx.module)


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(M, K, N, m, k, n, n_aie_cols, dtype_in_str, dtype_out_str):

    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = None
    if dtype_in_str == "bf16":
        dtype_in = T.bf16
    elif dtype_in_str == "i16":
        dtype_in = T.i16
    dtype_out = None
    if dtype_out_str == "bf16":
        dtype_out = T.bf16
    elif dtype_out_str == "i16":
        dtype_out = T.i16
    elif dtype_out_str == "f32":
        dtype_out = T.f32
    elif dtype_out_str == "i32":
        dtype_out = T.i32

    if dtype_in_str == "bf16":
        r = 4
        s = 8
        t = 4
    elif dtype_in_str == "i16":
        r = 4
        s = 4
        t = 4

    # Input matrix A:
    # Conceptually, we divide input A into (m * n_rows, k)-sized blocks. These
    # blocks are _broadcast_ across AIE core columns, then _distributed_ across
    # rows, s.t. each of the n_rows compute cores in a column receives a
    # contiguous (m, k)-sized block of A.
    assert (
        M % (m * n_aie_rows) == 0
    ), """A must be tileable into (m * n_aie_rows, k)-sized blocks"""

    # Both A and B are tiled in the K dimension into size k.
    assert K % k == 0

    # Input matrix B:
    # Conceptually, we do the same as with A, but instead of broadcasting
    # across columns we broadcast across rows and distribute across columns.
    assert (
        N % (n * n_aie_cols) == 0
    ), """B must be tileable into (k, n * n_aie_cols)-sized blocks"""

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    # If you get errors during CDO generation due to running out of program
    # memory, it may be because too much code is generated due to ObjectFIFO
    # loop unrollings. Reducing the depth to 1 here will work around that at
    # a big performance cost.
    fifo_depth = 2

    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    n_A_tiles_per_shim = n_aie_rows // n_aie_cols

    dev = None
    if n_aie_cols == 1:
        dev = AIEDevice.npu1_1col
    elif n_aie_cols == 2:
        dev = AIEDevice.npu1_2col
    elif n_aie_cols == 4:
        dev = AIEDevice.npu1_4col

    @device(dev)
    def device_body():
        A_l2_memref_ty = T.memref(m * k * n_A_tiles_per_shim, dtype_in())
        B_l2_memref_ty = T.memref(k * n, dtype_in())
        C_l2_memref_ty = T.memref(m * n * n_aie_rows, dtype_out())
        A_l1_memref_ty = T.memref(m, k, dtype_in())
        B_l1_memref_ty = T.memref(k, n, dtype_in())
        C_l1_memref_ty = T.memref(m, n, dtype_out())

        # AIE Core Function declarations
        zero_scalar = external_func(
            f"zero_scalar_{dtype_out_str}", inputs=[C_l1_memref_ty]
        )
        zero = external_func(f"zero_{dtype_out_str}", inputs=[C_l1_memref_ty])
        matmul_scalar = external_func(
            f"matmul_scalar_{dtype_in_str}_{dtype_out_str}",
            inputs=[A_l1_memref_ty, B_l1_memref_ty, C_l1_memref_ty],
        )
        matmul = external_func(
            f"matmul_{dtype_in_str}_{dtype_out_str}",
            inputs=[A_l1_memref_ty, B_l1_memref_ty, C_l1_memref_ty],
        )

        # Tile declarations as tile[row][col]
        tiles = [
            [tile(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)
        ]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        # AIE-array data movement with object fifos
        A_l3l2_fifos = [None] * n_aie_cols
        A_l2l1_fifos = [None] * n_aie_rows

        B_l3l2_fifos = [None] * n_aie_cols
        B_l2l1_fifos = [None] * n_aie_cols

        C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l2l3_fifos = [None] * n_aie_cols

        # Input A
        for row in range(n_aie_rows):
            A_l2l1_fifos[row] = object_fifo(
                f"A_L2L1_{row}",
                mem_tiles[row // n_A_tiles_per_shim],
                core_tiles[row][0:n_aie_cols],  # broadcast along one row
                fifo_depth,
                A_l1_memref_ty,
                [
                    (m // r, r * k),
                    (k // s, s),
                    (r, k),
                    (s, 1),
                ],
            )
        for col in range(n_aie_cols):
            A_l3l2_fifos[col] = object_fifo(
                f"A_L3L2_{col}",
                shim_tiles[col],
                mem_tiles[col],
                fifo_depth,
                A_l2_memref_ty,
            )
            # If n_cols == n_rows, n_A_tiles_per_shim is 1 and
            # this simply links a_l3l2_fifos[col] to a_l2l1_fifos[row] directly,
            # where col == row.
            # If n_cols < n_rows, each column receives multiple rows of
            # tiles; distribute it along rows of AIE cores.
            start_row = col * n_A_tiles_per_shim
            stop_row = start_row + n_A_tiles_per_shim
            if stop_row - start_row > 1:
                of_offsets = [m * k * i for i in range(stop_row - start_row)]
            else:
                of_offsets = []
            object_fifo_link(
                A_l3l2_fifos[col],
                [A_l2l1_fifos[row] for row in range(start_row, stop_row)],
                [],
                of_offsets,
            )

        # Input B
        for col in range(n_aie_cols):
            B_l3l2_fifos[col] = object_fifo(
                f"B_L3L2_{col}",
                shim_tiles[col],
                mem_tiles[col],
                fifo_depth,
                B_l2_memref_ty,
            )
            B_l2l1_fifos[col] = object_fifo(
                f"B_L2L1_{col}",
                mem_tiles[col],
                [
                    core_tiles[j][col] for j in range(n_aie_rows)
                ],  # broadcast along one column
                fifo_depth,
                B_l1_memref_ty,
                [
                    (k // s, s * n),
                    (n // t, t),
                    (s, n),
                    (t, 1),
                ],
            )
            object_fifo_link(B_l3l2_fifos[col], B_l2l1_fifos[col])

        # Output C
        for col in range(n_aie_cols):
            for row in range(n_aie_rows):
                C_l1l2_fifos[row][col] = object_fifo(
                    f"C_L1L2_{col}_{row}",
                    core_tiles[row][col],
                    mem_tiles[col],
                    fifo_depth,
                    C_l1_memref_ty,
                )
            C_l2l3_fifos[col] = object_fifo(
                f"C_L2L3_{col}",
                mem_tiles[col],
                shim_tiles[col],
                fifo_depth,
                C_l2_memref_ty,
                [
                    (m // r, r * n),
                    (r, t),
                    (n // t, r * t),
                    (t, 1),
                ],
            )
            if n_aie_rows > 1:
                of_offsets = [N * i for i in range(n_aie_rows)]
            else:
                of_offsets = []
            object_fifo_link(
                [C_l1l2_fifos[j][col] for j in range(n_aie_rows)],
                C_l2l3_fifos[col],
                of_offsets,
                [],
            )  # join along one column

        # Set up compute tiles
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):

                @core(core_tiles[row][col], f"mm_{m}x{k}x{n}.o")
                def core_body():
                    for _ in for_(0xFFFFFFFF):
                        loop = (
                            for_(n_tiles_per_core) if n_tiles_per_core > 1 else range(1)
                        )  # Workaround for issue #1547
                        for _ in loop:
                            elem_out = C_l1l2_fifos[row][col].acquire(
                                ObjectFifoPort.Produce, 1
                            )
                            call(zero, [elem_out])

                            for _ in for_(K // k):
                                elem_in_a = A_l2l1_fifos[row].acquire(
                                    ObjectFifoPort.Consume, 1
                                )
                                elem_in_b = B_l2l1_fifos[col].acquire(
                                    ObjectFifoPort.Consume, 1
                                )
                                call(matmul, [elem_in_a, elem_in_b, elem_out])
                                A_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                                B_l2l1_fifos[col].release(ObjectFifoPort.Consume, 1)
                                yield_([])

                            C_l1l2_fifos[row][col].release(ObjectFifoPort.Produce, 1)
                            yield_([])

                        if n_tiles_per_core > 1:  # workaround for issue #1547
                            yield_([])

        # To/from AIE-array data movement
        @runtime_sequence(
            T.memref(M * K, dtype_in()),
            T.memref(K * N, dtype_in()),
            T.memref(M * N, dtype_out()),
        )
        def sequence(A, B, C):

            c_tasks = [None] * n_aie_cols
            for col in range(n_aie_cols):
                c_task_repeat = 0
                c_task = dma_configure_task_for(C_l2l3_fifos[col],
                                                repeat_count=c_task_repeat,
                                                issue_token=True)
                C_aie_col_offset = (
                    col * n
                )
                C_offset = C_aie_col_offset
                with bds(c_task) as bd:
                    with bd[0]:
                        dma_bd(
                            C,
                            offset=C_offset,
                            len=M*N//n_aie_cols,
                            dimensions=[
                                (N // n // n_aie_cols, n * n_aie_cols),
                                (M, N),
                                (n, 1)
                            ]
                        )
                        EndOp()
                dma_start_task(c_task)
                c_tasks[col] = c_task

            for output_col_iter in range(N // n // n_aie_cols):
                
                a_tasks = [None] * n_aie_cols
                b_tasks = [None] * n_aie_cols

                for col in range(n_aie_cols):
                    # A
                    A_aie_col_offset = (
                        col * n_A_tiles_per_shim * m * K
                    )
                    A_offset = A_aie_col_offset
                    a_task_repeat = M // m // n_aie_rows - 1
                    a_task = dma_configure_task_for(A_l3l2_fifos[col],
                                                    repeat_count=a_task_repeat,
                                                    issue_token=True)
                    with bds(a_task) as bd:
                        with bd[0]:
                            dma_bd(
                                A,
                                offset=A_offset,
                                len=K * m * n_A_tiles_per_shim,
                                dimensions=[
                                    (M // m // n_aie_rows,   m * K * n_aie_rows),
                                    (K // k,                 k),
                                    (m * n_A_tiles_per_shim, K),
                                    (k,                      1)
                                ]
                            )
                            EndOp()
                    dma_start_task(a_task)
                    a_tasks[col] = a_task

                    # B
                    B_col_offset = (
                        output_col_iter * n_aie_cols * n
                    )
                    B_aie_col_offset = (
                        col * n
                    )
                    B_offset = B_col_offset + B_aie_col_offset
                    b_task_repeat = M // m // n_aie_rows - 1
                    b_task = dma_configure_task_for(B_l3l2_fifos[col],
                                                    repeat_count=b_task_repeat,
                                                    issue_token=True)
                    with bds(b_task) as bd:
                        with bd[0]:
                            dma_bd(
                                B,
                                offset=B_offset,
                                len=K*n,
                                dimensions=[
                                    (1, 0),
                                    (1, 0),  
                                    (K, N),
                                    (n, 1)
                                ]
                            )
                            EndOp()
                    dma_start_task(b_task)
                    b_tasks[col] = b_task
                
                for col in range(n_aie_cols):
                    dma_await_task(a_tasks[col])
                    dma_await_task(b_tasks[col])

            for col in range(n_aie_cols):
                dma_await_task(c_tasks[col])


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
