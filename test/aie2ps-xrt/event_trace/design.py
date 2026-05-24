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

N = 1024
TILE_WIDTH = 256
TRACE_SIZE = 8192

dev = AIEDevice.xcve3858


def my_event_trace():
    n_cores = N // TILE_WIDTH

    @device(dev)
    def device_body():
        in1_ty = np.ndarray[(TILE_WIDTH,), np.dtype[np.int32]]
        in2_ty = np.ndarray[(1,), np.dtype[np.int32]]
        out_ty = np.ndarray[(TILE_WIDTH,), np.dtype[np.int32]]
        all_in1_ty = np.ndarray[(N,), np.dtype[np.int32]]
        all_in2_ty = np.ndarray[(1,), np.dtype[np.int32]]
        all_out_ty = np.ndarray[(N,), np.dtype[np.int32]]
        trace_ty = np.ndarray[(TRACE_SIZE,), np.dtype[np.int32]]

        # Kernel declaration
        kernel = external_func(
            "vector_scalar_mul_vector",
            inputs=[in1_ty, out_ty, in2_ty, np.int32],
            link_with="scale.o",
        )

        # Tiles -- row 0: shim, row 1: memtile, row 3: first core row
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile = tile(0, 3)

        # ObjectFifos -- shim <-> memtile <-> core
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

        # ================================================================
        # TRACE CONFIGURATION
        # ================================================================

        # Core trace (Trace0)
        @trace(ComputeTile, "core_trace")
        def _():
            trace_mode(TraceMode.EventTime)
            trace_packet(8, TracePacketType.Core)
            trace_event("INSTR_EVENT_0")
            trace_event("INSTR_EVENT_1")
            trace_event("INSTR_VECTOR")
            trace_event("PORT_RUNNING_0")
            trace_event("PORT_RUNNING_1")
            trace_event("INSTR_LOCK_ACQUIRE_REQ")
            trace_event("INSTR_LOCK_RELEASE_REQ")
            trace_event("LOCK_STALL")
            trace_port(0, port=WireBundle.DMA, channel=0, direction=DMAChannelDir.S2MM)
            trace_port(1, port=WireBundle.DMA, channel=0, direction=DMAChannelDir.MM2S)
            trace_start(broadcast=8)
            trace_stop(broadcast=9)

        # Memory trace (Trace1)
        @trace(ComputeTile, "mem_trace")
        def _():
            trace_packet(9, TracePacketType.Mem)
            trace_event("DMA_S2MM_0_START_TASK")
            trace_event("DMA_S2MM_1_START_TASK")
            trace_event("DMA_MM2S_0_START_TASK")
            trace_event("DMA_S2MM_0_FINISHED_TASK")
            trace_event("DMA_S2MM_1_FINISHED_TASK")
            trace_event("DMA_MM2S_0_FINISHED_TASK")
            trace_event("DMA_S2MM_0_STREAM_STARVATION")
            trace_event("DMA_S2MM_1_STREAM_STARVATION")
            trace_start(event="BROADCAST_8")
            trace_stop(event="BROADCAST_9")

        # MemTile trace
        @trace(MemTile, "memtile_trace")
        def _():
            trace_packet(10, TracePacketType.MemTile)
            trace_event("PORT_RUNNING_0")
            trace_event("PORT_RUNNING_1")
            trace_event("PORT_RUNNING_2")
            trace_event("PORT_RUNNING_3")
            trace_event("PORT_RUNNING_4")
            trace_event("PORT_RUNNING_5")
            trace_event("PORT_RUNNING_6")
            trace_event("PORT_RUNNING_7")
            trace_port(0, port=WireBundle.DMA, channel=0, direction=DMAChannelDir.MM2S)
            trace_port(1, port=WireBundle.DMA, channel=1, direction=DMAChannelDir.MM2S)
            trace_port(2, port=WireBundle.DMA, channel=0, direction=DMAChannelDir.S2MM)
            trace_port(3, port=WireBundle.DMA, channel=1, direction=DMAChannelDir.S2MM)
            trace_start(broadcast=8)
            trace_stop(broadcast=9)

        # Shim tile trace
        @trace(ShimTile, "shim_trace")
        def _():
            trace_packet(11, TracePacketType.ShimTile)
            trace_event("NOC0_DMA_S2MM_0_START_TASK")
            trace_event("NOC0_DMA_S2MM_1_START_TASK")
            trace_event("NOC0_DMA_MM2S_0_START_TASK")
            trace_event("NOC0_DMA_S2MM_0_FINISHED_TASK")
            trace_event("NOC0_DMA_S2MM_1_FINISHED_TASK")
            trace_event("NOC0_DMA_MM2S_0_FINISHED_TASK")
            trace_event("NOC0_DMA_S2MM_0_STREAM_STARVATION")
            trace_event("NOC0_DMA_S2MM_1_STREAM_STARVATION")
            trace_start(event="TRUE")
            trace_stop(event="NONE")

        # ================================================================
        # RUNTIME SEQUENCE
        # ================================================================

        @runtime_sequence(all_in1_ty, all_in2_ty, all_out_ty, trace_ty)
        def sequence(in1, in2, out, trace_buf):
            # Activate trace configuration
            trace_start_config("core_trace")
            trace_start_config("mem_trace")
            trace_start_config("memtile_trace")
            trace_start_config("shim_trace")

            # Configure trace output buffer (arg_idx=3 = 4th runtime arg)
            trace_host_config(buffer_size=TRACE_SIZE, arg_idx=3)

            # Data transfers
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


with mlir_mod_ctx() as ctx:
    my_event_trace()
