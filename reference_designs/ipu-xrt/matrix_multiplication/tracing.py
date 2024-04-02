import enum

import aie.dialects.aie as aie_
import aie.dialects.aiex as aiex_
import aie.dialects.scf as scf_

"""
Common code for getting a trace out of a compute core into DDR.
For now, a lot of this is hard-coded.
"""


class TraceEvent(enum.Enum):
    TRUE = 0x01
    STREAM_STALL = 0x18
    CORE_EVENT_0 = 0x21
    CORE_EVENT_1 = 0x22
    VECTOR_INST = 0x25
    LOCK_ACQ_REQ = 0x2C
    LOCK_REL_REQ = 0x2D
    LOCK_STALL = 0x1A
    CORE_PORT_RUNNING_0 = 0x4F
    CORE_PORT_RUNNING_1 = 0x4B


def get_trace_event_values(events: list[TraceEvent]):
    """Turns a list of events into values for the trace control registers."""
    assert len(events) <= 8
    out_0 = 0
    out_1 = 0
    for i, e in enumerate(events[0:4]):
        assert e.value <= 0xFF
        out_0 |= e.value << 8 * i
    for i, e in enumerate(events[4:]):
        assert e.value <= 0xFF
        out_1 |= e.value << 8 * i
    return out_0, out_1


def trace_flow(compute_tile: aie_.TileOp, shim_tile: aie_.TileOp):
    """Set up a circuit-switched flow from core to shim for tracing information"""
    assert shim_tile.row.value == 0
    # aie_.WireBundle.DMA corresponds to DMA_S2MM_1 below in trace_setup
    aie_.flow(compute_tile, aie_.WireBundle.Trace, 0, shim_tile, aie_.WireBundle.DMA, 1)


def trace_setup(
    compute_tile: aie_.TileOp,
    shim_tile: aie_.TileOp,
    trace_size,
    trace_offset,
    trace_bd_id,
    events,
):
    assert shim_tile.row.value == 0

    events_0, events_1 = get_trace_event_values(events)
    # Configure tracing, see https://github.com/Xilinx/mlir-aie/blob/resnet/docs/Tracing.md
    # 0x340D0: Trace Control 0
    #          0xAABB---C
    #            AA        <- Event to stop trace capture
    #              BB      <- Event to start trace capture
    #                   C  <- Trace mode, 00=event=time, 01=event-PC, 10=execution
    # Configure so that "Event 1" (always true) causes tracing to start
    aiex_.ipu_write32(
        column=compute_tile.col.value,
        row=compute_tile.row.value,
        address=0x340D0,
        value=0x00010000,
    )
    # 0x340D4: Trace Control 1
    aiex_.ipu_write32(
        column=compute_tile.col.value,
        row=compute_tile.row.value,
        address=0x340D4,
        value=0x00000000,
    )
    # 0x340E0: Trace Event Group 1  (Which events to trace)
    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
    aiex_.ipu_write32(
        column=compute_tile.col.value,
        row=compute_tile.row.value,
        address=0x340E0,
        value=events_0,
        # value=0x4B222125,
    )
    # 0x340E4: Trace Event Group 2  (Which events to trace)
    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
    aiex_.ipu_write32(
        column=compute_tile.col.value,
        row=compute_tile.row.value,
        address=0x340E4,
        value=events_1,
        # value=0x2D2C1A4F,
    )
    # 0x3FF00: Stream Switch Event Port Selection 0
    aiex_.ipu_write32(
        column=compute_tile.col.value,
        row=compute_tile.row.value,
        address=0x3FF00,
        value=0x00000121,
    )

    # Configure a buffer descriptor to write tracing information that has been routed into this shim tile
    # out to host DDR memory
    aiex_.ipu_writebd_shimtile(
        column=shim_tile.col.value,
        column_num=1,
        bd_id=trace_bd_id,
        buffer_length=trace_size,
        buffer_offset=trace_offset,
        enable_packet=0,
        out_of_order_id=0,
        packet_id=0,
        packet_type=0,
        d0_size=0,
        d0_stride=0,
        d1_size=0,
        d1_stride=0,
        d2_stride=0,
        ddr_id=2,
        iteration_current=0,
        iteration_size=0,
        iteration_stride=0,
        lock_acq_enable=0,
        lock_acq_id=0,
        lock_acq_val=0,
        lock_rel_id=0,
        lock_rel_val=0,
        next_bd=0,
        use_next_bd=0,
        valid_bd=1,
    )

    # Start the BD
    # 0x1D20C: DMA_S2MM_1_TASK_QUEUE
    # DMA_S2MM_1 corresponds to aie_.WireBundle.DMA, 1 above in trace_flow()
    aiex_.ipu_write32(
        column=shim_tile.col.value,
        row=shim_tile.row.value,
        address=0x1D20C,
        value=trace_bd_id,
    )
