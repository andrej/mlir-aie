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
import aie.dialects.memref as memref
import aie.utils.trace as trace_utils
from functools import reduce
import operator

# tracing, common need to be imported after the above because they may emit
# instructions and need the context
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))
import common


# ##########################################################################
# Helpers
# ##########################################################################

def is_shim_tile(tile: TileOp):
    return tile.row.value == 0

def is_mem_tile(tile: TileOp):
    return tile.row.value == 1

def create_dma(tile: TileOp):
    if is_mem_tile(tile):
        #dmaOp = memtile_dma(stream.srcTile, loc=None, ip=None)()
        return MemTileDMAOp(T.index(), tile)
    elif tile.row.value >= 2:
        #dmaOp = mem(stream.srcTile, loc=None, ip=None)()
        return MemOp(T.index(), tile)
    else:
        return None

def get_memref_size(m: MemRefType):
    return reduce(operator.mul, [m.get_dim_size(i) for i in range(len(m.shape))], 1)

def ceildiv(a, b):
    return (a + b - 1) // b


# ##########################################################################
# Stream
# ##########################################################################

class Stream:
    """
    Generic struct to keep track of relevant ops for a circular-buffered 
    (e.g. ping-ponged) stream of data from a source tile to a destination tile.
    """
    def __init__(self, name, src, dst, n_buffers=2, memref=None):
        srcTile, srcDMAChannel = src
        dstTile, dstDMAChannel = dst

        self.name = name
        self.srcTile : TileOp = srcTile
        self.srcDMAChannel = srcDMAChannel
        self.dstTile : TileOp = dstTile
        self.dstDMAChannel = dstDMAChannel
        self.srcProdLock : LockOp = None  # number of buffers filled by core ready for sending to stream
        self.srcConsLock : LockOp = None  # number of free buffers, ready for core to write data into
        self.dstProdLock : LockOp = None  # number of free buffers to receive data into
        self.dstConsLock : LockOp = None  # number of filled buffers ready for consumption by core
        self.srcMemref : T.MemRefType = memref
        self.dstMemref : T.MemRefType = memref
        self.srcBuffers : list[BufferOp] = []
        self.dstBuffers : list[BufferOp] = []
        self.nBuffers = n_buffers # ping-pong
    
    def generate_src_locks(self):
        self.srcProdLock = lock(self.srcTile, 
                                sym_name=f"{self.name}_src_prod", 
                                init=self.nBuffers)
        self.srcConsLock = lock(self.srcTile, 
                                sym_name=f"{self.name}_src_cons", 
                                init=0)
    
    def generate_dst_locks(self):
        self.dstProdLock = lock(self.dstTile, 
                                sym_name=f"{self.name}_dst_prod", 
                                init=self.nBuffers)
        self.dstConsLock = lock(self.dstTile, 
                                sym_name=f"{self.name}_dst_cons", 
                                init=0)
    
    @staticmethod
    def generate_buffers(n_buffers, name, memref_ty, tile, dir, idx):
        ret = [None] * n_buffers
        if is_shim_tile(tile):
            ret[0] = memref.global_(name, memref_ty, sym_visibility="public")
            ShimDMAAllocationOp(name, dir, idx, tile.col.value)
        else:
            for j in range(n_buffers):
                ret[j] = buffer(tile, shape=memref_ty.shape, dtype=memref_ty.element_type, name=f"{name}_{j}")
        return ret

    def generate_src_buffers(self):
        self.srcBuffers = self.generate_buffers(self.nBuffers, 
                                                f"{self.name}_src_buffer", 
                                                self.srcMemref, 
                                                self.srcTile, 
                                                DMAChannelDir.MM2S,
                                                self.srcDMAChannel)
    
    def generate_dst_buffers(self):
        self.dstBuffers = self.generate_buffers(self.nBuffers,
                                                f"{self.name}_dst_buffer",
                                                self.dstMemref,
                                                self.dstTile,
                                                DMAChannelDir.S2MM,
                                                self.srcDMAChannel)

    def generate_flow(self):
        flow(self.srcTile,  WireBundle.DMA, self.srcDMAChannel,
             self.dstTile,  WireBundle.DMA, self.dstDMAChannel)
    
    def generate(self):
        self.generate_src_buffers()
        self.generate_dst_buffers()
        self.generate_src_locks()
        self.generate_dst_locks()
        self.generate_flow()


class DMAChain:
    """
    Generic struct to create and keep track of a chain of BDs for a single DMA.
    """
    def __init__(self, parent_op):
        self.blocks : list[Block] = []
        self.dmaStartOps : list[DMAStartOp] = []
        self.parent_op = parent_op

    def append_blocks(self, n_blocks):
        n = n_blocks
        if len(self.blocks) == 0:
            self.blocks = [Block.create_at_start(self.parent_op.regions[0])]
            n -= 1
        for _ in range(n):
            self.blocks.append(self.blocks[-1].create_after())
        return self.blocks[-n_blocks:len(self.blocks)]
    
    def append_start_op(self, startOp : DMAStartOp):
        self.dmaStartOps.append(startOp)
        if len(self.dmaStartOps) > 1:
            prevOp = self.dmaStartOps[-2]
            prevOp.successors[1] = InsertionPoint.current.block
    
    def append_end(self):
        assert len(self.dmaStartOps) > 0
        blocks = self.append_blocks(1)
        with InsertionPoint(blocks[0]), Location.unknown():
            EndOp()
        prevOp = self.dmaStartOps[-1]
        prevOp.successors[1] = blocks[0]



# ##########################################################################
# Main
# ##########################################################################

def main():
    argparser = common.get_default_argparser(1024, 1024, 1)
    argparser.add_argument("--cores", type=int, default=1)
    args = argparser.parse_args()
    assert args.N == 1  # matrix-VECTOR multiplication
    assert 0 < args.cores <= 4
    my_matmul(args.M, args.K, args.cores, args.trace)


# ##########################################################################
# Design
# ##########################################################################

def my_matmul(M, K, n_cores, trace_sz):
    assert(n_cores == 1)

    m = 32
    k = 32
    word_size_in = 2
    word_size_out = 4

    A_sz = M * K * word_size_in
    B_sz = K * word_size_in
    C_sz = M * word_size_out

    n_buffers_per_stream = 2  # ping-pong
    n_dma_blocks_per_stream = (n_buffers_per_stream  # mm2s and s2mm for each buffer
                               + 1)   # a start block
    
    split_A = True
    use_B_dummy = False

    # The trace we set up is on the left-most column and goes out the left-
    # most shim tile. When doing parallel transfers, the pathfinder routes
    # multiple channels in the leftmost column as well, and as a result
    # we run out of channels. To fix this, we move the memory tile that
    # moves the data one column to the right.
    trace_fix_offset = 0
    if trace_sz > 0:
        trace_fix_offset = 1

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_4col)
        def device_body():

            # ########################################
            # Declarations
            # ########################################

            memRef_inA_ty = T.memref(m * k, T.bf16())
            memRef_inA_partial_ty = T.memref(m * k // 2, T.bf16())
            memRef_inB_ty = T.memref(k, T.bf16())
            memRef_outC_ty = T.memref(m, T.f32())

            # AIE Core Function declarations
            zero_scalar = external_func("zero_scalar_f32", inputs=[memRef_outC_ty])
            zero = external_func("zero_vectorized_f32", inputs=[memRef_outC_ty])
            matvec_scalar = external_func(
                "matvec_scalar_bf16_f32",
                inputs=[memRef_inA_ty, memRef_inB_ty, memRef_outC_ty],
            )
            matvec = external_func(
                "matvec_vectorized_bf16_f32",
                inputs=[memRef_inA_ty, memRef_inB_ty, memRef_outC_ty],
            )
            passthrough_a = external_func(
                "passthrough_a",
                inputs=[T.IntegerType.get_signless(32), memRef_inA_ty, memRef_inB_ty, memRef_outC_ty],
            )
            passthrough_b = external_func(
                "passthrough_b",
                inputs=[memRef_inA_ty, memRef_inB_ty, memRef_outC_ty],
            )

            # Tile declarations
            ShimTiles = [tile(col, 0) for col in range(4)]
            MemTiles  = [tile(col, 1) for col in range(4)]
            CoreTiles = [tile(col, 2) for col in range(4)]

            # ########################################
            # Stream setup (Locks & Buffers)
            # ########################################

            A1_L3L2 : Stream = None
            if split_A:
                A2_L3L2 : Stream = None
            B_L3L2      : Stream = None
            A1_L2L1     : Stream = None
            A2_L2L1     : Stream = None
            if not use_B_dummy:
                B_L2L1      : Stream = None
            C_L1L3      : Stream = None
            
            ShimTile = ShimTiles[0]
            MemTile = MemTiles[trace_fix_offset]
            CoreTile = CoreTiles[0]

            mem_dummy_lock = lock(MemTile, init=10)
            core_dummy_lock = lock(CoreTile, init=10)

            A1_L3L2 = Stream(f"A1_L3L2", (ShimTile, 0), (MemTile, 0),  
                             memref=memRef_inA_ty if not split_A else
                                    memRef_inA_partial_ty,  
                             n_buffers=n_buffers_per_stream)
            A1_L3L2.generate()
            if split_A:
                A2_L3L2 = Stream(f"A2_L3L2", (ShimTile, 1), (MemTile, 2),  
                                 memref=memRef_inA_partial_ty, 
                                 n_buffers=n_buffers_per_stream)
                A2_L3L2.generate()

            if not use_B_dummy:
                B_L3L2 = Stream(f"B_L3L2", (ShimTiles[1], 0), (MemTile, 1),  memref=memRef_inB_ty,  n_buffers=n_buffers_per_stream)
                B_L3L2.generate()

            A1_L2L1 = Stream(f"A1_L2L1", (MemTile, 0),  (CoreTile, 0), memref=memRef_inA_ty,  n_buffers=n_buffers_per_stream)
            A1_L2L1.generate()
            # Flow and locks for the second half of A 
            # A1's buffer is used for both halves!
            if split_A:
                A2_L2L1 = Stream(f"A2_L2L1", (MemTile, 1), (CoreTile, 1), memref=memRef_inA_ty, n_buffers=n_buffers_per_stream)
                A2_L2L1.generate_src_locks()
                A2_L2L1.generate_dst_locks()
                A2_L2L1.generate_flow()

            # Transfer of B is fused with stream for A1 above
            if not use_B_dummy:
                B_L2L1 = Stream(f"B_L2L1", (MemTile, 0),  (CoreTile, 0), memref=memRef_inB_ty,  n_buffers=n_buffers_per_stream)
                B_L2L1.generate_dst_buffers()
                B_L2L1.generate_dst_locks()
            else:
                B_dummy = buffer(CoreTile, memRef_inB_ty.shape, memRef_inB_ty.element_type, name="b_dummy")

            C_L1L3 = Stream(f"C_L1L3", (CoreTile, 0), (ShimTile, 0), memref=memRef_outC_ty, n_buffers=n_buffers_per_stream)
            C_L1L3.generate()

            # DMA blocks must be defined after locks and buffers, since they
            # will refer to them.
            MemTileDMA = create_dma(MemTiles[trace_fix_offset])
            CoreTileDMA = create_dma(CoreTile)
            MemTileDMAChain = DMAChain(MemTileDMA)
            CoreTileDMAChain = DMAChain(CoreTileDMA)
            
            # ########################################
            # Data Movement - Matrix A + B (fused)
            # ########################################

            # Buffer Descriptors
            # L3 (ShimTile) stream --> L2 (MemTile) memory
            # L3 side is handled in sequence() function.
            A_halves = ([A1_L3L2] if not split_A else
                        [A1_L3L2, A2_L3L2])
            for A_half in A_halves:
                blocks = MemTileDMAChain.append_blocks(n_dma_blocks_per_stream)
                with InsertionPoint(blocks[0]), Location.unknown():
                    MemTileDMAChain.append_start_op(DMAStartOp(DMAChannelDir.S2MM, channel_index=A_half.dstDMAChannel, dest=blocks[1]))
                for j in range(0, n_buffers_per_stream):
                    with InsertionPoint(blocks[j+1]), Location.unknown():
                        use_lock(A_half.dstProdLock, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(A_half.dstBuffers[j], 
                               offset=0, 
                               len=get_memref_size(A_half.dstMemref))
                        use_lock(A_half.dstConsLock, LockAction.Release, value=1)
                        NextBDOp(blocks[j+2] if j+2<len(blocks) else blocks[1])

            if not use_B_dummy:
                blocks = MemTileDMAChain.append_blocks(n_dma_blocks_per_stream)
                with InsertionPoint(blocks[0]), Location.unknown():
                    MemTileDMAChain.append_start_op(DMAStartOp(DMAChannelDir.S2MM, channel_index=B_L3L2.dstDMAChannel, dest=blocks[1]))
                for j in range(0, n_buffers_per_stream):
                    with InsertionPoint(blocks[j+1]), Location.unknown():
                        use_lock(B_L3L2.dstProdLock, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(B_L3L2.dstBuffers[j], 
                               offset=0, 
                               len=get_memref_size(B_L3L2.dstMemref))
                        use_lock(B_L3L2.dstConsLock, LockAction.Release, value=1)
                        NextBDOp(blocks[j+2] if j+2<len(blocks) else blocks[1])

            # L2 (MemTile) memory --> stream towards L1 
            blocks = MemTileDMAChain.append_blocks(n_dma_blocks_per_stream + n_buffers_per_stream if not use_B_dummy else
                                                   n_dma_blocks_per_stream)
            with InsertionPoint(blocks[0]), Location.unknown():
                MemTileDMAChain.append_start_op(DMAStartOp(DMAChannelDir.MM2S, channel_index=A1_L2L1.srcDMAChannel, dest=blocks[1]))
            for j in range(0, n_buffers_per_stream):
                # Prepend vector b in stream of (first half of) A  (fused transfer)
                if not use_B_dummy:
                    with InsertionPoint(blocks[2*j+1]), Location.unknown():
                        use_lock(B_L3L2.dstConsLock, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(B_L3L2.dstBuffers[j], 
                               offset=0, 
                               len=get_memref_size(B_L3L2.dstMemref))
                        use_lock(B_L3L2.dstProdLock, LockAction.Release, value=1)
                        NextBDOp(blocks[2*j+2])
                with InsertionPoint(blocks[2*j+2] if not use_B_dummy else blocks[j+1]), Location.unknown():
                    use_lock(A1_L3L2.dstConsLock, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(A1_L3L2.dstBuffers[j], 
                           offset=0, 
                           len=get_memref_size(A1_L3L2.dstMemref))
                    use_lock(A1_L3L2.dstProdLock, LockAction.Release, value=1)
                    if not use_B_dummy:
                        NextBDOp(blocks[2*j+3] if 2*j+3<len(blocks) else blocks[1])
                    else:
                        NextBDOp(blocks[j+2] if j+2<len(blocks) else blocks[1])

            # Second half of A gets its own channel
            if split_A:
                blocks = MemTileDMAChain.append_blocks(n_dma_blocks_per_stream)
                with InsertionPoint(blocks[0]), Location.unknown():
                    MemTileDMAChain.append_start_op(DMAStartOp(DMAChannelDir.MM2S, channel_index=A2_L2L1.srcDMAChannel, dest=blocks[1]))
                for j in range(0, n_buffers_per_stream):
                    with InsertionPoint(blocks[j+1]), Location.unknown():
                        use_lock(A2_L3L2.dstConsLock, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(A2_L3L2.dstBuffers[j], 
                               offset=0,
                               len=get_memref_size(A2_L3L2.dstMemref))
                        use_lock(A2_L3L2.dstProdLock, LockAction.Release, value=1)
                        NextBDOp(blocks[j+2] if j+2<len(blocks) else blocks[1])

            # L2 (MemTile) stream --> L1 (CoreTile) memory
            # Fused movement of A and b through same channel
            # Second half of A gets its own channel
            blocks = CoreTileDMAChain.append_blocks(n_dma_blocks_per_stream + n_buffers_per_stream if not use_B_dummy else
                                                    n_dma_blocks_per_stream)
            with InsertionPoint(blocks[0]), Location.unknown():
                CoreTileDMAChain.append_start_op(DMAStartOp(DMAChannelDir.S2MM, channel_index=A1_L2L1.dstDMAChannel, dest=blocks[1]))
            for j in range(0, n_buffers_per_stream):
                if not use_B_dummy:
                    with InsertionPoint(blocks[2*j+1]), Location.unknown():
                        use_lock(B_L2L1.dstProdLock, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(B_L2L1.dstBuffers[j], 
                               offset=0, 
                               len=get_memref_size(memRef_inB_ty))
                        use_lock(B_L2L1.dstConsLock, LockAction.Release, value=1)
                        NextBDOp(blocks[2*j+2])
                with InsertionPoint(blocks[2*j+2] if not use_B_dummy else blocks[j+1]), Location.unknown():
                    use_lock(A1_L2L1.dstProdLock, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(A1_L2L1.dstBuffers[j], 
                           offset=0, 
                           len=get_memref_size(memRef_inA_ty)//2 if split_A else
                               get_memref_size(memRef_inA_ty),
                           dimensions=[(m//2 if split_A else m, 2), (k//2, 2*m), (2, 1)])
                    use_lock(A1_L2L1.dstConsLock, LockAction.Release, value=1)
                    if not use_B_dummy:
                        NextBDOp(blocks[2*j+3] if 2*j+3<len(blocks) else blocks[1])
                    else:
                        NextBDOp(blocks[j+2] if j+2<len(blocks) else blocks[1])
            # Separate channel for second half of A
            if split_A:
                blocks = CoreTileDMAChain.append_blocks(n_dma_blocks_per_stream)
                with InsertionPoint(blocks[0]), Location.unknown():
                    CoreTileDMAChain.append_start_op(DMAStartOp(DMAChannelDir.S2MM, channel_index=A2_L2L1.dstDMAChannel, dest=blocks[1]))
                for j in range(0, n_buffers_per_stream):
                    with InsertionPoint(blocks[j+1]), Location.unknown():
                        use_lock(A2_L2L1.dstProdLock, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(A1_L2L1.dstBuffers[j],  # fuse into one buffer (A1_L2L1's buffer)
                               offset=m//2*2,
                               len=get_memref_size(memRef_inA_ty)//2,
                               dimensions=[(m//2, 2), (k//2, 2*m), (2, 1)])
                        use_lock(A2_L2L1.dstConsLock, LockAction.Release, value=1)
                        NextBDOp(blocks[j+2] if j+2<len(blocks) else blocks[1])


            # ########################################
            # Data Movement - Matrix C
            # ########################################

            # L1 (CoreTile) memory --> L1 (CoreTile) stream
            blocks = CoreTileDMAChain.append_blocks(n_dma_blocks_per_stream)
            with InsertionPoint(blocks[0]), Location.unknown():
                CoreTileDMAChain.append_start_op(DMAStartOp(DMAChannelDir.MM2S, channel_index=C_L1L3.srcDMAChannel, 
                                                    dest=blocks[1]))
            for j in range(0, n_buffers_per_stream):
                with InsertionPoint(blocks[j+1]), Location.unknown():
                    use_lock(C_L1L3.srcConsLock, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(C_L1L3.srcBuffers[j], offset=0, len=get_memref_size(C_L1L3.srcMemref))
                    use_lock(C_L1L3.srcProdLock, LockAction.Release, value=1)
                    NextBDOp(blocks[j+2] if j<n_buffers_per_stream-1 else blocks[1])

            # Add end ops to all chains
            for chain in [CoreTileDMAChain, MemTileDMAChain]:
                if len(chain.blocks) > 0:
                    chain.append_end()

            # Set up a circuit-switched flow from core to shim for tracing information
            if trace_sz > 0:
                flow(CoreTiles[0], WireBundle.Trace, 0, ShimTiles[0], WireBundle.DMA, 1)

            # ########################################
            # Compute Cores
            # ########################################
            @core(CoreTiles[0], "mv.o")
            def core_body():
                assert(K // k % n_buffers_per_stream == 0)
                for _ in for_(0xFFFFFFFF):
                    for elem_out in C_L1L3.srcBuffers:
                        use_lock(C_L1L3.srcProdLock, LockAction.AcquireGreaterEqual, value=1)
                        call(zero, [elem_out])
                        for _ in for_(K // k // n_buffers_per_stream):
                            for j in range(n_buffers_per_stream):
                                use_lock(A1_L2L1.dstConsLock, LockAction.AcquireGreaterEqual, value=1)
                                if split_A:
                                    use_lock(A2_L2L1.dstConsLock, LockAction.AcquireGreaterEqual, value=1)
                                if not use_B_dummy:
                                    use_lock(B_L2L1.dstConsLock,  LockAction.AcquireGreaterEqual, value=1)
                                elem_in_a = A1_L2L1.dstBuffers[j]
                                elem_in_b = B_L2L1.dstBuffers[j] if not use_B_dummy else B_dummy
                                call(matvec, [elem_in_a, elem_in_b, elem_out])
                                #call(passthrough_a, [16, elem_in_a, elem_in_b, elem_out])
                                #call(passthrough_b, [elem_in_a, elem_in_b, elem_out])
                                use_lock(A1_L2L1.dstProdLock, LockAction.Release, value=1)
                                if split_A:
                                    use_lock(A2_L2L1.dstProdLock, LockAction.Release, value=1)
                                if not use_B_dummy:
                                    use_lock(B_L2L1.dstProdLock,  LockAction.Release, value=1)
                            yield_([])
                        use_lock(C_L1L3.srcConsLock, LockAction.Release, value=1)
                    yield_([])


            # ########################################
            # External Data Movement
            # ########################################

            @FuncOp.from_py_func(
                T.memref(A_sz // 4, T.i32()),
                T.memref(B_sz // 4, T.i32()),
                T.memref(C_sz // 4, T.i32()),
            )
            def sequence(A, B, C):
                # Tracing config
                if trace_sz > 0:
                    trace_utils.configure_simple_tracing_aie2(
                        CoreTiles[0],
                        ShimTiles[0],
                        ddr_id=2,
                        size=trace_sz,
                        offset=C_sz,
                    )

                # Repeat entire B vector M // m times.
                # If M // m  > 64, we need to split it up into separate transfers.
                transfer_sz = 64
                for transfer_i in range((M // m + transfer_sz - 1) // transfer_sz):
                    B_repeats = min(transfer_sz, M // m - transfer_i * transfer_sz)
                    transfer_offset = transfer_i * transfer_sz

                    # Each core is responsible for M rows of the output C.

                    # C
                    C_offset = transfer_offset * m * word_size_out // 4
                    npu_dma_memcpy_nd(
                        metadata=f"{C_L1L3.name}_dst_buffer",
                        bd_id=0,
                        mem=C,
                        offsets=[0, 0, 0, C_offset],
                        sizes=[1, 1, 1, B_repeats * m * word_size_out // 4],
                        strides=[0, 0, 0],
                    )

                    # A
                    A_offset = transfer_offset * K * word_size_in // 4
                    A_halves = ([A1_L3L2] if not split_A else
                                [A1_L3L2, A2_L3L2])
                    for idx, A_half in enumerate(A_halves):
                        npu_dma_memcpy_nd(
                            metadata=f"{A_half.name}_src_buffer",
                            bd_id=2+idx,
                            mem=A,
                            offsets=[0, 0, 0, A_offset + (idx*m//2*K*word_size_in//4)],
                            sizes=[B_repeats, 
                                    K // k, 
                                    m // len(A_halves), 
                                    k * word_size_in // 4],
                            strides=[m * K * word_size_in // 4, 
                                     k * word_size_in // 4, 
                                     K * word_size_in // 4,
                                     # implicitly 1
                                    ],
                        )

                    # B
                    if not use_B_dummy:
                        npu_dma_memcpy_nd(
                            metadata=f"{B_L3L2.name}_src_buffer",
                            bd_id=1,
                            mem=B,
                            sizes=[B_repeats, 1, 1, K * word_size_in // 4],
                            strides=[0, 0, 0],
                        )

                    npu_sync(column=0, row=0, direction=0, channel=0)
                    #npu_sync(column=1, row=0, direction=0, channel=0)

    print(ctx.module)


if __name__ == "__main__":
    main()
