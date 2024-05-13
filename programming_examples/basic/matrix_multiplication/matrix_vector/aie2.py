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
            ShimDMAAllocationOp(name, dir, idx, 0)
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
        # Shim -> Mem
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
    m = 32
    k = 32
    word_size_in = 2
    word_size_out = 4

    A_sz = M * K * word_size_in
    B_sz = K * word_size_in
    C_sz = M * word_size_out

    use_B_dummy = False

    n_buffers_per_stream = 1  # ping-pong
    n_dma_blocks_per_stream = (n_buffers_per_stream  # mm2s and s2mm for each buffer
                               + 1)   # a start block

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

        @device(AIEDevice.npu1_4col)
        def device_body():

            # ########################################
            # Declarations
            # ########################################

            memRef_inA_ty = T.memref(m * k, T.bf16())
            memRef_inB_ty = T.memref(k, T.bf16())
            memRef_inA_inB_ty = T.memref(m * k + k, T.bf16())
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
                inputs=[memRef_inA_ty, memRef_inB_ty, memRef_outC_ty],
            )
            passthrough_b = external_func(
                "passthrough_b",
                inputs=[memRef_inA_ty, memRef_inB_ty, memRef_outC_ty],
            )

            # Tile declarations
            ShimTiles = [tile(col, 0) for col in range(n_cores)]
            MemTiles  = [tile(col, 1) for col in range(4)]
            CoreTiles = [tile(col, 2) for col in range(n_cores)]

            # ########################################
            # Stream setup (Locks & Buffers)
            # ########################################

            A_L3L2 : list[Stream] = [None] * n_cores
            #mem_dummy_locks : list[LockOp] = [None] * n_cores
            B_L3L2 : list[Stream] = [None] * n_cores
            A_L2L1 : list[Stream] = [None] * n_cores
            B_L2L1 : list[Stream] = [None] * n_cores
            dummy_locks : list[LockOp] = [None] * n_cores
            mem_dummy_locks : list[LockOp] = [None] * n_cores
            C_L1L3 : List[Stream] = [None] * n_cores
            for i in range(n_cores):
                ShimTile = ShimTiles[i]
                MemTile = MemTiles[i+trace_fix_offset]
                CoreTile = CoreTiles[i]

                A_L3L2[i] = Stream(f"A_L3L2_{i}", (ShimTile, 0), (MemTile, 0),  memref=memRef_inA_ty,  n_buffers=n_buffers_per_stream)
                B_L3L2[i] = Stream(f"B_L3L2_{i}", (ShimTile, 1), (MemTile, 1),  memref=memRef_inB_ty,  n_buffers=n_buffers_per_stream)
                A_L3L2[i].generate()
                B_L3L2[i].generate()

                #A_B_L2L1[i] = Stream(f"A_B_L2L1_{i}", (MemTile, 0),  (CoreTile, 0), memref=memRef_inA_inB_ty,  n_buffers=n_buffers_per_stream)
                #A_B_L2L1[i].generate_flow()
                #A_B_L2L1[i].generate_src_locks()
                #A_B_L2L1[i].generate_dst_locks()
                #A_B_L2L1[i].generate_src_buffers()
                #A_B_L2L1[i].dstBuffers = {
                #    "A" : [buffer(CoreTile, shape=memRef_inA_ty.shape, dtype=memRef_inA_ty.element_type, name=f"A_L2L1_{j}")
                #           for j in range(n_buffers_per_stream)],
                #    "B" : [buffer(CoreTile, shape=memRef_inB_ty.shape, dtype=memRef_inB_ty.element_type, name=f"B_L2L1_{j}")
                #           for j in range(n_buffers_per_stream)]
                #}

                A_L2L1[i] = Stream(f"A_L2L1_{i}", (MemTile, 0),  (CoreTile, 0), memref=memRef_inA_ty,  n_buffers=n_buffers_per_stream)
                A_L2L1[i].generate()
                B_L2L1[i] = Stream(f"B_L2L1_{i}", (MemTile, 0),  (CoreTile, 0), memref=memRef_inB_ty,  n_buffers=n_buffers_per_stream)
                B_L2L1[i].generate()
                #B_L2L1[i].generate_src_locks()
                #B_L2L1[i].generate_dst_locks()
                #B_L2L1[i].generate_src_buffers()
                #B_L2L1[i].generate_dst_buffers()
                #A_L2L1[i].generate_src_locks()
                #A_L2L1[i].generate_dst_locks()
                #A_L2L1[i].generate_src_buffers()

                C_L1L3[i] = Stream(f"C_L1L3_{i}", (CoreTile, 0), (ShimTile, 0), memref=memRef_outC_ty, n_buffers=n_buffers_per_stream)
                C_L1L3[i].generate()
                dummy_locks[i] = lock(CoreTile, init=10)
                mem_dummy_locks[i] = lock(MemTile, init=10)

            # DMA blocks must be defined after locks and buffers, since they
            # will refer to them.
            MemTileDMAs = [None] * 4
            MemTileDMAs[trace_fix_offset] = create_dma(MemTiles[trace_fix_offset])
            CoreTileDMAs = [create_dma(CoreTile) for CoreTile in CoreTiles]
            MemTileDMAChains = [DMAChain(MemTileDMA) for MemTileDMA in MemTileDMAs]
            CoreTileDMAChains = [DMAChain(CoreTileDMA) for CoreTileDMA in CoreTileDMAs]
            
            # ########################################
            # Data Movement - Matrix A + B (fused)
            # ########################################

            for i in range(n_cores):
                ShimTile = ShimTiles[i]
                MemTile = MemTiles[trace_fix_offset]
                MemTileDMAChain = MemTileDMAChains[trace_fix_offset]
                CoreTile = CoreTiles[i]
                CoreTileDMAChain = CoreTileDMAChains[i]

                # Buffer Descriptors
                # L3 (ShimTile) stream --> L2 (MemTile) memory
                # L3 side is handled in sequence() function.
                blocks = MemTileDMAChain.append_blocks(n_dma_blocks_per_stream)
                with InsertionPoint(blocks[0]), Location.unknown():
                    MemTileDMAChain.append_start_op(DMAStartOp(DMAChannelDir.S2MM, channel_index=A_L3L2[i].dstDMAChannel, dest=blocks[1]))
                for j in range(0, n_buffers_per_stream):
                    with InsertionPoint(blocks[j+1]), Location.unknown():
                        use_lock(A_L3L2[i].dstProdLock, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(A_L3L2[i].dstBuffers[j], offset=0, len=get_memref_size(A_L3L2[i].dstMemref))
                        use_lock(A_L3L2[i].dstConsLock, LockAction.Release, value=1)
                        NextBDOp(blocks[j+2] if j<n_buffers_per_stream-1 else blocks[1])

                blocks = MemTileDMAChain.append_blocks(n_dma_blocks_per_stream)
                with InsertionPoint(blocks[0]), Location.unknown():
                    MemTileDMAChain.append_start_op(DMAStartOp(DMAChannelDir.S2MM, channel_index=B_L3L2[i].dstDMAChannel, dest=blocks[1]))
                for j in range(0, n_buffers_per_stream):
                    with InsertionPoint(blocks[j+1]), Location.unknown():
                        use_lock(B_L3L2[i].dstProdLock, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(B_L3L2[i].dstBuffers[j], offset=0, len=get_memref_size(B_L3L2[i].dstMemref))
                        use_lock(B_L3L2[i].dstConsLock, LockAction.Release, value=1)
                        NextBDOp(blocks[j+2] if j<n_buffers_per_stream-1 else blocks[1])

                # L2 (MemTile) memory --> stream towards L1 
                blocks = MemTileDMAChain.append_blocks(n_dma_blocks_per_stream + n_buffers_per_stream)
                with InsertionPoint(blocks[0]), Location.unknown():
                    MemTileDMAChain.append_start_op(DMAStartOp(DMAChannelDir.MM2S, channel_index=A_L2L1[i].srcDMAChannel, dest=blocks[1]))
                for j in range(0, n_buffers_per_stream):
                    with InsertionPoint(blocks[2*j+1]), Location.unknown():
                        use_lock(A_L3L2[i].dstConsLock, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(A_L3L2[i].dstBuffers[j], offset=0, len=get_memref_size(A_L3L2[i].dstMemref),
                               #dimensions=[(m, k), (k, 1)]
                               )
                        use_lock(A_L3L2[i].dstProdLock, LockAction.Release, value=1)
                        NextBDOp(blocks[2*j+2])
                    with InsertionPoint(blocks[2*j+2]), Location.unknown():
                        use_lock(B_L3L2[i].dstConsLock, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(B_L3L2[i].dstBuffers[j], offset=0, len=get_memref_size(B_L3L2[i].dstMemref))
                        use_lock(B_L3L2[i].dstProdLock, LockAction.Release, value=1)
                        NextBDOp(blocks[2*j+3] if 2*j+3<len(blocks) else blocks[1])

                # L2 (MemTile) stream --> L1 (CoreTile) memory
                blocks = CoreTileDMAChain.append_blocks(n_dma_blocks_per_stream + n_buffers_per_stream)
                with InsertionPoint(blocks[0]), Location.unknown():
                    CoreTileDMAChain.append_start_op(DMAStartOp(DMAChannelDir.S2MM, channel_index=A_L2L1[i].dstDMAChannel, dest=blocks[1]))
                for j in range(0, n_buffers_per_stream):
                    with InsertionPoint(blocks[2*j+1]), Location.unknown():
                        use_lock(A_L2L1[i].dstProdLock, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(A_L2L1[i].dstBuffers[j], offset=0, len=get_memref_size(memRef_inA_ty))
                        use_lock(A_L2L1[i].dstConsLock, LockAction.Release, value=1)
                        NextBDOp(blocks[2*j+2])
                    with InsertionPoint(blocks[2*j+2]), Location.unknown():
                        use_lock(B_L2L1[i].dstProdLock, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(B_L2L1[i].dstBuffers[j], offset=0, len=get_memref_size(memRef_inB_ty))
                        use_lock(B_L2L1[i].dstConsLock, LockAction.Release, value=1)
                        NextBDOp(blocks[2*j+3] if 2*j+3<len(blocks) else blocks[1])


            # ########################################
            # Data Movement - Matrix C
            # ########################################
            for i in range(n_cores):
                ShimTile = ShimTiles[i]
                CoreTile = CoreTiles[i]
                CoreTileDMAChain = CoreTileDMAChains[i]

                # L1 (CoreTile) memory --> L1 (CoreTile) stream
                blocks = CoreTileDMAChain.append_blocks(n_dma_blocks_per_stream)
                with InsertionPoint(blocks[0]), Location.unknown():
                    CoreTileDMAChain.append_start_op(DMAStartOp(DMAChannelDir.MM2S, channel_index=0, 
                                                     dest=blocks[1]))
                for j in range(0, n_buffers_per_stream):
                    with InsertionPoint(blocks[j+1]), Location.unknown():
                        use_lock(C_L1L3[i].srcConsLock, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(C_L1L3[i].srcBuffers[j], offset=0, len=get_memref_size(C_L1L3[i].srcMemref))
                        use_lock(C_L1L3[i].srcProdLock, LockAction.Release, value=1)
                        NextBDOp(blocks[j+2] if j<n_buffers_per_stream-1 else blocks[1])

            # Add end ops to all chains
            for chains in [CoreTileDMAChains, MemTileDMAChains]:
                for chain in chains:
                    if len(chain.blocks) > 0:
                        chain.append_end()

            # Set up a circuit-switched flow from core to shim for tracing information
            if trace_sz > 0:
                flow(CoreTiles[0], WireBundle.Trace, 0, ShimTiles[0], WireBundle.DMA, 1)

            # ########################################
            # Compute Cores
            # ########################################
            for i in range(n_cores):
                # Compute tile i
                @core(CoreTiles[i], "mv.o")
                def core_body():
                    for _ in for_(0xFFFFFFFF):
                        for elem_out in C_L1L3[i].srcBuffers:
                            use_lock(C_L1L3[i].srcProdLock, LockAction.AcquireGreaterEqual, value=1)
                            call(zero, [elem_out])
                            for _ in for_(K // k // n_buffers_per_stream):
                                for j in range(n_buffers_per_stream):
                                    use_lock(A_L2L1[i].dstConsLock, LockAction.AcquireGreaterEqual, value=1)
                                    use_lock(B_L2L1[i].dstConsLock, LockAction.AcquireGreaterEqual, value=1)
                                    elem_in_a = A_L2L1[i].dstBuffers[j]
                                    elem_in_b = B_L2L1[i].dstBuffers[j]
                                    call(matvec_scalar, [elem_in_a, elem_in_b, elem_out])
                                    use_lock(A_L2L1[i].dstProdLock, LockAction.Release, value=1)
                                    use_lock(B_L2L1[i].dstProdLock, LockAction.Release, value=1)
                                yield_([])
                            use_lock(C_L1L3[i].srcConsLock, LockAction.Release, value=1)
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

                    # Each core is responsible for M // n_cores rows of the output C.
                    for i in range(n_cores):
                        # C
                        C_offset = (
                            transfer_offset * m * word_size_out // 4
                            + i * (M // n_cores) * word_size_out // 4
                        )
                        npu_dma_memcpy_nd(
                            metadata=f"{C_L1L3[i].name}_dst_buffer",
                            bd_id=0,
                            mem=C,
                            offsets=[0, 0, 0, C_offset],
                            sizes=[1, 1, 1, B_repeats * m * word_size_out // 4],
                            strides=[0, 0, 0],
                        )

                        # A
                        A_offset = (
                            transfer_offset * K * word_size_in // 4
                            + i * (M // n_cores) * K * word_size_in // 4
                        )
                        npu_dma_memcpy_nd(
                            metadata=f"{A_L3L2[i].name}_src_buffer",
                            bd_id=1,
                            mem=A,
                            offsets=[0, 0, 0, A_offset],
                            sizes=[B_repeats, K // k, m, k * word_size_in // 4],
                            strides=[
                                m * K * word_size_in // 4,
                                k * word_size_in // 4,
                                K * word_size_in // 4,
                            ],
                        )

                        # B
                        npu_dma_memcpy_nd(
                            metadata=f"{B_L3L2[i].name}_src_buffer",
                            bd_id=2,
                            mem=B,
                            sizes=[B_repeats, 1, 1, K * word_size_in // 4],
                            strides=[0, 0, 0],
                        )

                    for i in range(n_cores):
                        npu_sync(column=i, row=0, direction=0, channel=0)

    print(ctx.module)


if __name__ == "__main__":
    main()
