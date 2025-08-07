//===- AIEWrite32SequenceAnalysis.cpp ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/AIEWrite32SequenceAnalysis.h"

using namespace mlir;

// TODO: move these / get from elsewhere
// at the very least these must go in target model to support other architectures
// this is for aie2p
#define CORE_DMA_S2MM_0_Ctrl        0x1DE00
#define CORE_DMA_S2MM_0_Start_Queue 0x1DE04
#define CORE_DMA_S2MM_1_Ctrl        0x1DE08
#define CORE_DMA_S2MM_1_Start_Queue 0x1DE0C
#define CORE_DMA_MM2S_0_Ctrl        0x1DE10
#define CORE_DMA_MM2S_0_Start_Queue 0x1DE14
#define CORE_DMA_MM2S_1_Ctrl        0x1DE18
#define CORE_DMA_MM2S_1_Start_Queue 0x1DE1C

#define MEM_DMA_S2MM_0_Ctrl         0xA0600
#define MEM_DMA_S2MM_0_Start_Queue  0xA0604
#define MEM_DMA_S2MM_1_Ctrl         0xA0608
#define MEM_DMA_S2MM_1_Start_Queue  0xA060C
#define MEM_DMA_S2MM_2_Ctrl         0xA0610
#define MEM_DMA_S2MM_2_Start_Queue  0xA0614
#define MEM_DMA_S2MM_3_Ctrl         0xA0618
#define MEM_DMA_S2MM_3_Start_Queue  0xA061C
#define MEM_DMA_S2MM_4_Ctrl         0xA0620
#define MEM_DMA_S2MM_4_Start_Queue  0xA0624
#define MEM_DMA_S2MM_5_Ctrl         0xA0628
#define MEM_DMA_S2MM_5_Start_Queue  0xA062C
#define MEM_DMA_MM2S_0_Ctrl         0xA0630
#define MEM_DMA_MM2S_0_Start_Queue  0xA0634
#define MEM_DMA_MM2S_1_Ctrl         0xA0638
#define MEM_DMA_MM2S_1_Start_Queue  0xA063C
#define MEM_DMA_MM2S_2_Ctrl         0xA0640
#define MEM_DMA_MM2S_2_Start_Queue  0xA0644
#define MEM_DMA_MM2S_3_Ctrl         0xA0648
#define MEM_DMA_MM2S_3_Start_Queue  0xA064C
#define MEM_DMA_MM2S_4_Ctrl         0xA0650
#define MEM_DMA_MM2S_4_Start_Queue  0xA0654
#define MEM_DMA_MM2S_5_Ctrl         0xA0658
#define MEM_DMA_MM2S_5_Start_Queue  0xA065C

#define NOC_DMA_S2MM_0_Ctrl         0x1D200
#define NOC_DMA_S2MM_0_Start_Queue  0x1D204
#define NOC_DMA_S2MM_1_Ctrl         0x1D208
#define NOC_DMA_S2MM_1_Start_Queue  0x1D20C
#define NOC_DMA_MM2S_0_Ctrl         0x1D210
#define NOC_DMA_MM2S_0_Start_Queue  0x1D214
#define NOC_DMA_MM2S_1_Ctrl         0x1D218
#define NOC_DMA_MM2S_1_Start_Queue  0x1D21C

namespace xilinx::AIEX {

// We should not coalesce writes across the following operations, because
// these operations will cause the NPU to execute for some time and execution 
// could thus see older writes. If we coalesce the older writes with the newer 
// ones, this would violate the intended semantics of the program.
inline bool isWriteBarrier(Operation *op) {
  if(llvm::isa<NpuAddressPatchOp, NpuSyncOp>(op)) {
    // These operations cause the firmware to wait for AIE execution. This
    // gives the AIEs time to observe writes, so we cannot remove writes
    // across these barriers.
    return true;
  } else if (NpuWrite32Op writeOp = llvm::dyn_cast<NpuWrite32Op>(op)) {
    // Thse register writes cause the NPU to start executing things, e.g.
    // start a queue of tasks on a DMA. We cannot coalesce or remove these,
    // as multiple writes would mean starting multiple things.
    return false;
    std::optional<uint32_t> maybeAddr = writeOp.getAbsoluteAddress();
    if (maybeAddr.has_value()) {
        uint32_t value = writeOp.getValue();
        uint32_t addr = *maybeAddr;
        uint32_t addr_lower_20 = addr & 0xFFFFF;
        switch(addr_lower_20) {
            case CORE_DMA_S2MM_0_Start_Queue:
            case CORE_DMA_S2MM_1_Start_Queue:
            case CORE_DMA_MM2S_0_Start_Queue:
            case CORE_DMA_MM2S_1_Start_Queue:
            case MEM_DMA_S2MM_0_Start_Queue:
            case MEM_DMA_S2MM_1_Start_Queue:
            case MEM_DMA_S2MM_2_Start_Queue:
            case MEM_DMA_S2MM_3_Start_Queue:
            case MEM_DMA_S2MM_4_Start_Queue:
            case MEM_DMA_S2MM_5_Start_Queue:
            case MEM_DMA_MM2S_0_Start_Queue:
            case MEM_DMA_MM2S_1_Start_Queue:
            case MEM_DMA_MM2S_2_Start_Queue:
            case MEM_DMA_MM2S_3_Start_Queue:
            case MEM_DMA_MM2S_4_Start_Queue:
            case MEM_DMA_MM2S_5_Start_Queue:
            case NOC_DMA_S2MM_0_Start_Queue:
            case NOC_DMA_S2MM_1_Start_Queue:
            case NOC_DMA_MM2S_0_Start_Queue:
            case NOC_DMA_MM2S_1_Start_Queue:
                return true;
            case CORE_DMA_S2MM_0_Ctrl:
            case CORE_DMA_S2MM_1_Ctrl:
            case CORE_DMA_MM2S_0_Ctrl:
            case CORE_DMA_MM2S_1_Ctrl:
            case MEM_DMA_S2MM_0_Ctrl:
            case MEM_DMA_S2MM_1_Ctrl:
            case MEM_DMA_S2MM_2_Ctrl:
            case MEM_DMA_S2MM_3_Ctrl:
            case MEM_DMA_S2MM_4_Ctrl:
            case MEM_DMA_S2MM_5_Ctrl:
            case MEM_DMA_MM2S_0_Ctrl:
            case MEM_DMA_MM2S_1_Ctrl:
            case MEM_DMA_MM2S_2_Ctrl:
            case MEM_DMA_MM2S_3_Ctrl:
            case MEM_DMA_MM2S_4_Ctrl:
            case MEM_DMA_MM2S_5_Ctrl:
            case NOC_DMA_S2MM_0_Ctrl:
            case NOC_DMA_S2MM_1_Ctrl:
            case NOC_DMA_MM2S_0_Ctrl:
            case NOC_DMA_MM2S_1_Ctrl:
                return value & 0b10; // reset bit set  FIXME check if this is really bit at position 1 or 0
        }
    }
  }
  return false;
}

AIEWrite32SequenceAnalysis::AIEWrite32SequenceAnalysis(Operation *op, AnalysisManager &am) 
  : analysisManager(am), writeSequences()
{
    RuntimeSequenceOp runtimeSequenceOp = llvm::dyn_cast<RuntimeSequenceOp>(op);
    DominanceInfo dominanceInfo(runtimeSequenceOp);
    if (!runtimeSequenceOp) {
        op->emitError("AIEWrite32SequenceAnalysis can only be called on aiex.runtime_sequence operations.");
        return;
    }

    // This implementation is simple because we assume that the runtime 
    // sequence is sequential/linear with just a single block.
    writeSequences.emplace_back();
    std::vector<NpuWrite32Op> currentWriteSequence = {};
    for (Operation &op : runtimeSequenceOp.getOps()) {
        if (isWriteBarrier(&op)) {
            writeSequences.emplace_back();
        } else if (NpuWrite32Op writeOp = llvm::dyn_cast<NpuWrite32Op>(op))  {
            writeSequences.back().push_back(writeOp); // somehow causes heap bounds violation?
        }
    }

    for (std::vector<NpuWrite32Op> &seq : writeSequences) {
        std::sort(seq.begin(), seq.end(),
            [&](NpuWrite32Op a, NpuWrite32Op b) {
                uint32_t addrA = a.getAbsoluteAddress().value_or(0);
                uint32_t addrB = b.getAbsoluteAddress().value_or(0);
                if (addrA < addrB) {
                    return true;
                } else if (addrA == addrB) {
                    return dominanceInfo.properlyDominates(a, b);
                } else {
                    return false;
                }
            });
    }

}

}