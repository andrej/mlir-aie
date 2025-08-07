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

namespace xilinx::AIEX {

// We should not coalesce writes across the following operations, because
// these operations will cause the NPU to execute for some time and execution 
// on the cores/NPU could thus see older writes and do something based on them.
inline bool isWriteBarrier(Operation *op) {
  return llvm::isa<NpuSyncOp>(op);
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
            writeSequences.back().push_back(writeOp);
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