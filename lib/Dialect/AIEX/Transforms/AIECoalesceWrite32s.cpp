//===- AIECoalesceWrite32s.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <algorithm>
#include <cstdint>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

// We should not coalesce writes across the following operations, because
// these operations will cause the NPU to execute for some time and execution 
// could thus see older writes. If we coalesce the older writes with the newer 
// ones, this would violate the intended semantics of the program.
inline bool isCoalescingBarrier(Operation *op) {
  return llvm::isa<NpuPushQueueOp, NpuAddressPatchOp, NpuDmaWaitOp, NpuSyncOp>(op);
}

struct AIEWrite32SequenceAnalysis {
  AnalysisManager &analysisManager;
  
  std::vector<std::vector<NpuWrite32Op>> writeSequences;
  std::map<NpuWrite32Op, std::vector<NpuWrite32Op> &> whichWriteSequence;

  AIEWrite32SequenceAnalysis(Operation *op, AnalysisManager &am) 
  : analysisManager(am) 
  {
    RuntimeSequenceOp runtimeSequenceOp = llvm::dyn_cast<RuntimeSequenceOp>(op);
    if (!runtimeSequenceOp) {
      op->emitError("AIEWrite32SequenceAnalysis can only be called on aiex.runtime_sequence operations.");
      return;
    }

    // This implementation is simple because we assume that the runtime 
    // sequence is sequential/linear with just a single block.
    std::vector<NpuWrite32Op> &currentSequence = writeSequences.emplace_back();
    for (Operation *op : runtimeSequenceOp.getOps()) {
        if (isCoalescingBarrier(op)) {
            currentSequence = writeSequences.emplace_back();
        } else if (NpuWrite32Op writeOp = llvm::dyn_cast<NpuWrite32Op>(op))  {
            currentSequence.push_back(writeOp);
            whichWriteSequence[writeOp] = currentSequence;
        }
    }

    for (std::vector<NpuWrite32Op> &seq : writeSequences) {
       std::sort(seq.begin(), seq.end(),
          [](NpuWrite32Op a, NpuWrite32Op b) {
            return a.getAbsoluteAddress() < b.getAbsoluteAddress();
          });
    }

  }

}

struct AIECoalesceWrite32sPass : AIECoalesceWrite32sBase<AIECoalesceWrite32sPass> {
  void runOnOperation() override {
    AIE::DeviceOp deviceOp = getOperation();
    OpBuilder builder(deviceOp);

    for (RuntimeSequenceOp runtimeSequenceOp : deviceOp.getOps<RuntimeSequenceOp>()) {
      AnalysisManager am = getAnalysisManager().nest(runtimeSequenceOp);
      AIEWrite32SequenceAnalysis sa = am.getAnalysis<AIEWrite32SequenceAnalysis>();
      for (std::vector<NpuWrite32Op> &writeSequence : sa.writeSequences) {
        if (writeSequence.size() < 2) {
          continue; // Nothing to coalesce
        }

        uint32_t base_address = writeSequence.front().getAbsoluteAddress().value();
        std::vector<uint32_t> data;
        std::vector<NpuWrite32Op> toErase;
        for (NpuWrite32Op writeOp : writeSequence) {
          if (writeOp.getAbsoluteAddress().value() == base_address + data.size() * sizeof(uint32_t)) {
            data.push_back(writeOp.getValue());
            toErase.push_back(writeOp);
          } else {
            if (data.size() > 1) {
              // Want at least two writes to coalesce
              memref::GlobalOp dataOp = getOrCreateDataMemref(builder, writeOp.getLoc(), data);
              builder.create<NpuBlockWriteOp>(
                writeOp.getLoc(),
                base_address,
                dataOp.getSymName(),
                nullptr, // buffer
                nullptr, // row
                nullptr  // column
              );
            } else {
                toErase.clear();
            }
          }
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIECoalesceWrite32sPass() {
  return std::make_unique<AIECoalesceWrite32sPass>();
}