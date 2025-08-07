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
#include "aie/Dialect/AIEX/AIEUtils.h"

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

struct Write32SequenceAnalysis {
  AnalysisManager &analysisManager;
  
  std::vector<std::vector<NpuWrite32Op>> writeSequences;

  Write32SequenceAnalysis(Operation *op, AnalysisManager &am) 
  : analysisManager(am) 
  {
    RuntimeSequenceOp runtimeSequenceOp = llvm::dyn_cast<RuntimeSequenceOp>(op);
    if (!runtimeSequenceOp) {
      op->emitError("Write32SequenceAnalysis can only be called on aiex.runtime_sequence operations.");
      return;
    }

    // This implementation is simple because we assume that the runtime 
    // sequence is sequential/linear with just a single block.
    writeSequences.emplace_back();
    for (Operation &op : runtimeSequenceOp.getOps()) {
        if (isCoalescingBarrier(&op)) {
            writeSequences.emplace_back();
        } else if (NpuWrite32Op writeOp = llvm::dyn_cast<NpuWrite32Op>(op))  {
            writeSequences.back().push_back(writeOp);
        }
    }

    for (std::vector<NpuWrite32Op> &seq : writeSequences) {
       std::sort(seq.begin(), seq.end(),
          [](NpuWrite32Op a, NpuWrite32Op b) {
            return a.getAbsoluteAddress() < b.getAbsoluteAddress();
          });
    }

  }

};

// Assumes the vector of input write32s passed in is contiguous, and ordered
// in order of increasing address.
void coalesceContiguousWrite32s(OpBuilder &builder, std::vector<NpuWrite32Op> &contiguousWriteOps) {
  std::vector<uint32_t> data;
  for (NpuWrite32Op &writeOp : contiguousWriteOps) {
    data.push_back(writeOp.getValue());
  }
  uint32_t base_address = contiguousWriteOps[0].getAbsoluteAddress().value();
  memref::GlobalOp dataOp = nullptr;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    AIE::DeviceOp parentDeviceOp = contiguousWriteOps[0]->getParentOfType<AIE::DeviceOp>();
    builder.setInsertionPointToStart(parentDeviceOp.getBody());
    dataOp = getOrCreateDataMemref(builder, parentDeviceOp, contiguousWriteOps[0].getLoc(), data);
  }
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(contiguousWriteOps[0]);
    memref::GetGlobalOp getDataOp = builder.create<memref::GetGlobalOp>(contiguousWriteOps[0].getLoc(), dataOp.getType(), dataOp.getName());
    builder.create<NpuBlockWriteOp>(
      contiguousWriteOps[0].getLoc(),
      base_address,
      getDataOp.getResult(),
      nullptr, // buffer
      nullptr, // row
      nullptr  // column
    );
  }
  for (NpuWrite32Op &writeOp : contiguousWriteOps) {
    writeOp.erase();
  }
}

struct AIECoalesceWrite32sPass : AIECoalesceWrite32sBase<AIECoalesceWrite32sPass> {
  void runOnOperation() override {
    AIE::DeviceOp deviceOp = getOperation();
    OpBuilder builder(deviceOp);

    for (RuntimeSequenceOp runtimeSequenceOp : deviceOp.getOps<RuntimeSequenceOp>()) {
      AnalysisManager am = getAnalysisManager().nest(runtimeSequenceOp);
      Write32SequenceAnalysis sa = am.getAnalysis<Write32SequenceAnalysis>();
      for (std::vector<NpuWrite32Op> &writeSequence : sa.writeSequences) {
        if (writeSequence.size() < 2) {
          continue; // Nothing to coalesce
        }

        uint32_t base_address = writeSequence.front().getAbsoluteAddress().value();
        std::vector<NpuWrite32Op> contiguousSequence;
        for (NpuWrite32Op writeOp : writeSequence) {
          if (writeOp.getAbsoluteAddress().value() != base_address + contiguousSequence.size() * sizeof(uint32_t)) {
            // Contiguous sequence (if any) has been interrupted
            if (contiguousSequence.size() > 1) {
              coalesceContiguousWrite32s(builder, contiguousSequence);
            }
            contiguousSequence.clear();
            base_address = writeOp.getAbsoluteAddress().value();
          }
          contiguousSequence.push_back(writeOp);
        } // end for each write
        if (contiguousSequence.size() > 1) {
          coalesceContiguousWrite32s(builder, contiguousSequence);
        }
      }  // end for each write sequence
    } // end for each runtime sequence op
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIECoalesceWrite32sPass() {
  return std::make_unique<AIECoalesceWrite32sPass>();
}