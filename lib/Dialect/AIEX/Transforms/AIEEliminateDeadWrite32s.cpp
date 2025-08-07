//===- AIEEliminateDeadWrite32s.cpp -----------------------------*- C++ -*-===//
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
#include "aie/Dialect/AIEX/AIEWrite32SequenceAnalysis.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include <iterator>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

struct AIEEliminateDeadWrite32sPass : AIEEliminateDeadWrite32sBase<AIEEliminateDeadWrite32sPass> {
  void runOnOperation() override {
    AIE::DeviceOp deviceOp = getOperation();
    OpBuilder builder(deviceOp);

    for (RuntimeSequenceOp runtimeSequenceOp : deviceOp.getOps<RuntimeSequenceOp>()) {
      AnalysisManager am = getAnalysisManager().nest(runtimeSequenceOp);
      AIEWrite32SequenceAnalysis writtenRegs = am.getAnalysis<AIEWrite32SequenceAnalysis>();
      for (std::vector<NpuWrite32Op> &writeSequence : writtenRegs.writeSequences) {
        // writes in writesequence are ordered by address, so it suffices to
        // only compare neighboring writes
        // iterate over all NpuWrite32Ops in the sequence, skipping the first one
        for (auto it = writeSequence.begin(); it != writeSequence.end(); ++it) {
          NpuWrite32Op &currentOp = *it;
          auto nextOpIt = std::next(it);
          if (nextOpIt != writeSequence.end()) {
            NpuWrite32Op &nextOp = *nextOpIt;
            // If the next write is to the same address, we can eliminate the current write
            if (currentOp.getAbsoluteAddress() == nextOp.getAbsoluteAddress()) {
              currentOp.erase();
            }
          }
        }
      } // end for each write sequence
    } // end for each runtime sequence
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIEEliminateDeadWrite32sPass() {
  return std::make_unique<AIEEliminateDeadWrite32sPass>();
}