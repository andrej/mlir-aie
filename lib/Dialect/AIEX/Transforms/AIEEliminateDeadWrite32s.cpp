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

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <algorithm>
#include <cstdint>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

struct EliminateDeadWrite32sPattern : OpConversionPattern<NpuBlockWriteOp> {
  using OpConversionPattern::OpConversionPattern;

  EliminateDeadWrite32sPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuBlockWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return failure();
  }
};


struct AIEEliminateDeadWrite32sPass : AIEEliminateDeadWrite32sBase<AIEEliminateDeadWrite32sPass> {
  void runOnOperation() override {
    AIE::DeviceOp deviceOp = getOperation();

    // target.addLegalDialect<AIEXDialect>();

    ConversionTarget target(getContext());
    target.addIllegalOp<NpuBlockWriteOp>();

    RewritePatternSet patterns(&getContext());
    patterns.insert<EliminateDeadWrite32sPattern>(&getContext());

    if (failed(applyPartialConversion(deviceOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIEEliminateDeadWrite32sPass() {
  return std::make_unique<AIEEliminateDeadWrite32sPass>();
}