//===- AIESeparateBlockwrites.cpp -------------------------------*- C++ -*-===//
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
#include "aie/Dialect/AIEX/Utils/AIEUtils.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <algorithm>
#include <cstdint>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

struct SeparateBlockwritesPattern : OpConversionPattern<NpuBlockWriteOp> {
  using OpConversionPattern::OpConversionPattern;

  SeparateBlockwritesPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit) {}

  LogicalResult
  matchAndRewrite(NpuBlockWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    std::optional<uint32_t> base_address = op.getAbsoluteAddress();
    if (!base_address.has_value()) {
        op.emitError("expected address");
        return failure();
    }
    
    DenseIntElementsAttr data = op.getDataWords();
    if (!data) {
      op.emitError("expected data");
      return failure();
    }

    // Split the block write into individual writes
    for (auto [i, word] : llvm::enumerate(data)) {
      uint32_t address = *base_address + i * sizeof(uint32_t);
      uint32_t value = word.getZExtValue();
      rewriter.create<NpuWrite32Op>(
        op.getLoc(), 
        address, 
        value,
        nullptr,
        nullptr,
        nullptr
      );
    }

    rewriter.eraseOp(op.getOperation());

    return success();
  }
};


struct AIESeparateBlockwritesPass : AIESeparateBlockwritesBase<AIESeparateBlockwritesPass> {

  void runOnOperation() override {
    AIE::DeviceOp deviceOp = getOperation();

    // target.addLegalDialect<AIEXDialect>();

    ConversionTarget target(getContext());
    target.addLegalOp<NpuWrite32Op>();
    target.addIllegalOp<NpuBlockWriteOp>();

    RewritePatternSet patterns(&getContext());
    patterns.insert<SeparateBlockwritesPattern>(&getContext());

    if (failed(applyPartialConversion(deviceOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>> AIEX::createAIESeparateBlockwritesPass() {
  return std::make_unique<AIESeparateBlockwritesPass>();
}
