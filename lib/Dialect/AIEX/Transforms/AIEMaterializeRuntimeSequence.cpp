//===- AIEMaterializeRuntimeSequence.cpp -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "aie-materialize-runtime-sequence"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

struct RuntimeCallGraphCyclicityAnalysis {
  AnalysisManager &analysisManager;

  // if invalid, analysis failed and results should not be considered
  bool isValid = false;

  // Call graph is cyclic
  bool isCyclic = false;

  RuntimeCallGraphCyclicityAnalysis(Operation *op, AnalysisManager &am) 
  : analysisManager(am) 
  {
    RuntimeSequenceOp runtimeSequenceOp = llvm::dyn_cast<RuntimeSequenceOp>(op);
    if (!runtimeSequenceOp) {
      op->emitError("RuntimeCallGraphCyclicityAnalysis can only be called on aiex.runtime_sequence operations.");
      return;
    }
    llvm::SetVector<RuntimeSequenceOp> visited = {};
    llvm::SetVector<RunOp> todo;
    for (RunOp runOp : runtimeSequenceOp.getOps<RunOp>()) {
      todo.insert(runOp);
    }
    while(!todo.empty()) {
      RunOp curOp = todo.pop_back_val();
      if (RuntimeSequenceOp calleeRuntimeSequence = curOp.getCalleeRuntimeSequenceOp()) {
        if (visited.contains(calleeRuntimeSequence)) {
          continue;
        }
        visited.insert(calleeRuntimeSequence);
        for (RunOp runOp : calleeRuntimeSequence.getOps<RunOp>()) {
          todo.insert(runOp);
        }
      }
    }
    isCyclic = visited.contains(runtimeSequenceOp);
    isValid = true;
  }
};

// Analysis that gives all the write32s that have occured since the last
// configure op A (or the start of the runtime sequence if this is the first
// configure op) up to a given configure op B. These are the registers that 
// might need to be reset before reconfiguration.
struct WrittenRegistersAnalysis {
  AnalysisManager &analysisManager;
  std::map<ConfigureOp, std::vector<NpuWrite32Op>> writtenRegistersUpto;
  std::map<ConfigureOp, std::vector<NpuMaskWrite32Op>> writtenMaskedRegistersUpto;

  WrittenRegistersAnalysis(Operation *op, AnalysisManager &am)
      : analysisManager(am) {
    RuntimeSequenceOp runtimeSequenceOp = llvm::dyn_cast<RuntimeSequenceOp>(op);
    if (!runtimeSequenceOp) {
      op->emitError("WrittenRegistersAnalysis can only be called on aiex.runtime_sequence operations.");
      return;
    }

    std::vector<NpuWrite32Op> currentWrittenRegisters;
    std::vector<NpuMaskWrite32Op> currentWrittenMaskedRegisters;
    for (Operation &op : runtimeSequenceOp.getOps()) {
      if (NpuWrite32Op writeOp = llvm::dyn_cast<NpuWrite32Op>(op)) {
        currentWrittenRegisters.push_back(writeOp);
      } else if(NpuMaskWrite32Op maskWriteOp = llvm::dyn_cast<NpuMaskWrite32Op>(op)) {
        currentWrittenMaskedRegisters.push_back(maskWriteOp);
      } else if(ConfigureOp configureOp = llvm::dyn_cast<ConfigureOp>(op)) {
        writtenRegistersUpto[configureOp] = currentWrittenRegisters;
        writtenMaskedRegistersUpto[configureOp] = currentWrittenMaskedRegisters;
        currentWrittenRegisters.clear();
        currentWrittenMaskedRegisters.clear();
      }
    }
  }

};

struct InsertResetOpsPattern : RewritePattern {

  WrittenRegistersAnalysis &writtenRegs;

  InsertResetOpsPattern(MLIRContext *context, WrittenRegistersAnalysis &writtenRegs, PatternBenefit benefit = 1)
      : RewritePattern(ConfigureOp::getOperationName(), benefit, context), writtenRegs(writtenRegs) {}

  LogicalResult
  matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    ConfigureOp configureOp = llvm::dyn_cast<ConfigureOp>(op);
    if (!configureOp) {
      return failure();
    }
    int nInsertedResetOps = 0;
    for (NpuWrite32Op writeOp : writtenRegs.writtenRegistersUpto[configureOp]) {
      // Insert reset ops before each write op
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(configureOp);
      rewriter.create<NpuWrite32Op>(configureOp.getLoc(), writeOp.getAbsoluteAddress().value(), 0, nullptr, nullptr, nullptr);
      nInsertedResetOps++;
    }
    for (NpuMaskWrite32Op maskWriteOp : writtenRegs.writtenMaskedRegistersUpto[configureOp]) {
      // Insert reset ops before each mask write op
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(configureOp);
      rewriter.create<NpuMaskWrite32Op>(configureOp.getLoc(), maskWriteOp.getAbsoluteAddress().value(), 0, maskWriteOp.getMask(), nullptr, nullptr, nullptr);
      nInsertedResetOps++;
    }
    if (nInsertedResetOps > 0) {
      return success();
    }
    return failure();
  }
};

// Turn aie.configure @device into aie.run %.. @configure
// TODO: add check that liveness of two aie.configures do not overlap
// (i.e., when we configure A, then configure B, cannot call runtime sequence of A after configuring B)
// TODO: add code to remove repeated @configure ops
struct InsertRunConfigureForConfigurePattern : RewritePattern {

  InsertRunConfigureForConfigurePattern(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(ConfigureOp::getOperationName(), benefit, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, 
                  PatternRewriter &rewriter) const override {
    ConfigureOp configureOp = llvm::dyn_cast<ConfigureOp>(op);
    if (!configureOp) {
      return failure();
    }

    AIE::DeviceOp referencedDevice = configureOp.getReferencedDeviceOp();
    if (!referencedDevice) {
      configureOp.emitError("Referenced symbol is not a device");
      return failure();
    }
    Operation *maybeConfigurationRuntimeSequence = SymbolTable::lookupSymbolIn(referencedDevice, "configure");
    if (!maybeConfigurationRuntimeSequence) {
      auto err = configureOp.emitError("No @configure runtime sequence found for this configuration");
      err.attachNote(referencedDevice.getLoc()) << "This device does not have a @configure runtime sequence";
      err.attachNote() << "Call `aiecc.py --aie-generate-txn` / `aie-opt -convert-aie-to-transaction` first to generate a configuration sequence from your design.";
      return failure();
    }
    RuntimeSequenceOp configureRuntimeSequence = llvm::dyn_cast<RuntimeSequenceOp>(maybeConfigurationRuntimeSequence);
    if (!configureRuntimeSequence) {
      configureOp.emitError("Referenced @configure symbol is not a runtime sequence");
      return failure();
    }
    // Insert a new op of type AIEX::RunOp
    rewriter.setInsertionPointAfter(configureOp);
    rewriter.create<AIEX::RunOp>(
      configureOp.getLoc(), 
      configureOp.getResult(),
      "configure",
      ::mlir::ValueRange({})
    );
    return success();
  }
};


// Inlines the definitions of all symbols referenced in the givent operation 
// at the current insertion point in the given rewriter, unless the symbol
// definition is in the "previouslyInlinedSymbolMap" map. While inlining,
// symbols will be renamed to have a unique name.
void inlineReferencedSymbolDefinitions(
    PatternRewriter &rewriter, 
    Operation *op, 
    Operation *lookupFrom,
    IRMapping argMap,
    llvm::DenseMap<SymbolRefAttr, SymbolRefAttr> &previouslyInlinedSymbolMap) {
  MLIRContext *ctx = op->getContext();
  for (NamedAttribute namedAttr : op->getAttrs()) {
    Attribute attr = namedAttr.getValue();
    auto newAttr = attr.replace(
      [&](SymbolRefAttr oldSymbolRef) {
        SymbolRefAttr newSymbolRef;
        if (!previouslyInlinedSymbolMap.count(oldSymbolRef)) {
          llvm::StringRef oldName = oldSymbolRef.getRootReference().getValue();
          std::string uniqueName = oldName.str();
          unsigned uniquingCounter = 0;
          while (SymbolTable::lookupNearestSymbolFrom(op, StringAttr::get(ctx, uniqueName))) {
            uniqueName = oldName.str() + "_" + std::to_string(uniquingCounter);
            uniquingCounter++;
          }
          newSymbolRef = SymbolRefAttr::get(ctx, uniqueName);
          previouslyInlinedSymbolMap[oldSymbolRef] = newSymbolRef;

          // Add the new symbol definition
          Operation *symbolDefOp = SymbolTable::lookupNearestSymbolFrom(lookupFrom, oldSymbolRef);
          Operation *clonedSymbolDefOp = rewriter.clone(*symbolDefOp, argMap);
          clonedSymbolDefOp->setAttr(SymbolTable::getSymbolAttrName(), StringAttr::get(ctx, uniqueName));
        } else {
          newSymbolRef = previouslyInlinedSymbolMap[oldSymbolRef];
        }
        return std::make_pair(newSymbolRef, WalkResult::advance());
      });
    op->setAttr(namedAttr.getName(), newAttr);
  }
}

struct InlineRuntimeCallsPattern : RewritePattern {

  InlineRuntimeCallsPattern(MLIRContext *ctx)
      : RewritePattern(RunOp::getOperationName(), PatternBenefit(1),
                       ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // matching logic
    RunOp runOp = llvm::dyn_cast<RunOp>(op);
    if (!runOp) {
      return failure();
    }
    AIE::DeviceOp calleeDevice = runOp.getCalleeDeviceOp();
    RuntimeSequenceOp calleeRuntimeSequence = runOp.getCalleeRuntimeSequenceOp();
    if (!calleeDevice, !calleeRuntimeSequence) {
      return failure();
    }
    if (!calleeRuntimeSequence.getOps<RunOp>().empty()) {
      return failure();
    }

    // rewrite logic
    Region &calleeBody = calleeRuntimeSequence.getBody();
    AIE::DeviceOp callerDevice = op->getParentOfType<AIE::DeviceOp>();
    if (!callerDevice) {
      runOp.emitError() << "needs to be in a DeviceOp";
      return failure();
    }
    Region &callerDeviceBody = callerDevice.getBodyRegion();
    IRMapping argMap;
    ValueRange values = runOp.getArgs();
    for (unsigned i = 0, n = calleeBody.getNumArguments(); i < n; i++) {
      BlockArgument arg = calleeBody.getArgument(i);
      Value val = values[i];
      if(arg.getType() != val.getType()) {
        return runOp.emitOpError() << "argument " << i << " type mismatch: "
                                   << " expected " << arg.getType()
                                   << " but got " << val.getType();
      }
      argMap.map(arg, val);
    }
    llvm::DenseMap<SymbolRefAttr, SymbolRefAttr> previouslyInlinedSymbolMap;
    rewriter.setInsertionPoint(runOp);
    mlir::OpBuilder::InsertPoint clonedOpInsertionPoint = rewriter.saveInsertionPoint();
    mlir::Block &callerDeviceBodyFirstBlock = callerDeviceBody.front();
    mlir::OpBuilder::InsertPoint clonedSymbolInsertionPoint(&callerDeviceBodyFirstBlock, callerDeviceBodyFirstBlock.begin());
    for (Operation &op : calleeBody.getOps()) {
      rewriter.restoreInsertionPoint(clonedOpInsertionPoint);
      Operation *clonedOp = rewriter.clone(op, argMap);
      clonedOpInsertionPoint = rewriter.saveInsertionPoint();

      rewriter.restoreInsertionPoint(clonedSymbolInsertionPoint);
      inlineReferencedSymbolDefinitions(
        rewriter, clonedOp, calleeRuntimeSequence.getOperation(), argMap, previouslyInlinedSymbolMap);
      clonedSymbolInsertionPoint = rewriter.saveInsertionPoint();
    }

    rewriter.eraseOp(runOp);

    return success();
  }
};

struct AIEMaterializeRuntimeSequencePass
    : AIEMaterializeRuntimeSequenceBase<AIEMaterializeRuntimeSequencePass> {
  void runOnOperation() override {
    AIE::DeviceOp deviceOp = getOperation();

    // Turn aie.configure to aie.run @configure
    for (RuntimeSequenceOp runtimeSequenceOp : deviceOp.getOps<RuntimeSequenceOp>()) {
      AnalysisManager am = getAnalysisManager().nest(runtimeSequenceOp);
      RuntimeCallGraphCyclicityAnalysis cyclicity = am.getAnalysis<RuntimeCallGraphCyclicityAnalysis>();
      if (!cyclicity.isValid) {
        return signalPassFailure();
      }
      if (cyclicity.isCyclic) {
        runtimeSequenceOp.emitError("Runtime sequence contains a cycle");
        return signalPassFailure();
      }
      RewritePatternSet patterns(&getContext());
      patterns.insert<InsertRunConfigureForConfigurePattern>(&getContext());
      walkAndApplyPatterns(runtimeSequenceOp, std::move(patterns));
    }

    // Greedily inline all runtime sequences that can be inlined;
    // this will start with runtime sequences that do not call other runtime
    // sequences (leaves); once their callers inline them, the callers can
    // be inlined as well, and so on
    MLIRContext *ctx = &getContext();
    AIE::DeviceOp device = getOperation();
    GreedyRewriteConfig rewriter_config = GreedyRewriteConfig();
    rewriter_config.setRegionSimplificationLevel(
        GreedySimplifyRegionLevel::Disabled);

    RewritePatternSet patterns(ctx);
    patterns.insert<InlineRuntimeCallsPattern>(ctx);
    if (failed(applyPatternsGreedily(device, std::move(patterns), rewriter_config))) {
      signalPassFailure();
    }

    // After inlining configure ops, insert reset ops.
    for (RuntimeSequenceOp runtimeSequenceOp : deviceOp.getOps<RuntimeSequenceOp>()) {
      AnalysisManager am = getAnalysisManager().nest(runtimeSequenceOp);
      WrittenRegistersAnalysis writtenRegs = am.getAnalysis<WrittenRegistersAnalysis>();
      RewritePatternSet patterns(&getContext());
      patterns.insert<InsertResetOpsPattern>(&getContext(), writtenRegs);
      walkAndApplyPatterns(runtimeSequenceOp, std::move(patterns));
    }

  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEMaterializeRuntimeSequencePass() {
  return std::make_unique<AIEMaterializeRuntimeSequencePass>();
}