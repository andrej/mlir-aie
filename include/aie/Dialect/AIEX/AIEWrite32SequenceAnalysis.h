//===- AIEWrite32SequenceAnalysis.h -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/AnalysisManager.h"
#include <vector>

namespace xilinx::AIEX {

struct AIEWrite32SequenceAnalysis {
  mlir::AnalysisManager &analysisManager;
  std::vector<std::vector<NpuWrite32Op>> writeSequences;
  AIEWrite32SequenceAnalysis(mlir::Operation *op, mlir::AnalysisManager &am);
};

}