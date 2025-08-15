//===- AIEToConfiguration.h -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_CONVERSION_AIETOCONFIGURATION_AIETOCONFIGURATION_H
#define AIE_CONVERSION_AIETOCONFIGURATION_AIETOCONFIGURATION_H

#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Targets/AIERT.h"

extern "C" {
#include "xaiengine/xaiegbl_defs.h"
// above needs to go first for u32, u64 typedefs
#include "xaiengine/xaie_txn.h"
}

#include <vector>
#include <memory>
#include <cstdint>
#include <optional>

namespace xilinx::AIE {

class DeviceOp;

std::unique_ptr<mlir::OperationPass<xilinx::AIE::DeviceOp>>
createConvertAIEToTransactionPass();

std::unique_ptr<mlir::OperationPass<xilinx::AIE::DeviceOp>>
createConvertAIEToControlPacketsPass();

std::optional<mlir::ModuleOp>
convertTransactionBinaryToMLIR(mlir::MLIRContext *ctx,
                               std::vector<uint8_t> &binary);

// An TransactionBinaryOperation encapulates an aie-rt TnxCmd struct
struct TransactionBinaryOperation {
  struct XAie_TxnCmd cmd;
  TransactionBinaryOperation(XAie_TxnOpcode opc, uint32_t mask, uint64_t addr,
                             uint32_t value, const uint8_t *data,
                             uint32_t size) {
    cmd.Opcode = opc;
    cmd.Mask = mask;
    cmd.RegOff = addr;
    cmd.Value = value;
    cmd.DataPtr = reinterpret_cast<uint64_t>(data);
    cmd.Size = size;
  }
};

std::optional<int>
parseTransactionBinary(const std::vector<uint8_t> &data,
                       std::vector<TransactionBinaryOperation> &ops);

// Translate vector of TransactionBinaryOperation to a sequence of transaction
// ops (npu.write32, npu.maskwrite32, npu.blockwrite).
llvm::LogicalResult
emitTransactionOps(mlir::OpBuilder &builder,
                   std::vector<TransactionBinaryOperation> &operations,
                   std::vector<mlir::memref::GlobalOp> &global_data);

// an enum to represent the output type of the transaction binary
enum OutputType {
  Transaction,
  ControlPacket,
};

llvm::LogicalResult convertTransactionOpsToMLIR(
    mlir::OpBuilder builder, DeviceOp device, OutputType outputType,
    std::vector<TransactionBinaryOperation> &operations,
    xilinx::AIEX::RuntimeSequenceOp seq = nullptr);


} // namespace xilinx::AIE

#endif // AIE_CONVERSION_AIETOCONFIGURATION_AIETOCONFIGURATION_H
