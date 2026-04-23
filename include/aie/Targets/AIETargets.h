//===- AIETargets.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_TARGETS_AIETARGETS_H
#define AIE_TARGETS_AIETARGETS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace xilinx {
namespace AIE {

/// Metadata describing the byte offset of a notable instruction within the NPU
/// instruction binary (LOAD_PDI or Write32/RTP).
struct NpuInstrOffset {
  enum Kind { LoadPdi, Write32, AddressPatch };
  Kind kind;
  uint32_t offset_bytes;          // byte offset of the instruction start
  // LOAD_PDI fields:
  uint32_t pdi_id = 0;
  uint32_t address_field_offset_bytes = 0;
  uint32_t size_field_offset_bytes = 0;
  // Write32 fields:
  uint32_t value_field_offset_bytes = 0;
  std::string name; // RTP buffer name (empty if not from an RTP write)
  // AddressPatch fields:
  uint32_t arg_idx = 0;
  uint32_t arg_plus = 0;
};

mlir::LogicalResult AIETranslateToXAIEV2(mlir::ModuleOp module,
                                         llvm::raw_ostream &output,
                                         llvm::StringRef deviceName = "");
mlir::LogicalResult AIETranslateToHSA(mlir::ModuleOp module,
                                      llvm::raw_ostream &output,
                                      llvm::StringRef deviceName = "");
mlir::LogicalResult AIEFlowsToJSON(mlir::ModuleOp module,
                                   llvm::raw_ostream &output,
                                   llvm::StringRef deviceName = "");
mlir::LogicalResult ADFGenerateCPPGraph(mlir::ModuleOp module,
                                        llvm::raw_ostream &output);
mlir::LogicalResult AIETranslateSCSimConfig(mlir::ModuleOp module,
                                            llvm::raw_ostream &output,
                                            llvm::StringRef deviceName = "");
mlir::LogicalResult AIETranslateShimSolution(mlir::ModuleOp module,
                                             llvm::raw_ostream &,
                                             llvm::StringRef deviceName = "");
mlir::LogicalResult AIETranslateGraphXPE(mlir::ModuleOp module,
                                         llvm::raw_ostream &, llvm::StringRef);
mlir::LogicalResult AIETranslateNpuToBinary(
    mlir::ModuleOp, std::vector<uint32_t> &, llvm::StringRef deviceName = "",
    llvm::StringRef sequenceName = "",
    std::vector<NpuInstrOffset> *offsets = nullptr);
mlir::LogicalResult AIETranslateToUcDma(mlir::ModuleOp module,
                                        llvm::raw_ostream &output);
mlir::LogicalResult AIETranslateToUcDma(mlir::ModuleOp, std::string &assembly);
mlir::LogicalResult
AIETranslateControlPacketsToUI32Vec(mlir::ModuleOp, std::vector<uint32_t> &,
                                    llvm::StringRef deviceName = "",
                                    llvm::StringRef sequenceName = "");
mlir::LogicalResult AIETranslateToLdScript(mlir::ModuleOp module,
                                           llvm::raw_ostream &output,
                                           int tileCol, int tileRow,
                                           llvm::StringRef deviceName = "");
mlir::LogicalResult AIETranslateToBCF(mlir::ModuleOp module,
                                      llvm::raw_ostream &output, int tileCol,
                                      int tileRow,
                                      llvm::StringRef deviceName = "");
mlir::LogicalResult
AIELLVMLink(llvm::raw_ostream &output, std::vector<std::string> Files,
            bool DisableDITypeMap = false, bool NoVerify = false,
            bool Internalize = false, bool OnlyNeeded = false,
            bool PreserveAssemblyUseListOrder = false, bool Verbose = false);

mlir::LogicalResult AIETranslateToCDODirect(
    mlir::ModuleOp m, llvm::StringRef workDirPath, llvm::StringRef deviceName,
    bool bigEndian = false, bool emitUnified = false, bool cdoDebug = false,
    bool aieSim = false, bool xaieDebug = false, bool enableCores = true);

mlir::LogicalResult AIETranslateToTargetArch(mlir::ModuleOp module,
                                             llvm::raw_ostream &output,
                                             llvm::StringRef deviceName);

} // namespace AIE

namespace aievec {

/// Translates the AIE vector dialect MLIR to C++ code.
mlir::LogicalResult translateAIEVecToCpp(mlir::Operation *op, bool aie2,
                                         mlir::raw_ostream &os);

} // namespace aievec
} // namespace xilinx

#endif
