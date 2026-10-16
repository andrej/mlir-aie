//===- AIELocalizeLocks.cpp ---------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIELOCALIZELOCKS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-localize-locks"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIELocalizeLocksPass
    : xilinx::AIE::impl::AIELocalizeLocksBase<AIELocalizeLocksPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
  void runOnOperation() override {

    DeviceOp deviceOp = getOperation();

    for (auto coreOp : deviceOp.getOps<CoreOp>()) {
      // Collect the locks used in this core.
      const auto &targetModel = getTargetModel(coreOp);

      auto thisTile = cast<TileOp>(coreOp.getTile().getDefiningOp());
      int col = thisTile.colIndex();
      int row = thisTile.rowIndex();

      coreOp.walk([&](UseLockOp useLock) {
        LockOp lock = useLock.getLockOp();
        if (!lock)
          return;
        TileOp tile = lock.getTileOp();
        int dstCol = tile.colIndex();
        int dstRow = tile.rowIndex();
        auto lockIndexOffset =
            targetModel.getLockLocalBaseIndex(col, row, dstCol, dstRow);
        if (!lockIndexOffset)
          llvm_unreachable("Found illegal lock user!");
        int localLockIndex = *lockIndexOffset + lock.getLockIDValue();
        useLock->setAttr("local_lock_id",
                         IntegerAttr::get(IntegerType::get(&getContext(), 32),
                                          localLockIndex));
      });
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIELocalizeLocksPass() {
  return std::make_unique<AIELocalizeLocksPass>();
}