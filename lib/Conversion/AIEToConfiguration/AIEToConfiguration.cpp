//===- AIEToConfiguration.h -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"

#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Targets/AIERT.h"

#include "llvm/Support/Debug.h"
#include <llvm/ADT/APInt.h>

extern "C" {
#include "xaiengine/xaiegbl_defs.h"
// above needs to go first for u32, u64 typedefs
#include "xaiengine/xaie_txn.h"
}

#include <cstring>
#include <optional>
#include <utility>
#include <vector>

namespace xilinx {
#define GEN_PASS_DEF_CONVERTAIETOCONTROLPACKETS
#define GEN_PASS_DEF_CONVERTAIETOTRANSACTION
#include "aie/Conversion/Passes.h.inc"
} // namespace xilinx

#define DEBUG_TYPE "aie-convert-to-config"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

// A TransactionBinaryOperation encapsulates an aie-rt XAie_TxnCmd struct and
// any additional metadata needed for custom operations that do not map cleanly
// onto the core command fields.
struct TransactionBinaryOperation {
  struct XAie_TxnCmd cmd = {};

  struct SyncPayload {
    int32_t column;
    int32_t row;
    int32_t direction;
    int32_t channel;
    int32_t columnCount;
    int32_t rowCount;
  };

  struct LoadPdiPayload {
    uint32_t id;
    uint32_t size;
    uint64_t address;
  };

  struct AddressPatchPayload {
    uint32_t action;
    uint32_t addr;
    int32_t argIdx;
    int32_t argPlus;
  };

  std::optional<SyncPayload> sync;
  std::optional<LoadPdiPayload> loadPdi;
  std::optional<AddressPatchPayload> addressPatch;

  TransactionBinaryOperation() = default;

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

constexpr size_t kTxnHeaderBytes = 16;

struct TxnPreemptHeader {
  uint8_t opcode;
  uint8_t level;
  uint16_t reserved;
};

struct TxnLoadPdiHeader {
  uint8_t opcode;
  uint8_t padding;
  uint16_t id;
  uint32_t size;
  uint64_t address;
};
} // namespace

// Parse a TXN binary blob. On success return the number of columns from the
// header and a vector of parsed operations. On failure return std::nullopt.
static std::optional<int>
parseTransactionBinary(const std::vector<uint8_t> &data,
                       std::vector<TransactionBinaryOperation> &ops) {

  if (data.size() < kTxnHeaderBytes) {
    llvm::errs() << "Transaction binary is too small for header\n";
    return std::nullopt;
  }

  uint32_t major = data[0];
  uint32_t minor = data[1];
  uint32_t num_cols = data[4];

  uint32_t num_ops, txn_size;
  std::memcpy(&num_ops, &data[8], 4);
  std::memcpy(&txn_size, &data[12], 4);

  LLVM_DEBUG(llvm::dbgs() << "Major: " << major << "\n");
  LLVM_DEBUG(llvm::dbgs() << "Minor: " << minor << "\n");
  LLVM_DEBUG(llvm::dbgs() << "DevGen: " << data[2] << "\n");
  LLVM_DEBUG(llvm::dbgs() << "NumRows: " << data[3] << "\n");
  LLVM_DEBUG(llvm::dbgs() << "NumCols: " << num_cols << "\n");
  LLVM_DEBUG(llvm::dbgs() << "NumMemTileRows: " << data[5] << "\n");
  LLVM_DEBUG(llvm::dbgs() << "NumOps: " << num_ops << "\n");
  LLVM_DEBUG(llvm::dbgs() << "TxnSize: " << txn_size << " bytes\n");

  size_t i = kTxnHeaderBytes;

  auto requireBytes = [&](size_t offset, size_t length) -> bool {
    if (offset + length > data.size()) {
      llvm::errs() << "Transaction binary truncated while parsing opcode\n";
      return false;
    }
    return true;
  };

  auto read32 = [&](size_t offset) -> uint32_t {
    uint32_t value;
    std::memcpy(&value, data.data() + offset, sizeof(uint32_t));
    return value;
  };

  // Convert opcode from uint8 to a validated opcode byte
  auto convertOpcode = [](uint8_t opc) -> std::optional<uint8_t> {
    switch (opc) {
    case static_cast<uint8_t>(XAie_TxnOpcode::XAIE_IO_WRITE):
    case static_cast<uint8_t>(XAie_TxnOpcode::XAIE_IO_BLOCKWRITE):
    case static_cast<uint8_t>(XAie_TxnOpcode::XAIE_IO_MASKWRITE):
    case 0x6: // XAie_TxnOpcode::XAIE_IO_PREEMPT
    case 0x8: // XAie_TxnOpcode::XAIE_IO_LOAD_PDI
    case static_cast<uint8_t>(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_TCT):
    case static_cast<uint8_t>(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_DDR_PATCH):
      return opc;
    default:
      llvm::errs() << "Unhandled opcode: " << std::to_string(opc) << "\n";
      return std::nullopt;
    }
  };

  // Parse the binary blob. There are two versions supported, 0.1 and 1.0.
  // For both versions, build a list of TransactionBinaryOperation objects
  // representing the parsed operations.
  if (major == 0 && minor == 1) {
    while (i < data.size()) {
      auto maybeOpcode = convertOpcode(data[i]);
      if (!maybeOpcode)
        return std::nullopt;
      XAie_TxnOpcode opcode = static_cast<XAie_TxnOpcode>(*maybeOpcode);
      LLVM_DEBUG(llvm::dbgs() << "opcode: " + std::to_string(opcode) << "\n");

      TransactionBinaryOperation op;
      op.cmd.Opcode = opcode;

      switch (opcode) {
      case XAie_TxnOpcode::XAIE_IO_WRITE: {
        LLVM_DEBUG(llvm::dbgs() << "opcode: WRITE (0x00)\n");
        if (!requireBytes(i, 24))
          return std::nullopt;
        uint32_t addrLo = read32(i + 8);
        uint32_t addrHi = read32(i + 12);
        uint32_t value = read32(i + 16);
        uint32_t opSize = read32(i + 20);
        if (!requireBytes(i, opSize))
          return std::nullopt;
        uint64_t addr = (static_cast<uint64_t>(addrHi) << 32) | addrLo;
        op.cmd.RegOff = addr;
        op.cmd.Value = value;
        op.cmd.Size = 0;
        i += opSize;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_BLOCKWRITE: {
        LLVM_DEBUG(llvm::dbgs() << "opcode: BLOCKWRITE (0x01)\n");
        if (!requireBytes(i, 16))
          return std::nullopt;
        uint32_t addr = read32(i + 8);
        uint32_t opSize = read32(i + 12);
        if (opSize < 16 || !requireBytes(i, opSize))
          return std::nullopt;
        const uint8_t *payload = data.data() + i + 16;
        uint32_t payloadBytes = opSize - 16;
        op.cmd.RegOff = addr;
        op.cmd.DataPtr = reinterpret_cast<uint64_t>(payload);
        op.cmd.Size = payloadBytes;
        i += opSize;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_MASKWRITE: {
        LLVM_DEBUG(llvm::dbgs() << "opcode: MASKWRITE (0x03)\n");
        if (!requireBytes(i, 28))
          return std::nullopt;
        uint32_t addrLo = read32(i + 8);
        uint32_t addrHi = read32(i + 12);
        uint32_t value = read32(i + 16);
        uint32_t mask = read32(i + 20);
        uint32_t opSize = read32(i + 24);
        if (!requireBytes(i, opSize))
          return std::nullopt;
        uint64_t addr = (static_cast<uint64_t>(addrHi) << 32) | addrLo;
        op.cmd.RegOff = addr;
        op.cmd.Value = value;
        op.cmd.Mask = mask;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_TCT: {
        uint32_t opSize = read32(i + 4);
        if (opSize < 16 || !requireBytes(i, opSize))
          return std::nullopt;
        uint32_t descriptor = read32(i + 8);
        uint32_t config = read32(i + 12);
        TransactionBinaryOperation::SyncPayload payload{
            /*column=*/static_cast<int32_t>((descriptor >> 16) & 0xff),
            /*row=*/static_cast<int32_t>((descriptor >> 8) & 0xff),
            /*direction=*/static_cast<int32_t>(descriptor & 0xff),
            /*channel=*/static_cast<int32_t>((config >> 24) & 0xff),
            /*columnCount=*/static_cast<int32_t>((config >> 16) & 0xff),
            /*rowCount=*/static_cast<int32_t>((config >> 8) & 0xff)};
        op.sync = payload;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      case 0x8: { // XAie_TxnOpcode::XAIE_IO_LOAD_PDI
        LLVM_DEBUG(llvm::dbgs() << "opcode: LOAD_PDI (0x08)\n");
        constexpr size_t opSize = sizeof(TxnLoadPdiHeader);
        if (!requireBytes(i, opSize))
          return std::nullopt;
        TxnLoadPdiHeader header;
        std::memcpy(&header, data.data() + i, opSize);
        TransactionBinaryOperation::LoadPdiPayload payload{
            header.id, header.size, header.address};
        op.loadPdi = payload;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_DDR_PATCH: {
        uint32_t opSize = read32(i + 4);
        if (opSize < 44 || !requireBytes(i, opSize))
          return std::nullopt;
        uint32_t action = read32(i + 20);
        uint32_t addr = read32(i + 24);
        int32_t argIdx = static_cast<int32_t>(read32(i + 32));
        int32_t argPlus = static_cast<int32_t>(read32(i + 40));
        TransactionBinaryOperation::AddressPatchPayload payload{
            action, addr, argIdx, argPlus};
        op.addressPatch = payload;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      case 0x6: { // XAie_TxnOpcode::XAIE_IO_PREEMPT
        LLVM_DEBUG(llvm::dbgs() << "opcode: PREEMPT (0x06)\n");
        constexpr size_t opSize = sizeof(TxnPreemptHeader);
        if (!requireBytes(i, opSize))
          return std::nullopt;
        auto header =
            reinterpret_cast<const TxnPreemptHeader *>(data.data() + i);
        op.cmd.Value = header->level;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      default:
        llvm::errs() << "Unhandled opcode: " << std::to_string(opcode)
                     << " for v0.1 transaction\n";
        return std::nullopt;
      }

      ops.push_back(std::move(op));
    }
  } else if (major == 1 && minor == 0) {
    while (i < data.size()) {
      auto maybeOpcode = convertOpcode(data[i]);
      if (!maybeOpcode)
        return std::nullopt;
      XAie_TxnOpcode opcode = static_cast<XAie_TxnOpcode>(*maybeOpcode);
      LLVM_DEBUG(llvm::dbgs() << "opcode: " + std::to_string(opcode) << "\n");

      TransactionBinaryOperation op;
      op.cmd.Opcode = opcode;

      switch (opcode) {
      case XAie_TxnOpcode::XAIE_IO_WRITE: {
        LLVM_DEBUG(llvm::dbgs() << "opcode: WRITE (0x00)\n");
        if (!requireBytes(i, 12))
          return std::nullopt;
        uint32_t addr = read32(i + 4);
        uint32_t value = read32(i + 8);
        op.cmd.RegOff = addr;
        op.cmd.Value = value;
        op.cmd.Size = 0;
        i += 12;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_BLOCKWRITE: {
        LLVM_DEBUG(llvm::dbgs() << "opcode: BLOCKWRITE (0x01)\n");
        if (!requireBytes(i, 12))
          return std::nullopt;
        uint32_t addr = read32(i + 4);
        uint32_t opSize = read32(i + 8);
        if (opSize < 12 || !requireBytes(i, opSize))
          return std::nullopt;
        const uint8_t *payload = data.data() + i + 12;
        uint32_t payloadBytes = opSize - 12;
        op.cmd.RegOff = addr;
        op.cmd.DataPtr = reinterpret_cast<uint64_t>(payload);
        op.cmd.Size = payloadBytes;
        i += opSize;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_MASKWRITE: {
        LLVM_DEBUG(llvm::dbgs() << "opcode: MASKWRITE (0x03)\n");
        if (!requireBytes(i, 16))
          return std::nullopt;
        uint32_t addr = read32(i + 4);
        uint32_t value = read32(i + 8);
        uint32_t mask = read32(i + 12);
        op.cmd.RegOff = addr;
        op.cmd.Value = value;
        op.cmd.Mask = mask;
        op.cmd.Size = 0;
        i += 16;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_TCT: {
        uint32_t opSize = read32(i + 4);
        if (opSize < 16 || !requireBytes(i, opSize))
          return std::nullopt;
        uint32_t descriptor = read32(i + 8);
        uint32_t config = read32(i + 12);
        TransactionBinaryOperation::SyncPayload payload{
            /*column=*/static_cast<int32_t>((descriptor >> 16) & 0xff),
            /*row=*/static_cast<int32_t>((descriptor >> 8) & 0xff),
            /*direction=*/static_cast<int32_t>(descriptor & 0xff),
            /*channel=*/static_cast<int32_t>((config >> 24) & 0xff),
            /*columnCount=*/static_cast<int32_t>((config >> 16) & 0xff),
            /*rowCount=*/static_cast<int32_t>((config >> 8) & 0xff)};
        op.sync = payload;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      case 0x8: { // XAie_TxnOpcode::XAIE_IO_LOAD_PDI
        LLVM_DEBUG(llvm::dbgs() << "opcode: LOAD_PDI (0x08)\n");
        constexpr size_t opSize = sizeof(TxnLoadPdiHeader);
        if (!requireBytes(i, opSize))
          return std::nullopt;
        TxnLoadPdiHeader header;
        std::memcpy(&header, data.data() + i, opSize);
        TransactionBinaryOperation::LoadPdiPayload payload{
            header.id, header.size, header.address};
        op.loadPdi = payload;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_DDR_PATCH: {
        uint32_t opSize = read32(i + 4);
        if (opSize < 44 || !requireBytes(i, opSize))
          return std::nullopt;
        uint32_t action = read32(i + 20);
        uint32_t addr = read32(i + 24);
        int32_t argIdx = static_cast<int32_t>(read32(i + 32));
        int32_t argPlus = static_cast<int32_t>(read32(i + 40));
        TransactionBinaryOperation::AddressPatchPayload payload{
            action, addr, argIdx, argPlus};
        op.addressPatch = payload;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      case 0x6: { // XAie_TxnOpcode::XAIE_IO_PREEMPT
        LLVM_DEBUG(llvm::dbgs() << "opcode: PREEMPT (0x06)\n");
        constexpr size_t opSize = sizeof(TxnPreemptHeader);
        if (!requireBytes(i, opSize))
          return std::nullopt;
        auto header =
            reinterpret_cast<const TxnPreemptHeader *>(data.data() + i);
        op.cmd.Value = header->level;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      default:
        llvm::errs() << "Unhandled opcode: " << std::to_string(opcode)
                     << " for v1.0 transaction\n";
        return std::nullopt;
      }

      ops.push_back(std::move(op));
    }
  } else {
    llvm::errs() << "Unsupported TXN binary version: " << major << "." << minor
                 << "\n";
    return std::nullopt;
  }

  return num_cols;
}

static LogicalResult generateTransactions(AIERTControl &ctl,
                                          const StringRef workDirPath,
                                          DeviceOp &targetOp, bool aieSim,
                                          bool enableElfs, bool enableInit,
                                          bool enableCores) {
  if (enableElfs && !targetOp.getOps<CoreOp>().empty() &&
      failed(ctl.addAieElfs(targetOp, workDirPath, aieSim)))
    return failure();
  if (enableInit && failed(ctl.addInitConfig(targetOp)))
    return failure();
  if (enableCores && !targetOp.getOps<CoreOp>().empty() &&
      failed(ctl.addCoreEnable(targetOp)))
    return failure();
  return success();
}

// DMA Control register base offsets (relative to tile base)
// For shim tiles (row 0):
static constexpr uint32_t kDMA_Control_Base_Shim = 0x1D200;
// For core tiles (row >= 1):
static constexpr uint32_t kDMA_Control_Base_Core = 0x1DE00;
// Queue register is at control + 0x4

// Lock value register base (runtime lock set operations)
static constexpr uint32_t kLock_Value_Base = 0x1F000;
static constexpr uint32_t kLock_Stride = 0x10; // 16 bytes per lock

// Try to parse a write32 operation as a runtime lock set operation.
// Returns true if this is a lock set, false otherwise.
static bool tryParseLockSet(uint32_t address, uint32_t value,
                            int32_t &column, int32_t &row,
                            int32_t &lock_id, int32_t &lock_value) {
  // Extract column and row from address
  column = (address >> 25) & 0xFF;
  row = (address >> 20) & 0x1F;

  // Get the offset within the tile
  uint32_t tile_base = (column << 25) | (row << 20);
  uint32_t offset = address - tile_base;

  // Check if this is in the lock value register range
  if (offset >= kLock_Value_Base && offset < (kLock_Value_Base + 0x1000)) {
    uint32_t lock_offset = offset - kLock_Value_Base;

    // Lock ID is determined by offset / stride
    lock_id = lock_offset / kLock_Stride;

    // Only recognize writes to the first register of each lock (offset % stride == 0)
    if (lock_offset % kLock_Stride == 0) {
      lock_value = static_cast<int32_t>(value);
      return true;
    }
  }

  return false;
}

// Try to parse a write32 operation as an RTP (Runtime Parameter) write.
// Returns true if this is an RTP write, false otherwise.
// RTP buffers are typically in the tile's local memory space.
// Known RTP offsets observed: 0x9400, 0xCA00, 0xCA04
static bool tryParseRtpWrite(uint32_t address, uint32_t value,
                             int32_t &column, int32_t &row,
                             uint32_t &offset, int32_t &rtp_value) {
  // Extract column and row from address
  column = (address >> 25) & 0xFF;
  row = (address >> 20) & 0x1F;

  // Get the offset within the tile
  uint32_t tile_base = (column << 25) | (row << 20);
  offset = address - tile_base;

  // RTP buffers are typically in data memory region (0x8000-0xFFFF range)
  // Known offsets: 0x9400, 0xCA00, 0xCA04
  if (offset >= 0x8000 && offset < 0x10000) {
    // This could be an RTP write
    // For now, we identify specific known offsets
    if (offset == 0x9400 || offset == 0xCA00 || offset == 0xCA04) {
      rtp_value = static_cast<int32_t>(value);
      return true;
    }
  }

  return false;
}

// Try to parse a write32/maskwrite32 operation as a DMA channel enable/control operation.
// Returns true if this is a DMA channel control write, false otherwise.
// These are writes to the DMA channel control registers (offset 0x1DE00/0x1DE08/etc.)
// that enable/disable DMA channels or configure their control bits.
static bool tryParseDMAChannelControl(uint32_t address, uint32_t mask, uint32_t value,
                                       int32_t &column, int32_t &row,
                                       int32_t &direction, int32_t &channel) {
  // Extract column and row from address
  column = (address >> 25) & 0xFF;
  row = (address >> 20) & 0x1F;

  // Get the offset within the tile
  uint32_t tile_base = (column << 25) | (row << 20);
  uint32_t offset = address - tile_base;

  // Determine the control base based on tile type
  uint32_t control_base;
  if (row == 0) {
    // Shim tile
    control_base = kDMA_Control_Base_Shim;
  } else {
    // Core or mem tile
    control_base = kDMA_Control_Base_Core;
  }

  // DMA control registers:
  // control_base + (channel * 8) + (MM2S ? 0x10 : 0)
  // Try to match S2MM channels
  if (offset == control_base + 0 * 8) {
    direction = 0; // S2MM
    channel = 0;
  } else if (offset == control_base + 1 * 8) {
    direction = 0; // S2MM
    channel = 1;
  } else if (offset == control_base + 0x10 + 0 * 8) {
    direction = 1; // MM2S
    channel = 0;
  } else if (offset == control_base + 0x10 + 1 * 8) {
    direction = 1; // MM2S
    channel = 1;
  } else {
    return false;
  }

  return true;
}

// Try to parse a maskwrite32 operation as a DMA queue operation (start/repeat queue).
// Returns true if this is a queue write, false otherwise.
// Queue registers are at control_base + (channel * 8) + (MM2S ? 0x10 : 0) + 0x4
static bool tryParseDMAQueueWrite(uint32_t address, uint32_t mask, uint32_t value,
                                   int32_t &column, int32_t &row,
                                   int32_t &direction, int32_t &channel,
                                   bool &is_repeat_queue) {
  // Extract column and row from address
  column = (address >> 25) & 0xFF;
  row = (address >> 20) & 0x1F;

  // Get the offset within the tile
  uint32_t tile_base = (column << 25) | (row << 20);
  uint32_t offset = address - tile_base;

  // Determine the control base based on tile type
  uint32_t control_base;
  if (row == 0) {
    // Shim tile
    control_base = kDMA_Control_Base_Shim;
  } else {
    // Core or mem tile
    control_base = kDMA_Control_Base_Core;
  }

  // Queue registers (same pattern as tryParseQueuePush):
  // Queue: control_base + (channel * 8) + (MM2S ? 0x10 : 0) + 0x4
  // Note: We currently don't distinguish start vs repeat queue in maskwrite operations
  // Both use the same register address with different value encodings

  // Try to match S2MM queues
  if (offset == control_base + 0 * 8 + 0x4) {
    direction = 0; // S2MM
    channel = 0;
    is_repeat_queue = false; // Assume start queue
  } else if (offset == control_base + 1 * 8 + 0x4) {
    direction = 0; // S2MM
    channel = 1;
    is_repeat_queue = false;
  } else if (offset == control_base + 0x10 + 0 * 8 + 0x4) {
    direction = 1; // MM2S
    channel = 0;
    is_repeat_queue = false;
  } else if (offset == control_base + 0x10 + 1 * 8 + 0x4) {
    direction = 1; // MM2S
    channel = 1;
    is_repeat_queue = false;
  } else {
    return false;
  }

  return true;
}

// Try to parse a write32 operation as a DMA reset operation.
// Returns true if this is a DMA reset write, false otherwise.
static bool tryParseDMAReset(uint32_t address, uint32_t value,
                              int32_t &column, int32_t &row) {
  // Extract column and row from address
  column = (address >> 25) & 0xFF;
  row = (address >> 20) & 0x1F;

  // Get the offset within the tile
  uint32_t tile_base = (column << 25) | (row << 20);
  uint32_t offset = address - tile_base;

  // DMA reset register is at 0x20000
  if (offset == 0x20000) {
    return true;
  }

  return false;
}

// Try to parse a maskwrite32 operation as a core control operation (enable/disable).
// Returns true if this is a core control write, false otherwise.
static bool tryParseCoreControl(uint32_t address, uint32_t mask, uint32_t value,
                                 int32_t &column, int32_t &row) {
  // Extract column and row from address
  column = (address >> 25) & 0xFF;
  row = (address >> 20) & 0x1F;

  // Get the offset within the tile
  uint32_t tile_base = (column << 25) | (row << 20);
  uint32_t offset = address - tile_base;

  // Core control register is at 0x32000
  if (offset == 0x32000) {
    return true;
  }

  return false;
}

// Try to parse a maskwrite32 operation as a DMA Controller_ID configuration.
// Returns true if this is a Controller_ID write, false otherwise.
// The Controller_ID field (bits 15:8) of DMA_S2MM/MM2S control registers
// is used for task-complete-token configuration.
static bool tryParseDMAControllerID(uint32_t address, uint32_t mask, uint32_t value,
                                    int32_t &column, int32_t &row,
                                    int32_t &direction, int32_t &channel,
                                    int32_t &controller_id) {
  // Extract column and row from address
  column = (address >> 25) & 0xFF;
  row = (address >> 20) & 0x1F;

  // Get the offset within the tile
  uint32_t tile_base = (column << 25) | (row << 20);
  uint32_t offset = address - tile_base;

  // Determine the control base based on tile type
  uint32_t control_base;
  if (row == 0) {
    // Shim tile
    control_base = kDMA_Control_Base_Shim;
  } else {
    // Core or mem tile
    control_base = kDMA_Control_Base_Core;
  }

  // DMA control registers:
  // control_base + (channel * 8) + (MM2S ? 0x10 : 0)
  // Controller_ID field is at bits 15:8

  // Check if mask includes Controller_ID bits (bits 15:8 = 0xFF00)
  if ((mask & 0xFF00) == 0) {
    return false; // Not touching Controller_ID field
  }

  // Try to match S2MM channels
  if (offset == control_base + 0 * 8) {
    direction = 0; // S2MM
    channel = 0;
  } else if (offset == control_base + 1 * 8) {
    direction = 0; // S2MM
    channel = 1;
  } else if (offset == control_base + 0x10 + 0 * 8) {
    direction = 1; // MM2S
    channel = 0;
  } else if (offset == control_base + 0x10 + 1 * 8) {
    direction = 1; // MM2S
    channel = 1;
  } else {
    return false;
  }

  // Extract Controller_ID from value (bits 15:8)
  controller_id = (value >> 8) & 0xFF;

  return true;
}

// Try to parse a write32 operation as a queue push operation.
// Returns true if this is a queue push, false otherwise.
// If true, populates the output parameters.
static bool tryParseQueuePush(uint32_t address, uint32_t value,
                              int32_t &column, int32_t &row,
                              int32_t &direction, int32_t &channel,
                              bool &issue_token, int32_t &repeat_count,
                              int32_t &bd_id) {
  // Extract column from tile base address
  column = (address >> 25) & 0xFF;

  // Extract row from tile base address
  row = (address >> 20) & 0x1F;

  // Get the offset within the tile
  uint32_t tile_base = (column << 25) | (row << 20);
  uint32_t offset = address - tile_base;

  // Determine the control base based on tile type
  uint32_t control_base;
  if (row == 0) {
    // Shim tile
    control_base = kDMA_Control_Base_Shim;
  } else {
    // Core or mem tile
    control_base = kDMA_Control_Base_Core;
  }

  // Queue register is control + 0x4
  // Check if this offset matches a queue register pattern:
  // control_base + (channel * 8) + (MM2S ? 0x10 : 0) + 0x4

  // Try to match S2MM channels
  if (offset == control_base + 0 * 8 + 0x4) {
    direction = 0; // S2MM
    channel = 0;
  } else if (offset == control_base + 1 * 8 + 0x4) {
    direction = 0; // S2MM
    channel = 1;
  } else if (offset == control_base + 0x10 + 0 * 8 + 0x4) {
    direction = 1; // MM2S
    channel = 0;
  } else if (offset == control_base + 0x10 + 1 * 8 + 0x4) {
    direction = 1; // MM2S
    channel = 1;
  } else {
    return false;
  }

  // Decode the value
  bd_id = value & 0xF;
  repeat_count = (value >> 16) & 0xFF;
  issue_token = (value & 0x80000000) != 0;

  return true;
}

// Try to parse a blockwrite operation as a BD configuration.
// Returns true if this is a BD blockwrite, false otherwise.
static bool tryParseBDBlockwrite(uint32_t address, const uint32_t *data,
                                 uint32_t size_in_words,
                                 int32_t &column, int32_t &row,
                                 int32_t &bd_id,
                                 std::array<int32_t, 32> &bd_fields) {
  // Extract column and row from address
  column = (address >> 25) & 0xFF;
  row = (address >> 20) & 0x1F;

  // Check if this is a shim tile (row 0) or mem tile (row 1)
  if (row != 0 && row != 1)
    return false;

  uint32_t tile_base = (column << 25) | (row << 20);
  uint32_t offset = address - tile_base;

  // Check if this is a BD base address
  // Shim tile DMA BD base: 0x1D000, mem tile: 0xA0000
  uint32_t bd_base = (row == 0) ? 0x1D000 : 0xA0000;

  // Each BD is 0x20 (32) bytes = 8 words
  if ((offset < bd_base) || ((offset - bd_base) % 0x20 != 0))
    return false;

  bd_id = (offset - bd_base) / 0x20;

  // We need at least 6-8 words for a BD
  if (size_in_words < 6)
    return false;

  // Copy the BD data (up to 8 words)
  bd_fields.fill(0);
  for (uint32_t i = 0; i < std::min(size_in_words, 8u); i++) {
    bd_fields[i] = data[i];
  }

  return true;
}

// Check if a blockwrite is a zero-value shim DMA BD initialization.
// These writes are used to disable/initialize unused BDs with zero values.
// Returns true if this should be suppressed (not emitted as raw blockwrite).
static bool isZeroValueShimDMABDInit(uint32_t address, const uint32_t *data,
                                     uint32_t size_in_words) {
  // Extract column and row from address
  int32_t row = (address >> 20) & 0x1F;

  // Only applies to shim tiles (row 0)
  if (row != 0)
    return false;

  uint32_t tile_base = ((address >> 25) & 0xFF) << 25;
  uint32_t offset = address - tile_base;

  // Check if this is in the shim DMA BD address range
  // Shim tile DMA BD base: 0x1D000, each BD is 0x20 bytes
  uint32_t bd_base = 0x1D000;
  uint32_t bd_end = bd_base + (16 * 0x20); // Assuming max 16 BDs

  if (offset < bd_base || offset >= bd_end)
    return false;

  // Check if offset is aligned to BD boundary
  if ((offset - bd_base) % 0x20 != 0)
    return false;

  // Check if all data values are zero (initialization)
  for (uint32_t i = 0; i < size_in_words; i++) {
    if (data[i] != 0)
      return false;
  }

  return true;
}

// Parse BD fields from the raw register values
static void parseBDFields(const std::array<int32_t, 32> &bd_data,
                          int32_t &buffer_length, int32_t &buffer_offset,
                          int32_t &enable_packet, int32_t &out_of_order_id,
                          int32_t &packet_id, int32_t &packet_type,
                          int32_t &d0_size, int32_t &d0_stride,
                          int32_t &d1_size, int32_t &d1_stride,
                          int32_t &d2_size, int32_t &d2_stride,
                          int32_t &iteration_current, int32_t &iteration_size,
                          int32_t &iteration_stride, int32_t &next_bd,
                          int32_t &use_next_bd, int32_t &valid_bd,
                          int32_t &lock_rel_val, int32_t &lock_rel_id,
                          int32_t &lock_acq_enable, int32_t &lock_acq_val,
                          int32_t &lock_acq_id, int32_t &burst_length) {
  // BD word 0: buffer_length
  buffer_length = bd_data[0];

  // BD word 1: buffer_offset (will be patched by address_patch)
  buffer_offset = bd_data[1];

  // BD word 2: packet info
  uint32_t word2 = bd_data[2];
  enable_packet = (word2 >> 30) & 0x1;
  out_of_order_id = (word2 >> 24) & 0x3F;
  packet_id = (word2 >> 16) & 0x1F;
  packet_type = word2 & 0x7;

  // BD word 3: d0_size and d0_stride
  uint32_t word3 = bd_data[3];
  d0_size = (word3 >> 20) & 0x3FF;  // Fixed: should be 10 bits, not 12
  d0_stride = word3 & 0xFFFFF;

  // BD word 4: burst_length, d1_size, d1_stride
  uint32_t word4 = bd_data[4];
  // Burst length is encoded in bits [31:30]:
  // 00 = 64 bytes, 01 = 128 bytes, 10 = 256 bytes
  uint32_t burst_encoding = (word4 >> 30) & 0x3;
  const int burst_lengths[] = {64, 128, 256, 0};  // 0 for undefined encoding (11)
  burst_length = burst_lengths[burst_encoding];
  d1_size = (word4 >> 20) & 0x3FF;  // Fixed: should be 10 bits, not 8
  d1_stride = word4 & 0xFFFFF;

  // BD word 5: d2_stride (and AXCache)
  uint32_t word5 = bd_data[5];
  d2_stride = word5 & 0xFFFFF;

  // BD word 6: iteration info
  uint32_t word6 = bd_data[6];
  iteration_current = (word6 >> 26) & 0x3F;  // Fixed: should be bits [31:26]
  iteration_size = (word6 >> 20) & 0x3F;     // Fixed: should be bits [25:20]
  iteration_stride = word6 & 0xFFFFF;        // Fixed: should be 20 bits, not 6

  // BD word 7: control and lock info
  uint32_t word7 = bd_data[7];
  next_bd = (word7 >> 27) & 0x1F;
  use_next_bd = (word7 >> 26) & 0x1;
  valid_bd = (word7 >> 25) & 0x1;
  lock_rel_val = (word7 >> 18) & 0x7F;
  lock_rel_id = (word7 >> 13) & 0x1F;
  lock_acq_enable = (word7 >> 12) & 0x1;
  lock_acq_val = (word7 >> 5) & 0x7F;
  lock_acq_id = word7 & 0x1F;

  // Default values for fields not in basic BD
  d2_size = 0;
}

// Translate vector of TransactionBinaryOperation to a sequence of transaction
// ops (npu.write32, npu.maskwrite32, npu.blockwrite).
static LogicalResult
emitTransactionOps(OpBuilder &builder,
                   std::vector<TransactionBinaryOperation> &operations,
                   std::vector<memref::GlobalOp> &global_data) {

  auto loc = builder.getUnknownLoc();
  auto ctx = builder.getContext();

  // create the txn ops
  for (auto [op, payload] : llvm::zip(operations, global_data)) {
    llvm::errs() << "DEBUG: Processing opcode " << (int)op.cmd.Opcode << " at address 0x" << llvm::utohexstr(op.cmd.RegOff) << "\n";

    if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_WRITE) {
      // Try to lift write32 to queue push
      int32_t column, row, direction, channel, repeat_count, bd_id;
      bool issue_token;

      if (tryParseQueuePush(op.cmd.RegOff, op.cmd.Value,
                           column, row, direction, channel,
                           issue_token, repeat_count, bd_id)) {
        // Emit NpuPushQueueOp
        auto dirAttr = AIE::DMAChannelDirAttr::get(
            ctx, direction == 0 ? AIE::DMAChannelDir::S2MM : AIE::DMAChannelDir::MM2S);
        AIEX::NpuPushQueueOp::create(
            builder, loc,
            builder.getI32IntegerAttr(column),
            builder.getI32IntegerAttr(row),
            dirAttr,
            builder.getI32IntegerAttr(channel),
            builder.getBoolAttr(issue_token),
            builder.getI32IntegerAttr(repeat_count),
            builder.getI32IntegerAttr(bd_id));
      } else {
        // Try to parse as a lock set operation
        int32_t lock_id, lock_value;
        if (tryParseLockSet(op.cmd.RegOff, op.cmd.Value,
                           column, row, lock_id, lock_value)) {
          // This is a runtime lock set operation
          // Since we can't emit aiex.set_lock without SSA lock values in the decompiler,
          // emit a descriptive comment operation or suppress the write
          // For now, we suppress it (don't emit anything) as these are runtime control operations
          // that will be reconstructed during recompilation
          llvm::errs() << "Note: Suppressing runtime lock set operation at tile("
                       << column << "," << row << ") lock " << lock_id
                       << " value " << lock_value << "\n";
          // Skip emission - don't emit raw write32
        } else {
          // Try to parse as an RTP write operation
          uint32_t offset;
          int32_t rtp_value;
          if (tryParseRtpWrite(op.cmd.RegOff, op.cmd.Value,
                              column, row, offset, rtp_value)) {
            // This is an RTP (Runtime Parameter) write
            // Since we can't emit aiex.npu.rtp_write without buffer symbols in the decompiler,
            // we suppress it here and document it
            llvm::errs() << "Note: Suppressing RTP write at tile(" << column << "," << row
                         << ") offset 0x" << llvm::utohexstr(offset)
                         << " value " << rtp_value << "\n";
            // Skip emission - don't emit raw write32
          } else {
            // Try to parse as a DMA reset operation
            if (tryParseDMAReset(op.cmd.RegOff, op.cmd.Value, column, row)) {
              // This is a DMA reset operation
              // Suppress it as it's a runtime configuration that will be reconstructed
              llvm::errs() << "Note: Suppressing DMA reset at tile(" << column << "," << row << ")\n";
              // Skip emission - don't emit raw write32
            } else {
              // Emit raw write32 for unrecognized operations
              AIEX::NpuWrite32Op::create(builder, loc, op.cmd.RegOff, op.cmd.Value,
                                         nullptr, nullptr, nullptr);
            }
          }
        }
      }
    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_BLOCKWRITE) {
      // Try to lift blockwrite to BD configuration
      int32_t column, row, bd_id;
      std::array<int32_t, 32> bd_fields;

      const uint32_t *data = reinterpret_cast<const uint32_t *>(op.cmd.DataPtr);
      uint32_t size_in_words = op.cmd.Size / 4;

      if (tryParseBDBlockwrite(op.cmd.RegOff, data, size_in_words,
                              column, row, bd_id, bd_fields)) {
        // Parse the BD fields
        int32_t buffer_length, buffer_offset, enable_packet, out_of_order_id;
        int32_t packet_id, packet_type, d0_size, d0_stride;
        int32_t d1_size, d1_stride, d2_size, d2_stride;
        int32_t iteration_current, iteration_size, iteration_stride;
        int32_t next_bd, use_next_bd, valid_bd;
        int32_t lock_rel_val, lock_rel_id, lock_acq_enable;
        int32_t lock_acq_val, lock_acq_id, burst_length;

        parseBDFields(bd_fields, buffer_length, buffer_offset,
                     enable_packet, out_of_order_id, packet_id, packet_type,
                     d0_size, d0_stride, d1_size, d1_stride, d2_size, d2_stride,
                     iteration_current, iteration_size, iteration_stride,
                     next_bd, use_next_bd, valid_bd,
                     lock_rel_val, lock_rel_id, lock_acq_enable,
                     lock_acq_val, lock_acq_id, burst_length);

        // Emit NpuWriteBdOp with all fields (matching the signature)
        // Setting zero padding fields to 0
        int32_t d0_zero_before = 0, d1_zero_before = 0, d2_zero_before = 0;
        int32_t d0_zero_after = 0, d1_zero_after = 0, d2_zero_after = 0;

        AIEX::NpuWriteBdOp::create(
            builder, loc,
            builder.getI32IntegerAttr(column),
            builder.getI32IntegerAttr(bd_id),
            builder.getI32IntegerAttr(buffer_length),
            builder.getI32IntegerAttr(buffer_offset),
            builder.getI32IntegerAttr(enable_packet),
            builder.getI32IntegerAttr(out_of_order_id),
            builder.getI32IntegerAttr(packet_id),
            builder.getI32IntegerAttr(packet_type),
            builder.getI32IntegerAttr(d0_size),
            builder.getI32IntegerAttr(d0_stride),
            builder.getI32IntegerAttr(d1_size),
            builder.getI32IntegerAttr(d1_stride),
            builder.getI32IntegerAttr(d2_size),
            builder.getI32IntegerAttr(d2_stride),
            builder.getI32IntegerAttr(iteration_current),
            builder.getI32IntegerAttr(iteration_size),
            builder.getI32IntegerAttr(iteration_stride),
            builder.getI32IntegerAttr(next_bd),
            builder.getI32IntegerAttr(row),
            builder.getI32IntegerAttr(use_next_bd),
            builder.getI32IntegerAttr(valid_bd),
            builder.getI32IntegerAttr(lock_rel_val),
            builder.getI32IntegerAttr(lock_rel_id),
            builder.getI32IntegerAttr(lock_acq_enable),
            builder.getI32IntegerAttr(lock_acq_val),
            builder.getI32IntegerAttr(lock_acq_id),
            builder.getI32IntegerAttr(d0_zero_before),
            builder.getI32IntegerAttr(d1_zero_before),
            builder.getI32IntegerAttr(d2_zero_before),
            builder.getI32IntegerAttr(d0_zero_after),
            builder.getI32IntegerAttr(d1_zero_after),
            builder.getI32IntegerAttr(d2_zero_after),
            builder.getI32IntegerAttr(burst_length));
      } else {
        // Check if this is a zero-value shim DMA BD initialization
        const uint32_t *data = reinterpret_cast<const uint32_t *>(op.cmd.DataPtr);
        uint32_t size_in_words = op.cmd.Size / 4;

        if (isZeroValueShimDMABDInit(op.cmd.RegOff, data, size_in_words)) {
          // Suppress zero-value shim DMA BD initialization writes
          int32_t column = (op.cmd.RegOff >> 25) & 0xFF;
          int32_t row = (op.cmd.RegOff >> 20) & 0x1F;
          uint32_t tile_base = (column << 25) | (row << 20);
          uint32_t offset = op.cmd.RegOff - tile_base;
          int32_t bd_id = (offset - 0x1D000) / 0x20;

          llvm::errs() << "Note: Suppressing shim DMA BD zero-value initialization at tile("
                       << column << "," << row << ") BD " << bd_id << "\n";
          // Skip emission - don't emit raw blockwrite
        } else {
          // Emit raw blockwrite for non-zero or non-BD writes
          auto memref = memref::GetGlobalOp::create(builder, loc, payload.getType(),
                                                    payload.getName());
          AIEX::NpuBlockWriteOp::create(
              builder, loc, builder.getUI32IntegerAttr(op.cmd.RegOff),
              memref.getResult(), nullptr, nullptr, nullptr);
        }
      }
    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_MASKWRITE) {
      llvm::errs() << "DEBUG: Processing maskwrite32 at address 0x" << llvm::utohexstr(op.cmd.RegOff) << "\n";
      // Try to parse as a lock operation first (locks use maskwrite for acquire/release)
      int32_t column, row, lock_id, lock_value;
      if (tryParseLockSet(op.cmd.RegOff, op.cmd.Value, column, row, lock_id, lock_value)) {
        // This is a runtime lock operation
        // Suppress it as it will be reconstructed during recompilation
        llvm::errs() << "Note: Suppressing runtime lock operation (maskwrite) at tile("
                     << column << "," << row << ") lock " << lock_id
                     << " value " << lock_value << "\n";
        // Skip emission - don't emit raw maskwrite32
      } else {
        // Try to parse as a DMA Controller_ID configuration
        int32_t direction, channel, controller_id;
        if (tryParseDMAControllerID(op.cmd.RegOff, op.cmd.Mask, op.cmd.Value,
                                     column, row, direction, channel, controller_id)) {
          // This is a DMA Controller_ID configuration (task-complete-token)
          // Since this is a runtime configuration that will be reconstructed during recompilation,
          // we suppress it here and document it
          llvm::errs() << "Note: Suppressing DMA Controller_ID write at tile(" << column << "," << row
                       << ") " << (direction == 0 ? "S2MM" : "MM2S") << " channel " << channel
                       << " controller_id " << controller_id << "\n";
          // Skip emission - don't emit raw maskwrite32
        } else {
          // Try to parse as a DMA queue write operation
          bool is_repeat_queue;
          if (tryParseDMAQueueWrite(op.cmd.RegOff, op.cmd.Mask, op.cmd.Value,
                                     column, row, direction, channel, is_repeat_queue)) {
            // This is a DMA queue write (start or repeat queue)
            // Suppress it as it will be reconstructed during recompilation
            llvm::errs() << "Note: Suppressing DMA " << (is_repeat_queue ? "repeat" : "start")
                         << " queue write at tile(" << column << "," << row
                         << ") " << (direction == 0 ? "S2MM" : "MM2S") << " channel " << channel << "\n";
            // Skip emission - don't emit raw maskwrite32
          } else {
            // Try to parse as a DMA channel control operation
            if (tryParseDMAChannelControl(op.cmd.RegOff, op.cmd.Mask, op.cmd.Value,
                                           column, row, direction, channel)) {
              // This is a DMA channel control operation (enable/disable)
              // Suppress it as it will be reconstructed during recompilation
              llvm::errs() << "Note: Suppressing DMA channel control write at tile(" << column << "," << row
                           << ") " << (direction == 0 ? "S2MM" : "MM2S") << " channel " << channel << "\n";
              // Skip emission - don't emit raw maskwrite32
            } else {
              // Try to parse as a core control operation
              if (tryParseCoreControl(op.cmd.RegOff, op.cmd.Mask, op.cmd.Value, column, row)) {
                // This is a core control operation (enable/disable)
                // Suppress it as it will be reconstructed during recompilation
                llvm::errs() << "Note: Suppressing core control write at tile(" << column << "," << row << ")\n";
                // Skip emission - don't emit raw maskwrite32
              } else {
                // Emit raw maskwrite32 for unrecognized operations
                AIEX::NpuMaskWrite32Op::create(builder, loc, op.cmd.RegOff, op.cmd.Value,
                                               op.cmd.Mask, nullptr, nullptr, nullptr);
              }
            }
          }
        }
      }
    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_TCT) {
      if (!op.sync) {
        llvm::errs() << "Missing sync payload while emitting transaction\n";
        return failure();
      }
      const TransactionBinaryOperation::SyncPayload &sync = *op.sync;
      AIEX::NpuSyncOp::create(builder, loc,
                              builder.getI32IntegerAttr(sync.column),
                              builder.getI32IntegerAttr(sync.row),
                              builder.getI32IntegerAttr(sync.direction),
                              builder.getI32IntegerAttr(sync.channel),
                              builder.getI32IntegerAttr(sync.columnCount),
                              builder.getI32IntegerAttr(sync.rowCount));
    } else if (op.cmd.Opcode == 0x8 /* XAie_TxnOpcode::XAIE_IO_LOAD_PDI */) {
      if (!op.loadPdi) {
        llvm::errs() << "Missing load_pdi payload while emitting transaction\n";
        return failure();
      }
      const TransactionBinaryOperation::LoadPdiPayload &payloadInfo =
          *op.loadPdi;
      auto idAttr =
          builder.getI32IntegerAttr(static_cast<int32_t>(payloadInfo.id));
      IntegerAttr sizeAttr =
          builder.getI32IntegerAttr(static_cast<int32_t>(payloadInfo.size));

      auto ui64Ty =
          IntegerType::get(builder.getContext(), 64, IntegerType::Unsigned);
      IntegerAttr addressAttr =
          IntegerAttr::get(ui64Ty, llvm::APInt(64, payloadInfo.address));

      AIEX::NpuLoadPdiOp::create(builder, loc, nullptr, idAttr, sizeAttr,
                                 addressAttr);
    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_DDR_PATCH) {
      if (!op.addressPatch) {
        llvm::errs()
            << "Missing address_patch payload while emitting transaction\n";
        return failure();
      }
      const TransactionBinaryOperation::AddressPatchPayload &patch =
          *op.addressPatch;
      AIEX::NpuAddressPatchOp::create(builder, loc,
                                      builder.getUI32IntegerAttr(patch.addr),
                                      builder.getI32IntegerAttr(patch.argIdx),
                                      builder.getI32IntegerAttr(patch.argPlus));
    } else if (op.cmd.Opcode == 0x6 /*  XAie_TxnOpcode::XAIE_IO_PREEMPT */) {
      auto ui8Ty =
          IntegerType::get(builder.getContext(), 8, IntegerType::Unsigned);
      auto levelAttr = IntegerAttr::get(ui8Ty, llvm::APInt(8, op.cmd.Value));
      AIEX::NpuPreemptOp::create(builder, loc, levelAttr);
    } else {
      llvm::errs() << "Unhandled txn opcode: " << op.cmd.Opcode << "\n";
      return failure();
    }
  }
  return success();
}

// Translate vector of TransactionBinaryOperation to a sequence of control
// packet ops.
static LogicalResult
emitControlPacketOps(OpBuilder &builder,
                     std::vector<TransactionBinaryOperation> &operations,
                     std::vector<memref::GlobalOp> &global_data) {

  auto loc = builder.getUnknownLoc();
  auto ctx = builder.getContext();

  // create the control packet ops
  for (auto [op, payload] : llvm::zip(operations, global_data)) {

    if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_WRITE) {
      AIEX::NpuControlPacketOp::create(
          builder, loc, builder.getUI32IntegerAttr(op.cmd.RegOff), nullptr,
          /*opcode*/ builder.getI32IntegerAttr(0),
          /*stream_id*/ builder.getI32IntegerAttr(0),
          DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>(op.cmd.Value)));
    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_BLOCKWRITE) {
      if (!payload.getInitialValue())
        continue;
      auto blockWriteData =
          dyn_cast<DenseIntElementsAttr>(*payload.getInitialValue());
      if (!blockWriteData) {
        payload.emitError(
            "Global symbol initial value is not a dense int array");
        break;
      }
      auto blockWriteDataValues = blockWriteData.getValues<int32_t>();
      // Split block write data into beats of 4 or less, in int32_t.
      int currAddr = op.cmd.RegOff;
      for (size_t i = 0; i < blockWriteDataValues.size(); i += 4) {
        auto last = std::min(blockWriteDataValues.size(), i + 4);
        SmallVector<int32_t> splitData =
            SmallVector<int32_t>(blockWriteDataValues.begin() + i,
                                 blockWriteDataValues.begin() + last);
        AIEX::NpuControlPacketOp::create(
            builder, loc, builder.getUI32IntegerAttr(currAddr), nullptr,
            /*opcode*/ builder.getI32IntegerAttr(0),
            /*stream_id*/ builder.getI32IntegerAttr(0),
            DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>(splitData)));
        currAddr += splitData.size() * sizeof(int32_t);
      }

    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_MASKWRITE) {
      AIEX::NpuControlPacketOp::create(
          builder, loc, builder.getUI32IntegerAttr(op.cmd.RegOff), nullptr,
          /*opcode*/ builder.getI32IntegerAttr(0),
          /*stream_id*/ builder.getI32IntegerAttr(0),
          DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>(op.cmd.Value)));
    } else {
      llvm::errs() << "Unhandled txn opcode: " << op.cmd.Opcode << "\n";
      return failure();
    }
  }
  return success();
}

// Perform bitwise or on consecutive control packets operating on the same
// address, to resolve the lack of mask write in control packets.
LogicalResult orConsecutiveWritesOnSameAddr(Block *body) {
  SmallVector<AIEX::NpuControlPacketOp> ctrlPktOps;
  body->walk(
      [&](AIEX::NpuControlPacketOp cpOp) { ctrlPktOps.push_back(cpOp); });
  if (ctrlPktOps.empty())
    return success();

  SmallVector<Operation *> erased;
  int addrBuffer = ctrlPktOps[0].getAddress();
  AIEX::NpuControlPacketOp ctrlPktBuffer = ctrlPktOps[0];
  for (size_t i = 1; i < ctrlPktOps.size(); i++) {
    int currentAddrBuffer = ctrlPktOps[i].getAddress();
    if (addrBuffer != currentAddrBuffer) {
      addrBuffer = currentAddrBuffer;
      ctrlPktBuffer = ctrlPktOps[i];
      continue;
    }
    auto bufferedData = ctrlPktBuffer.getData().value();
    auto currentData = ctrlPktOps[i].getData().value();
    SmallVector<int> newData;
    for (unsigned j = 0; j < std::max(bufferedData.size(), currentData.size());
         j++) {
      if (j < std::min(bufferedData.size(), currentData.size())) {
        newData.push_back(bufferedData[j] | currentData[j]);
        continue;
      }
      newData.push_back(j < bufferedData.size() ? bufferedData[j]
                                                : currentData[j]);
    }
    ctrlPktBuffer.getProperties().data = DenseI32ArrayAttr::get(
        ctrlPktBuffer->getContext(), ArrayRef<int>{newData});
    erased.push_back(ctrlPktOps[i]);
  }

  for (auto e : erased)
    e->erase();

  return success();
}

// Take transaction operations and insert them at the _current_ insertion point
// of the supplied builder.
static LogicalResult convertTransactionOpsToMLIR(
    OpBuilder builder, AIE::AIEToConfigurationOutputType outputType,
    std::vector<TransactionBinaryOperation> &operations,
    std::string blockwrite_prefix = "config_blockwrite_data_") {

  auto loc = builder.getUnknownLoc();

  // for each blockwrite in the binary, create a GlobalOp with the data at the
  // device level
  std::vector<memref::GlobalOp> global_data;
  {
    DeviceOp device =
        llvm::dyn_cast<DeviceOp>(builder.getBlock()->getParentOp());
    if (!device) {
      device = builder.getBlock()->getParentOp()->getParentOfType<DeviceOp>();
    }
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(device.getBody());
    int id = 0;
    for (auto &op : operations) {
      if (op.cmd.Opcode != XAIE_IO_BLOCKWRITE) {
        global_data.push_back(nullptr);
        continue;
      }
      uint32_t size = op.cmd.Size / 4;
      const uint32_t *d = reinterpret_cast<const uint32_t *>(op.cmd.DataPtr);
      std::vector<uint32_t> data32(d, d + size);

      std::string name = blockwrite_prefix;
      do {
        name = blockwrite_prefix + std::to_string(id++);
      } while (device.lookupSymbol(name));

      MemRefType memrefType = MemRefType::get({size}, builder.getI32Type());
      TensorType tensorType =
          RankedTensorType::get({size}, builder.getI32Type());
      auto global = memref::GlobalOp::create(
          builder, loc, name, builder.getStringAttr("private"), memrefType,
          DenseElementsAttr::get<uint32_t>(tensorType, data32), true, nullptr);
      global_data.push_back(global);
    }
  }

  // create the txn ops
  if (outputType == AIE::AIEToConfigurationOutputType::Transaction) {
    if (failed(emitTransactionOps(builder, operations, global_data)))
      return failure();
  } else if (outputType == AIE::AIEToConfigurationOutputType::ControlPacket) {
    if (failed(emitControlPacketOps(builder, operations, global_data)))
      return failure();
    // resolve mask writes; control packet doesn't natively support mask write.
    if (failed(orConsecutiveWritesOnSameAddr(builder.getBlock())))
      return failure();
  } else {
    llvm_unreachable("bad output type");
  }

  return success();
}

// Convert (disassemble) a transaction binary to MLIR. On success return a new
// ModuleOp containing a DeviceOp containing a runtime sequence with the
// transaction binary encoded as a sequence of npu.write32, npu.maskwrite32 and
// npu.blockwrite operations. On failure return std::nullopt.
std::optional<mlir::ModuleOp>
xilinx::AIE::convertTransactionBinaryToMLIR(mlir::MLIRContext *ctx,
                                            std::vector<uint8_t> &binary) {

  // parse the binary
  std::vector<TransactionBinaryOperation> operations;
  auto c = parseTransactionBinary(binary, operations);
  if (!c) {
    llvm::errs() << "Failed to parse binary\n";
    return std::nullopt;
  }
  int columns = *c;

  auto loc = mlir::UnknownLoc::get(ctx);

  // create a new ModuleOp and set the insertion point
  auto module = ModuleOp::create(loc);
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());

  // create aie.device
  // Map column count to device type (columns is 1-indexed)
  AIEDevice deviceType;
  switch (columns) {
    case 1: deviceType = AIEDevice::npu1_1col; break;
    case 2: deviceType = AIEDevice::npu1_2col; break;
    case 3: deviceType = AIEDevice::npu1_3col; break;
    case 4:
    default:
      // For 4+ columns, use npu1 (generic 4-column device)
      deviceType = AIEDevice::npu1;
      if (columns > 4) {
        llvm::errs() << "Warning: Transaction binary indicates " << columns
                     << " columns, using npu1 (4-column) device model\n";
      }
      break;
  }
  auto device = DeviceOp::create(builder, loc, deviceType,
                                 DeviceOp::getDefaultDeviceName());
  device.getRegion().emplaceBlock();
  DeviceOp::ensureTerminator(device.getBodyRegion(), builder, loc);
  builder.setInsertionPointToStart(device.getBody());

  // convert the parsed ops to MLIR
  if (failed(convertTransactionOpsToMLIR(
          builder, AIE::AIEToConfigurationOutputType::Transaction, operations)))
    return std::nullopt;

  return module;
}

LogicalResult xilinx::AIE::generateAndInsertConfigOps(
    OpBuilder &builder, xilinx::AIE::DeviceOp device, llvm::StringRef clElfDir,
    AIE::AIEToConfigurationOutputType outputType,
    std::string blockwrite_prefix) {
  const AIETargetModel &targetModel =
      (const AIETargetModel &)device.getTargetModel();

  if (!targetModel.hasProperty(AIETargetModel::IsNPU))
    return failure();

  bool aieSim = false;
  bool xaieDebug = false;

  AIERTControl ctl(targetModel);
  if (failed(ctl.setIOBackend(aieSim, xaieDebug)))
    return failure();

  // start collecting transactions
  ctl.startTransaction();

  bool generateElfs = true;
  if (failed(generateTransactions(ctl, clElfDir, device, aieSim, generateElfs,
                                  true, true)))
    return failure();

  // Export the transactions to a binary buffer
  std::vector<uint8_t> txn_data = ctl.exportSerializedTransaction();

  // parse the binary data
  std::vector<TransactionBinaryOperation> operations;
  if (!parseTransactionBinary(txn_data, operations)) {
    llvm::errs() << "Failed to parse binary\n";
    return failure();
  }

  if (failed(convertTransactionOpsToMLIR(builder, outputType, operations,
                                         blockwrite_prefix))) {
    return failure();
  }

  return success();
}

static LogicalResult
convertAIEToConfiguration(AIE::DeviceOp device, StringRef clElfDir,
                          AIE::AIEToConfigurationOutputType outputType) {

  OpBuilder builder(device.getBodyRegion());
  // search for aiex.configure ops in runtime sequences by walking the device
  // and collect them in a vector. If there are none, create a new runtime
  // sequence. Otherwise assume the insertion point is the first
  // aiex.configure op.
  auto loc = builder.getUnknownLoc();
  SmallVector<AIEX::ConfigureOp> configureOps;
  device.walk([&](AIEX::ConfigureOp op) { configureOps.push_back(op); });

  if (configureOps.empty()) {
    // create aiex.runtime_sequence
    int id = 0;
    std::string seq_name = "configure";
    while (device.lookupSymbol(seq_name))
      seq_name = "configure" + std::to_string(id++);
    StringAttr seq_sym_name = builder.getStringAttr(seq_name);
    auto seq = AIE::RuntimeSequenceOp::create(builder, loc, seq_sym_name);
    seq.getBody().push_back(new Block);
    builder.setInsertionPointToStart(&seq.getBody().front());
  } else {
    builder.setInsertionPoint(configureOps.front());
  }

  // convert the parsed ops to MLIR
  if (failed(generateAndInsertConfigOps(builder, device, clElfDir, outputType)))
    return failure();

  // If we chose the first aiex.configure as insertion point, erase it
  // and inline its child operations.
  if (!configureOps.empty()) {
    // splice the body into the current insertion point
    builder.getBlock()->getOperations().splice(
        builder.getInsertionPoint(),
        configureOps.front().getBody().front().getOperations());
    configureOps.front().erase();
  }

  return success();
}

namespace {

template <typename BaseClass, AIE::AIEToConfigurationOutputType MyOutputType>
struct ConvertAIEToConfigurationPass : BaseClass {
  std::string &ref_clElfDir;
  std::string &ref_clDeviceName;
  ConvertAIEToConfigurationPass(std::string &clElfDir,
                                std::string &clDeviceName)
      : ref_clElfDir(clElfDir), ref_clDeviceName(clDeviceName) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, AIEX::AIEXDialect>();
  }

  void runOnOperation() override {
    AIE::DeviceOp deviceOp = BaseClass::getOperation();
    if (!ref_clDeviceName.empty() &&
        deviceOp.getSymName() != ref_clDeviceName) {
      return;
    }
    if (failed(
            convertAIEToConfiguration(deviceOp, ref_clElfDir, MyOutputType))) {
      return BaseClass::signalPassFailure();
    }
  }
};

struct ConvertAIEToTransactionPass
    : ConvertAIEToConfigurationPass<
          xilinx::impl::ConvertAIEToTransactionBase<
              ConvertAIEToTransactionPass>,
          AIE::AIEToConfigurationOutputType::Transaction> {
  ConvertAIEToTransactionPass()
      : ConvertAIEToConfigurationPass<
            xilinx::impl::ConvertAIEToTransactionBase<
                ConvertAIEToTransactionPass>,
            AIE::AIEToConfigurationOutputType::Transaction>(clElfDir,
                                                            clDeviceName) {}
};

struct ConvertAIEToControlPacketsPass
    : ConvertAIEToConfigurationPass<
          xilinx::impl::ConvertAIEToControlPacketsBase<
              ConvertAIEToControlPacketsPass>,
          AIE::AIEToConfigurationOutputType::ControlPacket> {
  ConvertAIEToControlPacketsPass()
      : ConvertAIEToConfigurationPass<
            xilinx::impl::ConvertAIEToControlPacketsBase<
                ConvertAIEToControlPacketsPass>,
            AIE::AIEToConfigurationOutputType::ControlPacket>(clElfDir,
                                                              clDeviceName) {}
};

} // end anonymous namespace

std::unique_ptr<mlir::OperationPass<xilinx::AIE::DeviceOp>>
xilinx::AIE::createConvertAIEToTransactionPass() {
  return std::make_unique<ConvertAIEToTransactionPass>();
}

std::unique_ptr<mlir::OperationPass<xilinx::AIE::DeviceOp>>
xilinx::AIE::createConvertAIEToControlPacketsPass() {
  return std::make_unique<ConvertAIEToControlPacketsPass>();
}
