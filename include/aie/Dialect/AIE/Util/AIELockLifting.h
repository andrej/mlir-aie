//===- AIELockLifting.h - Lock Semantic Lifting -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Semantic lifting of lock register writes to high-level aie.lock operations.
// This transforms raw lock initialization writes (via maskwrite32) into
// meaningful lock declarations with init values.
//===----------------------------------------------------------------------===//

#ifndef AIE_LOCK_LIFTING_H
#define AIE_LOCK_LIFTING_H

#include <cstdint>
#include <map>
#include <optional>

namespace xilinx {
namespace AIE {

//===----------------------------------------------------------------------===//
// Lock Address Info - Result of parsing a lock address
//===----------------------------------------------------------------------===//

struct LockAddressInfo {
  bool isLockRegister = false;
  int column = 0;
  int row = 0;
  int lockId = 0;

  bool operator==(const LockAddressInfo &other) const {
    return column == other.column && row == other.row && lockId == other.lockId;
  }
};

//===----------------------------------------------------------------------===//
// Parsed Lock Configuration
//===----------------------------------------------------------------------===//

struct ParsedLockConfig {
  int column = 0;
  int row = 0;
  int lockId = 0;
  int initValue = 0;  // Lock initialization value

  bool operator<(const ParsedLockConfig &other) const {
    if (column != other.column) return column < other.column;
    if (row != other.row) return row < other.row;
    return lockId < other.lockId;
  }
};

//===----------------------------------------------------------------------===//
// Lock Address Parser
//===----------------------------------------------------------------------===//

class LockAddressParser {
public:
  /// Configure for specific device
  LockAddressParser(int numMemTileRows = 1) : numMemTileRows_(numMemTileRows) {}

  /// Parse an absolute address to determine if it's a lock register
  LockAddressInfo parse(uint32_t addr) const;

  /// Check if an address falls within any lock register range
  bool isLockAddress(uint32_t addr) const;

private:
  int numMemTileRows_;

  // Lock region constants for AIE2/NPU
  // Lock registers are at offset 0x1DE00 within each tile
  // Each lock has 2 registers (8 bytes), but init value is in first register
  static constexpr uint32_t kLockRegionBase = 0x1DE00;
  static constexpr uint32_t kLockRegionEnd = 0x1E000;  // Conservative upper bound
  static constexpr uint32_t kLockStride = 0x8;  // 8 bytes per lock
  static constexpr uint32_t kTileAddrShift = 20;  // 0x100000 per tile row
};

//===----------------------------------------------------------------------===//
// Lock Accumulator - Accumulates lock register writes into complete configs
//===----------------------------------------------------------------------===//

class LockAccumulator {
public:
  /// Key for identifying a specific lock
  struct LockKey {
    int col;
    int row;
    int lockId;

    bool operator<(const LockKey &other) const {
      if (col != other.col) return col < other.col;
      if (row != other.row) return row < other.row;
      return lockId < other.lockId;
    }
  };

  /// Tracked lock state during accumulation
  struct LockState {
    int firstNonZeroValue = 0;  // First non-zero value written with mask != 0
    bool hasEnable = false;     // Whether mask=0, value=1 write was seen
  };

  LockAccumulator() = default;

  /// Add a maskwrite to a lock register
  /// Returns completed lock config if this write sets the init value
  std::optional<ParsedLockConfig> addMaskWrite(uint32_t addr, uint32_t value,
                                                uint32_t mask,
                                                const LockAddressParser &parser);

  /// Get all accumulated locks with final init values
  std::map<LockKey, ParsedLockConfig> getAllLocks() const;

  /// Check if there are any locks
  bool hasLocks() const { return !lockStates_.empty(); }

  /// Get the number of locks
  size_t lockCount() const { return lockStates_.size(); }

private:
  /// Map of lock states being tracked
  std::map<LockKey, LockState> lockStates_;
};

} // namespace AIE
} // namespace xilinx

#endif // AIE_LOCK_LIFTING_H
