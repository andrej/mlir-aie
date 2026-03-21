//===- AIELockLifting.cpp - Lock Semantic Lifting ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Util/AIELockLifting.h"

namespace xilinx {
namespace AIE {

//===----------------------------------------------------------------------===//
// Lock Address Parser Implementation
//===----------------------------------------------------------------------===//

LockAddressInfo LockAddressParser::parse(uint32_t addr) const {
  LockAddressInfo info;
  info.isLockRegister = false;

  // Extract tile coordinates from address
  // Address format: row << 20 | col << (col_shift) | offset
  // For NPU/AIE2, each row is 0x100000 apart
  int row = (addr >> kTileAddrShift) & 0x7;  // Bits 20-22 for row
  int col = 0;  // For 1-col NPU designs, column is always 0

  // Calculate tile base and offset within tile
  uint32_t tileBase = row << kTileAddrShift;
  uint32_t offset = addr - tileBase;

  // Check if offset is within lock region
  if (offset >= kLockRegionBase && offset < kLockRegionEnd) {
    uint32_t lockOffset = offset - kLockRegionBase;

    // Each lock has 8 bytes (2 registers)
    // Lock ID is determined by which 8-byte block we're in
    int lockId = lockOffset / kLockStride;

    // Only consider this a lock register if it's the first register of the pair
    // (lock init values are in the first 4-byte register of each 8-byte block)
    if (lockOffset % kLockStride == 0) {
      info.isLockRegister = true;
      info.column = col;
      info.row = row;
      info.lockId = lockId;
    }
  }

  return info;
}

bool LockAddressParser::isLockAddress(uint32_t addr) const {
  return parse(addr).isLockRegister;
}

//===----------------------------------------------------------------------===//
// Lock Accumulator Implementation
//===----------------------------------------------------------------------===//

std::optional<ParsedLockConfig>
LockAccumulator::addMaskWrite(uint32_t addr, uint32_t value, uint32_t mask,
                               const LockAddressParser &parser) {
  // Parse the address
  auto addrInfo = parser.parse(addr);
  if (!addrInfo.isLockRegister) {
    return std::nullopt;
  }

  // Create lock key
  LockKey key{addrInfo.column, addrInfo.row, addrInfo.lockId};

  // Lock initialization pattern observed in xclbins:
  // 1. maskwrite with mask=2, value=N - sets a value
  // 2. maskwrite with mask=2, value=0 - resets value
  // 3. maskwrite with mask=0, value=1 - enable bit indicating lock should use non-zero init
  //
  // Strategy: Track first non-zero value AND whether enable bit is set

  auto &state = lockStates_[key];

  if (mask != 0 && value != 0 && state.firstNonZeroValue == 0) {
    // First non-zero value write - save it
    state.firstNonZeroValue = static_cast<int>(value);
  } else if (mask == 0 && value == 1) {
    // Enable bit - this lock should use its first non-zero value
    state.hasEnable = true;
  }

  return std::nullopt;  // Don't return individual configs, use getAllLocks instead
}

std::map<LockAccumulator::LockKey, ParsedLockConfig>
LockAccumulator::getAllLocks() const {
  std::map<LockKey, ParsedLockConfig> locks;

  for (const auto &[key, state] : lockStates_) {
    ParsedLockConfig config;
    config.column = key.col;
    config.row = key.row;
    config.lockId = key.lockId;

    // If lock has enable bit set, use first non-zero value; otherwise use 0
    config.initValue = state.hasEnable ? state.firstNonZeroValue : 0;

    locks[key] = config;
  }

  return locks;
}

} // namespace AIE
} // namespace xilinx
