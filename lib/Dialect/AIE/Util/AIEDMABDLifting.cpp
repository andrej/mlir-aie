//===- AIEDMABDLifting.cpp - DMA BD Semantic Lifting ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Util/AIEDMABDLifting.h"
#include "llvm/Support/Format.h"

using namespace xilinx::AIE;

//===----------------------------------------------------------------------===//
// BDFieldExtractor implementation
//===----------------------------------------------------------------------===//

ParsedBDConfig BDFieldExtractor::parseRegisters(const uint32_t regs[6]) {
  ParsedBDConfig config;

  // DMA_BDx_0: Base address and buffer length
  config.baseAddress = getBaseAddress(regs[0]);
  config.bufferLength = getBufferLength(regs[0]);

  // DMA_BDx_1: Packet configuration
  config.enableCompression = getEnableCompression(regs[1]);
  config.enablePacket = getEnablePacket(regs[1]);
  config.outOfOrderBdId = getOutOfOrderBdId(regs[1]);
  config.packetId = getPacketId(regs[1]);
  config.packetType = getPacketType(regs[1]);

  // DMA_BDx_2: D0 and D1 step sizes
  config.dimensions[0].stepSize = getD0Stepsize(regs[2]);
  config.dimensions[1].stepSize = getD1Stepsize(regs[2]);

  // DMA_BDx_3: D0 and D1 wraps, D2 step size
  config.dimensions[0].wrap = getD0Wrap(regs[3]);
  config.dimensions[1].wrap = getD1Wrap(regs[3]);
  config.dimensions[2].stepSize = getD2Stepsize(regs[3]);

  // DMA_BDx_4: Iteration control
  config.iterationCurrent = getIterationCurrent(regs[4]);
  config.iterationWrap = getIterationWrap(regs[4]);
  config.iterationStepSize = getIterationStepsize(regs[4]);

  // DMA_BDx_5: Locks and control
  config.tlastSuppress = getTlastSuppress(regs[5]);
  config.nextBd = getNextBd(regs[5]);
  config.useNextBd = getUseNextBd(regs[5]);
  config.validBd = getValidBd(regs[5]);

  // Lock release
  config.lockRelValue = getLockRelValue(regs[5]);
  config.lockRelId = getLockRelId(regs[5]);

  // Lock acquire
  config.lockAcquire.enabled = getLockAcqEnable(regs[5]);
  config.lockAcquire.lockId = getLockAcqId(regs[5]);
  config.lockAcquire.value = getLockAcqValue(regs[5]);

  return config;
}

//===----------------------------------------------------------------------===//
// BDAddressParser implementation
//===----------------------------------------------------------------------===//

BDAddressInfo BDAddressParser::parse(uint32_t addr) const {
  BDAddressInfo info;
  info.isBDRegister = false;

  // Extract tile coordinates from address
  // AIE2 formula: base + (col * 32 + row_offset) * 0x100000
  uint32_t tileOffset = (addr >> kTileAddrShift) & 0xFFF;
  uint32_t regOffset = addr & ((1 << kTileAddrShift) - 1);

  int column = tileOffset / 32;
  int rowPart = tileOffset % 32;

  // Check memory module BD region (compute tiles)
  if (regOffset >= kMemoryBDBase && regOffset < kMemoryBDEnd) {
    int bdOffset = regOffset - kMemoryBDBase;

    // Determine if this is shim (row 0) or compute tile
    if (rowPart == 0) {
      info.tileType = TileType::ShimNOC;
      info.row = 0;
    } else {
      info.tileType = TileType::Compute;
      info.row = rowPart;  // Use actual hardware row
    }

    info.isBDRegister = true;
    info.column = column;
    info.bdIndex = bdOffset / kBDSize;
    info.regIndex = (bdOffset % kBDSize) / 4;

    // Validate regIndex is in range
    if (info.regIndex > 5) {
      info.isBDRegister = false;
    }

    return info;
  }

  // Check memory tile BD region
  if (regOffset >= kMemTileBDBase && regOffset < kMemTileBDEnd) {
    int bdOffset = regOffset - kMemTileBDBase;

    info.isBDRegister = true;
    info.column = column;
    info.row = rowPart;  // Use actual hardware row
    info.tileType = TileType::MemoryTile;
    info.bdIndex = bdOffset / kBDSize;
    info.regIndex = (bdOffset % kBDSize) / 4;

    if (info.regIndex > 5) {
      info.isBDRegister = false;
    }

    return info;
  }

  return info;
}

bool BDAddressParser::isBDAddress(uint32_t addr) const {
  return parse(addr).isBDRegister;
}

//===----------------------------------------------------------------------===//
// BDAccumulator::BDKey implementation
//===----------------------------------------------------------------------===//

bool BDAccumulator::BDKey::operator<(const BDKey &other) const {
  if (col != other.col) return col < other.col;
  if (row != other.row) return row < other.row;
  if (static_cast<int>(type) != static_cast<int>(other.type))
    return static_cast<int>(type) < static_cast<int>(other.type);
  return bdIndex < other.bdIndex;
}

//===----------------------------------------------------------------------===//
// BDAccumulator::PendingBD implementation
//===----------------------------------------------------------------------===//

bool BDAccumulator::PendingBD::isComplete() const {
  // Complete if all 6 registers are written
  if (writeCount >= 6) return true;

  // Also complete if valid bit is explicitly set
  return hasValidBit();
}

bool BDAccumulator::PendingBD::hasValidBit() const {
  if (!registers[5].has_value()) return false;
  return BDFieldExtractor::getValidBd(registers[5].value());
}

//===----------------------------------------------------------------------===//
// BDAccumulator implementation
//===----------------------------------------------------------------------===//

std::optional<ParsedBDConfig>
BDAccumulator::addWrite(uint32_t addr, uint32_t value,
                        const BDAddressParser &parser) {
  auto addrInfo = parser.parse(addr);
  if (!addrInfo.isBDRegister) {
    return std::nullopt;
  }

  BDKey key{addrInfo.column, addrInfo.row, addrInfo.tileType,
            addrInfo.bdIndex};

  auto &pending = pendingBDs_[key];

  // Record this register write
  if (!pending.registers[addrInfo.regIndex].has_value()) {
    pending.writeCount++;
  }
  pending.registers[addrInfo.regIndex] = value;

  // Check if BD is now complete
  if (pending.isComplete()) {
    auto config = completeBD(key, pending);
    pendingBDs_.erase(key);
    return config;
  }

  return std::nullopt;
}

llvm::SmallVector<ParsedBDConfig> BDAccumulator::flush() {
  llvm::SmallVector<ParsedBDConfig> results;

  for (const auto &[key, pending] : pendingBDs_) {
    // Only emit if we have at least the address register
    if (pending.registers[0].has_value()) {
      results.push_back(completeBD(key, pending));
    }
  }

  pendingBDs_.clear();
  return results;
}

ParsedBDConfig BDAccumulator::completeBD(const BDKey &key,
                                          const PendingBD &pending) {
  // Build register array with defaults for missing values
  uint32_t regs[6] = {0, 0, 0, 0, 0, 0};
  for (int i = 0; i < 6; i++) {
    if (pending.registers[i].has_value()) {
      regs[i] = pending.registers[i].value();
    }
  }

  ParsedBDConfig config = BDFieldExtractor::parseRegisters(regs);

  // Fill in location info from key
  config.column = key.col;
  config.row = key.row;
  config.tileType = key.type;
  config.bdIndex = key.bdIndex;

  return config;
}

//===----------------------------------------------------------------------===//
// BDPrettyPrinter implementation
//===----------------------------------------------------------------------===//

void BDPrettyPrinter::printAsComment(llvm::raw_ostream &os,
                                      const ParsedBDConfig &bd) {
  os << "// DMA BD" << bd.bdIndex;

  // Tile location
  const char *typeStr = "";
  switch (bd.tileType) {
  case TileType::Compute: typeStr = "tile"; break;
  case TileType::MemoryTile: typeStr = "memtile"; break;
  case TileType::ShimNOC: typeStr = "shim_noc"; break;
  case TileType::ShimPL: typeStr = "shim_pl"; break;
  }
  os << " @ " << typeStr << "(" << bd.column << ", " << bd.row << ")";

  // Buffer info
  os << ": addr=0x" << llvm::format_hex_no_prefix(bd.baseAddress, 4);
  os << ", len=" << bd.bufferLength;

  // Dimensions if non-trivial
  if (bd.hasDimensions()) {
    os << ", dims=[";
    if (bd.dimensions[1].wrap != 0) {
      os << "<" << (int)bd.dimensions[1].wrap << ", "
         << bd.dimensions[2].stepSize << ">, ";
    }
    if (bd.dimensions[0].wrap != 0) {
      os << "<" << (int)bd.dimensions[0].wrap << ", "
         << bd.dimensions[1].stepSize << ">, ";
    }
    os << "<size, " << bd.dimensions[0].stepSize << ">]";
  }

  // Locks
  if (bd.hasLockAcquire()) {
    os << ", acq_lock(" << (int)bd.lockAcquire.lockId << ", "
       << (int)bd.lockAcquire.value << ")";
  }
  if (bd.hasLockRelease()) {
    os << ", rel_lock(" << (int)bd.lockRelId << ", "
       << (int)bd.lockRelValue << ")";
  }

  // Chaining
  if (bd.useNextBd) {
    os << ", next_bd=" << (int)bd.nextBd;
  }

  // Packet info
  if (bd.hasPacketHeader()) {
    os << ", packet(type=" << (int)bd.packetType
       << ", id=" << (int)bd.packetId << ")";
  }

  // Valid bit
  if (!bd.validBd) {
    os << " [INVALID]";
  }

  os << "\n";
}

void BDPrettyPrinter::printDimensions(llvm::raw_ostream &os,
                                       const ParsedBDConfig &bd) {
  if (!bd.hasDimensions()) {
    return;
  }

  os << "[";
  bool first = true;

  // Print from outermost to innermost
  if (bd.dimensions[1].wrap != 0) {
    os << "<" << (int)bd.dimensions[1].wrap << ", "
       << bd.dimensions[2].stepSize << ">";
    first = false;
  }

  if (bd.dimensions[0].wrap != 0) {
    if (!first) os << ", ";
    os << "<" << (int)bd.dimensions[0].wrap << ", "
       << bd.dimensions[1].stepSize << ">";
    first = false;
  }

  // Innermost dimension
  if (!first) os << ", ";
  os << "<wrap, " << bd.dimensions[0].stepSize << ">";

  os << "]";
}

void BDPrettyPrinter::printLockConfig(llvm::raw_ostream &os,
                                       const ParsedBDConfig &bd) {
  if (bd.hasLockAcquire()) {
    os << "aie.use_lock(%lock" << (int)bd.lockAcquire.lockId << ", ";
    if (bd.lockAcquire.value < 0) {
      os << "\"AcquireGreaterEqual\", " << -(int)bd.lockAcquire.value;
    } else {
      os << "\"Acquire\", " << (int)bd.lockAcquire.value;
    }
    os << ")\n";
  }

  if (bd.hasLockRelease()) {
    os << "aie.use_lock(%lock" << (int)bd.lockRelId << ", \"Release\", "
       << std::abs((int)bd.lockRelValue) << ")\n";
  }
}
