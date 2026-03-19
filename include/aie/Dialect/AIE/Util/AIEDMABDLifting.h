//===- AIEDMABDLifting.h - DMA BD Semantic Lifting --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Semantic lifting of DMA buffer descriptor register writes to high-level
// aie.dma_bd operations. This transforms raw register writes into meaningful
// data movement intent descriptions.
//===----------------------------------------------------------------------===//

#ifndef AIE_DMA_BD_LIFTING_H
#define AIE_DMA_BD_LIFTING_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace xilinx {
namespace AIE {

//===----------------------------------------------------------------------===//
// Tile Type Classification
//===----------------------------------------------------------------------===//

enum class TileType {
  Compute,    // AIE compute tiles with local memory
  MemoryTile, // Dedicated memory tiles
  ShimNOC,    // Shim tiles with NOC interface
  ShimPL      // Shim tiles with PL interface
};

//===----------------------------------------------------------------------===//
// BD Address Info - Result of parsing an address
//===----------------------------------------------------------------------===//

struct BDAddressInfo {
  bool isBDRegister = false;
  int column = 0;
  int row = 0;
  TileType tileType = TileType::Compute;
  int bdIndex = 0;    // BD number (0-15 for compute, 0-47 for memtile)
  int regIndex = 0;   // Register within BD (0-5)

  bool operator==(const BDAddressInfo &other) const {
    return column == other.column && row == other.row &&
           tileType == other.tileType && bdIndex == other.bdIndex;
  }
};

//===----------------------------------------------------------------------===//
// Dimension Descriptor - For N-D addressing
//===----------------------------------------------------------------------===//

struct DimensionDescriptor {
  uint16_t stepSize = 1;  // Stride in 32-bit words (actual value)
  uint8_t wrap = 0;       // Wrap count (0 = no wrap/linear)

  bool isLinear() const { return wrap == 0; }
};

//===----------------------------------------------------------------------===//
// Lock Configuration
//===----------------------------------------------------------------------===//

struct LockConfig {
  bool enabled = false;
  uint8_t lockId = 0;
  int8_t value = 0;  // Signed: negative means acquire_ge, positive means acq_eq
};

//===----------------------------------------------------------------------===//
// Parsed BD Configuration - Complete decoded BD state
//===----------------------------------------------------------------------===//

struct ParsedBDConfig {
  // Source location information
  int column = 0;
  int row = 0;
  TileType tileType = TileType::Compute;
  int bdIndex = 0;

  // DMA_BDx_0: Base address and length
  uint32_t baseAddress = 0;     // 32-bit word address within tile memory
  uint32_t bufferLength = 0;    // Transfer length in 32-bit words

  // DMA_BDx_1: Packet configuration
  bool enableCompression = false;
  bool enablePacket = false;
  uint8_t outOfOrderBdId = 0;
  uint8_t packetId = 0;
  uint8_t packetType = 0;

  // DMA_BDx_2, DMA_BDx_3: Dimension addressing
  // dimensions[0] = D0 (innermost), dimensions[1] = D1, dimensions[2] = D2
  std::array<DimensionDescriptor, 3> dimensions;

  // DMA_BDx_4: Iteration control
  uint8_t iterationCurrent = 0;
  uint8_t iterationWrap = 0;
  uint16_t iterationStepSize = 1;

  // DMA_BDx_5: Control and locks
  bool tlastSuppress = false;
  uint8_t nextBd = 0;
  bool useNextBd = false;
  bool validBd = false;

  // Lock acquire
  LockConfig lockAcquire;

  // Lock release
  uint8_t lockRelId = 0;
  int8_t lockRelValue = 0;  // 0 = no release

  // Helper methods
  bool hasLockAcquire() const { return lockAcquire.enabled; }
  bool hasLockRelease() const { return lockRelValue != 0; }
  bool hasDimensions() const {
    return dimensions[0].wrap != 0 || dimensions[1].wrap != 0;
  }
  bool hasIteration() const { return iterationWrap > 1; }
  bool hasPacketHeader() const { return enablePacket; }
};

//===----------------------------------------------------------------------===//
// BD Field Extractor - Bit field extraction utilities
//===----------------------------------------------------------------------===//

namespace BDFieldExtractor {

/// Extract unsigned bits from a value
inline uint32_t extractBits(uint32_t value, int highBit, int lowBit) {
  uint32_t mask = ((1u << (highBit - lowBit + 1)) - 1) << lowBit;
  return (value & mask) >> lowBit;
}

/// Extract signed bits (with sign extension)
inline int32_t extractSignedBits(uint32_t value, int highBit, int lowBit) {
  uint32_t raw = extractBits(value, highBit, lowBit);
  int width = highBit - lowBit + 1;
  if (raw & (1 << (width - 1))) {
    raw |= ~((1u << width) - 1);
  }
  return static_cast<int32_t>(raw);
}

// DMA_BDx_0 fields
inline uint32_t getBaseAddress(uint32_t reg0) {
  return extractBits(reg0, 27, 14);
}
inline uint32_t getBufferLength(uint32_t reg0) {
  return extractBits(reg0, 13, 0);
}

// DMA_BDx_1 fields
inline bool getEnableCompression(uint32_t reg1) {
  return (reg1 >> 31) & 1;
}
inline bool getEnablePacket(uint32_t reg1) {
  return (reg1 >> 30) & 1;
}
inline uint8_t getOutOfOrderBdId(uint32_t reg1) {
  return extractBits(reg1, 29, 24);
}
inline uint8_t getPacketId(uint32_t reg1) {
  return extractBits(reg1, 23, 19);
}
inline uint8_t getPacketType(uint32_t reg1) {
  return extractBits(reg1, 18, 16);
}

// DMA_BDx_2 fields
inline uint16_t getD0Stepsize(uint32_t reg2) {
  return extractBits(reg2, 12, 0) + 1;  // Encoded as actual-1
}
inline uint16_t getD1Stepsize(uint32_t reg2) {
  return extractBits(reg2, 25, 13) + 1;
}

// DMA_BDx_3 fields
inline uint8_t getD0Wrap(uint32_t reg3) {
  return extractBits(reg3, 20, 13);
}
inline uint8_t getD1Wrap(uint32_t reg3) {
  return extractBits(reg3, 28, 21);
}
inline uint16_t getD2Stepsize(uint32_t reg3) {
  return extractBits(reg3, 12, 0) + 1;
}

// DMA_BDx_4 fields
inline uint8_t getIterationCurrent(uint32_t reg4) {
  return extractBits(reg4, 24, 19);
}
inline uint8_t getIterationWrap(uint32_t reg4) {
  return extractBits(reg4, 18, 13) + 1;  // Encoded as actual-1
}
inline uint16_t getIterationStepsize(uint32_t reg4) {
  return extractBits(reg4, 12, 0) + 1;
}

// DMA_BDx_5 fields
inline bool getTlastSuppress(uint32_t reg5) {
  return (reg5 >> 31) & 1;
}
inline uint8_t getNextBd(uint32_t reg5) {
  return extractBits(reg5, 30, 27);
}
inline bool getUseNextBd(uint32_t reg5) {
  return (reg5 >> 26) & 1;
}
inline bool getValidBd(uint32_t reg5) {
  return (reg5 >> 25) & 1;
}
inline int8_t getLockRelValue(uint32_t reg5) {
  return static_cast<int8_t>(extractSignedBits(reg5, 24, 18));
}
inline uint8_t getLockRelId(uint32_t reg5) {
  return extractBits(reg5, 16, 13);
}
inline bool getLockAcqEnable(uint32_t reg5) {
  return (reg5 >> 12) & 1;
}
inline int8_t getLockAcqValue(uint32_t reg5) {
  return static_cast<int8_t>(extractSignedBits(reg5, 11, 5));
}
inline uint8_t getLockAcqId(uint32_t reg5) {
  return extractBits(reg5, 3, 0);
}

/// Parse all 6 BD registers into a ParsedBDConfig
ParsedBDConfig parseRegisters(const uint32_t regs[6]);

} // namespace BDFieldExtractor

//===----------------------------------------------------------------------===//
// BD Address Parser
//===----------------------------------------------------------------------===//

class BDAddressParser {
public:
  /// Configure for specific device
  BDAddressParser(int numMemTileRows = 1) : numMemTileRows_(numMemTileRows) {}

  /// Parse an absolute address to determine if it's a BD register
  BDAddressInfo parse(uint32_t addr) const;

  /// Check if an address falls within any BD register range
  bool isBDAddress(uint32_t addr) const;

private:
  int numMemTileRows_;

  // BD region constants for AIE2
  static constexpr uint32_t kMemoryBDBase = 0x1D000;
  static constexpr uint32_t kMemoryBDEnd = 0x1D200;
  static constexpr uint32_t kMemTileBDBase = 0xA0000;
  static constexpr uint32_t kMemTileBDEnd = 0xA0600;
  static constexpr uint32_t kBDSize = 0x20;  // 6 regs * 4 bytes, rounded
  static constexpr uint32_t kTileAddrShift = 20;  // 0x100000 per tile
};

//===----------------------------------------------------------------------===//
// BD Accumulator - Accumulates BD register writes into complete configs
//===----------------------------------------------------------------------===//

class BDAccumulator {
public:
  /// Key for identifying a specific BD
  struct BDKey {
    int col;
    int row;
    TileType type;
    int bdIndex;

    bool operator<(const BDKey &other) const;
  };

  /// State of a BD being accumulated
  struct PendingBD {
    std::array<std::optional<uint32_t>, 6> registers;
    int writeCount = 0;

    /// Check if BD configuration is complete
    bool isComplete() const;

    /// Check if valid bit is set (indicates intentional completion)
    bool hasValidBit() const;
  };

  BDAccumulator() = default;

  /// Add a write to the accumulator
  /// Returns completed BD config if this write completes a BD
  std::optional<ParsedBDConfig> addWrite(uint32_t addr, uint32_t value,
                                          const BDAddressParser &parser);

  /// Flush all pending BDs (for end of sequence)
  llvm::SmallVector<ParsedBDConfig> flush();

  /// Check if there are any pending BDs
  bool hasPending() const { return !pendingBDs_.empty(); }

  /// Get the number of pending BDs
  size_t pendingCount() const { return pendingBDs_.size(); }

private:
  std::map<BDKey, PendingBD> pendingBDs_;

  /// Complete a pending BD and return the parsed config
  ParsedBDConfig completeBD(const BDKey &key, const PendingBD &pending);
};

//===----------------------------------------------------------------------===//
// BD Pretty Printer - For annotated output
//===----------------------------------------------------------------------===//

class BDPrettyPrinter {
public:
  /// Print a parsed BD configuration as a comment
  static void printAsComment(llvm::raw_ostream &os, const ParsedBDConfig &bd);

  /// Print dimension layout in aie.dma_bd format
  static void printDimensions(llvm::raw_ostream &os, const ParsedBDConfig &bd);

  /// Print lock configuration
  static void printLockConfig(llvm::raw_ostream &os, const ParsedBDConfig &bd);
};

//===----------------------------------------------------------------------===//
// Semantic Lifting Options
//===----------------------------------------------------------------------===//

struct SemanticLiftingOptions {
  /// Enable semantic lifting (vs raw register output)
  bool enabled = true;

  /// Output mode
  enum class Mode {
    Lifted,    // Only emit aie.dma_bd operations
    Hybrid,    // Emit both lifted and raw (raw as comments)
    Annotated  // Emit raw with semantic annotations as comments
  };
  Mode mode = Mode::Lifted;

  /// Try to resolve buffer references
  bool resolveBuffers = true;

  /// Emit lock operations
  bool emitLocks = true;

  /// Emit BD chaining (next_bd)
  bool emitChaining = true;
};

} // namespace AIE
} // namespace xilinx

#endif // AIE_DMA_BD_LIFTING_H
