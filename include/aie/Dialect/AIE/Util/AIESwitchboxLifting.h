//===- AIESwitchboxLifting.h - Switchbox Semantic Lifting ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Semantic lifting of stream switch register writes to high-level
// aie.switchbox and aie.connect operations. This transforms raw register
// writes into meaningful routing topology descriptions.
//===----------------------------------------------------------------------===//

#ifndef AIE_SWITCHBOX_LIFTING_H
#define AIE_SWITCHBOX_LIFTING_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"  // For WireBundle enum
#include "aie/Dialect/AIE/Util/AIEDMABDLifting.h"  // For TileType enum
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace xilinx {
namespace AIE {

// Note: WireBundle enum is defined in AIEDialect.h

//===----------------------------------------------------------------------===//
// Switch Connection Info - Result of parsing a master config register
//===----------------------------------------------------------------------===//

struct SwitchConnectionInfo {
  bool isValid = false;        // Was this a valid switchbox register?
  int column = 0;              // Tile column
  int row = 0;                 // Tile row
  TileType tileType = TileType::Compute;

  // Master (destination) port info
  int masterPortIndex = 0;     // 0-22 for compute tiles
  WireBundle destBundle = WireBundle::Core;
  int destChannel = 0;         // Channel number

  // Configuration
  bool masterEnable = false;   // Bit 31
  bool packetMode = false;     // Bit 30
  bool dropHeader = false;     // Bit 7 (packet mode only)

  // Source info (circuit mode)
  int slavePortId = 0;         // Bits [4:0] - maps to source bundle/channel
  WireBundle sourceBundle = WireBundle::Core;
  int sourceChannel = 0;       // Decoded from slavePortId

  // Packet mode config (if packetMode = true)
  int arbiterId = 0;           // Bits [2:0]
  int mselEnable = 0;          // Bits [6:3]
};

//===----------------------------------------------------------------------===//
// Parsed Switchbox Config - Complete switchbox configuration for a tile
//===----------------------------------------------------------------------===//

struct ParsedSwitchboxConfig {
  // Location
  int column = 0;
  int row = 0;
  TileType tileType = TileType::Compute;

  // All active connections for this switchbox
  struct Connection {
    WireBundle sourceBundle = WireBundle::Core;
    int sourceChannel = 0;
    WireBundle destBundle = WireBundle::Core;
    int destChannel = 0;
    bool isPacketMode = false;
  };
  std::vector<Connection> connections;

  // Helper methods
  bool hasConnections() const { return !connections.empty(); }
  size_t connectionCount() const { return connections.size(); }
};

//===----------------------------------------------------------------------===//
// Switchbox Field Extractor - Bit field extraction utilities
//===----------------------------------------------------------------------===//

namespace SwitchFieldExtractor {

// Master config fields
inline bool getMasterEnable(uint32_t value) {
  return (value >> 31) & 1;
}

inline bool getPacketEnable(uint32_t value) {
  return (value >> 30) & 1;
}

inline bool getDropHeader(uint32_t value) {
  return (value >> 7) & 1;
}

inline uint8_t getSlavePortId(uint32_t value) {
  return value & 0x1F;  // Bits [4:0]
}

inline uint8_t getArbiterId(uint32_t value) {
  return value & 0x7;   // Bits [2:0] (packet mode)
}

inline uint8_t getMselEnable(uint32_t value) {
  return (value >> 3) & 0xF;  // Bits [6:3] (packet mode)
}

// Slave config fields
inline bool getSlaveEnable(uint32_t value) {
  return (value >> 31) & 1;
}

} // namespace SwitchFieldExtractor

//===----------------------------------------------------------------------===//
// Switch Address Parser - Detect and parse switchbox register addresses
//===----------------------------------------------------------------------===//

class SwitchAddressParser {
public:
  SwitchAddressParser(int numMemTileRows = 1);

  /// Parse address and return connection info if it's a master config register
  SwitchConnectionInfo parseMasterConfig(uint32_t addr) const;

  /// Check if address is any switchbox register
  bool isSwitchboxAddress(uint32_t addr) const;

  // Port mapping structure
  struct PortMapping {
    WireBundle bundle;
    int channel;
  };

  // Helper methods for port mapping
  PortMapping getMasterPortMapping(int portIndex, TileType tileType) const;
  PortMapping getSlavePortMapping(int portIndex, TileType tileType) const;

private:
  int numMemTileRows_;

  // Address region constants (compute tiles)
  static constexpr uint32_t kMasterConfigBase = 0x3F000;
  static constexpr uint32_t kMasterConfigEnd = 0x3F05C;   // 23 regs * 4 bytes
  static constexpr uint32_t kSlaveConfigBase = 0x3F100;
  static constexpr uint32_t kSlaveConfigEnd = 0x3F164;    // 25 regs * 4 bytes

  // Address region constants (memory tiles)
  static constexpr uint32_t kMemTileMasterConfigBase = 0xB0000;
  static constexpr uint32_t kMemTileMasterConfigEnd = 0xB0040;   // 17 regs * 4 bytes
  static constexpr uint32_t kMemTileSlaveConfigBase = 0xB0100;
  static constexpr uint32_t kMemTileSlaveConfigEnd = 0xB0144;    // 18 regs * 4 bytes

  static constexpr uint32_t kRegisterSize = 4;
  static constexpr uint32_t kTileAddrShift = 20;          // 0x100000 per tile

  // Port mapping tables
  // Master ports: 23 total (ports 0-22) for compute tiles
  static const PortMapping kMasterPortMap[23];

  // Slave ports: 25 total (ports 0-24) for compute tiles
  static const PortMapping kSlavePortMap[25];

  // MemTile master ports: 17 total (ports 0-16)
  static const PortMapping kMemTileMasterPortMap[17];

  // MemTile slave ports: 18 total (ports 0-17)
  static const PortMapping kMemTileSlavePortMap[18];

  // Shim tile master ports: 22 total (ports 0-21)
  static const PortMapping kShimTileMasterPortMap[22];

  // Shim tile slave ports: 23 total (ports 0-22)
  static const PortMapping kShimTileSlavePortMap[23];
};

//===----------------------------------------------------------------------===//
// Switchbox Accumulator - Collect connections into switchbox configs
//===----------------------------------------------------------------------===//

class SwitchboxAccumulator {
public:
  /// Key for identifying a specific switchbox (one per tile)
  struct SwitchboxKey {
    int col;
    int row;
    TileType type;

    bool operator<(const SwitchboxKey &other) const;
  };

  SwitchboxAccumulator() = default;

  /// Add a master config write
  /// Returns connection info if this is a valid switchbox connection
  std::optional<SwitchConnectionInfo> addMasterWrite(
      uint32_t addr, uint32_t value,
      const SwitchAddressParser &parser);

  /// Get all connections for a specific switchbox
  ParsedSwitchboxConfig getSwitchboxConfig(const SwitchboxKey &key) const;

  /// Get all accumulated switchboxes
  std::map<SwitchboxKey, ParsedSwitchboxConfig> getAll() const;

  /// Check if we have any switchbox configs
  bool hasConfigs() const { return !switchboxes_.empty(); }

  /// Get the number of configured switchboxes
  size_t switchboxCount() const { return switchboxes_.size(); }

private:
  std::map<SwitchboxKey, ParsedSwitchboxConfig> switchboxes_;
};

//===----------------------------------------------------------------------===//
// Switchbox Pretty Printer - For annotated output
//===----------------------------------------------------------------------===//

class SwitchboxPrettyPrinter {
public:
  /// Print a connection as a comment
  static void printConnectionAsComment(llvm::raw_ostream &os,
                                       const SwitchConnectionInfo &conn);

  /// Print a complete switchbox configuration
  static void printSwitchboxConfig(llvm::raw_ostream &os,
                                   const ParsedSwitchboxConfig &config);

  /// Convert WireBundle to string
  static const char* wireBundleToString(WireBundle bundle);
};

} // namespace AIE
} // namespace xilinx

#endif // AIE_SWITCHBOX_LIFTING_H
