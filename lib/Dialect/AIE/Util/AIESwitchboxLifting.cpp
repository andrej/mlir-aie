//===- AIESwitchboxLifting.cpp - Switchbox Semantic Lifting ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Util/AIESwitchboxLifting.h"
#include "llvm/Support/Format.h"

using namespace xilinx::AIE;

//===----------------------------------------------------------------------===//
// SwitchAddressParser - Port Mapping Tables
//===----------------------------------------------------------------------===//

// Master port mapping (23 ports) - Destinations
// Computed from: (offset - 0x3F000) / 4
const SwitchAddressParser::PortMapping
    SwitchAddressParser::kMasterPortMap[23] = {
  {WireBundle::Core, 0},        // 0: AIE_Core0 (0x3F000)
  {WireBundle::DMA, 0},         // 1: DMA0 (0x3F004)
  {WireBundle::DMA, 1},         // 2: DMA1 (0x3F008)
  {WireBundle::TileControl, 0}, // 3: Tile_Ctrl (0x3F00C)
  {WireBundle::FIFO, 0},        // 4: FIFO0 (0x3F010)
  {WireBundle::South, 0},       // 5: South0 (0x3F014)
  {WireBundle::South, 1},       // 6: South1 (0x3F018)
  {WireBundle::South, 2},       // 7: South2 (0x3F01C)
  {WireBundle::South, 3},       // 8: South3 (0x3F020)
  {WireBundle::West, 0},        // 9: West0 (0x3F024)
  {WireBundle::West, 1},        // 10: West1 (0x3F028)
  {WireBundle::West, 2},        // 11: West2 (0x3F02C)
  {WireBundle::West, 3},        // 12: West3 (0x3F030)
  {WireBundle::North, 0},       // 13: North0 (0x3F034)
  {WireBundle::North, 1},       // 14: North1 (0x3F038)
  {WireBundle::North, 2},       // 15: North2 (0x3F03C)
  {WireBundle::North, 3},       // 16: North3 (0x3F040)
  {WireBundle::North, 4},       // 17: North4 (0x3F044)
  {WireBundle::North, 5},       // 18: North5 (0x3F048)
  {WireBundle::East, 0},        // 19: East0 (0x3F04C)
  {WireBundle::East, 1},        // 20: East1 (0x3F050)
  {WireBundle::East, 2},        // 21: East2 (0x3F054)
  {WireBundle::East, 3},        // 22: East3 (0x3F058)
};

// Slave port mapping (25 ports) - Sources
// Computed from: (offset - 0x3F100) / 4
const SwitchAddressParser::PortMapping
    SwitchAddressParser::kSlavePortMap[25] = {
  {WireBundle::Core, 0},        // 0: AIE_Core0 (0x3F100)
  {WireBundle::DMA, 0},         // 1: DMA_0 (0x3F104)
  {WireBundle::DMA, 1},         // 2: DMA_1 (0x3F108)
  {WireBundle::TileControl, 0}, // 3: Tile_Ctrl (0x3F10C)
  {WireBundle::FIFO, 0},        // 4: FIFO_0 (0x3F110)
  {WireBundle::South, 0},       // 5: South_0 (0x3F114)
  {WireBundle::South, 1},       // 6: South_1 (0x3F118)
  {WireBundle::South, 2},       // 7: South_2 (0x3F11C)
  {WireBundle::South, 3},       // 8: South_3 (0x3F120)
  {WireBundle::South, 4},       // 9: South_4 (0x3F124)
  {WireBundle::South, 5},       // 10: South_5 (0x3F128)
  {WireBundle::West, 0},        // 11: West_0 (0x3F12C)
  {WireBundle::West, 1},        // 12: West_1 (0x3F130)
  {WireBundle::West, 2},        // 13: West_2 (0x3F134)
  {WireBundle::West, 3},        // 14: West_3 (0x3F138)
  {WireBundle::North, 0},       // 15: North_0 (0x3F13C)
  {WireBundle::North, 1},       // 16: North_1 (0x3F140)
  {WireBundle::North, 2},       // 17: North_2 (0x3F144)
  {WireBundle::North, 3},       // 18: North_3 (0x3F148)
  {WireBundle::East, 0},        // 19: East_0 (0x3F14C)
  {WireBundle::East, 1},        // 20: East_1 (0x3F150)
  {WireBundle::East, 2},        // 21: East_2 (0x3F154)
  {WireBundle::East, 3},        // 22: East_3 (0x3F158)
  {WireBundle::Trace, 0},       // 23: AIE_Trace (0x3F15C)
  {WireBundle::Trace, 1},       // 24: Mem_Trace (0x3F160)
};

// MemTile master port mapping (17 ports) - Destinations
// Computed from: (offset - 0xB0000) / 4
const SwitchAddressParser::PortMapping
    SwitchAddressParser::kMemTileMasterPortMap[17] = {
  {WireBundle::DMA, 0},         // 0: DMA0 (0xB0000)
  {WireBundle::DMA, 1},         // 1: DMA1 (0xB0004)
  {WireBundle::DMA, 2},         // 2: DMA2 (0xB0008)
  {WireBundle::DMA, 3},         // 3: DMA3 (0xB000C)
  {WireBundle::DMA, 4},         // 4: DMA4 (0xB0010)
  {WireBundle::DMA, 5},         // 5: DMA5 (0xB0014)
  {WireBundle::TileControl, 0}, // 6: Tile_Ctrl (0xB0018)
  {WireBundle::South, 0},       // 7: South0 (0xB001C)
  {WireBundle::South, 1},       // 8: South1 (0xB0020)
  {WireBundle::South, 2},       // 9: South2 (0xB0024)
  {WireBundle::South, 3},       // 10: South3 (0xB0028)
  {WireBundle::North, 0},       // 11: North0 (0xB002C)
  {WireBundle::North, 1},       // 12: North1 (0xB0030)
  {WireBundle::North, 2},       // 13: North2 (0xB0034)
  {WireBundle::North, 3},       // 14: North3 (0xB0038)
  {WireBundle::North, 4},       // 15: North4 (0xB003C)
  {WireBundle::North, 5},       // 16: North5 (0xB0040)
};

// MemTile slave port mapping (18 ports) - Sources
// Computed from: (offset - 0xB0100) / 4
const SwitchAddressParser::PortMapping
    SwitchAddressParser::kMemTileSlavePortMap[18] = {
  {WireBundle::DMA, 0},         // 0: DMA_0 (0xB0100)
  {WireBundle::DMA, 1},         // 1: DMA_1 (0xB0104)
  {WireBundle::DMA, 2},         // 2: DMA_2 (0xB0108)
  {WireBundle::DMA, 3},         // 3: DMA_3 (0xB010C)
  {WireBundle::DMA, 4},         // 4: DMA_4 (0xB0110)
  {WireBundle::DMA, 5},         // 5: DMA_5 (0xB0114)
  {WireBundle::TileControl, 0}, // 6: Tile_Ctrl (0xB0118)
  {WireBundle::South, 0},       // 7: South_0 (0xB011C)
  {WireBundle::South, 1},       // 8: South_1 (0xB0120)
  {WireBundle::South, 2},       // 9: South_2 (0xB0124)
  {WireBundle::South, 3},       // 10: South_3 (0xB0128)
  {WireBundle::North, 0},       // 11: North_0 (0xB012C)
  {WireBundle::North, 1},       // 12: North_1 (0xB0130)
  {WireBundle::North, 2},       // 13: North_2 (0xB0134)
  {WireBundle::North, 3},       // 14: North_3 (0xB0138)
  {WireBundle::North, 4},       // 15: North_4 (0xB013C)
  {WireBundle::North, 5},       // 16: North_5 (0xB0140)
  {WireBundle::Trace, 0},       // 17: Mem_Trace (0xB0144)
};

// Shim tile master port mapping (22 ports) - Destinations
// Computed from: (offset - 0x3F000) / 4
// Note: Shim tiles use same switchbox base as compute tiles but different port layout
const SwitchAddressParser::PortMapping
    SwitchAddressParser::kShimTileMasterPortMap[22] = {
  {WireBundle::TileControl, 0}, // 0: Tile_Ctrl (0x3F000)
  {WireBundle::FIFO, 0},        // 1: FIFO0 (0x3F004)
  {WireBundle::South, 0},       // 2: South0 (0x3F008)
  {WireBundle::South, 1},       // 3: South1 (0x3F00C)
  {WireBundle::South, 2},       // 4: South2 (0x3F010)
  {WireBundle::South, 3},       // 5: South3 (0x3F014)
  {WireBundle::South, 4},       // 6: South4 (0x3F018)
  {WireBundle::South, 5},       // 7: South5 (0x3F01C)
  {WireBundle::West, 0},        // 8: West0 (0x3F020)
  {WireBundle::West, 1},        // 9: West1 (0x3F024)
  {WireBundle::West, 2},        // 10: West2 (0x3F028)
  {WireBundle::West, 3},        // 11: West3 (0x3F02C)
  {WireBundle::North, 0},       // 12: North0 (0x3F030)
  {WireBundle::North, 1},       // 13: North1 (0x3F034)
  {WireBundle::North, 2},       // 14: North2 (0x3F038)
  {WireBundle::North, 3},       // 15: North3 (0x3F03C)
  {WireBundle::North, 4},       // 16: North4 (0x3F040)
  {WireBundle::North, 5},       // 17: North5 (0x3F044)
  {WireBundle::East, 0},        // 18: East0 (0x3F048)
  {WireBundle::East, 1},        // 19: East1 (0x3F04C)
  {WireBundle::East, 2},        // 20: East2 (0x3F050)
  {WireBundle::East, 3},        // 21: East3 (0x3F054)
};

// Shim tile slave port mapping (23 ports) - Sources
// Computed from: (offset - 0x3F100) / 4
const SwitchAddressParser::PortMapping
    SwitchAddressParser::kShimTileSlavePortMap[23] = {
  {WireBundle::TileControl, 0}, // 0: Tile_Ctrl (0x3F100)
  {WireBundle::FIFO, 0},        // 1: FIFO_0 (0x3F104)
  {WireBundle::South, 0},       // 2: South_0 (0x3F108)
  {WireBundle::South, 1},       // 3: South_1 (0x3F10C)
  {WireBundle::South, 2},       // 4: South_2 (0x3F110)
  {WireBundle::South, 3},       // 5: South_3 (0x3F114)
  {WireBundle::South, 4},       // 6: South_4 (0x3F118)
  {WireBundle::South, 5},       // 7: South_5 (0x3F11C)
  {WireBundle::South, 6},       // 8: South_6 (0x3F120)
  {WireBundle::South, 7},       // 9: South_7 (0x3F124)
  {WireBundle::West, 0},        // 10: West_0 (0x3F128)
  {WireBundle::West, 1},        // 11: West_1 (0x3F12C)
  {WireBundle::West, 2},        // 12: West_2 (0x3F130)
  {WireBundle::West, 3},        // 13: West_3 (0x3F134)
  {WireBundle::North, 0},       // 14: North_0 (0x3F138)
  {WireBundle::North, 1},       // 15: North_1 (0x3F13C)
  {WireBundle::North, 2},       // 16: North_2 (0x3F140)
  {WireBundle::North, 3},       // 17: North_3 (0x3F144)
  {WireBundle::East, 0},        // 18: East_0 (0x3F148)
  {WireBundle::East, 1},        // 19: East_1 (0x3F14C)
  {WireBundle::East, 2},        // 20: East_2 (0x3F150)
  {WireBundle::East, 3},        // 21: East_3 (0x3F154)
  {WireBundle::Trace, 0},       // 22: Trace (0x3F158)
};

//===----------------------------------------------------------------------===//
// SwitchAddressParser Implementation
//===----------------------------------------------------------------------===//

SwitchAddressParser::SwitchAddressParser(int numMemTileRows)
    : numMemTileRows_(numMemTileRows) {}

SwitchAddressParser::PortMapping
SwitchAddressParser::getMasterPortMapping(int portIndex, TileType tileType) const {
  if (tileType == TileType::MemoryTile) {
    if (portIndex < 0 || portIndex >= 17) {
      return {WireBundle::Core, 0};  // Invalid - return default
    }
    return kMemTileMasterPortMap[portIndex];
  } else if (tileType == TileType::ShimNOC || tileType == TileType::ShimPL) {
    // Shim tile
    if (portIndex < 0 || portIndex >= 22) {
      return {WireBundle::Core, 0};  // Invalid - return default
    }
    return kShimTileMasterPortMap[portIndex];
  } else {
    // Compute tile
    if (portIndex < 0 || portIndex >= 23) {
      return {WireBundle::Core, 0};  // Invalid - return default
    }
    return kMasterPortMap[portIndex];
  }
}

SwitchAddressParser::PortMapping
SwitchAddressParser::getSlavePortMapping(int portIndex, TileType tileType) const {
  if (tileType == TileType::MemoryTile) {
    if (portIndex < 0 || portIndex >= 18) {
      return {WireBundle::Core, 0};  // Invalid - return default
    }
    return kMemTileSlavePortMap[portIndex];
  } else if (tileType == TileType::ShimNOC || tileType == TileType::ShimPL) {
    // Shim tile
    if (portIndex < 0 || portIndex >= 23) {
      return {WireBundle::Core, 0};  // Invalid - return default
    }
    return kShimTileSlavePortMap[portIndex];
  } else {
    // Compute tile
    if (portIndex < 0 || portIndex >= 25) {
      return {WireBundle::Core, 0};  // Invalid - return default
    }
    return kSlavePortMap[portIndex];
  }
}

SwitchConnectionInfo
SwitchAddressParser::parseMasterConfig(uint32_t addr) const {
  SwitchConnectionInfo info;
  info.isValid = false;

  // Extract tile coordinates from address
  // AIE2 formula: base + (col * 32 + row_offset) * 0x100000
  uint32_t tileOffset = (addr >> kTileAddrShift) & 0xFFF;
  uint32_t regOffset = addr & ((1 << kTileAddrShift) - 1);

  int column = tileOffset / 32;
  int rowPart = tileOffset % 32;

  // Check if this is a MemTile master config register
  if (regOffset >= kMemTileMasterConfigBase && regOffset <= kMemTileMasterConfigEnd) {
    // Calculate master port index for MemTile
    int portOffset = regOffset - kMemTileMasterConfigBase;
    if (portOffset % kRegisterSize != 0) {
      return info;  // Not aligned to register boundary
    }

    int masterPortIndex = portOffset / kRegisterSize;
    if (masterPortIndex < 0 || masterPortIndex >= 17) {
      return info;  // Out of range
    }

    // Determine MemTile row
    if (rowPart >= 1 && rowPart <= numMemTileRows_) {
      info.tileType = TileType::MemoryTile;
      info.row = rowPart - 1;  // MemTile rows are 0-based internally
      info.isValid = true;
      info.column = column;
      info.masterPortIndex = masterPortIndex;

      // Get destination port mapping for MemTile
      PortMapping destMapping = getMasterPortMapping(masterPortIndex, TileType::MemoryTile);
      info.destBundle = destMapping.bundle;
      info.destChannel = destMapping.channel;

      return info;
    }
    return info;  // Invalid row for MemTile
  }

  // Check if this is a compute tile master config register
  if (regOffset >= kMasterConfigBase && regOffset <= kMasterConfigEnd) {
    // Calculate master port index
    int portOffset = regOffset - kMasterConfigBase;
    if (portOffset % kRegisterSize != 0) {
      return info;  // Not aligned to register boundary
    }

    int masterPortIndex = portOffset / kRegisterSize;

    // Determine tile type and row
    // For compute tiles: rowPart = numMemTileRows + 1 + actualRow
    // For shim tiles: rowPart = 0
    if (rowPart == 0) {
      // Shim tile
      // Shim tiles have 22 master ports (0-21)
      if (masterPortIndex < 0 || masterPortIndex >= 22) {
        return info;  // Out of range for Shim tile
      }
      info.tileType = TileType::ShimNOC;
      info.row = 0;
    } else if (rowPart <= numMemTileRows_) {
      // Memory tile - but we're in compute tile register range, invalid
      return info;
    } else {
      // Compute tile
      // Compute tiles have 23 master ports (0-22)
      if (masterPortIndex < 0 || masterPortIndex >= 23) {
        return info;  // Out of range for Compute tile
      }
      info.tileType = TileType::Compute;
      info.row = rowPart - (numMemTileRows_ + 1);
    }

    // Valid master config register
    info.isValid = true;
    info.column = column;
    info.masterPortIndex = masterPortIndex;

    // Get destination port mapping
    PortMapping destMapping = getMasterPortMapping(masterPortIndex, info.tileType);
    info.destBundle = destMapping.bundle;
    info.destChannel = destMapping.channel;

    return info;
  }

  return info;
}

bool SwitchAddressParser::isSwitchboxAddress(uint32_t addr) const {
  uint32_t regOffset = addr & ((1 << kTileAddrShift) - 1);

  // Check compute tile master config range
  if (regOffset >= kMasterConfigBase && regOffset <= kMasterConfigEnd) {
    return true;
  }

  // Check compute tile slave config range
  if (regOffset >= kSlaveConfigBase && regOffset <= kSlaveConfigEnd) {
    return true;
  }

  // Check MemTile master config range
  if (regOffset >= kMemTileMasterConfigBase && regOffset <= kMemTileMasterConfigEnd) {
    return true;
  }

  // Check MemTile slave config range
  if (regOffset >= kMemTileSlaveConfigBase && regOffset <= kMemTileSlaveConfigEnd) {
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// SwitchboxAccumulator::SwitchboxKey Implementation
//===----------------------------------------------------------------------===//

bool SwitchboxAccumulator::SwitchboxKey::operator<(
    const SwitchboxKey &other) const {
  if (col != other.col) return col < other.col;
  if (row != other.row) return row < other.row;
  return static_cast<int>(type) < static_cast<int>(other.type);
}

//===----------------------------------------------------------------------===//
// SwitchboxAccumulator Implementation
//===----------------------------------------------------------------------===//

std::optional<SwitchConnectionInfo>
SwitchboxAccumulator::addMasterWrite(uint32_t addr, uint32_t value,
                                     const SwitchAddressParser &parser) {
  // Parse the address to see if it's a master config register
  SwitchConnectionInfo connInfo = parser.parseMasterConfig(addr);
  if (!connInfo.isValid) {
    return std::nullopt;
  }

  // Extract configuration bits from the value
  connInfo.masterEnable = SwitchFieldExtractor::getMasterEnable(value);
  connInfo.packetMode = SwitchFieldExtractor::getPacketEnable(value);
  connInfo.dropHeader = SwitchFieldExtractor::getDropHeader(value);
  connInfo.slavePortId = SwitchFieldExtractor::getSlavePortId(value);

  // Only process enabled connections in circuit mode for now
  if (!connInfo.masterEnable) {
    return std::nullopt;
  }

  if (connInfo.packetMode) {
    // Packet mode - extract arbiter config
    connInfo.arbiterId = SwitchFieldExtractor::getArbiterId(value);
    connInfo.mselEnable = SwitchFieldExtractor::getMselEnable(value);
    // For now, skip packet mode connections
    return std::nullopt;
  }

  // Circuit mode - decode slave port ID to source bundle/channel
  SwitchAddressParser::PortMapping sourceMapping =
      parser.getSlavePortMapping(connInfo.slavePortId, connInfo.tileType);
  connInfo.sourceBundle = sourceMapping.bundle;
  connInfo.sourceChannel = sourceMapping.channel;

  // Add this connection to the switchbox configuration
  SwitchboxKey key{connInfo.column, connInfo.row, connInfo.tileType};

  ParsedSwitchboxConfig &config = switchboxes_[key];
  config.column = connInfo.column;
  config.row = connInfo.row;
  config.tileType = connInfo.tileType;

  // Create and add the connection
  ParsedSwitchboxConfig::Connection conn;
  conn.sourceBundle = connInfo.sourceBundle;
  conn.sourceChannel = connInfo.sourceChannel;
  conn.destBundle = connInfo.destBundle;
  conn.destChannel = connInfo.destChannel;
  conn.isPacketMode = connInfo.packetMode;

  config.connections.push_back(conn);

  return connInfo;
}

ParsedSwitchboxConfig
SwitchboxAccumulator::getSwitchboxConfig(const SwitchboxKey &key) const {
  auto it = switchboxes_.find(key);
  if (it != switchboxes_.end()) {
    return it->second;
  }
  // Return empty config
  ParsedSwitchboxConfig empty;
  empty.column = key.col;
  empty.row = key.row;
  empty.tileType = key.type;
  return empty;
}

std::map<SwitchboxAccumulator::SwitchboxKey, ParsedSwitchboxConfig>
SwitchboxAccumulator::getAll() const {
  return switchboxes_;
}

//===----------------------------------------------------------------------===//
// SwitchboxPrettyPrinter Implementation
//===----------------------------------------------------------------------===//

const char* SwitchboxPrettyPrinter::wireBundleToString(WireBundle bundle) {
  switch (bundle) {
    case WireBundle::Core: return "Core";
    case WireBundle::DMA: return "DMA";
    case WireBundle::FIFO: return "FIFO";
    case WireBundle::South: return "South";
    case WireBundle::West: return "West";
    case WireBundle::North: return "North";
    case WireBundle::East: return "East";
    case WireBundle::Trace: return "Trace";
    case WireBundle::TileControl: return "TileControl";
  }
  return "Unknown";
}

void SwitchboxPrettyPrinter::printConnectionAsComment(
    llvm::raw_ostream &os, const SwitchConnectionInfo &conn) {
  os << "// Switchbox(" << conn.column << ", " << conn.row << "): "
     << wireBundleToString(conn.sourceBundle) << ":" << conn.sourceChannel
     << " -> "
     << wireBundleToString(conn.destBundle) << ":" << conn.destChannel;

  if (conn.packetMode) {
    os << " [packet mode, arbiter=" << conn.arbiterId << "]";
  }

  os << "\n";
}

void SwitchboxPrettyPrinter::printSwitchboxConfig(
    llvm::raw_ostream &os, const ParsedSwitchboxConfig &config) {
  if (!config.hasConnections()) {
    return;
  }

  os << "// Switchbox configuration for tile(" << config.column << ", "
     << config.row << "):\n";

  for (const auto &conn : config.connections) {
    os << "//   " << wireBundleToString(conn.sourceBundle) << ":"
       << conn.sourceChannel << " -> "
       << wireBundleToString(conn.destBundle) << ":" << conn.destChannel;

    if (conn.isPacketMode) {
      os << " [packet]";
    }
    os << "\n";
  }
}
