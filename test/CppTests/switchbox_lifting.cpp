//===- switchbox_lifting.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Unit tests for Switchbox lifting infrastructure:
// - SwitchAddressParser: address decoding to identify switchbox registers
// - SwitchboxAccumulator: accumulating register writes into switchbox configs
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Util/AIESwitchboxLifting.h"

#include "mlir/IR/MLIRContext.h"

#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace xilinx::AIE;
using namespace mlir;

//===----------------------------------------------------------------------===//
// SwitchFieldExtractor Tests
//===----------------------------------------------------------------------===//

void test_switch_field_extraction() {
  std::cout << "Test: SwitchFieldExtractor - Master Config Fields\n";

  // Master config register format:
  // [31]    Master Enable
  // [30]    Packet Enable
  // [7]     Drop Header
  // [4:0]   Slave Port ID

  uint32_t value = (1u << 31) |  // Master enable
                   (1u << 30) |  // Packet enable
                   (1u << 7) |   // Drop header
                   0x15;         // Slave port = 21

  bool masterEn = SwitchFieldExtractor::getMasterEnable(value);
  bool packetEn = SwitchFieldExtractor::getPacketEnable(value);
  bool dropHdr = SwitchFieldExtractor::getDropHeader(value);
  uint8_t slavePort = SwitchFieldExtractor::getSlavePortId(value);

  if (!masterEn) throw std::runtime_error("getMasterEnable failed");
  if (!packetEn) throw std::runtime_error("getPacketEnable failed");
  if (!dropHdr) throw std::runtime_error("getDropHeader failed");
  if (slavePort != 0x15) {
    throw std::runtime_error("getSlavePortId failed: got " +
                            std::to_string(slavePort) + ", expected 21");
  }

  std::cout << "  ✓ Switch field extraction works\n";
}

void test_switch_packet_mode_fields() {
  std::cout << "Test: SwitchFieldExtractor - Packet Mode Fields\n";

  // Packet mode fields:
  // [6:3] MSEL Enable
  // [2:0] Arbiter ID

  uint32_t value = (0xA << 3) |  // MSEL = 0xA
                   0x5;           // Arbiter = 5

  uint8_t msel = SwitchFieldExtractor::getMselEnable(value);
  uint8_t arbiter = SwitchFieldExtractor::getArbiterId(value);

  if (msel != 0xA) throw std::runtime_error("getMselEnable failed");
  if (arbiter != 0x5) throw std::runtime_error("getArbiterId failed");

  std::cout << "  ✓ Packet mode field extraction works\n";
}

//===----------------------------------------------------------------------===//
// SwitchAddressParser Tests
//===----------------------------------------------------------------------===//

void test_switch_address_parser_master_config() {
  std::cout << "Test: SwitchAddressParser - Master Config Register\n";

  SwitchAddressParser parser(1);

  // Master config base: 0x3F000
  // Port 5 (South0): 0x3F000 + (5 * 4) = 0x3F014
  // Tile (2, 3): base 0x300000
  // Full address: 0x300000 + 0x3F014 = 0x33F014

  uint32_t addr = 0x300000 + 0x3F000 + (5 * 4);

  // Create a connection: South0 -> Core0
  // Master enable, circuit mode, slave port 0 (Core)
  SwitchConnectionInfo conn = parser.parseMasterConfig(addr);

  if (!conn.isValid) {
    throw std::runtime_error("Failed to identify master config register");
  }
  if (conn.column != 3) {
    throw std::runtime_error("Wrong column: got " + std::to_string(conn.column));
  }
  if (conn.row != 2) {
    throw std::runtime_error("Wrong row: got " + std::to_string(conn.row));
  }
  if (conn.masterPortIndex != 5) {
    throw std::runtime_error("Wrong master port index");
  }
  // Port 5 should map to South, 0
  if (conn.destBundle != WireBundle::South) {
    throw std::runtime_error("Wrong dest bundle");
  }
  if (conn.destChannel != 0) {
    throw std::runtime_error("Wrong dest channel");
  }

  std::cout << "  ✓ Master config address parsing works\n";
}

void test_switch_address_parser_port_mapping() {
  std::cout << "Test: SwitchAddressParser - Port Mapping Tables\n";

  SwitchAddressParser parser(1);

  // Test master port 0 (Core) at tile (0,2)
  uint32_t addr = 0x200000 + 0x3F000;  // Port 0
  SwitchConnectionInfo conn = parser.parseMasterConfig(addr);

  if (conn.destBundle != WireBundle::Core || conn.destChannel != 0) {
    throw std::runtime_error("Master port 0 should map to Core:0");
  }

  // Test master port 1 (DMA0) at tile (0,2)
  addr = 0x200000 + 0x3F004;  // Port 1
  conn = parser.parseMasterConfig(addr);

  if (conn.destBundle != WireBundle::DMA || conn.destChannel != 0) {
    throw std::runtime_error("Master port 1 should map to DMA:0");
  }

  // Test master port 13 (North0) at tile (0,2)
  addr = 0x200000 + 0x3F034;  // Port 13
  conn = parser.parseMasterConfig(addr);

  if (conn.destBundle != WireBundle::North || conn.destChannel != 0) {
    throw std::runtime_error("Master port 13 should map to North:0");
  }

  std::cout << "  ✓ Port mapping tables work correctly\n";
}

void test_switch_address_parser_non_switch() {
  std::cout << "Test: SwitchAddressParser - Non-Switch Address\n";

  SwitchAddressParser parser(1);

  // Test a random address that's not a switchbox register
  uint32_t addr = 0x12345678;

  if (parser.isSwitchboxAddress(addr)) {
    throw std::runtime_error("Incorrectly identified non-switch address");
  }

  std::cout << "  ✓ Non-switch address correctly rejected\n";
}

//===----------------------------------------------------------------------===//
// SwitchboxAccumulator Tests
//===----------------------------------------------------------------------===//

void test_switchbox_accumulator_single_connection() {
  std::cout << "Test: SwitchboxAccumulator - Single Connection\n";

  SwitchboxAccumulator accum;
  SwitchAddressParser parser(1);

  // Tile (1, 2): connect DMA0 (master port 1) to Core (slave port 0)
  uint32_t tileBase = 0x200000;  // Tile (1,2)
  uint32_t addr = tileBase + 0x3F000 + (1 * 4);  // Master port 1 (DMA0)
  uint32_t value = (1u << 31) | 0;  // Enable, source = Core (slave 0)

  auto result = accum.addMasterWrite(addr, value, parser);

  if (!result.has_value()) {
    throw std::runtime_error("Failed to add master write");
  }

  SwitchConnectionInfo conn = *result;
  if (!conn.isValid) throw std::runtime_error("Invalid connection");
  if (conn.column != 1) throw std::runtime_error("Wrong column");
  if (conn.row != 2) throw std::runtime_error("Wrong row");
  if (!conn.masterEnable) throw std::runtime_error("Master not enabled");

  std::cout << "  ✓ Single connection accumulation works\n";
}

void test_switchbox_accumulator_multiple_connections() {
  std::cout << "Test: SwitchboxAccumulator - Multiple Connections\n";

  SwitchboxAccumulator accum;
  SwitchAddressParser parser(1);

  uint32_t tileBase = 0x100000;  // Tile (1,1)

  // Add multiple connections for the same switchbox
  // Connection 1: DMA0 <- Core
  accum.addMasterWrite(tileBase + 0x3F004, (1u << 31) | 0, parser);

  // Connection 2: North0 <- DMA0
  accum.addMasterWrite(tileBase + 0x3F034, (1u << 31) | 1, parser);

  // Connection 3: South0 <- DMA1
  accum.addMasterWrite(tileBase + 0x3F014, (1u << 31) | 2, parser);

  // Get the complete switchbox config
  SwitchboxAccumulator::SwitchboxKey key{1, 1, TileType::Compute};
  ParsedSwitchboxConfig config = accum.getSwitchboxConfig(key);

  if (config.connectionCount() != 3) {
    throw std::runtime_error("Expected 3 connections, got " +
                            std::to_string(config.connectionCount()));
  }

  std::cout << "  ✓ Multiple connection accumulation works\n";
}

void test_switchbox_accumulator_multiple_tiles() {
  std::cout << "Test: SwitchboxAccumulator - Multiple Tiles\n";

  SwitchboxAccumulator accum;
  SwitchAddressParser parser(1);

  // Add connections for different tiles
  // Tile (0, 2)
  accum.addMasterWrite(0x200000 + 0x3F000, (1u << 31) | 0, parser);

  // Tile (1, 2)
  accum.addMasterWrite(0x300000 + 0x3F000, (1u << 31) | 1, parser);

  // Tile (2, 3)
  accum.addMasterWrite(0x500000 + 0x3F000, (1u << 31) | 2, parser);

  if (accum.switchboxCount() != 3) {
    throw std::runtime_error("Expected 3 switchboxes, got " +
                            std::to_string(accum.switchboxCount()));
  }

  std::cout << "  ✓ Multiple tile tracking works\n";
}

void test_switchbox_accumulator_get_all() {
  std::cout << "Test: SwitchboxAccumulator - Get All Switchboxes\n";

  SwitchboxAccumulator accum;
  SwitchAddressParser parser(1);

  // Add connections for 2 tiles
  accum.addMasterWrite(0x200000 + 0x3F000, (1u << 31) | 0, parser);
  accum.addMasterWrite(0x200000 + 0x3F004, (1u << 31) | 1, parser);
  accum.addMasterWrite(0x300000 + 0x3F000, (1u << 31) | 2, parser);

  auto allConfigs = accum.getAll();

  if (allConfigs.size() != 2) {
    throw std::runtime_error("Expected 2 switchbox configs");
  }

  // Verify first switchbox has 2 connections
  bool foundTileWith2Conns = false;
  for (const auto &[key, config] : allConfigs) {
    if (config.connectionCount() == 2) {
      foundTileWith2Conns = true;
      break;
    }
  }

  if (!foundTileWith2Conns) {
    throw std::runtime_error("Expected to find tile with 2 connections");
  }

  std::cout << "  ✓ Get all switchboxes works\n";
}

void test_switchbox_accumulator_packet_mode() {
  std::cout << "Test: SwitchboxAccumulator - Packet Mode Connection\n";

  SwitchboxAccumulator accum;
  SwitchAddressParser parser(1);

  // Packet mode connection: enable bit 30
  uint32_t addr = 0x200000 + 0x3F000;
  uint32_t value = (1u << 31) |  // Master enable
                   (1u << 30) |  // Packet mode
                   0x5;          // Arbiter/MSEL config

  auto result = accum.addMasterWrite(addr, value, parser);

  if (!result.has_value()) {
    throw std::runtime_error("Failed to add packet mode write");
  }

  if (!result->packetMode) {
    throw std::runtime_error("Packet mode not detected");
  }

  std::cout << "  ✓ Packet mode connection works\n";
}

//===----------------------------------------------------------------------===//
// Shim Tile Tests
//===----------------------------------------------------------------------===//

void test_shim_tile_detection() {
  std::cout << "Test: SwitchAddressParser - Shim Tile Detection\n";

  SwitchAddressParser parser(1);

  // Shim tiles are at row 0
  // Tile (2, 0): tileOffset = 2 * 32 + 0 = 64 (0x40)
  // Base address: 0x40 << 20 = 0x4000000
  // Master port 0 (Tile_Ctrl): base + 0x3F000
  uint32_t addr = 0x4000000 + 0x3F000;

  SwitchConnectionInfo conn = parser.parseMasterConfig(addr);

  if (!conn.isValid) {
    throw std::runtime_error("Failed to identify shim tile switchbox register");
  }
  if (conn.column != 2) {
    throw std::runtime_error("Wrong column: got " + std::to_string(conn.column) + ", expected 2");
  }
  if (conn.row != 0) {
    throw std::runtime_error("Wrong row: got " + std::to_string(conn.row) + ", expected 0");
  }
  if (conn.tileType != TileType::ShimNOC) {
    throw std::runtime_error("Wrong tile type: expected ShimNOC");
  }

  std::cout << "  ✓ Shim tile detection works\n";
}

void test_shim_tile_port_mapping() {
  std::cout << "Test: SwitchAddressParser - Shim Tile Port Mapping\n";

  SwitchAddressParser parser(1);

  // Shim tile (1, 0)
  uint32_t tileBase = 0x2000000;  // Column 1, row 0

  // Test master port 0 (Tile_Ctrl) for Shim
  uint32_t addr = tileBase + 0x3F000;  // Port 0
  SwitchConnectionInfo conn = parser.parseMasterConfig(addr);

  if (conn.destBundle != WireBundle::TileControl || conn.destChannel != 0) {
    throw std::runtime_error("Shim master port 0 should map to TileControl:0");
  }

  // Test master port 1 (FIFO0) for Shim
  addr = tileBase + 0x3F004;  // Port 1
  conn = parser.parseMasterConfig(addr);

  if (conn.destBundle != WireBundle::FIFO || conn.destChannel != 0) {
    throw std::runtime_error("Shim master port 1 should map to FIFO:0");
  }

  // Test master port 2 (South0) for Shim
  addr = tileBase + 0x3F008;  // Port 2
  conn = parser.parseMasterConfig(addr);

  if (conn.destBundle != WireBundle::South || conn.destChannel != 0) {
    throw std::runtime_error("Shim master port 2 should map to South:0");
  }

  // Test master port 12 (North0) for Shim
  addr = tileBase + 0x3F030;  // Port 12
  conn = parser.parseMasterConfig(addr);

  if (conn.destBundle != WireBundle::North || conn.destChannel != 0) {
    throw std::runtime_error("Shim master port 12 should map to North:0");
  }

  std::cout << "  ✓ Shim tile port mapping works correctly\n";
}

void test_shim_tile_connection() {
  std::cout << "Test: SwitchboxAccumulator - Shim Tile Connection\n";

  SwitchboxAccumulator accum;
  SwitchAddressParser parser(1);

  // Shim tile (3, 0): connect North0 (master port 12) to South_0 (slave port 2)
  uint32_t tileBase = 0x6000000;  // Column 3, row 0
  uint32_t addr = tileBase + 0x3F000 + (12 * 4);  // Master port 12 (North0)
  uint32_t value = (1u << 31) | 2;  // Enable, source = South_0 (slave 2)

  auto result = accum.addMasterWrite(addr, value, parser);

  if (!result.has_value()) {
    throw std::runtime_error("Failed to add shim tile master write");
  }

  SwitchConnectionInfo conn = *result;
  if (!conn.isValid) throw std::runtime_error("Invalid connection");
  if (conn.column != 3) throw std::runtime_error("Wrong column");
  if (conn.row != 0) throw std::runtime_error("Wrong row");
  if (conn.tileType != TileType::ShimNOC) throw std::runtime_error("Wrong tile type");
  if (!conn.masterEnable) throw std::runtime_error("Master not enabled");

  // Verify the connection was accumulated
  SwitchboxAccumulator::SwitchboxKey key{3, 0, TileType::ShimNOC};
  ParsedSwitchboxConfig config = accum.getSwitchboxConfig(key);

  if (config.connectionCount() != 1) {
    throw std::runtime_error("Expected 1 connection in shim switchbox");
  }

  std::cout << "  ✓ Shim tile connection accumulation works\n";
}

//===----------------------------------------------------------------------===//
// Main Test Runner
//===----------------------------------------------------------------------===//

int main() {
  try {
    std::cout << "\n=== Switchbox Lifting Unit Tests ===\n\n";

    // SwitchFieldExtractor tests
    test_switch_field_extraction();
    test_switch_packet_mode_fields();

    // SwitchAddressParser tests
    test_switch_address_parser_master_config();
    test_switch_address_parser_port_mapping();
    test_switch_address_parser_non_switch();

    // SwitchboxAccumulator tests
    test_switchbox_accumulator_single_connection();
    test_switchbox_accumulator_multiple_connections();
    test_switchbox_accumulator_multiple_tiles();
    test_switchbox_accumulator_get_all();
    test_switchbox_accumulator_packet_mode();

    // Shim tile tests
    test_shim_tile_detection();
    test_shim_tile_port_mapping();
    test_shim_tile_connection();

    std::cout << "\n✓ All switchbox lifting tests passed!\n";
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "\n✗ Test failed: " << e.what() << "\n";
    return 1;
  }
}
