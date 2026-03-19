//===- flow_reconstruction.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Unit tests for Flow Reconstruction from switchbox connections:
// - FlowReconstructionGraph: graph building and flow tracing
// - ReconstructedFlow: reconstructed end-to-end data paths
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Util/AIEFlowReconstruction.h"
#include "aie/Dialect/AIE/Util/AIESwitchboxLifting.h"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace xilinx::AIE;

//===----------------------------------------------------------------------===//
// Test Utilities
//===----------------------------------------------------------------------===//

void assertFlowExists(const std::vector<ReconstructedFlow> &flows,
                      int srcCol, int srcRow, WireBundle srcBundle, int srcCh,
                      int dstCol, int dstRow, WireBundle dstBundle, int dstCh,
                      const std::string &testName) {
  for (const auto &f : flows) {
    if (f.sourceCol == srcCol && f.sourceRow == srcRow &&
        f.sourceBundle == srcBundle && f.sourceChannel == srcCh &&
        f.destCol == dstCol && f.destRow == dstRow &&
        f.destBundle == dstBundle && f.destChannel == dstCh) {
      return;  // Found
    }
  }
  throw std::runtime_error(testName + ": Expected flow not found");
}

//===----------------------------------------------------------------------===//
// Basic Graph Construction Tests
//===----------------------------------------------------------------------===//

void test_empty_graph() {
  std::cout << "Test: Empty Graph\n";

  FlowReconstructionGraph graph;

  if (graph.hasConnections()) {
    throw std::runtime_error("Empty graph should have no connections");
  }

  auto flows = graph.reconstructFlows();
  if (!flows.empty()) {
    throw std::runtime_error("Empty graph should produce no flows");
  }

  std::cout << "  PASS\n";
}

void test_single_connection_same_tile() {
  std::cout << "Test: Single Connection Same Tile (DMA to Core)\n";

  FlowReconstructionGraph graph;

  // Tile (0, 2): DMA:0 -> Core:0
  graph.addConnection(0, 2, WireBundle::DMA, 0, WireBundle::Core, 0);

  auto flows = graph.reconstructFlows();

  if (flows.size() != 1) {
    throw std::runtime_error("Expected 1 flow, got " +
                             std::to_string(flows.size()));
  }

  assertFlowExists(flows, 0, 2, WireBundle::DMA, 0,
                         0, 2, WireBundle::Core, 0, "Single connection");

  std::cout << "  PASS\n";
}

void test_two_hop_vertical_flow() {
  std::cout << "Test: Two-Hop Vertical Flow (DMA through intermediate tile)\n";

  FlowReconstructionGraph graph;

  // Tile (0, 0): DMA:0 -> North:0
  graph.addConnection(0, 0, WireBundle::DMA, 0, WireBundle::North, 0);

  // Tile (0, 1): South:0 -> North:0 (pass-through)
  graph.addConnection(0, 1, WireBundle::South, 0, WireBundle::North, 0);

  // Tile (0, 2): South:0 -> DMA:0
  graph.addConnection(0, 2, WireBundle::South, 0, WireBundle::DMA, 0);

  auto flows = graph.reconstructFlows();

  if (flows.size() != 1) {
    throw std::runtime_error("Expected 1 flow, got " +
                             std::to_string(flows.size()));
  }

  assertFlowExists(flows, 0, 0, WireBundle::DMA, 0,
                         0, 2, WireBundle::DMA, 0, "Two-hop flow");

  // Check hop count
  if (flows[0].hopCount != 2) {
    throw std::runtime_error("Expected 2 hops, got " +
                             std::to_string(flows[0].hopCount));
  }

  std::cout << "  PASS\n";
}

void test_horizontal_flow() {
  std::cout << "Test: Horizontal Flow (West to East)\n";

  FlowReconstructionGraph graph;

  // Tile (0, 1): DMA:0 -> East:0
  graph.addConnection(0, 1, WireBundle::DMA, 0, WireBundle::East, 0);

  // Tile (1, 1): West:0 -> Core:0
  graph.addConnection(1, 1, WireBundle::West, 0, WireBundle::Core, 0);

  auto flows = graph.reconstructFlows();

  if (flows.size() != 1) {
    throw std::runtime_error("Expected 1 flow, got " +
                             std::to_string(flows.size()));
  }

  assertFlowExists(flows, 0, 1, WireBundle::DMA, 0,
                         1, 1, WireBundle::Core, 0, "Horizontal flow");

  std::cout << "  PASS\n";
}

//===----------------------------------------------------------------------===//
// Broadcast Pattern Tests
//===----------------------------------------------------------------------===//

void test_broadcast_same_tile() {
  std::cout << "Test: Broadcast Within Same Tile\n";

  FlowReconstructionGraph graph;

  // Tile (0, 2): DMA:0 -> Core:0
  graph.addConnection(0, 2, WireBundle::DMA, 0, WireBundle::Core, 0);

  // Tile (0, 2): DMA:0 -> FIFO:0 (same source, different dest)
  graph.addConnection(0, 2, WireBundle::DMA, 0, WireBundle::FIFO, 0);

  auto flows = graph.reconstructFlows();

  if (flows.size() != 2) {
    throw std::runtime_error("Expected 2 flows (broadcast), got " +
                             std::to_string(flows.size()));
  }

  assertFlowExists(flows, 0, 2, WireBundle::DMA, 0,
                         0, 2, WireBundle::Core, 0, "Broadcast to Core");
  assertFlowExists(flows, 0, 2, WireBundle::DMA, 0,
                         0, 2, WireBundle::FIFO, 0, "Broadcast to FIFO");

  std::cout << "  PASS\n";
}

void test_broadcast_to_multiple_tiles() {
  std::cout << "Test: Broadcast to Multiple Tiles\n";

  FlowReconstructionGraph graph;

  // Tile (1, 1): DMA:0 -> North:0 AND East:0
  graph.addConnection(1, 1, WireBundle::DMA, 0, WireBundle::North, 0);
  graph.addConnection(1, 1, WireBundle::DMA, 0, WireBundle::East, 0);

  // Tile (1, 2): South:0 -> DMA:0
  graph.addConnection(1, 2, WireBundle::South, 0, WireBundle::DMA, 0);

  // Tile (2, 1): West:0 -> Core:0
  graph.addConnection(2, 1, WireBundle::West, 0, WireBundle::Core, 0);

  auto flows = graph.reconstructFlows();

  if (flows.size() != 2) {
    throw std::runtime_error("Expected 2 flows (broadcast), got " +
                             std::to_string(flows.size()));
  }

  assertFlowExists(flows, 1, 1, WireBundle::DMA, 0,
                         1, 2, WireBundle::DMA, 0, "Broadcast to (1,2)");
  assertFlowExists(flows, 1, 1, WireBundle::DMA, 0,
                         2, 1, WireBundle::Core, 0, "Broadcast to (2,1)");

  std::cout << "  PASS\n";
}

//===----------------------------------------------------------------------===//
// Multiple Independent Flows Tests
//===----------------------------------------------------------------------===//

void test_multiple_independent_flows() {
  std::cout << "Test: Multiple Independent Flows\n";

  FlowReconstructionGraph graph;

  // Flow 1: Tile (0, 1) DMA:0 -> Tile (0, 2) Core:0
  graph.addConnection(0, 1, WireBundle::DMA, 0, WireBundle::North, 0);
  graph.addConnection(0, 2, WireBundle::South, 0, WireBundle::Core, 0);

  // Flow 2: Tile (1, 1) Core:0 -> Tile (1, 1) DMA:0 (local)
  graph.addConnection(1, 1, WireBundle::Core, 0, WireBundle::DMA, 0);

  auto flows = graph.reconstructFlows();

  if (flows.size() != 2) {
    throw std::runtime_error("Expected 2 independent flows, got " +
                             std::to_string(flows.size()));
  }

  assertFlowExists(flows, 0, 1, WireBundle::DMA, 0,
                         0, 2, WireBundle::Core, 0, "Flow 1");
  assertFlowExists(flows, 1, 1, WireBundle::Core, 0,
                         1, 1, WireBundle::DMA, 0, "Flow 2");

  std::cout << "  PASS\n";
}

//===----------------------------------------------------------------------===//
// Integration with SwitchboxAccumulator Tests
//===----------------------------------------------------------------------===//

void test_integration_with_parsed_config() {
  std::cout << "Test: Integration with ParsedSwitchboxConfig\n";

  FlowReconstructionGraph graph;

  // Create a parsed switchbox config
  ParsedSwitchboxConfig config;
  config.column = 0;
  config.row = 2;

  // Add connections
  ParsedSwitchboxConfig::Connection conn1;
  conn1.sourceBundle = WireBundle::DMA;
  conn1.sourceChannel = 0;
  conn1.destBundle = WireBundle::North;
  conn1.destChannel = 0;
  config.connections.push_back(conn1);

  ParsedSwitchboxConfig::Connection conn2;
  conn2.sourceBundle = WireBundle::South;
  conn2.sourceChannel = 1;
  conn2.destBundle = WireBundle::Core;
  conn2.destChannel = 0;
  config.connections.push_back(conn2);

  // Add config to graph
  graph.addSwitchboxConfig(config);

  // Verify graph state
  if (!graph.hasConnections()) {
    throw std::runtime_error("Graph should have connections");
  }

  auto tiles = graph.getActiveTiles();
  if (tiles.size() != 1) {
    throw std::runtime_error("Expected 1 active tile");
  }
  if (tiles.find({0, 2}) == tiles.end()) {
    throw std::runtime_error("Tile (0,2) should be active");
  }

  std::cout << "  PASS\n";
}

void test_flow_from_accumulator() {
  std::cout << "Test: Flow Reconstruction from SwitchboxAccumulator\n";

  SwitchboxAccumulator accum;
  SwitchAddressParser parser(1);

  // Simulate a two-hop flow: (0,2) DMA:0 -> (0,3) DMA:0
  // Tile (0, 2): DMA:0 -> North:0
  // Master port 13 (North0) <- slave port 1 (DMA0)
  uint32_t tile02Base = 0x200000;  // Tile (0,2) with numMemTileRows=1
  accum.addMasterWrite(tile02Base + 0x3F034, (1u << 31) | 1, parser);

  // Tile (0, 3): South:0 -> DMA:0
  // Master port 1 (DMA0) <- slave port 5 (South0)
  uint32_t tile03Base = 0x300000;  // Tile (0,3) with numMemTileRows=1
  accum.addMasterWrite(tile03Base + 0x3F004, (1u << 31) | 5, parser);

  // Build reconstruction graph from accumulator
  FlowReconstructionGraph graph;
  for (const auto &[key, config] : accum.getAll()) {
    graph.addSwitchboxConfig(config);
  }

  // Reconstruct flows
  auto flows = graph.reconstructFlows();

  if (flows.size() != 1) {
    throw std::runtime_error("Expected 1 flow, got " +
                             std::to_string(flows.size()));
  }

  // The flow should be: (0,2) DMA:0 -> (0,3) DMA:0
  assertFlowExists(flows, 0, 2, WireBundle::DMA, 0,
                         0, 3, WireBundle::DMA, 0, "Accumulator flow");

  std::cout << "  PASS\n";
}

//===----------------------------------------------------------------------===//
// Edge Cases Tests
//===----------------------------------------------------------------------===//

void test_loopback_same_port() {
  std::cout << "Test: Loopback on Same Tile (Core:0 -> Core:0)\n";

  FlowReconstructionGraph graph;

  // Self-loop: Core:0 output connects back to Core:0 input
  // This is a valid configuration in AIE
  graph.addConnection(0, 2, WireBundle::Core, 0, WireBundle::Core, 0);

  auto flows = graph.reconstructFlows();

  if (flows.size() != 1) {
    throw std::runtime_error("Expected 1 loopback flow, got " +
                             std::to_string(flows.size()));
  }

  assertFlowExists(flows, 0, 2, WireBundle::Core, 0,
                         0, 2, WireBundle::Core, 0, "Loopback");

  std::cout << "  PASS\n";
}

void test_dead_end_no_sink() {
  std::cout << "Test: Dead End (no sink connected)\n";

  FlowReconstructionGraph graph;

  // Source exists but no sink is connected
  graph.addConnection(0, 2, WireBundle::DMA, 0, WireBundle::North, 0);
  // No tile (0, 3) connections - dead end

  auto flows = graph.reconstructFlows();

  // Should produce no flows since there's no reachable sink
  if (!flows.empty()) {
    throw std::runtime_error("Dead end should produce no flows, got " +
                             std::to_string(flows.size()));
  }

  std::cout << "  PASS\n";
}

void test_multiple_channels_same_bundle() {
  std::cout << "Test: Multiple Channels on Same Bundle\n";

  FlowReconstructionGraph graph;

  // Two independent channels on DMA
  graph.addConnection(0, 2, WireBundle::DMA, 0, WireBundle::Core, 0);
  graph.addConnection(0, 2, WireBundle::DMA, 1, WireBundle::Core, 1);

  auto flows = graph.reconstructFlows();

  if (flows.size() != 2) {
    throw std::runtime_error("Expected 2 flows, got " +
                             std::to_string(flows.size()));
  }

  assertFlowExists(flows, 0, 2, WireBundle::DMA, 0,
                         0, 2, WireBundle::Core, 0, "Channel 0");
  assertFlowExists(flows, 0, 2, WireBundle::DMA, 1,
                         0, 2, WireBundle::Core, 1, "Channel 1");

  std::cout << "  PASS\n";
}

//===----------------------------------------------------------------------===//
// Pretty Printing Tests
//===----------------------------------------------------------------------===//

void test_flow_printing() {
  std::cout << "Test: Flow Pretty Printing\n";

  ReconstructedFlow flow;
  flow.sourceCol = 0;
  flow.sourceRow = 0;
  flow.sourceBundle = WireBundle::DMA;
  flow.sourceChannel = 0;
  flow.destCol = 0;
  flow.destRow = 2;
  flow.destBundle = WireBundle::Core;
  flow.destChannel = 1;
  flow.hopCount = 2;

  std::string output;
  llvm::raw_string_ostream os(output);

  FlowReconstructionEmitter::printFlowAsComment(os, flow);

  // Check that output contains expected elements
  if (output.find("aie.flow") == std::string::npos) {
    throw std::runtime_error("Output should contain 'aie.flow'");
  }
  if (output.find("DMA : 0") == std::string::npos) {
    throw std::runtime_error("Output should contain 'DMA : 0'");
  }
  if (output.find("Core : 1") == std::string::npos) {
    throw std::runtime_error("Output should contain 'Core : 1'");
  }
  if (output.find("2 hop") == std::string::npos) {
    throw std::runtime_error("Output should contain hop count");
  }

  std::cout << "  PASS\n";
}

//===----------------------------------------------------------------------===//
// Main Test Runner
//===----------------------------------------------------------------------===//

int main() {
  try {
    std::cout << "\n=== Flow Reconstruction Unit Tests ===\n\n";

    // Basic graph construction tests
    test_empty_graph();
    test_single_connection_same_tile();
    test_two_hop_vertical_flow();
    test_horizontal_flow();

    // Broadcast pattern tests
    test_broadcast_same_tile();
    test_broadcast_to_multiple_tiles();

    // Multiple flows tests
    test_multiple_independent_flows();

    // Integration tests
    test_integration_with_parsed_config();
    test_flow_from_accumulator();

    // Edge case tests
    test_loopback_same_port();
    test_dead_end_no_sink();
    test_multiple_channels_same_bundle();

    // Pretty printing tests
    test_flow_printing();

    std::cout << "\nAll " << 13 << " flow reconstruction tests passed!\n";
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "\nTest failed: " << e.what() << "\n";
    return 1;
  }
}
