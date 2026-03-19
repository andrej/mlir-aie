//===- AIEFlowReconstruction.cpp - Flow Reconstruction from Switchboxes --===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Util/AIEFlowReconstruction.h"

#include <queue>
#include <set>

using namespace xilinx::AIE;

//===----------------------------------------------------------------------===//
// ReconstructedFlow Implementation
//===----------------------------------------------------------------------===//

bool ReconstructedFlow::operator==(const ReconstructedFlow &other) const {
  return sourceCol == other.sourceCol &&
         sourceRow == other.sourceRow &&
         sourceBundle == other.sourceBundle &&
         sourceChannel == other.sourceChannel &&
         destCol == other.destCol &&
         destRow == other.destRow &&
         destBundle == other.destBundle &&
         destChannel == other.destChannel;
}

//===----------------------------------------------------------------------===//
// FlowReconstructionGraph::Node Implementation
//===----------------------------------------------------------------------===//

bool FlowReconstructionGraph::Node::operator<(const Node &other) const {
  if (col != other.col) return col < other.col;
  if (row != other.row) return row < other.row;
  if (static_cast<int>(bundle) != static_cast<int>(other.bundle))
    return static_cast<int>(bundle) < static_cast<int>(other.bundle);
  if (channel != other.channel) return channel < other.channel;
  return isOutput < other.isOutput;
}

bool FlowReconstructionGraph::Node::operator==(const Node &other) const {
  return col == other.col && row == other.row &&
         bundle == other.bundle && channel == other.channel &&
         isOutput == other.isOutput;
}

//===----------------------------------------------------------------------===//
// FlowReconstructionGraph Implementation
//===----------------------------------------------------------------------===//

bool FlowReconstructionGraph::isEndpointBundle(WireBundle bundle) {
  switch (bundle) {
    case WireBundle::DMA:
    case WireBundle::Core:
    case WireBundle::FIFO:
    case WireBundle::Trace:
    case WireBundle::TileControl:
      return true;
    default:
      return false;
  }
}

bool FlowReconstructionGraph::isPassThroughBundle(WireBundle bundle) {
  switch (bundle) {
    case WireBundle::North:
    case WireBundle::South:
    case WireBundle::East:
    case WireBundle::West:
      return true;
    default:
      return false;
  }
}

WireBundle FlowReconstructionGraph::getOppositeBundle(WireBundle bundle) {
  switch (bundle) {
    case WireBundle::North: return WireBundle::South;
    case WireBundle::South: return WireBundle::North;
    case WireBundle::East: return WireBundle::West;
    case WireBundle::West: return WireBundle::East;
    default: return bundle;  // Endpoints don't have opposites
  }
}

std::optional<FlowReconstructionGraph::Node>
FlowReconstructionGraph::getNeighborNode(int col, int row,
                                          WireBundle bundle, int channel) {
  // Calculate neighbor tile coordinates based on direction
  int neighborCol = col;
  int neighborRow = row;

  switch (bundle) {
    case WireBundle::North:
      neighborRow = row + 1;
      break;
    case WireBundle::South:
      neighborRow = row - 1;
      if (neighborRow < 0) return std::nullopt;  // Below shim row
      break;
    case WireBundle::East:
      neighborCol = col + 1;
      break;
    case WireBundle::West:
      neighborCol = col - 1;
      if (neighborCol < 0) return std::nullopt;  // Beyond grid
      break;
    default:
      return std::nullopt;  // Not a pass-through bundle
  }

  // The neighbor node is the input on the opposite side
  WireBundle neighborBundle = getOppositeBundle(bundle);

  return Node{neighborCol, neighborRow, neighborBundle, channel, false};
}

void FlowReconstructionGraph::addEdge(const Node &from, const Node &to) {
  edges_[from].push_back(to);
}

void FlowReconstructionGraph::addSwitchboxConfig(
    const ParsedSwitchboxConfig &config) {
  for (const auto &conn : config.connections) {
    addConnection(config.column, config.row,
                  conn.sourceBundle, conn.sourceChannel,
                  conn.destBundle, conn.destChannel);
  }
}

void FlowReconstructionGraph::addConnection(
    int col, int row,
    WireBundle sourceBundle, int sourceChannel,
    WireBundle destBundle, int destChannel) {

  // Track active tiles
  activeTiles_.insert({col, row});

  // Create intra-switchbox edge: source (input to switchbox) -> dest (output)
  Node srcNode{col, row, sourceBundle, sourceChannel, false};
  Node dstNode{col, row, destBundle, destChannel, true};
  addEdge(srcNode, dstNode);

  // Track flow sources (endpoint outputs -> data originates here)
  if (isEndpointBundle(sourceBundle)) {
    flowSources_.push_back(srcNode);
  }

  // Track flow sinks (endpoint inputs -> data terminates here)
  if (isEndpointBundle(destBundle)) {
    flowSinks_.push_back(dstNode);
  }

  // Add inter-tile edge for pass-through destinations
  if (isPassThroughBundle(destBundle)) {
    auto neighbor = getNeighborNode(col, row, destBundle, destChannel);
    if (neighbor.has_value()) {
      addEdge(dstNode, *neighbor);
    }
  }
}

std::vector<ReconstructedFlow>
FlowReconstructionGraph::reconstructFlows() const {
  std::vector<ReconstructedFlow> flows;

  // For each flow source, BFS to find all reachable sinks
  for (const auto &source : flowSources_) {
    // BFS state
    std::queue<std::pair<Node, int>> worklist;  // (node, hop count)
    std::set<Node> visited;

    worklist.push({source, 0});

    while (!worklist.empty()) {
      auto [current, hops] = worklist.front();
      worklist.pop();

      // Skip if already visited
      if (visited.count(current)) continue;
      visited.insert(current);

      // Check if current is an output to an endpoint (flow sink)
      if (current.isOutput && isEndpointBundle(current.bundle)) {
        // Found a sink - record the flow
        ReconstructedFlow flow;
        flow.sourceCol = source.col;
        flow.sourceRow = source.row;
        flow.sourceBundle = source.bundle;
        flow.sourceChannel = source.channel;
        flow.destCol = current.col;
        flow.destRow = current.row;
        flow.destBundle = current.bundle;
        flow.destChannel = current.channel;
        flow.hopCount = hops;

        flows.push_back(flow);
        continue;  // Don't traverse past endpoints
      }

      // Continue traversal through successors
      auto it = edges_.find(current);
      if (it != edges_.end()) {
        for (const auto &next : it->second) {
          if (!visited.count(next)) {
            // Increment hop count when crossing tile boundary
            int newHops = hops;
            if (current.col != next.col || current.row != next.row) {
              newHops++;
            }
            worklist.push({next, newHops});
          }
        }
      }
    }
  }

  return flows;
}

//===----------------------------------------------------------------------===//
// FlowReconstructionEmitter Implementation
//===----------------------------------------------------------------------===//

const char* FlowReconstructionEmitter::wireBundleToString(WireBundle bundle) {
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

void FlowReconstructionEmitter::printFlowAsComment(
    llvm::raw_ostream &os, const ReconstructedFlow &flow) {
  os << "// aie.flow(%tile_" << flow.sourceCol << "_" << flow.sourceRow
     << ", " << wireBundleToString(flow.sourceBundle) << " : "
     << flow.sourceChannel << ", %tile_" << flow.destCol << "_"
     << flow.destRow << ", " << wireBundleToString(flow.destBundle)
     << " : " << flow.destChannel << ")";

  if (flow.hopCount > 0) {
    os << "  // " << flow.hopCount << " hop(s)";
  }

  os << "\n";
}

void FlowReconstructionEmitter::printFlows(
    llvm::raw_ostream &os, const std::vector<ReconstructedFlow> &flows) {
  if (flows.empty()) {
    os << "// No flows reconstructed\n";
    return;
  }

  os << "// === Reconstructed Flows (" << flows.size() << " total) ===\n";
  for (const auto &flow : flows) {
    printFlowAsComment(os, flow);
  }
  os << "// === End Reconstructed Flows ===\n";
}
