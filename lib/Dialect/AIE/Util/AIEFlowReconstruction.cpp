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

void FlowReconstructionGraph::addShimMuxConfig(
    const ParsedShimMuxConfig &config) {
  // Shim mux connections are at row 0 (shim tiles)
  // Special handling: shim_mux North port connects to switchbox South port
  // at the SAME tile (not to the neighbor tile)
  int col = config.column;
  int row = 0;

  for (const auto &conn : config.connections) {
    WireBundle srcBundle, dstBundle;
    int srcChannel, dstChannel;

    // Map ShimMuxSource to WireBundle (same logic as emitShimMuxForTile)
    WireBundle localBundle;
    switch (conn.source) {
      case ShimMuxSource::PL:
        localBundle = WireBundle::South;  // PL connects via South
        break;
      case ShimMuxSource::DMA:
        localBundle = WireBundle::DMA;
        break;
      case ShimMuxSource::NOC:
        // NoC uses special addressing - skip for now
        continue;
      default:
        continue;  // Skip invalid sources
    }

    if (conn.isInput) {
      // Mux: Input stream receives from local resource (DMA/PL) and sends North
      // Example: DMA:0 -> North:3
      // The North output connects to the switchbox South input at the SAME tile
      srcBundle = localBundle;
      srcChannel = 0;  // DMA/PL channel 0
      dstBundle = WireBundle::North;
      dstChannel = conn.streamIndex;

      // Add connection within shim_mux
      addConnection(col, row, srcBundle, srcChannel, dstBundle, dstChannel);

      // Add explicit connection from shim_mux North to switchbox South
      // (at the same tile, not neighbor)
      Node shimMuxNorthOutput{col, row, WireBundle::North, dstChannel, true};
      Node switchboxSouthInput{col, row, WireBundle::South, dstChannel, false};
      addEdge(shimMuxNorthOutput, switchboxSouthInput);

    } else {
      // Demux: Output stream receives from North and sends to local resource
      // Example: North:2 -> DMA:0
      // The switchbox South output connects to shim_mux North input at the SAME tile
      srcBundle = WireBundle::North;
      srcChannel = conn.streamIndex;
      dstBundle = localBundle;
      dstChannel = 0;  // DMA/PL channel 0

      // Add explicit connection from switchbox South to shim_mux North
      // (at the same tile, not neighbor)
      Node switchboxSouthOutput{col, row, WireBundle::South, srcChannel, true};
      Node shimMuxNorthInput{col, row, WireBundle::North, srcChannel, false};
      addEdge(switchboxSouthOutput, shimMuxNorthInput);

      // Add connection within shim_mux
      addConnection(col, row, srcBundle, srcChannel, dstBundle, dstChannel);
    }
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

  llvm::errs() << "DEBUG: Flow reconstruction starting\n";
  llvm::errs() << "DEBUG: Number of flow sources: " << flowSources_.size() << "\n";
  for (const auto &src : flowSources_) {
    llvm::errs() << "DEBUG: Source at (" << src.col << "," << src.row << ") "
                 << (int)src.bundle << ":" << src.channel << " isOutput=" << src.isOutput << "\n";
  }
  llvm::errs() << "DEBUG: Number of edges: " << edges_.size() << "\n";

  // Pre-process: synthesize missing pass-through edges for memory tiles.
  // Memory tiles often only have routing configured in one direction in the CDO.
  // We synthesize the reverse direction to enable bidirectional flow reconstruction.
  std::map<Node, std::vector<Node>> synthesizedEdges = edges_;

  // For each existing output node to a pass-through port, check if the incoming
  // data from the neighbor would have a path through this tile
  for (const auto &[node, successors] : edges_) {
    // Look for outputs to pass-through ports (data leaving this tile)
    if (node.isOutput && isPassThroughBundle(node.bundle)) {
      // This is data exiting via a pass-through port (e.g., South:1 output)
      // The neighbor tile's opposite input receives this data
      auto neighborInput = getNeighborNode(node.col, node.row, node.bundle, node.channel);
      if (neighborInput.has_value()) {
        // Check if this neighbor input has any outgoing edges
        if (synthesizedEdges.find(*neighborInput) == synthesizedEdges.end()) {
          // The neighbor input has no edges - synthesize pass-through
          WireBundle oppositeBundle = getOppositeBundle(neighborInput->bundle);
          Node neighborOutput{neighborInput->col, neighborInput->row, oppositeBundle,
                             neighborInput->channel, true};

          // Add the intra-tile pass-through edge
          synthesizedEdges[*neighborInput].push_back(neighborOutput);

          // Add the inter-tile edge to the next neighbor
          auto nextNeighbor = getNeighborNode(neighborOutput.col, neighborOutput.row,
                                             neighborOutput.bundle, neighborOutput.channel);
          if (nextNeighbor.has_value()) {
            synthesizedEdges[neighborOutput].push_back(*nextNeighbor);
          }
        }
      }
    }
  }

  // For each flow source, BFS to find all reachable sinks using synthesized edges
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
      auto it = synthesizedEdges.find(current);
      if (it != synthesizedEdges.end()) {
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
