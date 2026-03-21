//===- AIEFlowReconstruction.h - Flow Reconstruction from Switchboxes ----===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Reconstruct aie.flow operations from parsed switchbox connections.
// This traces data paths through intermediate switchboxes to show end-to-end
// routing topology rather than individual switchbox-level connections.
//===----------------------------------------------------------------------===//

#ifndef AIE_FLOW_RECONSTRUCTION_H
#define AIE_FLOW_RECONSTRUCTION_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Util/AIESwitchboxLifting.h"

#include <map>
#include <optional>
#include <queue>
#include <set>
#include <vector>

namespace xilinx {
namespace AIE {

//===----------------------------------------------------------------------===//
// ReconstructedFlow - A reconstructed end-to-end data path
//===----------------------------------------------------------------------===//

struct ReconstructedFlow {
  // Source endpoint
  int sourceCol = 0;
  int sourceRow = 0;
  WireBundle sourceBundle = WireBundle::DMA;
  int sourceChannel = 0;

  // Destination endpoint
  int destCol = 0;
  int destRow = 0;
  WireBundle destBundle = WireBundle::DMA;
  int destChannel = 0;

  // Path information (for debugging/visualization)
  int hopCount = 0;  // Number of intermediate switchbox hops

  bool operator==(const ReconstructedFlow &other) const;
};

//===----------------------------------------------------------------------===//
// FlowReconstructionGraph - Graph-based flow reconstruction
//===----------------------------------------------------------------------===//

class FlowReconstructionGraph {
public:
  FlowReconstructionGraph() = default;

  /// Add all connections from a parsed switchbox config
  void addSwitchboxConfig(const ParsedSwitchboxConfig &config);

  /// Add all connections from a parsed shim mux config
  void addShimMuxConfig(const ParsedShimMuxConfig &config);

  /// Add a single connection
  void addConnection(int col, int row,
                     WireBundle sourceBundle, int sourceChannel,
                     WireBundle destBundle, int destChannel);

  /// Reconstruct all flows from the accumulated switchbox connections
  std::vector<ReconstructedFlow> reconstructFlows() const;

  /// Check if the graph has any connections
  bool hasConnections() const { return !edges_.empty(); }

  /// Get the number of edges in the graph
  size_t edgeCount() const { return edges_.size(); }

  /// Get all active tiles (tiles with switchbox configurations)
  std::set<std::pair<int, int>> getActiveTiles() const { return activeTiles_; }

private:
  /// Node in the routing graph
  /// Represents a unique (tile, port, direction) tuple
  struct Node {
    int col;
    int row;
    WireBundle bundle;
    int channel;
    bool isOutput;  // true = output from switchbox, false = input to switchbox

    bool operator<(const Node &other) const;
    bool operator==(const Node &other) const;
  };

  /// Check if a bundle is an endpoint (flow start/end)
  static bool isEndpointBundle(WireBundle bundle);

  /// Check if a bundle is a pass-through (inter-tile routing)
  static bool isPassThroughBundle(WireBundle bundle);

  /// Get the neighbor node that a pass-through output connects to
  static std::optional<Node> getNeighborNode(int col, int row,
                                              WireBundle bundle, int channel);

  /// Get the opposite bundle for inter-tile connections
  static WireBundle getOppositeBundle(WireBundle bundle);

  /// Add an edge to the graph
  void addEdge(const Node &from, const Node &to);

  /// Graph edges: map from node to list of successor nodes
  std::map<Node, std::vector<Node>> edges_;

  /// Track endpoint sources (DMA/Core outputs - flow origins)
  std::vector<Node> flowSources_;

  /// Track endpoint sinks (DMA/Core inputs - flow destinations)
  std::vector<Node> flowSinks_;

  /// Track active tiles
  std::set<std::pair<int, int>> activeTiles_;
};

//===----------------------------------------------------------------------===//
// FlowReconstructionEmitter - Emit reconstructed flows as MLIR operations
//===----------------------------------------------------------------------===//

class FlowReconstructionEmitter {
public:
  /// Convert WireBundle to string for pretty printing
  static const char* wireBundleToString(WireBundle bundle);

  /// Print a reconstructed flow as a comment
  static void printFlowAsComment(llvm::raw_ostream &os,
                                  const ReconstructedFlow &flow);

  /// Print all reconstructed flows
  static void printFlows(llvm::raw_ostream &os,
                         const std::vector<ReconstructedFlow> &flows);
};

} // namespace AIE
} // namespace xilinx

#endif // AIE_FLOW_RECONSTRUCTION_H
