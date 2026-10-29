//===- DesignGraph.h --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The dataflow graph of a placed and routed AIE design, independent of MLIR.
// Its JSON form is the interface between the IR frontend and everything that
// draws or queries the design.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_VISUALIZE_DESIGNGRAPH_H
#define AIE_VISUALIZE_DESIGNGRAPH_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace xilinx::AIE::viz {

/// Source file to the lines in it that an element stands for.
using SourceLines = std::map<std::string, std::set<unsigned>>;

void mergeSources(SourceLines &into, const SourceLines &from);

/// Node kinds. A switchbox port is directional: an `in` port is where a
/// stream enters the switch, an `out` port where it leaves.
namespace kind {
inline constexpr const char *Port = "port";
inline constexpr const char *DMA = "dma";
inline constexpr const char *Buffer = "buffer";
inline constexpr const char *Call = "call";
inline constexpr const char *Argument = "arg";
} // namespace kind

struct TileInfo {
  int col = 0;
  int row = 0;
  /// "shim", "mem" or "core".
  std::string type;
  /// Whether the design declares this tile.
  bool used = false;
  /// Stream switch ports by bundle name: {in, out}.
  std::map<std::string, std::pair<unsigned, unsigned>> ports;
  unsigned mm2sChannels = 0;
  unsigned s2mmChannels = 0;
};

struct Node {
  std::string id;
  std::string kind;
  std::string label;
  /// Hover text with what the label leaves out.
  std::string detail;
  /// The tile the node belongs to; runtime arguments have none.
  int col = -1;
  int row = -1;
  /// Ports: bundle, channel and "in"/"out". DMA channels: channel and
  /// "MM2S"/"S2MM". Runtime arguments: the argument index.
  std::string bundle;
  int channel = -1;
  std::string direction;
  /// Buffer size, or bytes a DMA channel or argument moves per run.
  uint64_t bytes = 0;
  /// Contracted away in this view. Hidden nodes stay listed so that a
  /// contracted edge can say which ones it passes through.
  bool hidden = false;
  /// Interchangeable nodes contraction merged into this one, itself first.
  std::vector<std::string> members;
  /// The node this one was merged into; this node is then hidden.
  std::string mergedInto;
  SourceLines sources;

  bool hasTile() const { return col >= 0; }
};

struct Edge {
  std::string source;
  std::string target;
  /// Route IDs (indices into Graph::flows) crossing this edge.
  std::set<unsigned> flows;
  /// Packet IDs crossing this edge. Empty for circuit-switched streams and
  /// for everything outside the stream network.
  std::set<unsigned> packets;
  /// Bytes moved per run of the design, if the IR says.
  uint64_t bytes = 0;
  SourceLines sources;
  /// The edges of the full graph this edge stands for, as (source, target).
  /// An uncontracted edge stands for itself.
  std::vector<std::pair<std::string, std::string>> segments;

  std::string id() const { return source + "->" + target; }
};

/// A routed stream, as aie.flow or aie.packet_flow states it.
struct Flow {
  unsigned id = 0;
  std::string label;
  std::optional<unsigned> packet;
  SourceLines sources;
};

struct Graph {
  std::string device;
  std::string target;
  std::vector<TileInfo> tiles;
  std::vector<Node> nodes;
  std::vector<Edge> edges;
  std::vector<Flow> flows;
  /// Node kinds contracted to produce this graph.
  std::vector<std::string> contracted;
  /// Source file contents, so a viewer can show what `sources` points at.
  std::map<std::string, std::string> files;
};

llvm::json::Value toJSON(const Graph &graph);
llvm::Expected<Graph> graphFromJSON(const llvm::json::Value &value);

/// Hides every node of the given kinds that passes exactly one stream
/// through: one incoming edge, one outgoing edge, and not back to where it
/// came from. Edges that come to join the same two nodes merge. Nodes of
/// those kinds that are interchangeable -- same tile, same neighbors in and
/// out, like the two buffers of a ping-pong pair -- merge into one first.
/// Runs to a fixpoint, so a node where streams branch, join, start or end
/// survives whatever is contracted. Nodes of those kinds with no edges are
/// hidden too.
Graph contract(const Graph &graph, llvm::ArrayRef<std::string> kinds);

struct RenderOptions {
  /// How an edge that stands for contracted nodes gets from its source to
  /// its target.
  enum class Routing {
    /// Along the physical path of every segment it stands for.
    Physical,
    /// Through each switchbox and tile its contracted nodes belong to, in
    /// order, by whatever path crosses the fewest other lines.
    MustHit,
    /// By whatever path crosses the fewest other lines.
    Free,
  };
  Routing routing = Routing::Physical;
  /// Scale line width by bytes moved instead of drawing every line alike.
  bool weighEdges = false;
  std::string title;
};

/// Draws the graph on a fixed layout of the device: every switchbox port,
/// DMA channel and inter-tile wire has its own place, so lines never share
/// a track. Contracted nodes are not drawn; edges through them still follow
/// the physical path they stand for.
llvm::Expected<std::string> renderSvg(const Graph &graph,
                                      const RenderOptions &options);

/// A self-contained HTML viewer over one or more named views of a design.
std::string renderHtml(llvm::ArrayRef<std::pair<std::string, Graph>> views,
                       llvm::StringRef title);

} // namespace xilinx::AIE::viz

#endif // AIE_VISUALIZE_DESIGNGRAPH_H
