//===- aie-visualize.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "aie/InitialAllDialect.h"
#include "aie/Target/LLVMIR/Dialect/XLLVM/XLLVMToLLVMIRTranslation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace xilinx;

static cl::opt<std::string> fileName(cl::Positional, cl::desc("<input mlir>"),
                                     cl::Required);
static cl::opt<bool> emitDot("emit-dot",
                             cl::desc("Emit a DOT route visualization"));
static cl::opt<std::string> emitDotPerFlow(
    "emit-dot-per-flow",
    cl::desc("Emit one DOT file per flow into the given directory"),
    cl::value_desc("directory"));
static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));
static cl::list<unsigned>
    highlightedFlows("highlight-flow",
                     cl::desc("Highlight a flow by its numeric ID"),
                     cl::CommaSeparated, cl::ZeroOrMore);
static cl::list<unsigned>
    onlyFlows("only-flow", cl::desc("Emit only a flow with this numeric ID"),
              cl::CommaSeparated, cl::ZeroOrMore);
static cl::opt<bool> showBuffers(
    "show-buffers",
    cl::desc("Show tile buffers and their DMA channel connections"));
static cl::opt<bool> showPacketIDs(
    "show-packet-ids",
    cl::desc("Show flow IDs and packet IDs on routed connections"));
static cl::opt<bool>
    showVias("show-vias",
             cl::desc("Show switchbox ports along routed connections"));
static cl::opt<bool> topologyOnly(
    "topology-only",
    cl::desc(
        "Show endpoint topology without route vias or fixed tile positions"));
static cl::opt<bool> followThroughBuffers(
    "follow-through-buffers",
    cl::desc("Group flows connected through buffers or shared endpoints"));

const std::string bold("\033[0;1m");
const std::string dim("\033[0;2m");
const std::string red("\033[0;31m");
const std::string green("\033[1;32m");
const std::string yellow("\033[1;33m");
const std::string blue("\033[1;34m");
const std::string cyan("\033[0;36m");
const std::string magenta("\033[0;35m");
const std::string bwhite("\033[0;47m");
const std::string reset("\033[0m");
const std::string bgray("\033[48;5;239m");

namespace {

struct PortNode {
  int col;
  int row;
  AIE::WireBundle bundle;
  int channel;
  std::optional<AIE::DMAChannelDir> dmaDirection;

  auto asTuple() const {
    return std::make_tuple(col, row, static_cast<int>(bundle), channel,
                           dmaDirection);
  }
  bool operator<(const PortNode &other) const {
    return asTuple() < other.asTuple();
  }
  bool operator==(const PortNode &other) const {
    return asTuple() == other.asTuple();
  }
};

struct Segment {
  PortNode source;
  PortNode dest;

  bool operator<(const Segment &other) const {
    return std::tie(source, dest) < std::tie(other.source, other.dest);
  }
};

struct FlowRoute {
  unsigned id;
  std::optional<int> packetID;
  std::optional<int> packetMask;
  std::vector<PortNode> points;
};

struct BufferInfo {
  AIE::BufferOp op;
  unsigned id;
  int col;
  int row;
  unsigned tileIndex;
};

using DMAChannelKey = std::tuple<int, int, AIE::DMAChannelDir, int>;
using DMAChannelBuffers = std::map<DMAChannelKey, std::vector<AIE::BufferOp>>;

struct FlowGroups {
  std::vector<unsigned> routeGroups;
  unsigned count;
};

struct BufferSegment {
  PortNode port;
  Operation *buffer;
  bool intoBuffer;

  bool operator<(const BufferSegment &other) const {
    return std::tie(port, buffer, intoBuffer) <
           std::tie(other.port, other.buffer, other.intoBuffer);
  }
};

static FailureOr<PortNode>
getPortNode(mlir::Value tile, AIE::WireBundle bundle, int channel,
            Operation *owner,
            std::optional<AIE::DMAChannelDir> dmaDirection = std::nullopt) {
  auto tileOp = dyn_cast_or_null<AIE::TileLike>(tile.getDefiningOp());
  if (!tileOp)
    return owner->emitOpError("route endpoint is not a tile-like operation");
  std::optional<int> col = tileOp.tryGetCol();
  std::optional<int> row = tileOp.tryGetRow();
  if (!col || !row)
    return owner->emitOpError("route endpoint has unresolved coordinates");
  if (bundle != AIE::WireBundle::DMA)
    dmaDirection = std::nullopt;
  return PortNode{*col, *row, bundle, channel, dmaDirection};
}

template <typename FlowTy>
static LogicalResult appendVias(FlowTy flow, std::vector<PortNode> &points) {
  if (flow.getVias().empty())
    return flow.emitOpError(
        "requires vias; run aie-find-flows with emit-vias=true");
  ArrayRef<int32_t> ingressBundles =
      flow.getViaIngressBundlesAttr().asArrayRef();
  ArrayRef<int32_t> ingressChannels =
      flow.getViaIngressChannelsAttr().asArrayRef();
  ArrayRef<int32_t> egressBundles = flow.getViaEgressBundlesAttr().asArrayRef();
  ArrayRef<int32_t> egressChannels =
      flow.getViaEgressChannelsAttr().asArrayRef();
  for (auto [index, tile] : llvm::enumerate(flow.getVias())) {
    FailureOr<PortNode> ingress =
        getPortNode(tile, static_cast<AIE::WireBundle>(ingressBundles[index]),
                    ingressChannels[index], flow, AIE::DMAChannelDir::MM2S);
    FailureOr<PortNode> egress =
        getPortNode(tile, static_cast<AIE::WireBundle>(egressBundles[index]),
                    egressChannels[index], flow, AIE::DMAChannelDir::S2MM);
    if (failed(ingress) || failed(egress))
      return failure();
    points.push_back(*ingress);
    points.push_back(*egress);
  }
  return success();
}

static FailureOr<std::vector<FlowRoute>> collectRoutes(AIE::DeviceOp device) {
  std::vector<FlowRoute> routes;
  for (Operation &operation : *device.getBody()) {
    if (auto flow = dyn_cast<AIE::FlowOp>(operation)) {
      FlowRoute route{
          static_cast<unsigned>(routes.size()), std::nullopt, std::nullopt, {}};
      FailureOr<PortNode> source =
          getPortNode(flow.getSource(), flow.getSourceBundle(),
                      flow.getSourceChannel(), flow, AIE::DMAChannelDir::MM2S);
      FailureOr<PortNode> dest =
          getPortNode(flow.getDest(), flow.getDestBundle(),
                      flow.getDestChannel(), flow, AIE::DMAChannelDir::S2MM);
      if (failed(source) || failed(dest))
        return failure();
      route.points.push_back(*source);
      if (!topologyOnly && failed(appendVias(flow, route.points)))
        return failure();
      route.points.push_back(*dest);
      routes.push_back(std::move(route));
      continue;
    }
    auto packetFlow = dyn_cast<AIE::PacketFlowOp>(operation);
    if (!packetFlow)
      continue;
    auto sources = packetFlow.getOps<AIE::PacketSourceOp>();
    auto dests = packetFlow.getOps<AIE::PacketDestOp>();
    if (!llvm::hasSingleElement(sources) || !llvm::hasSingleElement(dests))
      return packetFlow.emitOpError(
          "requires exactly one source and destination per routed section");
    AIE::PacketSourceOp sourceOp = *sources.begin();
    AIE::PacketDestOp destOp = *dests.begin();
    FlowRoute route{static_cast<unsigned>(routes.size()),
                    packetFlow.IDInt(),
                    packetFlow.getMask() ? std::optional<int>(static_cast<int>(
                                               *packetFlow.getMask()))
                                         : std::nullopt,
                    {}};
    FailureOr<PortNode> source = getPortNode(
        sourceOp.getTile(), sourceOp.getBundle(), sourceOp.getChannel(),
        packetFlow, AIE::DMAChannelDir::MM2S);
    FailureOr<PortNode> dest =
        getPortNode(destOp.getTile(), destOp.getBundle(), destOp.getChannel(),
                    packetFlow, AIE::DMAChannelDir::S2MM);
    if (failed(source) || failed(dest))
      return failure();
    route.points.push_back(*source);
    if (!topologyOnly && failed(appendVias(packetFlow, route.points)))
      return failure();
    route.points.push_back(*dest);
    routes.push_back(std::move(route));
  }
  return routes;
}

static FailureOr<std::vector<BufferInfo>> collectBuffers(AIE::DeviceOp device) {
  std::vector<BufferInfo> buffers;
  std::map<std::pair<int, int>, unsigned> tileCounts;
  for (AIE::BufferOp buffer : device.getOps<AIE::BufferOp>()) {
    AIE::TileOp tile = buffer.getTileOp();
    if (!tile)
      return buffer.emitOpError("buffer owner is not a resolved tile");
    std::pair<int, int> coordinate{tile.colIndex(), tile.rowIndex()};
    buffers.push_back({buffer, static_cast<unsigned>(buffers.size()),
                       coordinate.first, coordinate.second,
                       tileCounts[coordinate]++});
  }
  return buffers;
}

static DMAChannelBuffers collectDMAChannelBuffers(AIE::DeviceOp device) {
  DMAChannelBuffers channels;
  for (auto program : device.getOps<AIE::DmaBody>()) {
    auto tile =
        dyn_cast_or_null<AIE::TileLike>(program.getTile().getDefiningOp());
    if (!tile || !tile.tryGetCol() || !tile.tryGetRow())
      continue;
    for (Block &block : program.getDmaBody()) {
      for (AIE::DMAStartOp start : block.getOps<AIE::DMAStartOp>()) {
        if (start.getEndpoint())
          continue;
        DMAChannelKey key{*tile.tryGetCol(), *tile.tryGetRow(),
                          start.getChannelDir(), start.getChannelIndex()};
        std::vector<AIE::BufferOp> &buffers = channels[key];
        SmallVector<Block *> worklist{start.getDest()};
        llvm::SmallPtrSet<Block *, 8> visited;
        while (!worklist.empty()) {
          Block *current = worklist.pop_back_val();
          if (!current || !visited.insert(current).second)
            continue;
          for (AIE::DMABDOp descriptor : current->getOps<AIE::DMABDOp>()) {
            auto buffer = descriptor.getBuffer().getDefiningOp<AIE::BufferOp>();
            if (buffer && !llvm::is_contained(buffers, buffer))
              buffers.push_back(buffer);
          }
          Operation *terminator = current->getTerminator();
          if (isa<AIE::DMAStartOp>(terminator))
            continue;
          llvm::append_range(worklist, terminator->getSuccessors());
        }
      }
    }
  }
  return channels;
}

static FlowGroups collectFlowGroups(ArrayRef<FlowRoute> routes,
                                    const DMAChannelBuffers &channelBuffers) {
  FlowGroups groups;
  groups.routeGroups.resize(routes.size());
  if (!followThroughBuffers) {
    std::iota(groups.routeGroups.begin(), groups.routeGroups.end(), 0);
    groups.count = routes.size();
    return groups;
  }

  std::vector<unsigned> parents(routes.size());
  std::iota(parents.begin(), parents.end(), 0);
  auto findRoot = [&](unsigned route) {
    while (parents[route] != route) {
      parents[route] = parents[parents[route]];
      route = parents[route];
    }
    return route;
  };
  auto merge = [&](unsigned first, unsigned second) {
    unsigned firstRoot = findRoot(first);
    unsigned secondRoot = findRoot(second);
    if (firstRoot != secondRoot)
      parents[secondRoot] = firstRoot;
  };

  using EndpointKey =
      std::tuple<PortNode, std::optional<int>, std::optional<int>>;
  std::map<Operation *, std::vector<unsigned>> bufferRoutes;
  std::map<EndpointKey, std::vector<unsigned>> endpointRoutes;
  for (const FlowRoute &route : routes) {
    const PortNode &source = route.points.front();
    if (source.bundle != AIE::WireBundle::DMA &&
        source.bundle != AIE::WireBundle::Core)
      endpointRoutes[{source, route.packetID, route.packetMask}].push_back(
          route.id);
    if (source.bundle == AIE::WireBundle::DMA) {
      DMAChannelKey key{source.col, source.row, AIE::DMAChannelDir::MM2S,
                        source.channel};
      auto channels = channelBuffers.find(key);
      if (channels != channelBuffers.end())
        for (AIE::BufferOp buffer : channels->second)
          bufferRoutes[buffer].push_back(route.id);
    }
    const PortNode &dest = route.points.back();
    if (dest.bundle != AIE::WireBundle::DMA &&
        dest.bundle != AIE::WireBundle::Core)
      endpointRoutes[{dest, route.packetID, route.packetMask}].push_back(
          route.id);
    if (dest.bundle == AIE::WireBundle::DMA) {
      DMAChannelKey key{dest.col, dest.row, AIE::DMAChannelDir::S2MM,
                        dest.channel};
      auto channels = channelBuffers.find(key);
      if (channels != channelBuffers.end())
        for (AIE::BufferOp buffer : channels->second)
          bufferRoutes[buffer].push_back(route.id);
    }
  }
  for (const auto &[buffer, routes] : bufferRoutes) {
    for (unsigned route : ArrayRef(routes).drop_front())
      merge(routes.front(), route);
  }
  for (const auto &[endpoint, routes] : endpointRoutes) {
    for (unsigned route : ArrayRef(routes).drop_front())
      merge(routes.front(), route);
  }

  std::map<unsigned, unsigned> groupIDs;
  for (const FlowRoute &route : routes) {
    unsigned root = findRoot(route.id);
    auto [group, inserted] = groupIDs.try_emplace(root, groupIDs.size());
    groups.routeGroups[route.id] = group->second;
  }
  groups.count = groupIDs.size();
  return groups;
}

static std::string portNodeID(const PortNode &port) {
  std::string id = "p_" + std::to_string(port.col) + "_" +
                   std::to_string(port.row) + "_" +
                   std::to_string(static_cast<int>(port.bundle)) + "_" +
                   std::to_string(port.channel);
  if (port.dmaDirection)
    id += *port.dmaDirection == AIE::DMAChannelDir::MM2S ? "_m" : "_s";
  return id;
}

static std::string bufferNodeID(unsigned id) {
  return "buffer_" + std::to_string(id);
}

static std::string tileNodeID(int col, int row) {
  return "tile_" + std::to_string(col) + "_" + std::to_string(row);
}

static std::string escapeDotLabel(StringRef value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char character : value) {
    if (character == '\\' || character == '\"')
      escaped.push_back('\\');
    escaped.push_back(character);
  }
  return escaped;
}

static std::string bufferLabel(AIE::BufferOp buffer, unsigned id) {
  if (auto name = buffer->getAttrOfType<StringAttr>("sym_name"))
    return escapeDotLabel(name.getValue());
  return "buffer " + std::to_string(id);
}

static std::pair<double, double> portPosition(const PortNode &port) {
  double x = port.col * 3.0;
  double y = port.row * 3.0;
  double channelOffset = (port.channel - 1.5) * 0.18;
  switch (port.bundle) {
  case AIE::WireBundle::North:
    return {x + channelOffset, y + 0.92};
  case AIE::WireBundle::South:
    return {x + channelOffset, y - 0.92};
  case AIE::WireBundle::East:
    return {x + 0.92, y + channelOffset};
  case AIE::WireBundle::West:
    return {x - 0.92, y + channelOffset};
  case AIE::WireBundle::DMA:
    return {x + (port.dmaDirection == AIE::DMAChannelDir::S2MM ? -0.22 : -0.54),
            y + channelOffset};
  case AIE::WireBundle::Core:
    return {x + 0.38, y + channelOffset};
  default:
    return {x, y + channelOffset};
  }
}

static std::string flowColor(unsigned id) {
  static constexpr const char *colors[] = {
      "#d73027", "#4575b4", "#1a9850", "#984ea3", "#ff7f00",
      "#00a6a6", "#e7298a", "#6a3d9a", "#a6761d", "#1f78b4"};
  return colors[id % std::size(colors)];
}

static std::string shortPortName(const PortNode &port) {
  switch (port.bundle) {
  case AIE::WireBundle::DMA:
    return (port.dmaDirection == AIE::DMAChannelDir::S2MM ? "S2MM" : "MM2S") +
           std::to_string(port.channel);
  case AIE::WireBundle::Core:
    return "C" + std::to_string(port.channel);
  case AIE::WireBundle::North:
    return "N" + std::to_string(port.channel);
  case AIE::WireBundle::South:
    return "S" + std::to_string(port.channel);
  case AIE::WireBundle::East:
    return "E" + std::to_string(port.channel);
  case AIE::WireBundle::West:
    return "W" + std::to_string(port.channel);
  default:
    return stringifyWireBundle(port.bundle).str() +
           std::to_string(port.channel);
  }
}

static LogicalResult
emitRouteDot(AIE::DeviceOp device, raw_ostream &output,
             std::optional<unsigned> selectedFlow = std::nullopt) {
  FailureOr<std::vector<FlowRoute>> routes = collectRoutes(device);
  if (failed(routes))
    return failure();
  FailureOr<std::vector<BufferInfo>> buffers = collectBuffers(device);
  if (failed(buffers))
    return failure();
  DMAChannelBuffers channelBuffers = collectDMAChannelBuffers(device);
  FlowGroups groups = collectFlowGroups(*routes, channelBuffers);

  std::set<unsigned> highlights(highlightedFlows.begin(),
                                highlightedFlows.end());
  std::set<unsigned> only(onlyFlows.begin(), onlyFlows.end());
  if (selectedFlow)
    only = {*selectedFlow};
  auto validateIDs = [&](const std::set<unsigned> &ids,
                         StringRef option) -> LogicalResult {
    for (unsigned id : ids) {
      if (id >= groups.count) {
        device.emitOpError() << option << " references unknown flow " << id
                             << "; valid IDs are 0 through "
                             << (groups.count == 0 ? 0 : groups.count - 1);
        return failure();
      }
    }
    return success();
  };
  if (failed(validateIDs(highlights, "--highlight-flow")) ||
      failed(validateIDs(only, "--only-flow")))
    return failure();
  auto isVisible = [&](unsigned id) { return only.empty() || only.count(id); };
  auto isHighlighted = [&](unsigned id) {
    return highlights.empty() || highlights.count(id);
  };

  std::set<PortNode> ports;
  std::set<PortNode> guidePorts;
  std::set<std::pair<int, int>> topologyTiles;
  std::map<Segment, std::vector<const FlowRoute *>> segments;
  std::map<BufferSegment, std::vector<const FlowRoute *>> bufferSegments;
  std::map<unsigned, Segment> labeledSegments;
  std::set<Segment> finalSegments;
  for (const FlowRoute &route : *routes) {
    unsigned groupID = groups.routeGroups[route.id];
    if (!isVisible(groupID))
      continue;
    if (topologyOnly) {
      topologyTiles.insert(
          {route.points.front().col, route.points.front().row});
      topologyTiles.insert({route.points.back().col, route.points.back().row});
    } else if (showVias) {
      ports.insert(route.points.begin(), route.points.end());
    } else {
      ports.insert(route.points.front());
      ports.insert(route.points.back());
      guidePorts.insert(route.points.begin() + 1, route.points.end() - 1);
    }
    std::vector<Segment> routeSegments;
    std::vector<Segment> physicalLinks;
    for (auto pair : llvm::zip_equal(ArrayRef(route.points).drop_back(),
                                     ArrayRef(route.points).drop_front())) {
      const auto &[source, dest] = pair;
      if (source == dest)
        continue;
      Segment segment{source, dest};
      segments[segment].push_back(&route);
      routeSegments.push_back(segment);
      if (source.col != dest.col || source.row != dest.row)
        physicalLinks.push_back(segment);
    }
    ArrayRef<Segment> labelCandidates = physicalLinks.empty()
                                            ? ArrayRef(routeSegments)
                                            : ArrayRef(physicalLinks);
    if (!labelCandidates.empty())
      labeledSegments[route.id] = labelCandidates[labelCandidates.size() / 2];
    if (!routeSegments.empty())
      finalSegments.insert(routeSegments.back());
    if (topologyOnly || (!showBuffers && !followThroughBuffers))
      continue;
    const PortNode &source = route.points.front();
    if (source.bundle == AIE::WireBundle::DMA) {
      DMAChannelKey key{source.col, source.row, AIE::DMAChannelDir::MM2S,
                        source.channel};
      for (AIE::BufferOp buffer : channelBuffers[key])
        bufferSegments[{source, buffer, false}].push_back(&route);
    }
    const PortNode &dest = route.points.back();
    if (dest.bundle == AIE::WireBundle::DMA) {
      DMAChannelKey key{dest.col, dest.row, AIE::DMAChannelDir::S2MM,
                        dest.channel};
      for (AIE::BufferOp buffer : channelBuffers[key])
        bufferSegments[{dest, buffer, true}].push_back(&route);
    }
  }

  const AIE::AIETargetModel &model = device.getTargetModel();
  output << "digraph aie_routes {\n  graph [";
  if (!topologyOnly)
    output << "layout=neato, overlap=true, ";
  output << "outputorder=nodesfirst, bgcolor=\"white\", pad=\"0.45\"];\n"
         << "  node [fontname=\"Helvetica\"];\n"
         << "  edge [fontname=\"Helvetica\", fontsize=9, arrowsize=0.65];\n";
  for (int col = 0; col < model.columns(); ++col) {
    for (int row = 0; row < model.rows(); ++row) {
      if (topologyOnly && !topologyTiles.count({col, row}))
        continue;
      StringRef fill = model.isCoreTile(col, row)      ? "#eef7ee"
                       : model.isMemTile(col, row)     ? "#fff2df"
                       : model.isShimNOCTile(col, row) ? "#e8f1fb"
                                                       : "#f3edf8";
      output << "  " << tileNodeID(col, row) << " [shape=box";
      if (!topologyOnly)
        output << ", fixedsize=true, width=2.15, height=2.15, pos=\""
               << col * 3.0 << ',' << row * 3.0 << "!\"";
      output << ", label=\"(" << col << ',' << row
             << ")\", style=filled, fillcolor=\"" << fill
             << "\", color=\"#b8b8b8\", fontcolor=\"#555555\"];\n";
    }
  }
  if (!topologyOnly) {
    for (const PortNode &port : ports) {
      auto [x, y] = portPosition(port);
      output << "  " << portNodeID(port) << " [shape=point, width=0.09, pos=\""
             << x << ',' << y << "!\", xlabel=\"" << shortPortName(port)
             << "\"];\n";
    }
    for (const PortNode &port : guidePorts) {
      if (ports.count(port))
        continue;
      auto [x, y] = portPosition(port);
      output << "  " << portNodeID(port)
             << " [shape=point, width=0, height=0, pos=\"" << x << ',' << y
             << "!\", label=\"\"];\n";
    }
  }
  std::map<Operation *, unsigned> bufferIDs;
  std::set<Operation *> visibleBuffers;
  for (const auto &[segment, segmentRoutes] : bufferSegments)
    visibleBuffers.insert(segment.buffer);
  if (showBuffers && !topologyOnly) {
    std::map<std::pair<int, int>, unsigned> visibleTileCounts;
    for (const BufferInfo &buffer : *buffers) {
      if (!only.empty() && !visibleBuffers.count(buffer.op))
        continue;
      bufferIDs[buffer.op] = buffer.id;
      double x = buffer.col * 3.0;
      unsigned tileIndex = only.empty()
                               ? buffer.tileIndex
                               : visibleTileCounts[{buffer.col, buffer.row}]++;
      double y = buffer.row * 3.0 + (only.empty() ? 0.46 : 0.5) -
                 tileIndex * (only.empty() ? 0.34 : 0.48);
      output << "  " << bufferNodeID(buffer.id)
             << " [shape=box, fixedsize=true, width="
             << (only.empty() ? "1.35" : "1.7")
             << ", height=" << (only.empty() ? "0.34" : "0.42") << ", pos=\""
             << x << ',' << y << "!\"";
      output << ", label=\"" << bufferLabel(buffer.op, buffer.id)
             << "\", fontsize=8, style=filled, fillcolor=\"#ffffff\", "
                "color=\"#666666\"];\n";
    }
  } else if (!topologyOnly) {
    for (const BufferInfo &buffer : *buffers) {
      if (!visibleBuffers.count(buffer.op))
        continue;
      bufferIDs[buffer.op] = buffer.id;
      double y = buffer.row * 3.0 + 0.46 - buffer.tileIndex * 0.34;
      output << "  " << bufferNodeID(buffer.id)
             << " [shape=point, width=0, height=0, pos=\"" << buffer.col * 3.0
             << ',' << y << "!\", label=\"\"];\n";
    }
  }
  auto writeColors = [&](ArrayRef<const FlowRoute *> edgeRoutes) {
    bool anyHighlighted = false;
    bool first = true;
    std::set<unsigned> writtenGroups;
    for (const FlowRoute *route : edgeRoutes) {
      unsigned groupID = groups.routeGroups[route->id];
      if (!writtenGroups.insert(groupID).second)
        continue;
      if (!first)
        output << ':';
      if (isHighlighted(groupID)) {
        output << flowColor(groupID);
        anyHighlighted = true;
      } else {
        output << "#c2c2c2";
      }
      first = false;
    }
    if (first)
      output << "#c2c2c2";
    return anyHighlighted;
  };
  auto writeLabel = [&](const FlowRoute &route) {
    unsigned groupID = groups.routeGroups[route.id];
    std::string color = isHighlighted(groupID) ? flowColor(groupID) : "#c2c2c2";
    output << "<FONT COLOR=\"" << color << "\">F" << groupID;
    if (route.packetID) {
      output << " pkt=" << *route.packetID;
      if (route.packetMask)
        output << '/' << *route.packetMask;
    }
    output << "</FONT>";
  };
  for (const auto &[segment, segmentRoutes] : segments) {
    std::vector<const FlowRoute *> labels;
    for (const FlowRoute *route : segmentRoutes) {
      if (labeledSegments.at(route->id) < segment ||
          segment < labeledSegments.at(route->id))
        continue;
      labels.push_back(route);
    }
    auto nodeID = [&](const PortNode &port) {
      return topologyOnly ? tileNodeID(port.col, port.row) : portNodeID(port);
    };
    output << "  " << nodeID(segment.source) << " -> " << nodeID(segment.dest)
           << " [color=\"";
    bool anyHighlighted = writeColors(segmentRoutes);
    output << "\", penwidth=\"" << (anyHighlighted ? "2.4" : "1.2") << '\"';
    if (showPacketIDs && !labels.empty()) {
      output << ", label=<";
      for (auto [index, route] : llvm::enumerate(labels)) {
        if (index)
          output << "<BR/>";
        writeLabel(*route);
      }
      output << '>';
    }
    if (!topologyOnly && !showVias && !finalSegments.count(segment))
      output << ", arrowhead=none";
    output << "];\n";
  }
  for (const auto &[segment, segmentRoutes] : bufferSegments) {
    unsigned bufferID = bufferIDs.at(segment.buffer);
    std::string portID = topologyOnly
                             ? tileNodeID(segment.port.col, segment.port.row)
                             : portNodeID(segment.port);
    std::string source = segment.intoBuffer ? portID : bufferNodeID(bufferID);
    std::string dest = segment.intoBuffer ? bufferNodeID(bufferID) : portID;
    output << "  " << source << " -> " << dest << " [color=\"";
    bool anyHighlighted = writeColors(segmentRoutes);
    output << "\", penwidth=\"" << (anyHighlighted ? "2.4" : "1.2")
           << "\", style=dashed];\n";
  }
  output << "}\n";
  return success();
}

} // namespace

int main(int argc, char *argv[]) {
  cl::ParseCommandLineOptions(argc, argv);

  MLIRContext ctx;
  ParserConfig pcfg(&ctx);
  SourceMgr srcMgr;

  DialectRegistry registry;
  registry.insert<arith::ArithDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<vector::VectorDialect>();
  xilinx::registerAllDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  xilinx::xllvm::registerXLLVMDialectTranslation(registry);
  ctx.appendDialectRegistry(registry);

  OwningOpRef<ModuleOp> owning =
      parseSourceFile<ModuleOp>(fileName, srcMgr, pcfg);

  if (!owning)
    return 1;

  auto deviceOps = owning->getOps<AIE::DeviceOp>();
  if (!llvm::hasSingleElement(deviceOps))
    return 2;

  AIE::DeviceOp deviceOp = *deviceOps.begin();

  const xilinx::AIE::AIETargetModel &model = deviceOp.getTargetModel();

  model.validate();

  if (emitDot && !emitDotPerFlow.empty()) {
    errs() << "--emit-dot and --emit-dot-per-flow are mutually exclusive\n";
    return 3;
  }

  if (emitDot) {
    std::error_code error;
    ToolOutputFile output(outputFilename, error, sys::fs::OF_Text);
    if (error) {
      errs() << error.message() << '\n';
      return 3;
    }
    if (failed(emitRouteDot(deviceOp, output.os())))
      return 4;
    output.keep();
    return 0;
  }

  if (!emitDotPerFlow.empty()) {
    if (!onlyFlows.empty() || !highlightedFlows.empty() ||
        outputFilename.getNumOccurrences()) {
      errs() << "--emit-dot-per-flow cannot be combined with --only-flow, "
                "--highlight-flow, or -o\n";
      return 3;
    }
    FailureOr<std::vector<FlowRoute>> routes = collectRoutes(deviceOp);
    if (failed(routes))
      return 4;
    DMAChannelBuffers channelBuffers = collectDMAChannelBuffers(deviceOp);
    FlowGroups groups = collectFlowGroups(*routes, channelBuffers);
    std::error_code error = sys::fs::create_directories(emitDotPerFlow);
    if (error) {
      errs() << error.message() << '\n';
      return 3;
    }
    for (unsigned groupID = 0; groupID < groups.count; ++groupID) {
      SmallString<256> path(emitDotPerFlow);
      sys::path::append(path, "flow-" + std::to_string(groupID) + ".dot");
      ToolOutputFile output(path, error, sys::fs::OF_Text);
      if (error) {
        errs() << error.message() << '\n';
        return 3;
      }
      if (failed(emitRouteDot(deviceOp, output.os(), groupID)))
        return 4;
      output.keep();
    }
    return 0;
  }

  std::vector<bool> used(model.columns() * model.rows());
  for (int col = 0; col < model.columns(); col++) {
    for (int row = 0; row < model.rows(); row++) {
      used[col + model.columns() * row] = false;
    }
  }
  for (auto tile : deviceOp.getOps<AIE::TileOp>()) {
    used[tile.getCol() + model.columns() * tile.getRow()] = true;
  }

  std::cout << model.columns() << " Columns and " << model.rows() << " Rows\n";
  for (int row = model.rows() - 1; row >= 0; row--) {
    std::cout << reset << row % 10 << " ";
    for (int col = 0; col < model.columns(); col++) {
      if (used[col + model.columns() * row])
        std::cout << bgray;
      else
        std::cout << dim;
      std::string v = reset + ".";
      if (model.isCoreTile(col, row))
        v = green + 'C';
      else if (model.isMemTile(col, row))
        v = red + 'M';
      else if (model.isShimNOCTile(col, row))
        v = blue + 'D';
      else if (model.isShimPLTile(col, row))
        v = magenta + 'P';
      std::cout << v << reset;
    }
    std::cout << "\n";
  }

  std::cout << reset << "  ";
  for (int col = 0; col < model.columns(); col++)
    std::cout << col % 10;
  std::cout << "\n";

  std::cout << "  ";
  for (int col = 0; col < model.columns(); col++) {
    int coltens = col / 10;
    if (coltens > 0)
      std::cout << coltens;
    else
      std::cout << " ";
  }
  std::cout << "\n";

  return 0;
}
