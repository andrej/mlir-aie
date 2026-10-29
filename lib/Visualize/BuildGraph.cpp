//===- BuildGraph.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reads a placed AIE design into a DesignGraph: switchbox ports and the wires
// between them, DMA channels and the buffers their descriptors name, kernel
// calls and the buffers they touch, and the runtime sequence arguments the
// shim DMAs move.
//
//===----------------------------------------------------------------------===//

#include "aie/Visualize/AIEVisualize.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"

#include <map>
#include <set>
#include <tuple>

using namespace llvm;
using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE::viz;

namespace {

SourceLines sourcesOf(Location location) {
  SourceLines result;
  location->walk([&](Location nested) {
    if (auto file = dyn_cast<FileLineColLoc>(nested))
      result[file.getFilename().str()].insert(file.getLine());
    return WalkResult::advance();
  });
  return result;
}

/// The design's own IR, printed, and the line each op is printed on. Ops of
/// the copy the analysis runs on carry their line as an attribute, which
/// survives the passes that rewrite the copy; ops they create carry none.
struct PrintedIR {
  std::string name;
  std::string text;
  llvm::DenseMap<Operation *, unsigned> lines;
};
constexpr StringLiteral kPrintedLine = "aie.visualize.line";
thread_local const PrintedIR *printedIR = nullptr;

SourceLines sourcesOf(Operation *op) {
  SourceLines result = sourcesOf(op->getLoc());
  if (!printedIR)
    return result;
  if (auto line = op->getAttrOfType<IntegerAttr>(kPrintedLine))
    result[printedIR->name].insert(line.getInt());
  else if (auto found = printedIR->lines.find(op);
           found != printedIR->lines.end())
    result[printedIR->name].insert(found->second);
  return result;
}

std::string portID(int col, int row, bool in, AIE::WireBundle bundle,
                   int channel) {
  return formatv("port:{0},{1}:{2}:{3}{4}", col, row, in ? "in" : "out",
                 AIE::stringifyWireBundle(bundle), channel);
}

std::string dmaID(int col, int row, AIE::DMAChannelDir direction, int channel) {
  return formatv("dma:{0},{1}:{2}{3}", col, row,
                 AIE::stringifyDMAChannelDir(direction), channel);
}

std::string shortBundle(AIE::WireBundle bundle) {
  switch (bundle) {
  case AIE::WireBundle::North:
    return "N";
  case AIE::WireBundle::South:
    return "S";
  case AIE::WireBundle::East:
    return "E";
  case AIE::WireBundle::West:
    return "W";
  case AIE::WireBundle::TileControl:
    return "Ctrl";
  default:
    return AIE::stringifyWireBundle(bundle).str();
  }
}

uint64_t memrefBytes(mlir::Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref || !memref.hasStaticShape())
    return 0;
  return memref.getNumElements() * ((memref.getElementTypeBitWidth() + 7) / 8);
}

/// Drops the content hash some kernel symbols carry ("37882bf0_matmul").
StringRef stripKernelHash(StringRef callee) {
  size_t separator = callee.find('_');
  if (separator == StringRef::npos || separator < 8 || separator > 16)
    return callee;
  StringRef prefix = callee.take_front(separator);
  if (!all_of(prefix, isHexDigit) || !any_of(prefix, isDigit))
    return callee;
  return callee.drop_front(separator + 1);
}

std::optional<std::pair<int, int>> tileOf(Value tile) {
  auto op = dyn_cast_or_null<AIE::TileLike>(tile.getDefiningOp());
  if (!op)
    return std::nullopt;
  std::optional<int> col = op.tryGetCol();
  std::optional<int> row = op.tryGetRow();
  if (!col || !row)
    return std::nullopt;
  return std::make_pair(*col, *row);
}

/// A switch connection inside one tile, as (col, row, in-port, out-port).
using ConnectionKey = std::tuple<int, int, int, int, int, int>;

ConnectionKey connectionKey(int col, int row, AIE::WireBundle inBundle,
                            int inChannel, AIE::WireBundle outBundle,
                            int outChannel) {
  return {col,
          row,
          static_cast<int>(inBundle),
          inChannel,
          static_cast<int>(outBundle),
          outChannel};
}

/// Where the routing configuration that aie-find-flows lifts away was
/// written: switchbox connections, packet rules and shim mux connections.
struct ConfigurationSources {
  std::map<ConnectionKey, SourceLines> switchbox;
  std::map<ConnectionKey, SourceLines> shimMux;
};

ConfigurationSources collectConfigurationSources(AIE::DeviceOp device) {
  ConfigurationSources result;
  for (auto switchbox : device.getOps<AIE::SwitchboxOp>()) {
    std::optional<std::pair<int, int>> tile = tileOf(switchbox.getTile());
    if (!tile)
      continue;
    auto [col, row] = *tile;
    for (auto connect : switchbox.getOps<AIE::ConnectOp>()) {
      SourceLines &sources = result.switchbox[connectionKey(
          col, row, connect.getSourceBundle(), connect.sourceIndex(),
          connect.getDestBundle(), connect.destIndex())];
      mergeSources(sources, sourcesOf(connect));
      mergeSources(sources, sourcesOf(switchbox));
    }
    for (auto rules : switchbox.getOps<AIE::PacketRulesOp>())
      for (auto rule : rules.getOps<AIE::PacketRuleOp>()) {
        auto amsel = rule.getAmsel().getDefiningOp<AIE::AMSelOp>();
        if (!amsel)
          continue;
        for (auto master : switchbox.getOps<AIE::MasterSetOp>()) {
          if (!is_contained(master.getAmsels(), amsel.getResult()))
            continue;
          SourceLines &sources = result.switchbox[connectionKey(
              col, row, rules.getSourceBundle(), rules.getSourceChannel(),
              master.getDestBundle(), master.getDestChannel())];
          for (Operation *op :
               {rule.getOperation(), rules.getOperation(), amsel.getOperation(),
                master.getOperation(), switchbox.getOperation()})
            mergeSources(sources, sourcesOf(op));
        }
      }
  }
  for (auto mux : device.getOps<AIE::ShimMuxOp>()) {
    std::optional<std::pair<int, int>> tile = tileOf(mux.getTile());
    if (!tile)
      continue;
    for (auto connect : mux.getOps<AIE::ConnectOp>()) {
      SourceLines &sources = result.shimMux[connectionKey(
          tile->first, tile->second, connect.getSourceBundle(),
          connect.sourceIndex(), connect.getDestBundle(), connect.destIndex())];
      mergeSources(sources, sourcesOf(connect));
      mergeSources(sources, sourcesOf(mux));
    }
  }
  return result;
}

/// One switchbox a route crosses: the port it enters by and the port it
/// leaves by.
struct Hop {
  int col;
  int row;
  AIE::WireBundle inBundle;
  int inChannel;
  AIE::WireBundle outBundle;
  int outChannel;
};

struct Endpoint {
  int col;
  int row;
  AIE::WireBundle bundle;
  int channel;
};

struct Route {
  unsigned id;
  Endpoint source;
  Endpoint dest;
  std::vector<Hop> hops;
  std::optional<unsigned> packet;
  std::optional<unsigned> mask;
  std::set<unsigned> carried;
  Operation *op;
};

template <typename FlowTy>
LogicalResult appendHops(FlowTy flow, std::vector<Hop> &hops) {
  if (flow.getVias().empty())
    return success();
  ArrayRef<int32_t> inBundles = flow.getViaIngressBundlesAttr().asArrayRef();
  ArrayRef<int32_t> inChannels = flow.getViaIngressChannelsAttr().asArrayRef();
  ArrayRef<int32_t> outBundles = flow.getViaEgressBundlesAttr().asArrayRef();
  ArrayRef<int32_t> outChannels = flow.getViaEgressChannelsAttr().asArrayRef();
  for (auto [index, tile] : enumerate(flow.getVias())) {
    std::optional<std::pair<int, int>> coordinates = tileOf(tile);
    if (!coordinates)
      return flow.emitOpError("via is not a placed tile");
    hops.push_back(
        {coordinates->first, coordinates->second,
         static_cast<AIE::WireBundle>(inBundles[index]), inChannels[index],
         static_cast<AIE::WireBundle>(outBundles[index]), outChannels[index]});
  }
  return success();
}

FailureOr<Endpoint> endpointOf(Operation *owner, Value tile,
                               AIE::WireBundle bundle, int channel) {
  std::optional<std::pair<int, int>> coordinates = tileOf(tile);
  if (!coordinates)
    return owner->emitOpError("route endpoint is not a placed tile");
  return Endpoint{coordinates->first, coordinates->second, bundle, channel};
}

FailureOr<std::vector<Route>> collectRoutes(AIE::DeviceOp device) {
  std::vector<Route> routes;
  for (Operation &op : *device.getBody()) {
    if (auto flow = dyn_cast<AIE::FlowOp>(op)) {
      FailureOr<Endpoint> source =
          endpointOf(flow, flow.getSource(), flow.getSourceBundle(),
                     flow.getSourceChannel());
      FailureOr<Endpoint> dest = endpointOf(
          flow, flow.getDest(), flow.getDestBundle(), flow.getDestChannel());
      if (failed(source) || failed(dest))
        return failure();
      Route route{static_cast<unsigned>(routes.size()),
                  *source,
                  *dest,
                  {},
                  std::nullopt,
                  std::nullopt,
                  {},
                  flow};
      if (failed(appendHops(flow, route.hops)))
        return failure();
      routes.push_back(std::move(route));
      continue;
    }
    auto packetFlow = dyn_cast<AIE::PacketFlowOp>(op);
    if (!packetFlow)
      continue;
    auto sources = packetFlow.getOps<AIE::PacketSourceOp>();
    auto dests = packetFlow.getOps<AIE::PacketDestOp>();
    if (!hasSingleElement(sources))
      return packetFlow.emitOpError("expected exactly one packet source");
    AIE::PacketSourceOp sourceOp = *sources.begin();
    FailureOr<Endpoint> source =
        endpointOf(packetFlow, sourceOp.getTile(), sourceOp.getBundle(),
                   sourceOp.getChannel());
    if (failed(source))
      return failure();
    std::vector<Hop> hops;
    if (failed(appendHops(packetFlow, hops)))
      return failure();
    std::optional<unsigned> mask;
    if (std::optional<int8_t> value = packetFlow.getMask())
      mask = static_cast<uint8_t>(*value);
    for (AIE::PacketDestOp destOp : dests) {
      FailureOr<Endpoint> dest =
          endpointOf(packetFlow, destOp.getTile(), destOp.getBundle(),
                     destOp.getChannel());
      if (failed(dest))
        return failure();
      routes.push_back({static_cast<unsigned>(routes.size()),
                        *source,
                        *dest,
                        hops,
                        static_cast<uint8_t>(packetFlow.getID()),
                        mask,
                        {},
                        packetFlow});
    }
  }
  return routes;
}

/// A packet rule aie-find-flows leaves in a switchbox: where two packet
/// streams merge, no single flow can say so.
struct PacketRule {
  int col;
  int row;
  AIE::WireBundle inBundle;
  int inChannel;
  AIE::WireBundle outBundle;
  int outChannel;
  unsigned mask;
  unsigned value;
  SourceLines sources;
};

std::vector<PacketRule> collectPacketRules(AIE::DeviceOp device,
                                           bool controlOverlay) {
  AIE::AIEDialect::IsCtrlPktOverlayAttrHelper overlay(device.getContext());
  auto isOverlay = [&](Operation *op) {
    return overlay.isAttrPresent(op) || op->hasAttr("is_ctrl_pkt_overlay");
  };
  std::vector<PacketRule> rules;
  for (auto switchbox : device.getOps<AIE::SwitchboxOp>()) {
    std::optional<std::pair<int, int>> tile = tileOf(switchbox.getTile());
    if (!tile)
      continue;
    for (auto packetRules : switchbox.getOps<AIE::PacketRulesOp>())
      for (auto rule : packetRules.getOps<AIE::PacketRuleOp>()) {
        auto amsel = rule.getAmsel().getDefiningOp<AIE::AMSelOp>();
        if (!amsel)
          continue;
        for (auto master : switchbox.getOps<AIE::MasterSetOp>()) {
          if (!is_contained(master.getAmsels(), amsel.getResult()))
            continue;
          if (!controlOverlay &&
              (isOverlay(rule) || isOverlay(packetRules) || isOverlay(master)))
            continue;
          PacketRule entry{tile->first,
                           tile->second,
                           packetRules.getSourceBundle(),
                           packetRules.getSourceChannel(),
                           master.getDestBundle(),
                           master.getDestChannel(),
                           static_cast<uint8_t>(rule.getMask()),
                           static_cast<uint8_t>(rule.getValue()),
                           sourcesOf(rule)};
          mergeSources(entry.sources, sourcesOf(master));
          rules.push_back(std::move(entry));
        }
      }
  }
  return rules;
}

/// Follows a value back to the buffers it may be: through views, branch and
/// region arguments, yields and selects.
std::vector<AIE::BufferOp> traceBuffers(Value root) {
  std::vector<AIE::BufferOp> buffers;
  SmallVector<Value> worklist{root};
  llvm::SmallDenseSet<Value, 8> visited;
  auto pushYielded = [&](Operation *op, unsigned index) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        if (Operation *terminator = block.getTerminator())
          if (terminator->hasTrait<OpTrait::ReturnLike>() &&
              index < terminator->getNumOperands())
            worklist.push_back(terminator->getOperand(index));
  };
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    if (auto buffer = value.getDefiningOp<AIE::BufferOp>()) {
      buffers.push_back(buffer);
      continue;
    }
    if (Operation *definition = value.getDefiningOp()) {
      if (auto view = dyn_cast<ViewLikeOpInterface>(definition)) {
        worklist.push_back(view.getViewSource());
      } else if (auto select = dyn_cast<arith::SelectOp>(definition)) {
        worklist.push_back(select.getTrueValue());
        worklist.push_back(select.getFalseValue());
      } else if (definition->getNumRegions()) {
        unsigned index = cast<OpResult>(value).getResultNumber();
        pushYielded(definition, index);
        if (auto loop = dyn_cast<scf::ForOp>(definition))
          worklist.push_back(loop.getInitArgs()[index]);
      }
      continue;
    }
    auto argument = cast<BlockArgument>(value);
    Block *block = argument.getOwner();
    if (auto loop = dyn_cast<scf::ForOp>(block->getParentOp())) {
      unsigned index = argument.getArgNumber();
      if (index > 0) {
        worklist.push_back(loop.getInitArgs()[index - 1]);
        pushYielded(loop, index - 1);
      }
      continue;
    }
    for (Block *predecessor : block->getPredecessors()) {
      auto branch = dyn_cast<BranchOpInterface>(predecessor->getTerminator());
      if (!branch)
        continue;
      for (auto [index, successor] :
           enumerate(predecessor->getTerminator()->getSuccessors())) {
        if (successor != block)
          continue;
        SuccessorOperands operands = branch.getSuccessorOperands(index);
        if (argument.getArgNumber() < operands.size())
          if (Value incoming = operands[argument.getArgNumber()])
            worklist.push_back(incoming);
      }
    }
  }
  return buffers;
}

class GraphBuilder {
public:
  GraphBuilder(AIE::DeviceOp device, AIE::RuntimeSequenceOp sequence,
               const ConfigurationSources &configuration,
               const AIE::VisualizeOptions &options)
      : device(device), sequence(sequence), configuration(configuration),
        options(options), model(device.getTargetModel()) {}

  FailureOr<Graph> build();

private:
  Node &node(const std::string &id, StringRef kindName) {
    auto [entry, inserted] = nodeIndex.try_emplace(id, graph.nodes.size());
    if (inserted) {
      Node created;
      created.id = id;
      created.kind = kindName.str();
      graph.nodes.push_back(std::move(created));
    }
    return graph.nodes[entry->second];
  }

  Edge &edge(const std::string &source, const std::string &target) {
    auto [entry, inserted] =
        edgeIndex.try_emplace({source, target}, graph.edges.size());
    if (inserted) {
      Edge created;
      created.source = source;
      created.target = target;
      created.segments.emplace_back(source, target);
      graph.edges.push_back(std::move(created));
    }
    return graph.edges[entry->second];
  }

  std::string port(int col, int row, bool in, AIE::WireBundle bundle,
                   int channel) {
    std::string id = portID(col, row, in, bundle, channel);
    Node &created = node(id, kind::Port);
    created.col = col;
    created.row = row;
    created.bundle = AIE::stringifyWireBundle(bundle).str();
    created.channel = channel;
    created.direction = in ? "in" : "out";
    created.label = shortBundle(bundle) + std::to_string(channel);
    created.detail = formatv("({0},{1}) switchbox {2} {3} {4}", col, row,
                             in ? "input" : "output", created.bundle, channel);
    return id;
  }

  std::string dma(int col, int row, AIE::DMAChannelDir direction, int channel) {
    std::string id = dmaID(col, row, direction, channel);
    Node &created = node(id, kind::DMA);
    created.col = col;
    created.row = row;
    created.channel = channel;
    created.direction = AIE::stringifyDMAChannelDir(direction).str();
    created.label = created.direction + std::to_string(channel);
    created.detail = formatv("({0},{1}) DMA {2} channel {3}", col, row,
                             created.direction, channel);
    return id;
  }

  void collectTiles();
  void collectBuffers();
  void collectDMAPrograms();
  LogicalResult collectStreams();
  void collectRuntimeSequence();
  void collectCalls();
  void embedSources();

  AIE::DeviceOp device;
  AIE::RuntimeSequenceOp sequence;
  const ConfigurationSources &configuration;
  const AIE::VisualizeOptions &options;
  const AIE::AIETargetModel &model;

  Graph graph;
  std::map<std::string, size_t> nodeIndex;
  std::map<std::pair<std::string, std::string>, size_t> edgeIndex;
  std::map<Operation *, std::string> bufferIDs;
  /// Bytes each DMA channel moves per run, by node ID.
  std::map<std::string, uint64_t> channelBytes;
  std::set<Operation *> buffersFilledByDMA;
  std::set<Operation *> buffersDrainedByDMA;
};

void GraphBuilder::collectTiles() {
  int minCol = std::numeric_limits<int>::max();
  int maxCol = -1;
  int maxRow = 0;
  std::set<std::pair<int, int>> declared;
  for (auto tile : device.getOps<AIE::TileOp>()) {
    declared.insert({tile.colIndex(), tile.rowIndex()});
    minCol = std::min(minCol, tile.colIndex());
    maxCol = std::max(maxCol, tile.colIndex());
    maxRow = std::max(maxRow, tile.rowIndex());
  }
  if (maxCol < 0)
    return;
  maxRow = std::max(maxRow, model.rows() - 1);

  static constexpr AIE::WireBundle bundles[] = {
      AIE::WireBundle::Core,       AIE::WireBundle::DMA,
      AIE::WireBundle::FIFO,       AIE::WireBundle::South,
      AIE::WireBundle::West,       AIE::WireBundle::North,
      AIE::WireBundle::East,       AIE::WireBundle::Trace,
      AIE::WireBundle::TileControl};
  for (int col = minCol; col <= maxCol; ++col)
    for (int row = 0; row <= maxRow; ++row) {
      TileInfo tile;
      tile.col = col;
      tile.row = row;
      tile.used = declared.count({col, row});
      if (model.isShimNOCorPLTile(col, row))
        tile.type = "shim";
      else if (model.isMemTile(col, row))
        tile.type = "mem";
      else if (model.isCoreTile(col, row))
        tile.type = "core";
      else
        continue;
      for (AIE::WireBundle bundle : bundles) {
        unsigned in = model.getNumSourceSwitchboxConnections(col, row, bundle);
        unsigned out = model.getNumDestSwitchboxConnections(col, row, bundle);
        if (in || out)
          tile.ports[AIE::stringifyWireBundle(bundle).str()] = {in, out};
      }
      if (tile.type == "shim") {
        tile.mm2sChannels = model.getNumSourceShimMuxConnections(
            col, row, AIE::WireBundle::DMA);
        tile.s2mmChannels =
            model.getNumDestShimMuxConnections(col, row, AIE::WireBundle::DMA);
      } else {
        tile.mm2sChannels = model.getNumSourceSwitchboxConnections(
            col, row, AIE::WireBundle::DMA);
        tile.s2mmChannels = model.getNumDestSwitchboxConnections(
            col, row, AIE::WireBundle::DMA);
      }
      if (tile.used) {
        for (unsigned channel = 0; channel < tile.mm2sChannels; ++channel)
          dma(col, row, AIE::DMAChannelDir::MM2S, channel);
        for (unsigned channel = 0; channel < tile.s2mmChannels; ++channel)
          dma(col, row, AIE::DMAChannelDir::S2MM, channel);
      }
      graph.tiles.push_back(std::move(tile));
    }
}

void GraphBuilder::collectBuffers() {
  std::map<std::pair<int, int>, unsigned> perTile;
  for (auto buffer : device.getOps<AIE::BufferOp>()) {
    std::optional<std::pair<int, int>> tile = tileOf(buffer.getTile());
    if (!tile)
      continue;
    unsigned index = perTile[*tile]++;
    std::string name = buffer.hasName() ? buffer.name().getValue().str() : "";
    std::string id = name.empty() ? formatv("buffer:{0},{1}#{2}", tile->first,
                                            tile->second, index)
                                        .str()
                                  : "buffer:" + name;
    Node &created = node(id, kind::Buffer);
    created.col = tile->first;
    created.row = tile->second;
    created.label = name.empty() ? "buffer " + std::to_string(index) : name;
    std::string type;
    raw_string_ostream(type) << buffer.getType();
    created.detail = formatv("({0},{1}) buffer {2} : {3}", tile->first,
                             tile->second, created.label, type);
    created.bytes = memrefBytes(buffer.getType());
    // Its data then arrives with the configuration, not at run time.
    created.initialized = bool(buffer.getInitialValue());
    created.sources = sourcesOf(buffer);
    bufferIDs[buffer] = id;
  }
}

void GraphBuilder::collectDMAPrograms() {
  // Every block a channel's descriptor chain can reach, from the block a
  // start (or an aie.dma region) enters.
  auto walkChain = [](Block *entry,
                      llvm::function_ref<void(Operation *)> visit) {
    SmallVector<Block *> worklist{entry};
    SmallPtrSet<Block *, 8> visited;
    while (!worklist.empty()) {
      Block *block = worklist.pop_back_val();
      if (!block || !visited.insert(block).second)
        continue;
      for (Operation &op : *block)
        visit(&op);
      Operation *terminator = block->getTerminator();
      if (!terminator || isa<AIE::DMAStartOp>(terminator))
        continue;
      append_range(worklist, terminator->getSuccessors());
    }
  };

  for (auto program : device.getOps<AIE::DmaBody>()) {
    std::optional<std::pair<int, int>> tile = tileOf(program.getTile());
    if (!tile)
      continue;
    auto [col, row] = *tile;
    auto addChannel = [&](AIE::DMAChannelDir direction, int channel,
                          Operation *start, Block *entry) {
      std::string channelID = dma(col, row, direction, channel);
      mergeSources(node(channelID, kind::DMA).sources, sourcesOf(start));
      auto visit = [&](Operation *op) {
        mergeSources(node(channelID, kind::DMA).sources, sourcesOf(op));
        auto descriptor = dyn_cast<AIE::DMABDOp>(op);
        if (!descriptor)
          return;
        auto buffer = descriptor.getBuffer().getDefiningOp<AIE::BufferOp>();
        uint64_t bytes = descriptor.getLenInBytes();
        channelBytes[channelID] += bytes;
        if (!buffer || !bufferIDs.count(buffer))
          return;
        bool reads = direction == AIE::DMAChannelDir::MM2S;
        (reads ? buffersDrainedByDMA : buffersFilledByDMA).insert(buffer);
        Edge &access = reads ? edge(bufferIDs[buffer], channelID)
                             : edge(channelID, bufferIDs[buffer]);
        access.bytes += bytes;
        mergeSources(access.sources, sourcesOf(descriptor));
      };
      walkChain(entry, visit);
    };
    for (Block &block : program.getDmaBody()) {
      for (auto start : block.getOps<AIE::DMAStartOp>())
        if (!start.getEndpoint())
          addChannel(start.getChannelDir(), start.getChannelIndex(), start,
                     start.getDest());
      for (auto dmaOp : block.getOps<AIE::DMAOp>())
        for (Region &region : dmaOp.getBds())
          if (!region.empty())
            addChannel(dmaOp.getChannelDir(), dmaOp.getChannelIndex(), dmaOp,
                       &region.front());
    }
  }
}

LogicalResult GraphBuilder::collectStreams() {
  FailureOr<std::vector<Route>> collected = collectRoutes(device);
  if (failed(collected))
    return failure();
  std::vector<Route> routes = std::move(*collected);
  std::vector<PacketRule> rules =
      collectPacketRules(device, options.controlOverlay);

  // A flow claims every packet ID that agrees with its own on the bits its
  // mask selects; only IDs the design names are worth telling apart.
  std::set<unsigned> named;
  for (const Route &route : routes)
    if (route.packet)
      named.insert(*route.packet);
  for (const PacketRule &rule : rules)
    if (rule.mask == 0x1f)
      named.insert(rule.value);
  for (Route &route : routes) {
    if (!route.packet)
      continue;
    route.carried.insert(*route.packet);
    if (route.mask)
      for (unsigned packet : named)
        if ((packet & *route.mask) == *route.packet)
          route.carried.insert(packet);
  }

  auto isDMA = [](const Endpoint &end) {
    return end.bundle == AIE::WireBundle::DMA;
  };
  auto bytesOf = [&](const Route &route) -> uint64_t {
    if (isDMA(route.source)) {
      auto found = channelBytes.find(dmaID(route.source.col, route.source.row,
                                           AIE::DMAChannelDir::MM2S,
                                           route.source.channel));
      if (found != channelBytes.end() && found->second)
        return found->second;
    }
    if (isDMA(route.dest)) {
      auto found = channelBytes.find(dmaID(route.dest.col, route.dest.row,
                                           AIE::DMAChannelDir::S2MM,
                                           route.dest.channel));
      if (found != channelBytes.end())
        return found->second;
    }
    return 0;
  };

  std::map<std::string, std::set<unsigned>> portPackets;
  for (const Route &route : routes) {
    Flow flow;
    flow.id = route.id;
    flow.packet = route.packet;
    flow.sources = sourcesOf(route.op);
    flow.label = formatv(
        "({0},{1}) {2}:{3} -> ({4},{5}) {6}:{7}", route.source.col,
        route.source.row, AIE::stringifyWireBundle(route.source.bundle),
        route.source.channel, route.dest.col, route.dest.row,
        AIE::stringifyWireBundle(route.dest.bundle), route.dest.channel);
    if (route.packet)
      flow.label += formatv(" packet {0}", *route.packet).str();
    if (route.hops.empty()) {
      device.emitWarning() << "aie-visualize: flow " << flow.label
                           << " has no route; it is not drawn";
      graph.flows.push_back(std::move(flow));
      continue;
    }

    uint64_t bytes = bytesOf(route);
    auto addHopEdge = [&](const std::string &from, const std::string &to,
                          const SourceLines &extra) {
      Edge &hop = edge(from, to);
      hop.flows.insert(route.id);
      hop.packets.insert(route.carried.begin(), route.carried.end());
      hop.bytes = std::max(hop.bytes, bytes);
      mergeSources(hop.sources, flow.sources);
      mergeSources(hop.sources, extra);
      portPackets[from].insert(route.carried.begin(), route.carried.end());
      portPackets[to].insert(route.carried.begin(), route.carried.end());
    };
    auto muxSources = [&](int col, int row, AIE::WireBundle from, int fromCh,
                          AIE::WireBundle to, int toCh) {
      auto found = configuration.shimMux.find(
          connectionKey(col, row, from, fromCh, to, toCh));
      return found == configuration.shimMux.end() ? SourceLines{}
                                                  : found->second;
    };

    const Hop &first = route.hops.front();
    std::string previous;
    for (auto [index, hop] : enumerate(route.hops)) {
      std::string in =
          port(hop.col, hop.row, true, hop.inBundle, hop.inChannel);
      std::string out =
          port(hop.col, hop.row, false, hop.outBundle, hop.outChannel);
      if (!previous.empty())
        addHopEdge(previous, in, {});
      auto found = configuration.switchbox.find(
          connectionKey(hop.col, hop.row, hop.inBundle, hop.inChannel,
                        hop.outBundle, hop.outChannel));
      addHopEdge(in, out,
                 found == configuration.switchbox.end() ? SourceLines{}
                                                        : found->second);
      previous = out;
    }
    if (isDMA(route.source)) {
      std::string channel = dma(route.source.col, route.source.row,
                                AIE::DMAChannelDir::MM2S, route.source.channel);
      // A shim DMA reaches its switchbox through the shim mux, which the
      // shim's switchbox sees as its South side.
      addHopEdge(
          channel,
          port(first.col, first.row, true, first.inBundle, first.inChannel),
          muxSources(route.source.col, route.source.row, AIE::WireBundle::DMA,
                     route.source.channel, AIE::WireBundle::North,
                     first.inChannel));
    }
    if (isDMA(route.dest)) {
      const Hop &last = route.hops.back();
      std::string channel = dma(route.dest.col, route.dest.row,
                                AIE::DMAChannelDir::S2MM, route.dest.channel);
      addHopEdge(
          port(last.col, last.row, false, last.outBundle, last.outChannel),
          channel,
          muxSources(route.dest.col, route.dest.row, AIE::WireBundle::North,
                     last.outChannel, AIE::WireBundle::DMA,
                     route.dest.channel));
    }
    graph.flows.push_back(std::move(flow));
  }

  // Wires join an output port to the input port of the neighbor it faces.
  // Packet rules can leave a stream at a port no flow continues from, so
  // wires are read off the ports rather than off the flows.
  auto wires = [&]() {
    std::vector<std::pair<std::string, std::string>> result;
    for (const Node &candidate : graph.nodes) {
      if (candidate.kind != kind::Port || candidate.direction != "out")
        continue;
      std::optional<AIE::WireBundle> bundle =
          AIE::symbolizeWireBundle(candidate.bundle);
      if (!bundle)
        continue;
      std::optional<std::pair<AIE::TileID, AIE::Port>> other =
          AIE::getConnectingWireEndpoint(
              model, AIE::TileID{candidate.col, candidate.row},
              AIE::Port{*bundle, candidate.channel});
      if (!other)
        continue;
      std::string in = portID(other->first.col, other->first.row, true,
                              other->second.bundle, other->second.channel);
      if (nodeIndex.count(in))
        result.emplace_back(candidate.id, in);
    }
    return result;
  };

  std::vector<std::pair<std::string, std::string>> ruleEdges;
  for (const PacketRule &rule : rules) {
    std::string in =
        port(rule.col, rule.row, true, rule.inBundle, rule.inChannel);
    std::string out =
        port(rule.col, rule.row, false, rule.outBundle, rule.outChannel);
    ruleEdges.emplace_back(in, out);
  }
  // Carry each packet through the rules that accept it and across wires
  // until no port gains one.
  for (bool changed = true; changed;) {
    changed = false;
    for (auto [rule, ends] : zip_equal(rules, ruleEdges)) {
      std::set<unsigned> arriving = portPackets[ends.first];
      for (unsigned packet : arriving)
        if ((packet & rule.mask) == rule.value)
          changed |= portPackets[ends.second].insert(packet).second;
    }
    for (const auto &[from, to] : wires()) {
      std::set<unsigned> arriving = portPackets[from];
      for (unsigned packet : arriving)
        changed |= portPackets[to].insert(packet).second;
    }
  }
  for (auto [rule, ends] : zip_equal(rules, ruleEdges)) {
    Edge &connection = edge(ends.first, ends.second);
    for (unsigned packet : portPackets[ends.first])
      if ((packet & rule.mask) == rule.value)
        connection.packets.insert(packet);
    mergeSources(connection.sources, rule.sources);
  }
  for (const auto &[from, to] : wires()) {
    Edge &wire = edge(from, to);
    wire.packets.insert(portPackets[from].begin(), portPackets[from].end());
  }
  return success();
}

void GraphBuilder::collectRuntimeSequence() {
  if (!sequence)
    return;
  Block &body = sequence.getBody().front();
  auto names = sequence->getAttrOfType<ArrayAttr>("runtime_arg_names");
  for (BlockArgument argument : body.getArguments()) {
    unsigned index = argument.getArgNumber();
    Node &created = node("arg:" + std::to_string(index), kind::Argument);
    created.channel = index;
    created.label = "%arg" + std::to_string(index);
    if (names && index < names.size())
      if (auto name = dyn_cast<StringAttr>(names[index]))
        created.label = name.getValue().str();
    std::string type;
    raw_string_ostream(type) << argument.getType();
    created.detail =
        "runtime sequence argument " + created.label + " : " + type;
    created.sources = sourcesOf(argument.getLoc());
    mergeSources(created.sources, sourcesOf(sequence));
  }

  auto argumentOf = [&](Value value) -> std::optional<std::string> {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &body)
      return std::nullopt;
    return "arg:" + std::to_string(argument.getArgNumber());
  };
  using Channel = std::tuple<int, int, AIE::DMAChannelDir, int>;
  auto allocationChannels = [&](SymbolRefAttr symbol) {
    std::vector<Channel> channels;
    Operation *definition =
        SymbolTable::lookupSymbolIn(device, symbol.getRootReference());
    if (auto allocation =
            dyn_cast_or_null<AIE::ShimDMAAllocationOp>(definition)) {
      if (std::optional<std::pair<int, int>> tile =
              tileOf(allocation.getTile()))
        channels.emplace_back(tile->first, tile->second,
                              allocation.getChannelDir(),
                              allocation.getChannelIndex());
    }
    return channels;
  };
  auto use = [&](const Channel &channel, Value buffer, uint64_t bytes,
                 Operation *op) {
    std::optional<std::string> argument = argumentOf(buffer);
    if (!argument)
      return;
    auto [col, row, direction, index] = channel;
    std::string channelID = dma(col, row, direction, index);
    bool reads = direction == AIE::DMAChannelDir::MM2S;
    Edge &access =
        reads ? edge(*argument, channelID) : edge(channelID, *argument);
    access.bytes += bytes;
    SourceLines sources = sourcesOf(op);
    for (Operation *user : op->getUsers())
      mergeSources(sources, sourcesOf(user));
    mergeSources(access.sources, sources);
    mergeSources(node(channelID, kind::DMA).sources, sources);
    channelBytes[channelID] += bytes;
    node(*argument, kind::Argument).bytes += bytes;
  };

  sequence.walk([&](Operation *op) {
    if (auto memcpy = dyn_cast<AIEX::NpuDmaMemcpyNdOp>(op)) {
      uint64_t elements = 1;
      for (int64_t size : memcpy.getStaticSizes())
        elements = ShapedType::isDynamic(size) ? 0 : elements * size;
      auto type = dyn_cast<BaseMemRefType>(memcpy.getMemref().getType());
      uint64_t bytes =
          type ? elements * ((type.getElementTypeBitWidth() + 7) / 8) : 0;
      for (const Channel &channel : allocationChannels(memcpy.getMetadata()))
        use(channel, memcpy.getMemref(), bytes, memcpy);
      if (Operation *allocation = SymbolTable::lookupSymbolIn(
              device, memcpy.getMetadata().getRootReference()))
        for (const Channel &channel : allocationChannels(memcpy.getMetadata()))
          mergeSources(node(dma(std::get<0>(channel), std::get<1>(channel),
                                std::get<2>(channel), std::get<3>(channel)),
                            kind::DMA)
                           .sources,
                       sourcesOf(allocation));
      return;
    }
    std::vector<Channel> channels;
    Region *body = nullptr;
    if (auto task = dyn_cast<AIEX::DMAConfigureTaskOp>(op)) {
      if (std::optional<std::pair<int, int>> tile = tileOf(task.getTile()))
        channels.emplace_back(tile->first, tile->second, task.getDirection(),
                              static_cast<int>(task.getChannel()));
      body = &task.getBody();
    } else if (auto task = dyn_cast<AIEX::DMAConfigureTaskForOp>(op)) {
      channels = allocationChannels(task.getAlloc());
      body = &task.getBody();
    }
    if (!body)
      return;
    body->walk([&](AIE::DMABDOp descriptor) {
      for (const Channel &channel : channels)
        use(channel, descriptor.getBuffer(), descriptor.getLenInBytes(), op);
    });
  });
}

void GraphBuilder::collectCalls() {
  static const Regex bufferStem("^(.*)_buff_[0-9]+$");
  static const Regex lockRole("^(.*)_(prod|cons)_lock(_[0-9]+)?$");

  for (auto core : device.getOps<AIE::CoreOp>()) {
    std::optional<std::pair<int, int>> tile = tileOf(core.getTile());
    if (!tile)
      continue;
    auto [col, row] = *tile;

    // A call writes an objectFIFO's buffer if the lock of that FIFO it
    // acquired last is the producer lock, and reads it if it is the consumer
    // lock. One core can be both ends of a FIFO, so this is decided per call,
    // in program order.
    std::vector<std::tuple<Operation *, std::string, bool>> acquires;
    std::map<Operation *, size_t> position;
    core.walk<WalkOrder::PreOrder>([&](Operation *op) {
      position[op] = position.size();
      auto useLock = dyn_cast<AIE::UseLockOp>(op);
      if (!useLock || (!useLock.acquire() && !useLock.acquireGE()))
        return;
      auto lock = useLock.getLock().getDefiningOp<AIE::LockOp>();
      SmallVector<StringRef> match;
      if (lock && lock.hasName() &&
          lockRole.match(lock.name().getValue(), &match))
        acquires.emplace_back(op, match[1].str(), match[2] == "prod");
    });
    auto writes = [&](AIE::BufferOp buffer, Operation *call) {
      if (buffersDrainedByDMA.count(buffer))
        return true;
      if (buffersFilledByDMA.count(buffer))
        return false;
      SmallVector<StringRef> match;
      if (!buffer.hasName() ||
          !bufferStem.match(buffer.name().getValue(), &match))
        return false;
      std::optional<bool> producing;
      size_t before = position.count(call) ? position[call] : position.size();
      for (const auto &[op, stem, producer] : acquires)
        if (stem == match[1] && position[op] < before)
          producing = producer;
      return producing.value_or(false);
    };

    std::vector<std::pair<Operation *, std::string>> calls;
    core.walk([&](func::CallOp call) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (callee && callee.isExternal())
        calls.emplace_back(call, stripKernelHash(call.getCallee()).str());
    });
    // A core that calls no kernel computes in place; it reads and writes
    // the buffers its memory operations name.
    if (calls.empty())
      calls.emplace_back(core, "core");

    for (auto [index, entry] : enumerate(calls)) {
      auto [op, name] = entry;
      std::string id = formatv("call:{0},{1}#{2}", col, row, index);
      Node &created = node(id, kind::Call);
      created.col = col;
      created.row = row;
      created.label = name;
      created.detail =
          formatv("({0},{1}) {2}", col, row,
                  isa<func::CallOp>(op)
                      ? "call @" + cast<func::CallOp>(op).getCallee().str()
                      : std::string("core body"));
      created.sources = sourcesOf(op);

      std::vector<Value> operands;
      if (isa<func::CallOp>(op))
        append_range(operands, op->getOperands());
      else
        op->walk([&](Operation *inner) {
          for (Value operand : inner->getOperands())
            if (isa<BaseMemRefType>(operand.getType()))
              operands.push_back(operand);
        });
      std::set<Operation *> seen;
      for (Value operand : operands)
        for (AIE::BufferOp buffer : traceBuffers(operand)) {
          if (!bufferIDs.count(buffer) || !seen.insert(buffer).second)
            continue;
          Edge &access = writes(buffer, op) ? edge(id, bufferIDs[buffer])
                                            : edge(bufferIDs[buffer], id);
          access.bytes = std::max(access.bytes, memrefBytes(buffer.getType()));
          mergeSources(access.sources, sourcesOf(op));
        }
    }
  }
}

void GraphBuilder::embedSources() {
  std::set<std::string> files;
  auto note = [&](const SourceLines &sources) {
    for (const auto &[file, lines] : sources)
      files.insert(file);
  };
  for (const Node &each : graph.nodes)
    note(each.sources);
  for (const Edge &each : graph.edges)
    note(each.sources);
  for (const Flow &each : graph.flows)
    note(each.sources);

  static constexpr uint64_t kMaxSourceBytes = 16 << 20;
  for (const std::string &file : files) {
    SmallVector<std::string, 2> candidates{file};
    if (!sys::path::is_absolute(file) && !options.sourceDirectory.empty()) {
      SmallString<256> joined(options.sourceDirectory);
      sys::path::append(joined, file);
      candidates.push_back(std::string(joined));
    }
    for (const std::string &candidate : candidates) {
      uint64_t size = 0;
      if (sys::fs::file_size(candidate, size) || size > kMaxSourceBytes)
        continue;
      ErrorOr<std::unique_ptr<MemoryBuffer>> contents =
          MemoryBuffer::getFile(candidate);
      if (!contents)
        continue;
      graph.files[file] = (*contents)->getBuffer().str();
      break;
    }
  }
}

FailureOr<Graph> GraphBuilder::build() {
  graph.device = device.getSymName().str();
  graph.target = AIE::stringifyAIEDevice(device.getDevice()).str();
  collectTiles();
  collectBuffers();
  collectDMAPrograms();
  collectRuntimeSequence();
  if (failed(collectStreams()))
    return failure();
  collectCalls();

  for (Node &each : graph.nodes)
    if (each.kind == kind::DMA) {
      auto found = channelBytes.find(each.id);
      if (found != channelBytes.end())
        each.bytes = found->second;
    }
  if (options.embedSources)
    embedSources();
  if (printedIR)
    graph.files[printedIR->name] = printedIR->text;
  return std::move(graph);
}

FailureOr<std::pair<AIE::DeviceOp, AIE::RuntimeSequenceOp>>
selectDevice(ModuleOp module, const AIE::VisualizeOptions &options) {
  AIE::DeviceOp device;
  for (auto candidate : module.getOps<AIE::DeviceOp>()) {
    if (!options.deviceName.empty() &&
        candidate.getSymName() != options.deviceName)
      continue;
    if (device) {
      module.emitError("aie-visualize: the input holds several devices; "
                       "select one by name");
      return failure();
    }
    device = candidate;
  }
  if (!device) {
    module.emitError("aie-visualize: no matching aie.device");
    return failure();
  }
  AIE::RuntimeSequenceOp sequence;
  for (auto candidate : device.getOps<AIE::RuntimeSequenceOp>()) {
    if (!options.runtimeSequenceName.empty() &&
        candidate.getSymName() != options.runtimeSequenceName)
      continue;
    if (sequence) {
      if (options.runtimeSequenceName.empty())
        device.emitWarning("aie-visualize: the device holds several runtime "
                           "sequences; showing the first");
      break;
    }
    sequence = candidate;
  }
  if (!sequence && !options.runtimeSequenceName.empty()) {
    device.emitError("aie-visualize: no runtime sequence named ")
        << options.runtimeSequenceName;
    return failure();
  }
  return std::make_pair(device, sequence);
}

} // namespace

FailureOr<Graph> AIE::buildDesignGraph(ModuleOp module,
                                       const VisualizeOptions &options) {
  FailureOr<std::pair<DeviceOp, RuntimeSequenceOp>> selected =
      selectDevice(module, options);
  if (failed(selected))
    return failure();
  PrintedIR printed;
  if (!options.irName.empty()) {
    printed.name = options.irName;
    AsmState::LocationMap locations;
    AsmState state(module, OpPrintingFlags(), &locations);
    raw_string_ostream os(printed.text);
    module.print(os, state);
    for (const auto &[op, position] : locations)
      printed.lines[op] = position.first;
  }
  struct Scope {
    Scope(const PrintedIR *ir) { printedIR = ir; }
    ~Scope() { printedIR = nullptr; }
  } scope(options.irName.empty() ? nullptr : &printed);

  ConfigurationSources configuration =
      collectConfigurationSources(selected->first);

  // The analysis wants every stream as a flow carrying its own route, and
  // every kernel operand as the buffer it names. Lifting the switchbox
  // network can claim overlapping hops, which a design may not have but a
  // view of one may, so the copy is not verified.
  OwningOpRef<ModuleOp> copy(module.clone());
  if (!options.irName.empty()) {
    SmallVector<Operation *> originals, copies;
    module->walk([&](Operation *op) { originals.push_back(op); });
    copy->walk([&](Operation *op) { copies.push_back(op); });
    Builder builder(module.getContext());
    for (auto [original, cloned] : zip_equal(originals, copies))
      if (auto found = printed.lines.find(original);
          found != printed.lines.end())
        cloned->setAttr(kPrintedLine, builder.getI32IntegerAttr(found->second));
  }
  PassManager pm(module.getContext());
  pm.enableVerifier(false);
  OpPassManager &devicePasses = pm.nest<DeviceOp>();
  devicePasses.addPass(createMem2Reg());
  DeviceOp device = selected->first;
  bool routed = !device.getOps<SwitchboxOp>().empty();
  auto hasVias = [](auto flow) { return !flow.getVias().empty(); };
  bool lifted = !routed && llvm::all_of(device.getOps<FlowOp>(), hasVias) &&
                llvm::all_of(device.getOps<PacketFlowOp>(), hasVias);
  if (!lifted) {
    if (!routed)
      devicePasses.addPass(createAIEPathfinderPass());
    std::unique_ptr<Pass> findFlows = createAIEFindFlowsPass();
    if (failed(findFlows->initializeOptions(
            "emit-vias=true", [](const Twine &) { return failure(); })))
      return failure();
    devicePasses.addPass(std::move(findFlows));
  }
  if (failed(pm.run(*copy)))
    return failure();

  FailureOr<std::pair<DeviceOp, RuntimeSequenceOp>> analyzed =
      selectDevice(*copy, options);
  if (failed(analyzed))
    return failure();
  return GraphBuilder(analyzed->first, analyzed->second, configuration, options)
      .build();
}

std::vector<std::pair<std::string, AIE::viz::Graph>>
AIE::standardViews(const AIE::viz::Graph &graph) {
  return {
      {"Topology", viz::contract(graph, {kind::Port, kind::DMA, kind::Buffer})},
      {"Buffers", viz::contract(graph, {kind::Port, kind::DMA})},
      {"Physical", graph}};
}
