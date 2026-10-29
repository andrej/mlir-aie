//===- DesignGraph.cpp ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Visualize/DesignGraph.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace llvm;
using namespace xilinx::AIE::viz;

void xilinx::AIE::viz::mergeSources(SourceLines &into,
                                    const SourceLines &from) {
  for (const auto &[file, lines] : from)
    into[file].insert(lines.begin(), lines.end());
}

template <typename Range>
static json::Array toArray(const Range &values) {
  json::Array result;
  for (const auto &value : values)
    result.push_back(value);
  return result;
}

static json::Array sourcesToJSON(const SourceLines &sources) {
  json::Array result;
  for (const auto &[file, lines] : sources)
    result.push_back(json::Object{{"file", file}, {"lines", toArray(lines)}});
  return result;
}

json::Value xilinx::AIE::viz::toJSON(const Graph &graph) {
  json::Array tiles;
  for (const TileInfo &tile : graph.tiles) {
    json::Object ports;
    for (const auto &[bundle, counts] : tile.ports)
      ports[bundle] = json::Array{counts.first, counts.second};
    tiles.push_back(json::Object{{"col", tile.col},
                                 {"row", tile.row},
                                 {"type", tile.type},
                                 {"used", tile.used},
                                 {"ports", std::move(ports)},
                                 {"mm2s", tile.mm2sChannels},
                                 {"s2mm", tile.s2mmChannels}});
  }

  json::Array nodes;
  for (const Node &node : graph.nodes) {
    json::Object object{
        {"id", node.id}, {"kind", node.kind}, {"label", node.label}};
    if (!node.detail.empty())
      object["detail"] = node.detail;
    if (node.hasTile())
      object["tile"] = json::Array{node.col, node.row};
    if (!node.bundle.empty())
      object["bundle"] = node.bundle;
    if (node.channel >= 0)
      object["channel"] = node.channel;
    if (!node.direction.empty())
      object["direction"] = node.direction;
    if (node.bytes)
      object["bytes"] = static_cast<int64_t>(node.bytes);
    if (node.initialized)
      object["initialized"] = true;
    if (node.hidden)
      object["hidden"] = true;
    if (!node.members.empty())
      object["members"] = toArray(node.members);
    if (!node.mergedInto.empty())
      object["mergedInto"] = node.mergedInto;
    if (!node.sources.empty())
      object["sources"] = sourcesToJSON(node.sources);
    nodes.push_back(std::move(object));
  }

  json::Array edges;
  for (const Edge &edge : graph.edges) {
    json::Object object{
        {"id", edge.id()}, {"source", edge.source}, {"target", edge.target}};
    if (!edge.flows.empty())
      object["flows"] = toArray(edge.flows);
    if (!edge.packets.empty())
      object["packets"] = toArray(edge.packets);
    if (edge.bytes)
      object["bytes"] = static_cast<int64_t>(edge.bytes);
    if (!edge.sources.empty())
      object["sources"] = sourcesToJSON(edge.sources);
    bool standsForItself = edge.segments.size() == 1 &&
                           edge.segments.front().first == edge.source &&
                           edge.segments.front().second == edge.target;
    if (!standsForItself) {
      json::Array segments;
      for (const auto &[from, to] : edge.segments)
        segments.push_back(json::Array{from, to});
      object["segments"] = std::move(segments);
    }
    edges.push_back(std::move(object));
  }

  json::Array flows;
  for (const Flow &flow : graph.flows) {
    json::Object object{{"id", flow.id}, {"label", flow.label}};
    if (flow.packet)
      object["packet"] = *flow.packet;
    if (!flow.sources.empty())
      object["sources"] = sourcesToJSON(flow.sources);
    flows.push_back(std::move(object));
  }

  json::Object files;
  for (const auto &[path, text] : graph.files)
    files[path] = text;

  return json::Object{
      {"format", "aie-design-graph"},
      {"version", 1},
      {"device", graph.device},
      {"target", graph.target},
      {"contracted", toArray(graph.contracted)},
      {"tiles", std::move(tiles)},
      {"nodes", std::move(nodes)},
      {"edges", std::move(edges)},
      {"flows", std::move(flows)},
      {"files", std::move(files)},
  };
}

static Error malformed(const Twine &what) {
  return createStringError(inconvertibleErrorCode(),
                           "malformed design graph: " + what);
}

static SourceLines sourcesFromJSON(const json::Object &object) {
  SourceLines sources;
  if (const json::Array *array = object.getArray("sources"))
    for (const json::Value &entry : *array) {
      const json::Object *source = entry.getAsObject();
      if (!source)
        continue;
      std::optional<StringRef> file = source->getString("file");
      const json::Array *lines = source->getArray("lines");
      if (!file || !lines)
        continue;
      for (const json::Value &line : *lines)
        if (std::optional<int64_t> number = line.getAsInteger())
          sources[file->str()].insert(*number);
    }
  return sources;
}

template <typename T>
static std::set<T> setFromJSON(const json::Object &object, StringRef key) {
  std::set<T> values;
  if (const json::Array *array = object.getArray(key))
    for (const json::Value &value : *array)
      if (std::optional<int64_t> number = value.getAsInteger())
        values.insert(static_cast<T>(*number));
  return values;
}

Expected<Graph> xilinx::AIE::viz::graphFromJSON(const json::Value &value) {
  const json::Object *root = value.getAsObject();
  if (!root || root->getString("format") != "aie-design-graph")
    return malformed("not an aie-design-graph document");

  Graph graph;
  graph.device = root->getString("device").value_or("").str();
  graph.target = root->getString("target").value_or("").str();
  if (const json::Array *contracted = root->getArray("contracted"))
    for (const json::Value &kindName : *contracted)
      if (std::optional<StringRef> name = kindName.getAsString())
        graph.contracted.push_back(name->str());

  if (const json::Array *tiles = root->getArray("tiles"))
    for (const json::Value &entry : *tiles) {
      const json::Object *object = entry.getAsObject();
      if (!object)
        return malformed("tile is not an object");
      TileInfo tile;
      tile.col = object->getInteger("col").value_or(0);
      tile.row = object->getInteger("row").value_or(0);
      tile.type = object->getString("type").value_or("").str();
      tile.used = object->getBoolean("used").value_or(false);
      tile.mm2sChannels = object->getInteger("mm2s").value_or(0);
      tile.s2mmChannels = object->getInteger("s2mm").value_or(0);
      if (const json::Object *ports = object->getObject("ports"))
        for (const auto &[bundle, counts] : *ports) {
          const json::Array *pair = counts.getAsArray();
          if (!pair || pair->size() != 2)
            return malformed("port counts are not a pair");
          tile.ports[bundle.str()] = {(*pair)[0].getAsInteger().value_or(0),
                                      (*pair)[1].getAsInteger().value_or(0)};
        }
      graph.tiles.push_back(std::move(tile));
    }

  const json::Array *nodes = root->getArray("nodes");
  const json::Array *edges = root->getArray("edges");
  if (!nodes || !edges)
    return malformed("missing nodes or edges");
  for (const json::Value &entry : *nodes) {
    const json::Object *object = entry.getAsObject();
    if (!object || !object->getString("id") || !object->getString("kind"))
      return malformed("node without id or kind");
    Node node;
    node.id = object->getString("id")->str();
    node.kind = object->getString("kind")->str();
    node.label = object->getString("label").value_or("").str();
    node.detail = object->getString("detail").value_or("").str();
    if (const json::Array *tile = object->getArray("tile");
        tile && tile->size() == 2) {
      node.col = (*tile)[0].getAsInteger().value_or(-1);
      node.row = (*tile)[1].getAsInteger().value_or(-1);
    }
    node.bundle = object->getString("bundle").value_or("").str();
    node.channel = object->getInteger("channel").value_or(-1);
    node.direction = object->getString("direction").value_or("").str();
    node.bytes = object->getInteger("bytes").value_or(0);
    node.hidden = object->getBoolean("hidden").value_or(false);
    node.initialized = object->getBoolean("initialized").value_or(false);
    if (const json::Array *members = object->getArray("members"))
      for (const json::Value &member : *members)
        if (std::optional<StringRef> id = member.getAsString())
          node.members.push_back(id->str());
    node.mergedInto = object->getString("mergedInto").value_or("").str();
    node.sources = sourcesFromJSON(*object);
    graph.nodes.push_back(std::move(node));
  }
  for (const json::Value &entry : *edges) {
    const json::Object *object = entry.getAsObject();
    if (!object || !object->getString("source") || !object->getString("target"))
      return malformed("edge without source or target");
    Edge edge;
    edge.source = object->getString("source")->str();
    edge.target = object->getString("target")->str();
    edge.flows = setFromJSON<unsigned>(*object, "flows");
    edge.packets = setFromJSON<unsigned>(*object, "packets");
    edge.bytes = object->getInteger("bytes").value_or(0);
    edge.sources = sourcesFromJSON(*object);
    if (const json::Array *segments = object->getArray("segments")) {
      for (const json::Value &segment : *segments) {
        const json::Array *pair = segment.getAsArray();
        if (!pair || pair->size() != 2 || !(*pair)[0].getAsString() ||
            !(*pair)[1].getAsString())
          return malformed("segment is not a pair of node IDs");
        edge.segments.emplace_back((*pair)[0].getAsString()->str(),
                                   (*pair)[1].getAsString()->str());
      }
    } else {
      edge.segments.emplace_back(edge.source, edge.target);
    }
    graph.edges.push_back(std::move(edge));
  }

  if (const json::Array *flows = root->getArray("flows"))
    for (const json::Value &entry : *flows) {
      const json::Object *object = entry.getAsObject();
      if (!object)
        return malformed("flow is not an object");
      Flow flow;
      flow.id = object->getInteger("id").value_or(0);
      flow.label = object->getString("label").value_or("").str();
      if (std::optional<int64_t> packet = object->getInteger("packet"))
        flow.packet = *packet;
      flow.sources = sourcesFromJSON(*object);
      graph.flows.push_back(std::move(flow));
    }

  if (const json::Object *files = root->getObject("files"))
    for (const auto &[path, text] : *files)
      if (std::optional<StringRef> contents = text.getAsString())
        graph.files[path.str()] = contents->str();
  return graph;
}

namespace {
using EdgeKey = std::pair<std::string, std::string>;

/// Edges keyed by the nodes they join, with each node's incident keys
/// indexed so that splicing one node out stays cheap on large designs.
struct EdgeIndex {
  std::map<EdgeKey, Edge> edges;
  std::map<std::string, std::set<EdgeKey>> incoming;
  std::map<std::string, std::set<EdgeKey>> outgoing;

  void insert(Edge edge) {
    EdgeKey key{edge.source, edge.target};
    auto [entry, inserted] = edges.try_emplace(key, edge);
    if (inserted) {
      incoming[key.second].insert(key);
      outgoing[key.first].insert(key);
      return;
    }
    // Two paths that now join the same nodes are one edge carrying both.
    Edge &existing = entry->second;
    existing.flows.insert(edge.flows.begin(), edge.flows.end());
    existing.packets.insert(edge.packets.begin(), edge.packets.end());
    existing.bytes += edge.bytes;
    mergeSources(existing.sources, edge.sources);
    for (auto &segment : edge.segments)
      if (!llvm::is_contained(existing.segments, segment))
        existing.segments.push_back(std::move(segment));
  }

  /// Moves every edge of `from` to `to`.
  void redirect(const std::string &from, const std::string &to) {
    for (EdgeKey key : std::set<EdgeKey>(incoming[from])) {
      Edge edge = take(key);
      edge.target = to;
      insert(std::move(edge));
    }
    for (EdgeKey key : std::set<EdgeKey>(outgoing[from])) {
      Edge edge = take(key);
      edge.source = to;
      insert(std::move(edge));
    }
  }

  std::vector<std::string> neighbors(const std::string &id, bool in) {
    std::vector<std::string> result;
    for (const EdgeKey &key : in ? incoming[id] : outgoing[id])
      result.push_back(in ? key.first : key.second);
    return result;
  }

  Edge take(const EdgeKey &key) {
    Edge edge = std::move(edges.at(key));
    edges.erase(key);
    incoming[key.second].erase(key);
    outgoing[key.first].erase(key);
    return edge;
  }
};
/// "A_buff_0", "A_buff_1" -> "A_buff_{0,1}".
std::string groupLabel(ArrayRef<std::string> labels) {
  StringRef first = labels.front();
  size_t common = first.size();
  for (StringRef label : labels.drop_front()) {
    size_t i = 0;
    while (i < common && i < label.size() && label[i] == first[i])
      ++i;
    common = i;
  }
  if (common == 0)
    return first.str() + " +" + std::to_string(labels.size() - 1);
  std::string result = first.take_front(common).str() + "{";
  for (auto [index, label] : llvm::enumerate(labels))
    result += (index ? "," : "") + StringRef(label).drop_front(common).str();
  return result + "}";
}
} // namespace

Graph xilinx::AIE::viz::contract(const Graph &graph,
                                 ArrayRef<std::string> kinds) {
  Graph result = graph;
  result.contracted.assign(kinds.begin(), kinds.end());
  std::set<std::string> hiddenKinds(kinds.begin(), kinds.end());

  EdgeIndex index;
  for (const Edge &edge : graph.edges)
    index.insert(edge);

  std::map<std::string, size_t> position;
  for (auto [i, node] : llvm::enumerate(result.nodes))
    position[node.id] = i;

  for (bool changed = true; changed;) {
    changed = false;
    std::map<std::tuple<std::string, int, int, std::vector<std::string>,
                        std::vector<std::string>>,
             std::vector<Node *>>
        interchangeable;
    for (Node &node : result.nodes) {
      if (node.hidden || !hiddenKinds.count(node.kind))
        continue;
      std::vector<std::string> in = index.neighbors(node.id, true);
      std::vector<std::string> out = index.neighbors(node.id, false);
      if (in.empty() && out.empty())
        continue;
      interchangeable[{node.kind, node.col, node.row, in, out}].push_back(
          &node);
    }
    for (auto &[key, group] : interchangeable) {
      if (group.size() < 2)
        continue;
      Node &representative = *group.front();
      if (representative.members.empty())
        representative.members.push_back(representative.id);
      for (Node *member : llvm::drop_begin(group)) {
        index.redirect(member->id, representative.id);
        member->hidden = true;
        member->mergedInto = representative.id;
        mergeSources(representative.sources, member->sources);
        if (member->members.empty())
          representative.members.push_back(member->id);
        for (const std::string &inner : member->members) {
          representative.members.push_back(inner);
          result.nodes[position.at(inner)].mergedInto = representative.id;
        }
        member->members.clear();
      }
      std::vector<std::string> labels;
      for (const std::string &member : representative.members)
        labels.push_back(graph.nodes[position.at(member)].label);
      representative.label = groupLabel(labels);
      changed = true;
    }

    for (Node &node : result.nodes) {
      if (node.hidden || !hiddenKinds.count(node.kind))
        continue;
      const std::set<EdgeKey> &in = index.incoming[node.id];
      const std::set<EdgeKey> &out = index.outgoing[node.id];
      if (in.empty() && out.empty()) {
        node.hidden = true;
        continue;
      }
      if (in.size() != 1 || out.size() != 1)
        continue;
      EdgeKey before = *in.begin();
      EdgeKey after = *out.begin();
      if (before.first == after.second)
        continue;

      Edge first = index.take(before);
      Edge second = index.take(after);
      Edge spliced;
      spliced.source = first.source;
      spliced.target = second.target;
      spliced.flows = first.flows;
      spliced.flows.insert(second.flows.begin(), second.flows.end());
      spliced.packets = first.packets;
      spliced.packets.insert(second.packets.begin(), second.packets.end());
      // A chain moves no more than its narrowest link; a link that does not
      // say how much it moves does not narrow it.
      spliced.bytes = !first.bytes    ? second.bytes
                      : !second.bytes ? first.bytes
                                      : std::min(first.bytes, second.bytes);
      spliced.sources = first.sources;
      mergeSources(spliced.sources, second.sources);
      mergeSources(spliced.sources, node.sources);
      spliced.segments = std::move(first.segments);
      llvm::append_range(spliced.segments, second.segments);
      index.insert(std::move(spliced));
      node.hidden = true;
      changed = true;
    }
  }

  result.edges.clear();
  for (auto &[key, edge] : index.edges)
    result.edges.push_back(std::move(edge));
  return result;
}
