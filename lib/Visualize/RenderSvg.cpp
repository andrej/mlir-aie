//===- RenderSvg.cpp --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Draws a DesignGraph on a fixed layout of the device.
//
// Each tile is a cell with its switchbox in the lower left corner. Wires to
// the tile above run up the band over the switchbox, one lane per physical
// wire; wires to the tile to the right run along the band to the right of
// it. DMA channels sit level with the switchbox port they attach to, then
// come the tile's buffers, then its kernel calls. Runtime arguments sit
// below the array, each on a bus of its own. Since every wire and port has a
// place of its own, two streams never share a line, and no routing search is
// needed. Contracted nodes are left out of the drawing while the edges
// through them keep their physical path.
//
//===----------------------------------------------------------------------===//

#include "aie/Visualize/DesignGraph.h"

#include "OrthogonalRouter.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <functional>
#include <map>
#include <set>

using namespace llvm;
using namespace xilinx::AIE::viz;

namespace {

// Measurements in SVG user units (pixels at 100%).
constexpr double kLane = 6;      // Pitch between parallel wires.
constexpr double kSwitchPad = 8; // Switchbox edge to its first lane.
constexpr double kSlot = 17;     // Pitch of DMA channels.
constexpr double kDMAWidth = 46;
constexpr double kDMAHeight = 13;
constexpr double kSwitchToDMA = 24; // Room for the switchbox-to-DMA link.
constexpr double kColumnGap = 40;   // Least room for links between columns.
constexpr double kTrackPitch = 5;   // Pitch of the tracks in that room.
constexpr double kBufferHeight = 15;
constexpr double kBufferPitch = 20;
constexpr double kCallHeight = 17;
constexpr double kCallPitch = 23;
constexpr double kTitleHeight = 20;
constexpr double kCellPad = 10;
// Pitch of the lanes along a tile's right and top edges that carry links to
// a neighbor's memory.
constexpr double kGutterPitch = 4;
constexpr double kCharWidth = 5.6; // Mean advance of a 9.5px label glyph.
constexpr double kEndPitch = 4.5;  // Least room per line end on a side.
constexpr double kArrowLength = 7;
constexpr double kArrowWidth = 5;
constexpr double kLineWidth = 1.4;
constexpr double kMargin = 24;

struct Point {
  double x = 0;
  double y = 0;
};

struct Box {
  double x = 0;
  double y = 0;
  double w = 0;
  double h = 0;
  double right() const { return x + w; }
  double bottom() const { return y + h; }
  Point center() const { return {x + w / 2, y + h / 2}; }
};

enum class Side { Left, Right, Top, Bottom };

/// The unit vector pointing into a box through the given side.
Point inward(Side side) {
  switch (side) {
  case Side::Left:
    return {1, 0};
  case Side::Right:
    return {-1, 0};
  case Side::Top:
    return {0, 1};
  case Side::Bottom:
    return {0, -1};
  }
  return {0, 0};
}

std::string num(double value) {
  return formatv("{0:f1}", std::round(value * 10) / 10).str();
}

std::string escape(StringRef text) {
  std::string result;
  for (char c : text) {
    switch (c) {
    case '&':
      result += "&amp;";
      break;
    case '<':
      result += "&lt;";
      break;
    case '>':
      result += "&gt;";
      break;
    case '"':
      result += "&quot;";
      break;
    default:
      result += c;
    }
  }
  return result;
}

std::string fit(StringRef label, double width) {
  size_t room = std::max<size_t>(3, static_cast<size_t>(width / kCharWidth));
  if (label.size() <= room)
    return label.str();
  return label.take_front(room - 1).str() + "…";
}

std::string formatBytes(uint64_t bytes) {
  static constexpr const char *units[] = {"B", "KiB", "MiB", "GiB"};
  double value = static_cast<double>(bytes);
  unsigned unit = 0;
  while (value >= 1024 && unit + 1 < std::size(units)) {
    value /= 1024;
    ++unit;
  }
  return formatv(value < 10 && unit ? "{0:f1} {1}" : "{0:f0} {1}", value,
                 units[unit]);
}

/// A drawn piece of an edge: an SVG path and the direction it travels at its
/// end, for the arrowhead.
struct Stroke {
  std::string d;
  Point start;
  Point startHeading;
  Point end;
  Point heading;
  /// Where an arrowhead is drawn: at the end, or mid-way along a wire.
  std::optional<std::pair<Point, Point>> chevron;
};

std::string cubic(Point a, Point b, Point c, Point d) {
  return "M" + num(a.x) + "," + num(a.y) + " C" + num(b.x) + "," + num(b.y) +
         " " + num(c.x) + "," + num(c.y) + " " + num(d.x) + "," + num(d.y);
}

/// An orthogonal polyline with its corners rounded off.
std::string rounded(ArrayRef<Point> points, double radius = 4) {
  std::string d = "M" + num(points[0].x) + "," + num(points[0].y);
  for (size_t i = 1; i + 1 < points.size(); ++i) {
    Point p = points[i];
    Point before = points[i - 1];
    Point after = points[i + 1];
    double inLength = std::hypot(p.x - before.x, p.y - before.y);
    double outLength = std::hypot(after.x - p.x, after.y - p.y);
    double r = std::min({radius, inLength / 2, outLength / 2});
    if (inLength == 0 || outLength == 0) {
      continue;
    }
    Point start{p.x - (p.x - before.x) / inLength * r,
                p.y - (p.y - before.y) / inLength * r};
    Point end{p.x + (after.x - p.x) / outLength * r,
              p.y + (after.y - p.y) / outLength * r};
    d += " L" + num(start.x) + "," + num(start.y) + " Q" + num(p.x) + "," +
         num(p.y) + " " + num(end.x) + "," + num(end.y);
  }
  d += " L" + num(points.back().x) + "," + num(points.back().y);
  return d;
}

Point unit(Point from, Point to) {
  double length = std::hypot(to.x - from.x, to.y - from.y);
  if (length == 0)
    return {0, 0};
  return {(to.x - from.x) / length, (to.y - from.y) / length};
}

std::string arrowhead(Point tip, Point heading) {
  Point base{tip.x - heading.x * kArrowLength,
             tip.y - heading.y * kArrowLength};
  Point normal{-heading.y * kArrowWidth / 2, heading.x * kArrowWidth / 2};
  return "M" + num(tip.x) + "," + num(tip.y) + " L" + num(base.x + normal.x) +
         "," + num(base.y + normal.y) + " L" + num(base.x - normal.x) + "," +
         num(base.y - normal.y) + "Z";
}

/// Qualitative colors that stay apart on white, darkened where the usual
/// tableau tints would be too pale for thin lines.
constexpr const char *kPalette[] = {"#2f6db3", "#e0731b", "#2e9a46", "#cf3a3c",
                                    "#7d58b8", "#16969e", "#b8860b", "#d4508e",
                                    "#8a5a3c", "#4d7f16", "#5465d6", "#b03a8a"};
constexpr const char *kMixed = "#5f6368";

class Renderer {
public:
  Renderer(const Graph &graph, const RenderOptions &options)
      : graph(graph), options(options) {}

  Expected<std::string> render();

private:
  struct Cell {
    const TileInfo *tile = nullptr;
    Box cell;
    Box switchbox;
    Box content;
    /// Local ports (DMA and the like) on the switchbox's right side.
    std::vector<std::string> slots;
    double laneBandTop = 0;
    double dmaX = 0;
    double bufferX = 0;
    double callX = 0;
  };

  void index();
  void layout();
  void measure();
  void assignTracks();
  void place();
  void placeArguments();
  void assignAnchors();
  Point portPoint(const Node &port) const;
  Side portSide(const Node &port) const;
  Side sideFacing(const Node &node, const Node &other) const;
  Stroke strokeFor(const std::string &from, const std::string &to);
  void computeColors();
  Error routeEdges();
  /// The box a node is drawn as: its own, or the one around a merged group.
  Box drawnBoxOf(const Node &node) const;

  /// Whether a node is drawn, by itself or as part of the group it was
  /// merged into.
  bool isVisible(const std::string &id) const {
    const Node &node = *nodes.at(id);
    if (!node.hidden)
      return true;
    return !node.mergedInto.empty() && !nodes.at(node.mergedInto)->hidden;
  }
  bool isBox(const Node &node) const { return node.kind != kind::Port; }
  std::pair<int, int> tileOf(const Node &node) const {
    return {node.col, node.row};
  }

  const Graph &graph;
  const RenderOptions &options;

  std::map<std::string, const Node *> nodes;
  /// Every edge of the full graph, as the segments the view's edges list.
  std::vector<std::pair<std::string, std::string>> segments;
  std::map<std::pair<int, int>, Cell> cells;
  std::map<std::pair<int, int>, std::vector<std::string>> buffersByTile;
  std::map<std::pair<int, int>, std::vector<std::string>> callsByTile;
  std::map<std::pair<int, int>, std::vector<std::string>> dmasByTile;
  std::map<std::string, Box> boxes;
  /// Routes of edges that do not follow their physical path, by edge ID.
  std::map<std::string, std::vector<Point>> routes;
  std::map<std::string, double> boxHeight;
  /// Slot of each local port and DMA channel on its switchbox's right side.
  std::map<std::string, size_t> slotOf;
  std::map<std::pair<std::string, std::string>, size_t> segmentIndex;
  /// Anchor of each segment end on a box: (segment index, at source).
  std::map<std::pair<size_t, bool>, Point> anchors;
  std::map<std::string, std::string> colors;
  std::vector<std::pair<std::string, std::string>> legend;

  unsigned upLanes = 0, downLanes = 0;
  std::map<int, unsigned> eastLanes, westLanes;
  double switchWidth = 0;
  double bufferWidth = 0;
  double callWidth = 0;
  double cellWidth = 0;
  std::map<int, double> rowHeight;
  std::map<int, unsigned> rowSlots;
  int minCol = 0, maxCol = 0, maxRow = 0;
  double arrayBottom = 0;
  double width = 0, height = 0;

  /// Runtime argument geometry: the bus of each argument and direction
  /// (true: the argument is read), the stem joining it to its box, and the
  /// lane in the shim each argument-to-DMA segment drops down.
  std::map<std::pair<std::string, bool>, double> argumentBus;
  std::map<std::pair<std::string, bool>, double> argumentStem;
  std::map<size_t, double> argumentTap;
  /// The gutter lane of each link between two tiles' local resources.
  std::map<size_t, unsigned> gutterLane;
  /// Room between a tile's DMA and buffer columns and between its buffer and
  /// call columns, and the track each link between them runs down.
  /// Room along each tile's right and top edges for the gutter lanes.
  unsigned gutterLanes = 1;
  double rightGutter() const { return 8 + gutterLanes * kGutterPitch; }
  /// Room at the left of each column gap, before its first track, for the
  /// lanes of links to a neighbor's memory.
  double trackMargin() const { return 8 + gutterLanes * kGutterPitch; }
  double topGutter() const { return 6 + gutterLanes * kGutterPitch; }
  double dmaGap = kColumnGap;
  double callGap = kColumnGap;
  std::map<size_t, int> track;
  unsigned dmaTracks = 0, callTracks = 0;
  bool crossesTiles(const Node &a, const Node &b) const {
    return a.kind != kind::Port && b.kind != kind::Port &&
           a.kind != kind::Argument && b.kind != kind::Argument &&
           tileOf(a) != tileOf(b);
  }
};

void Renderer::index() {
  for (const Node &node : graph.nodes)
    nodes[node.id] = &node;
  std::set<std::pair<std::string, std::string>> seen;
  for (const Edge &edge : graph.edges)
    for (const auto &segment : edge.segments)
      if (nodes.count(segment.first) && nodes.count(segment.second) &&
          seen.insert(segment).second) {
        segmentIndex[segment] = segments.size();
        segments.push_back(segment);
      }
}

void Renderer::measure() {
  minCol = std::numeric_limits<int>::max();
  maxCol = 0;
  for (const TileInfo &tile : graph.tiles) {
    minCol = std::min(minCol, tile.col);
    maxCol = std::max(maxCol, tile.col);
    maxRow = std::max(maxRow, tile.row);
    auto count = [&](StringRef bundle, bool in) -> unsigned {
      auto found = tile.ports.find(bundle.str());
      if (found == tile.ports.end())
        return 0;
      return in ? found->second.first : found->second.second;
    };
    upLanes = std::max({upLanes, count("North", false),
                        tile.type == "shim" ? 0u : count("South", true)});
    downLanes = std::max({downLanes, count("North", true),
                          tile.type == "shim" ? 0u : count("South", false)});
    eastLanes[tile.row] = std::max(
        {eastLanes[tile.row], count("East", false), count("West", true)});
    westLanes[tile.row] = std::max(
        {westLanes[tile.row], count("West", false), count("East", true)});
    cells[{tile.col, tile.row}].tile = &tile;
  }
  if (cells.empty())
    minCol = 0;
  switchWidth = 2 * kSwitchPad + (upLanes + downLanes - 1) * kLane;

  for (const Node &node : graph.nodes) {
    if (!node.hasTile())
      continue;
    if (node.kind == kind::Buffer)
      buffersByTile[tileOf(node)].push_back(node.id);
    else if (node.kind == kind::Call)
      callsByTile[tileOf(node)].push_back(node.id);
    else if (node.kind == kind::DMA)
      dmasByTile[tileOf(node)].push_back(node.id);
  }
  // A box grows with the lines meeting its busiest side, so that line ends
  // never crowd together.
  std::map<std::pair<std::string, bool>, unsigned> ends;
  for (const auto &[from, to] : segments) {
    const Node &a = *nodes.at(from);
    const Node &b = *nodes.at(to);
    auto note = [&](const Node &self, const Node &other) {
      bool right = (self.kind == kind::Buffer && other.kind == kind::Call) ||
                   self.kind == kind::DMA;
      ++ends[{self.id, right}];
    };
    note(a, b);
    note(b, a);
  }
  for (const Node &node : graph.nodes) {
    double base = node.kind == kind::Call ? kCallHeight : kBufferHeight;
    unsigned most = std::max(ends[{node.id, false}], ends[{node.id, true}]);
    boxHeight[node.id] = std::max(base, (most + 1) * kEndPitch);
  }
  bufferWidth = 90;
  callWidth = 90;
  for (const Node &node : graph.nodes) {
    double needed = node.label.size() * kCharWidth + 12;
    if (node.kind == kind::Buffer)
      bufferWidth = std::max(bufferWidth, std::min(needed, 170.0));
    if (node.kind == kind::Call)
      callWidth = std::max(callWidth, std::min(needed, 170.0));
  }

  // Local slots: each DMA channel, with the switchbox port it attaches to,
  // then any other local port a stream uses.
  std::map<std::string, std::string> portOfDMA;
  for (const auto &[from, to] : segments) {
    const Node &a = *nodes.at(from);
    const Node &b = *nodes.at(to);
    if (a.kind == kind::DMA && b.kind == kind::Port)
      portOfDMA[from] = to;
    if (b.kind == kind::DMA && a.kind == kind::Port)
      portOfDMA[to] = from;
  }
  std::set<std::string> slotted;
  for (auto &[coordinates, cell] : cells) {
    std::vector<const Node *> channels;
    for (const std::string &id : dmasByTile[coordinates])
      channels.push_back(nodes.at(id));
    llvm::stable_sort(channels, [](const Node *a, const Node *b) {
      return std::make_tuple(a->direction != "S2MM", a->channel) <
             std::make_tuple(b->direction != "S2MM", b->channel);
    });
    for (const Node *channel : channels) {
      slotOf[channel->id] = cell.slots.size();
      slotted.insert(channel->id);
      auto port = portOfDMA.find(channel->id);
      if (port != portOfDMA.end()) {
        slotOf[port->second] = cell.slots.size();
        slotted.insert(port->second);
      }
      cell.slots.push_back(channel->id);
    }
  }
  for (const Node &node : graph.nodes) {
    if (node.kind != kind::Port || slotted.count(node.id))
      continue;
    auto found = cells.find(tileOf(node));
    if (found == cells.end())
      continue;
    bool sideBundle =
        node.bundle == "North" || node.bundle == "East" ||
        node.bundle == "West" ||
        (node.bundle == "South" && found->second.tile->type != "shim");
    if (!sideBundle) {
      slotOf[node.id] = found->second.slots.size();
      found->second.slots.push_back(node.id);
    }
  }

  cellWidth = switchWidth + kSwitchToDMA + kDMAWidth + dmaGap + bufferWidth +
              callGap + callWidth + kCellPad + rightGutter();
  for (auto &[coordinates, cell] : cells) {
    int row = coordinates.second;
    rowSlots[row] = std::max<unsigned>(rowSlots[row], cell.slots.size());
    auto stackHeight = [&](const std::vector<std::string> &ids, double pitch,
                           double base) {
      double total = 0;
      for (const std::string &id : ids)
        total += pitch - base + boxHeight[id];
      return total;
    };
    double stack = std::max(
        {cell.slots.size() * kSlot,
         stackHeight(buffersByTile[coordinates], kBufferPitch, kBufferHeight),
         stackHeight(callsByTile[coordinates], kCallPitch, kCallHeight)});
    double band = 2 * kSwitchPad + (eastLanes[row] + westLanes[row]) * kLane;
    rowHeight[row] = std::max(rowHeight[row], topGutter() + kTitleHeight +
                                                  stack + band + 2 * kCellPad);
  }
}

void Renderer::place() {
  double y = kMargin + 28;
  std::map<int, double> rowTop;
  for (int row = maxRow; row >= 0; --row) {
    rowTop[row] = y;
    y += rowHeight[row];
  }
  arrayBottom = y;

  for (auto &[coordinates, cell] : cells) {
    auto [col, row] = coordinates;
    cell.cell = {kMargin + (col - minCol) * cellWidth, rowTop[row], cellWidth,
                 rowHeight[row]};
    double bottom = cell.cell.bottom();
    unsigned lanes = eastLanes[row] + westLanes[row];
    cell.laneBandTop = bottom - 2 * kSwitchPad - lanes * kLane;
    double slotsHeight = cell.slots.size() * kSlot;
    double switchTop = cell.laneBandTop - slotsHeight - kSwitchPad;
    if (cell.slots.empty())
      switchTop = std::min(switchTop, bottom - 2 * kSwitchPad - kLane * 4);
    cell.switchbox = {cell.cell.x, switchTop, switchWidth, bottom - switchTop};
    cell.dmaX = cell.switchbox.right() + kSwitchToDMA;
    cell.bufferX = cell.dmaX + kDMAWidth + dmaGap;
    cell.callX = cell.bufferX + bufferWidth + callGap;
    cell.content = {cell.switchbox.right() + 6, cell.cell.y + topGutter(),
                    cell.cell.right() - cell.switchbox.right() - 6 - kCellPad -
                        rightGutter(),
                    cell.laneBandTop - cell.cell.y - topGutter()};

    for (auto [index, id] : enumerate(cell.slots)) {
      const Node &node = *nodes.at(id);
      if (node.kind != kind::DMA)
        continue;
      double slotY = cell.laneBandTop - slotsHeight + (index + 0.5) * kSlot;
      boxes[id] = {cell.dmaX, slotY - kDMAHeight / 2, kDMAWidth, kDMAHeight};
    }
  }

  // Order a tile's buffers by the height of the DMA channels they meet, and
  // its calls by the height of their buffers, so links cross as little as
  // the design allows.
  auto meanY = [&](const std::string &id) {
    double sum = 0;
    unsigned count = 0;
    for (const auto &[from, to] : segments) {
      const std::string *other = from == id ? &to : to == id ? &from : nullptr;
      if (!other)
        continue;
      auto box = boxes.find(*other);
      if (box == boxes.end())
        continue;
      sum += box->second.center().y;
      ++count;
    }
    return count ? sum / count : -1.0;
  };
  // Each box sits as close to the height of what it connects to as the
  // boxes above and below it allow.
  auto stack = [&](std::vector<std::string> &ids, double x, double w, double h,
                   double pitch, const Cell &cell) {
    if (ids.empty())
      return;
    double top = cell.content.y + kTitleHeight;
    double bottom = cell.laneBandTop;
    double gap = pitch - h;
    std::vector<std::pair<double, std::string>> ordered;
    for (const std::string &id : ids) {
      double target = meanY(id);
      ordered.emplace_back(target < 0 ? top : target - boxHeight[id] / 2, id);
    }
    llvm::sort(ordered);
    std::vector<double> ys;
    for (auto [index, entry] : enumerate(ordered))
      ys.push_back(std::max(
          entry.first,
          index ? ys.back() + boxHeight[ordered[index - 1].second] + gap
                : top));
    double next = bottom;
    for (int index = ys.size() - 1; index >= 0; --index) {
      double height = boxHeight[ordered[index].second];
      ys[index] = std::min(ys[index], next - height - gap / 2);
      next = ys[index] - gap / 2;
    }
    for (auto [y, entry] : zip_equal(ys, ordered))
      boxes[entry.second] = {x, y, w, boxHeight[entry.second]};
  };
  for (auto &[coordinates, cell] : cells) {
    stack(buffersByTile[coordinates], cell.bufferX, bufferWidth, kBufferHeight,
          kBufferPitch, cell);
    stack(callsByTile[coordinates], cell.callX, callWidth, kCallHeight,
          kCallPitch, cell);
  }
  std::map<std::pair<int, int>, unsigned> lanesTaken;
  for (auto [index, segment] : enumerate(segments)) {
    const Node &a = *nodes.at(segment.first);
    const Node &b = *nodes.at(segment.second);
    if (!crossesTiles(a, b) || !cells.count(tileOf(a)) ||
        !cells.count(tileOf(b)))
      continue;
    unsigned lane = std::max(lanesTaken[tileOf(a)], lanesTaken[tileOf(b)]);
    gutterLane[index] = lane;
    lanesTaken[tileOf(a)] = lanesTaken[tileOf(b)] = lane + 1;
  }
  placeArguments();
}

void Renderer::placeArguments() {
  std::vector<const Node *> arguments;
  for (const Node &node : graph.nodes)
    if (node.kind == kind::Argument)
      arguments.push_back(&node);
  if (arguments.empty()) {
    height = arrayBottom + kMargin;
    return;
  }

  // Each segment between an argument and a shim DMA drops from the DMA down
  // its own lane in the shim tile's empty content area. Segments leaving
  // lower on the DMA column take lanes further left, so no drop crosses
  // another segment's way out.
  std::map<std::pair<int, int>, std::vector<std::pair<double, size_t>>> drops;
  std::map<std::string, std::vector<double>> taps;
  for (auto [index, segment] : enumerate(segments)) {
    const Node &a = *nodes.at(segment.first);
    const Node &b = *nodes.at(segment.second);
    const Node *channel = a.kind == kind::Argument   ? &b
                          : b.kind == kind::Argument ? &a
                                                     : nullptr;
    if (!channel || !boxes.count(channel->id))
      continue;
    drops[tileOf(*channel)].emplace_back(boxes[channel->id].center().y, index);
  }
  for (auto &[coordinates, list] : drops) {
    llvm::stable_sort(
        list, [](const auto &a, const auto &b) { return a.first > b.first; });
    const Cell &cell = cells.at(coordinates);
    for (auto [lane, entry] : enumerate(list))
      argumentTap[entry.second] =
          cell.dmaX + kDMAWidth + 16 + lane * (kLane + 1);
  }

  // An argument the design reads and one it writes each get a bus; one it
  // both reads and writes gets two.
  std::set<std::pair<std::string, bool>> directions;
  for (const auto &[from, to] : segments) {
    if (nodes.at(from)->kind == kind::Argument)
      directions.insert({from, true});
    if (nodes.at(to)->kind == kind::Argument)
      directions.insert({to, false});
  }
  double busTop = arrayBottom + 14;
  unsigned buses = 0;
  for (const Node *argument : arguments)
    for (bool reads : {true, false})
      if (directions.count({argument->id, reads}))
        argumentBus[{argument->id, reads}] = busTop + buses++ * (kLane + 2);
  double boxTop = busTop + buses * (kLane + 2) + 14;

  for (auto [index, segment] : enumerate(segments)) {
    auto tap = argumentTap.find(index);
    if (tap == argumentTap.end())
      continue;
    const std::string &argument =
        nodes.at(segment.first)->kind == kind::Argument ? segment.first
                                                        : segment.second;
    taps[argument].push_back(tap->second);
  }

  // Each argument sits under the middle of the drops it feeds, moved right
  // where it would overlap its neighbor or put its own stem on a drop.
  std::set<double> dropXs;
  for (const auto &[segment, x] : argumentTap)
    dropXs.insert(x);
  std::vector<std::pair<double, const Node *>> ordered;
  for (const Node *argument : arguments) {
    std::vector<double> &xs = taps[argument->id];
    double center = xs.empty() ? kMargin
                               : (*std::min_element(xs.begin(), xs.end()) +
                                  *std::max_element(xs.begin(), xs.end())) /
                                     2;
    ordered.emplace_back(center, argument);
  }
  llvm::stable_sort(
      ordered, [](const auto &a, const auto &b) { return a.first < b.first; });
  double nextFree = kMargin;
  for (auto &[center, argument] : ordered) {
    double w = std::max(60.0, argument->label.size() * kCharWidth + 18);
    double x = std::max(center - w / 2, nextFree);
    auto clearOfDrops = [&](double stem) {
      return llvm::none_of(
          dropXs, [&](double drop) { return std::abs(drop - stem) < 4; });
    };
    bool both = directions.count({argument->id, true}) &&
                directions.count({argument->id, false});
    auto stems = [&](double left) {
      return both ? std::make_pair(left + w / 3, left + 2 * w / 3)
                  : std::make_pair(left + w / 2, left + w / 2);
    };
    while (!clearOfDrops(stems(x).first) || !clearOfDrops(stems(x).second))
      x += 5;
    boxes[argument->id] = {x, boxTop, w, 20};
    argumentStem[{argument->id, true}] = stems(x).first;
    argumentStem[{argument->id, false}] = stems(x).second;
    nextFree = x + w + 12;
  }
  height = boxTop + 20 + kMargin + 20;
}

Point Renderer::portPoint(const Node &port) const {
  const Cell &cell = cells.at(tileOf(port));
  const Box &box = cell.switchbox;
  bool in = port.direction == "in";
  unsigned channel = std::max(port.channel, 0);
  auto upX = [&](unsigned lane) {
    return box.x + kSwitchPad + std::min(lane, upLanes - 1) * kLane;
  };
  auto downX = [&](unsigned lane) {
    return box.x + kSwitchPad +
           (upLanes + std::min(lane, downLanes - 1)) * kLane;
  };
  int row = port.row;
  auto eastY = [&](unsigned lane) {
    return box.bottom() - kSwitchPad -
           std::min(lane, std::max(eastLanes.at(row), 1u) - 1) * kLane;
  };
  auto westY = [&](unsigned lane) {
    return box.bottom() - kSwitchPad -
           (eastLanes.at(row) +
            std::min(lane, std::max(westLanes.at(row), 1u) - 1)) *
               kLane;
  };
  if (port.bundle == "North")
    return {in ? downX(channel) : upX(channel), box.y};
  if (port.bundle == "South" && cell.tile->type != "shim")
    return {in ? upX(channel) : downX(channel), box.bottom()};
  if (port.bundle == "East")
    return {box.right(), in ? westY(channel) : eastY(channel)};
  if (port.bundle == "West")
    return {box.x, in ? eastY(channel) : westY(channel)};

  // A local port takes the slot of the DMA channel it attaches to, or a slot
  // of its own.
  auto slot = slotOf.find(port.id);
  if (slot == slotOf.end())
    return {box.right(), box.y + kSwitchPad};
  return {box.right(), cell.laneBandTop - cell.slots.size() * kSlot +
                           (slot->second + 0.5) * kSlot};
}

Side Renderer::portSide(const Node &port) const {
  const Cell &cell = cells.at(tileOf(port));
  if (port.bundle == "North")
    return Side::Top;
  if (port.bundle == "South" && cell.tile->type != "shim")
    return Side::Bottom;
  if (port.bundle == "West")
    return Side::Left;
  return Side::Right;
}

Side Renderer::sideFacing(const Node &node, const Node &other) const {
  if (node.kind == kind::Argument)
    return Side::Top;
  if (crossesTiles(node, other))
    return Side::Right;
  if (node.kind == kind::DMA)
    return other.kind == kind::Port ? Side::Left : Side::Right;
  if (node.kind == kind::Buffer)
    return other.kind == kind::Call ? Side::Right : Side::Left;
  return Side::Left;
}

void Renderer::assignAnchors() {
  // Ends meeting the same side of a box spread along it, in the order of
  // where their other ends lie, so that they neither overlap nor cross on
  // the way in.
  std::map<std::pair<std::string, Side>,
           std::vector<std::tuple<double, size_t, bool>>>
      ends;
  auto otherPosition = [&](const Node &other) -> Point {
    if (other.kind == kind::Port)
      return portPoint(other);
    return boxes.at(other.id).center();
  };
  for (auto [index, segment] : enumerate(segments)) {
    const Node &a = *nodes.at(segment.first);
    const Node &b = *nodes.at(segment.second);
    for (auto [self, other, atSource] :
         {std::make_tuple(&a, &b, true), std::make_tuple(&b, &a, false)}) {
      if (!isBox(*self) || !boxes.count(self->id))
        continue;
      if (other->kind != kind::Port && !boxes.count(other->id))
        continue;
      Side side = sideFacing(*self, *other);
      Point position = otherPosition(*other);
      double key =
          side == Side::Top || side == Side::Bottom ? position.x : position.y;
      if (self->kind == kind::Argument)
        key = argumentTap.count(index) ? argumentTap.at(index) : position.x;
      ends[{self->id, side}].emplace_back(key, index, atSource);
    }
  }
  for (auto &[key, list] : ends) {
    const Box &box = boxes.at(key.first);
    llvm::stable_sort(list, [](const auto &a, const auto &b) {
      return std::get<0>(a) < std::get<0>(b);
    });
    // An argument's segments leave from one stem per direction.
    if (nodes.at(key.first)->kind == kind::Argument) {
      for (auto &[unused, index, atSource] : list)
        anchors[{index, atSource}] = {argumentStem.at({key.first, atSource}),
                                      box.y};
      continue;
    }
    for (auto [position, entry] : enumerate(list)) {
      auto &[unused, index, atSource] = entry;
      double t = (position + 1.0) / (list.size() + 1.0);
      Point point;
      switch (key.second) {
      case Side::Left:
        point = {box.x, box.y + t * box.h};
        break;
      case Side::Right:
        point = {box.right(), box.y + t * box.h};
        break;
      case Side::Top:
        point = {box.x + t * box.w, box.y};
        break;
      case Side::Bottom:
        point = {box.x + t * box.w, box.bottom()};
        break;
      }
      anchors[{index, atSource}] = point;
    }
  }
}

Stroke Renderer::strokeFor(const std::string &from, const std::string &to) {
  size_t index = segmentIndex.at({from, to});
  const Node &a = *nodes.at(from);
  const Node &b = *nodes.at(to);
  auto endOf = [&](const Node &node, bool atSource) {
    if (node.kind == kind::Port)
      return portPoint(node);
    auto anchor = anchors.find({index, atSource});
    if (anchor != anchors.end())
      return anchor->second;
    return boxes.at(node.id).center();
  };
  Point start = endOf(a, true);
  Point end = endOf(b, false);
  Stroke stroke;
  stroke.start = start;
  stroke.end = end;

  // Inside a switchbox: leave each port heading into the box.
  if (a.kind == kind::Port && b.kind == kind::Port && tileOf(a) == tileOf(b)) {
    const Box &box = cells.at(tileOf(a)).switchbox;
    // Short leads keep connections nearly straight, so that two of them
    // cross at a clear angle instead of running side by side.
    double reach = std::min(10.0, std::min(box.w, box.h) * 0.2);
    Point in = inward(portSide(a));
    Point out = inward(portSide(b));
    if (portSide(a) == portSide(b)) {
      double across = portSide(a) == Side::Left || portSide(a) == Side::Right
                          ? box.w
                          : box.h;
      reach = std::min(6 + std::abs(start.x - end.x + start.y - end.y) * 0.6,
                       across * 0.75);
    }
    Point c1{start.x + in.x * reach, start.y + in.y * reach};
    Point c2{end.x + out.x * reach, end.y + out.y * reach};
    stroke.d = cubic(start, c1, c2, end);
    stroke.startHeading = unit(start, c1);
    stroke.heading = unit(c2, end);
    return stroke;
  }
  // A wire between neighboring switchboxes: straight down its lane.
  if (a.kind == kind::Port && b.kind == kind::Port) {
    std::vector<Point> points{start};
    if (start.x != end.x && start.y != end.y) {
      bool vertical = portSide(a) == Side::Top || portSide(a) == Side::Bottom;
      if (vertical) {
        double middle = (start.y + end.y) / 2;
        points.push_back({start.x, middle});
        points.push_back({end.x, middle});
      } else {
        double middle = (start.x + end.x) / 2;
        points.push_back({middle, start.y});
        points.push_back({middle, end.y});
      }
    }
    points.push_back(end);
    stroke.d = rounded(points);
    stroke.startHeading = unit(points[0], points[1]);
    stroke.heading = unit(points[points.size() - 2], end);
    Point middle{(start.x + end.x) / 2, (start.y + end.y) / 2};
    if (std::hypot(end.x - start.x, end.y - start.y) > 30)
      stroke.chevron = {{middle.x + stroke.heading.x * kArrowLength / 2,
                         middle.y + stroke.heading.y * kArrowLength / 2},
                        stroke.heading};
    return stroke;
  }
  // Between an argument and a shim DMA: out of the DMA, down the segment's
  // own lane, along the argument's bus, and down its stem.
  if (a.kind == kind::Argument || b.kind == kind::Argument) {
    const Node &argument = a.kind == kind::Argument ? a : b;
    Point channelEnd = a.kind == kind::Argument ? end : start;
    Point stem = a.kind == kind::Argument ? start : end;
    double lane =
        argumentTap.count(index) ? argumentTap.at(index) : channelEnd.x + 16;
    double bus = argumentBus.at({argument.id, a.kind == kind::Argument});
    std::vector<Point> points{
        channelEnd, {lane, channelEnd.y}, {lane, bus}, {stem.x, bus}, stem};
    if (a.kind == kind::Argument)
      std::reverse(points.begin(), points.end());
    stroke.d = rounded(points);
    stroke.startHeading = unit(points[0], points[1]);
    stroke.heading = unit(points[points.size() - 2], points.back());
    return stroke;
  }
  // To a neighbor's memory: each end leaves its box for a lane of its own
  // just right of its column, which is free of boxes, and climbs to its
  // tile's top gutter. Tiles side by side are joined along that gutter;
  // tiles above one another, through the right gutter of their column.
  if (crossesTiles(a, b)) {
    unsigned lane = gutterLane.count(index) ? gutterLane.at(index) : 0;
    double offset = 4 + lane * kGutterPitch;
    auto exitX = [&](const Node &node) {
      const Cell &cell = cells.at(tileOf(node));
      if (node.kind == kind::DMA)
        return cell.dmaX + kDMAWidth + offset;
      if (node.kind == kind::Buffer)
        return cell.bufferX + bufferWidth + offset;
      return cell.cell.right() - rightGutter() + offset;
    };
    auto topY = [&](const Node &node) {
      return cells.at(tileOf(node)).cell.y + offset;
    };
    std::vector<Point> points{start, {exitX(a), start.y}, {exitX(a), topY(a)}};
    if (a.row != b.row) {
      double gutter = cells.at(tileOf(a)).cell.right() - rightGutter() + offset;
      points.push_back({gutter, topY(a)});
      points.push_back({gutter, topY(b)});
    }
    points.push_back({exitX(b), topY(b)});
    points.push_back({exitX(b), end.y});
    points.push_back(end);
    stroke.d = rounded(points);
    stroke.startHeading = unit(points[0], points[1]);
    stroke.heading = unit(points[points.size() - 2], end);
    return stroke;
  }
  // Between columns of a tile: along the link's own track.
  if (auto found = track.find(index); found != track.end()) {
    if (found->second < 0) {
      stroke.d = "M" + num(start.x) + "," + num(start.y) + " L" + num(end.x) +
                 "," + num(end.y);
      stroke.heading = stroke.startHeading = unit(start, end);
      return stroke;
    }
    bool rightward = start.x < end.x;
    double gapLeft = rightward ? start.x : end.x;
    double x = gapLeft + trackMargin() + found->second * kTrackPitch;
    std::vector<Point> points{start, {x, start.y}, {x, end.y}, end};
    stroke.d = rounded(points, 3);
    stroke.startHeading = unit(points[0], points[1]);
    stroke.heading = unit(points[2], end);
    return stroke;
  }
  // Between a port and its DMA channel: leave and arrive level.
  Point exit = a.kind == kind::Port ? Point{1, 0}
                                    : Point{-inward(sideFacing(a, b)).x,
                                            -inward(sideFacing(a, b)).y};
  Point entry = b.kind == kind::Port ? Point{-1, 0} : inward(sideFacing(b, a));
  double reach = std::max(12.0, std::abs(end.x - start.x) * 0.5);
  if (start.y == end.y && exit.x == entry.x) {
    stroke.d = "M" + num(start.x) + "," + num(start.y) + " L" + num(end.x) +
               "," + num(end.y);
    stroke.heading = stroke.startHeading = unit(start, end);
    return stroke;
  }
  Point c1{start.x + exit.x * reach, start.y + exit.y * reach};
  Point c2{end.x - entry.x * reach, end.y - entry.y * reach};
  stroke.d = cubic(start, c1, c2, end);
  stroke.startHeading = exit;
  stroke.heading = entry;
  return stroke;
}

void Renderer::computeColors() {
  // An edge is colored after where its data comes from: a runtime argument,
  // a kernel, or a stream with no source in the design. Calls of one kernel
  // share a color. Where data from several sources mixes, the edge is gray.
  std::map<std::string, std::vector<const Edge *>> incoming;
  for (const Edge &edge : graph.edges)
    incoming[edge.target].push_back(&edge);
  std::map<std::string, std::set<std::string>> roots;
  std::set<std::string> visiting;
  std::function<const std::set<std::string> &(const std::string &)> rootsOf =
      [&](const std::string &id) -> const std::set<std::string> & {
    auto found = roots.find(id);
    if (found != roots.end())
      return found->second;
    const Node &node = *nodes.at(id);
    std::set<std::string> result;
    if (node.kind == kind::Call) {
      result.insert("kernel:" + node.label);
    } else if (node.kind == kind::Argument || incoming[id].empty() ||
               !visiting.insert(id).second) {
      result.insert(id);
    } else {
      for (const Edge *edge : incoming[id]) {
        const std::set<std::string> &upstream = rootsOf(edge->source);
        result.insert(upstream.begin(), upstream.end());
      }
      visiting.erase(id);
    }
    return roots[id] = std::move(result);
  };

  std::vector<std::string> order;
  for (const Node &node : graph.nodes)
    if (node.kind == kind::Argument)
      order.push_back(node.id);
  for (const Edge &edge : graph.edges) {
    const std::set<std::string> &sources = rootsOf(edge.source);
    if (sources.size() == 1 && !is_contained(order, *sources.begin()))
      order.push_back(*sources.begin());
  }
  std::map<std::string, std::string> rootColor;
  for (auto [index, root] : enumerate(order))
    rootColor[root] = kPalette[index % std::size(kPalette)];
  for (const Edge &edge : graph.edges) {
    const std::set<std::string> &sources = rootsOf(edge.source);
    colors[edge.id()] =
        sources.size() == 1 ? rootColor[*sources.begin()] : kMixed;
  }
  std::set<std::string> feeding;
  for (const Edge &edge : graph.edges)
    feeding.insert(edge.source);
  for (const std::string &root : order) {
    if (StringRef(root).starts_with("kernel:"))
      legend.emplace_back(rootColor[root], StringRef(root).drop_front(7).str());
    else if (nodes.at(root)->kind == kind::Argument && feeding.count(root))
      legend.emplace_back(rootColor[root], nodes.at(root)->label);
  }
}

void Renderer::layout() {
  cells.clear();
  buffersByTile.clear();
  callsByTile.clear();
  dmasByTile.clear();
  boxes.clear();
  boxHeight.clear();
  slotOf.clear();
  anchors.clear();
  rowHeight.clear();
  rowSlots.clear();
  eastLanes.clear();
  westLanes.clear();
  argumentBus.clear();
  argumentStem.clear();
  argumentTap.clear();
  gutterLane.clear();
  track.clear();
  upLanes = downLanes = 0;
  measure();
  place();
  assignAnchors();
  assignTracks();
}

/// Gives every link between two columns of a tile a vertical track of its
/// own, so that links cross only at right angles. Links whose vertical runs
/// overlap take different tracks (interval coloring); a link leaving at the
/// height another arrives stays left of it, so their horizontal runs do not
/// meet.
void Renderer::assignTracks() {
  struct Link {
    size_t index;
    double left, right;
  };
  std::map<std::pair<std::pair<int, int>, bool>, std::vector<Link>> gaps;
  for (auto [index, segment] : enumerate(segments)) {
    const Node &a = *nodes.at(segment.first);
    const Node &b = *nodes.at(segment.second);
    if (a.kind == kind::Port || b.kind == kind::Port ||
        a.kind == kind::Argument || b.kind == kind::Argument ||
        crossesTiles(a, b) || !anchors.count({index, true}) ||
        !anchors.count({index, false}))
      continue;
    bool toCalls = a.kind == kind::Call || b.kind == kind::Call;
    bool aLeft = boxes.at(a.id).x < boxes.at(b.id).x;
    double ya = anchors.at({index, true}).y;
    double yb = anchors.at({index, false}).y;
    gaps[{tileOf(a), toCalls}].push_back(
        {index, aLeft ? ya : yb, aLeft ? yb : ya});
  }
  for (auto &[key, links] : gaps) {
    llvm::sort(links, [](const Link &x, const Link &y) {
      return std::min(x.left, x.right) < std::min(y.left, y.right);
    });
    // A link leaving at the height another arrives must run left of it, or
    // their horizontal runs would share that height. Links are placed in an
    // order that respects this, each on the lowest free track right of every
    // link it must follow.
    size_t count = links.size();
    std::vector<std::vector<size_t>> after(count);
    std::vector<unsigned> waiting(count, 0);
    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < count; ++j)
        if (i != j && std::abs(links[i].left - links[j].right) < 2) {
          after[i].push_back(j);
          ++waiting[j];
        }
    std::vector<int> assigned(count, -1);
    std::vector<int> leftmost(count, 0);
    std::vector<bool> done(count, false);
    unsigned used = 0;
    for (size_t step = 0; step < count; ++step) {
      // The first link, in order of height, with nothing left to wait for;
      // a cycle of constraints is broken at its lowest link.
      size_t next = count;
      for (size_t i = 0; i < count && next == count; ++i)
        if (!done[i] && waiting[i] == 0)
          next = i;
      for (size_t i = 0; i < count && next == count; ++i)
        if (!done[i])
          next = i;
      done[next] = true;
      const Link &link = links[next];
      for (size_t j : after[next]) {
        --waiting[j];
      }
      if (std::abs(link.left - link.right) < 0.5) {
        track[link.index] = -1;
        continue;
      }
      double lo = std::min(link.left, link.right) - 3;
      double hi = std::max(link.left, link.right) + 3;
      int candidate = leftmost[next];
      for (;; ++candidate) {
        bool free = true;
        for (size_t i = 0; i < count && free; ++i) {
          if (assigned[i] != candidate)
            continue;
          double otherLo = std::min(links[i].left, links[i].right);
          double otherHi = std::max(links[i].left, links[i].right);
          free = !(otherLo < hi && lo < otherHi);
        }
        if (free)
          break;
      }
      assigned[next] = candidate;
      track[link.index] = candidate;
      used = std::max<unsigned>(used, candidate + 1);
      for (size_t j : after[next])
        leftmost[j] = std::max(leftmost[j], candidate + 1);
    }
    (key.second ? callTracks : dmaTracks) =
        std::max(key.second ? callTracks : dmaTracks, used);
  }
}

Expected<std::string> Renderer::render() {
  index();
  layout();
  // The gutters hold every neighbor link a tile has, and the rooms between
  // columns hold those links' lanes and every track of their busiest tile.
  unsigned neededLanes = 1;
  for (const auto &[segment, lane] : gutterLane)
    neededLanes = std::max(neededLanes, lane + 1);
  unsigned lanesBefore = gutterLanes;
  gutterLanes = neededLanes;
  double neededDMAGap =
      std::max(kColumnGap, trackMargin() + dmaTracks * kTrackPitch + 10);
  double neededCallGap =
      std::max(kColumnGap, trackMargin() + callTracks * kTrackPitch + 10);
  if (neededDMAGap != dmaGap || neededCallGap != callGap ||
      neededLanes != lanesBefore) {
    dmaGap = neededDMAGap;
    callGap = neededCallGap;
    layout();
  }
  computeColors();
  if (options.routing != RenderOptions::Routing::Physical)
    if (Error error = routeEdges())
      return std::move(error);
  width = 2 * kMargin + (maxCol - minCol + 1) * cellWidth;
  for (const auto &[id, box] : boxes)
    width = std::max(width, box.right() + kMargin);

  // The legend says what the colors mean, in rows under the drawing.
  std::string legendSvg;
  if (!legend.empty()) {
    legend.emplace_back(kMixed, "");
    double x = kMargin;
    double y = height - kMargin + 8;
    raw_string_ostream legendOs(legendSvg);
    legendOs << "<g class=\"legend\">\n";
    for (const auto &[color, label] : legend) {
      std::string text = label.empty() ? "mixed sources" : "from " + label;
      double entry = 26 + text.size() * kCharWidth * 1.1;
      if (x + entry > width - kMargin && x > kMargin) {
        x = kMargin;
        y += 16;
      }
      legendOs << "<rect x=\"" << num(x) << "\" y=\"" << num(y - 6)
               << "\" width=\"16\" height=\"4\" fill=\"" << color
               << "\"/><text x=\"" << num(x + 20) << "\" y=\"" << num(y)
               << "\">" << escape(text) << "</text>\n";
      x += entry + 12;
    }
    legendOs << "</g>\n";
    height = y + kMargin;
  }

  std::string out;
  raw_string_ostream os(out);
  os << "<svg xmlns=\"http://www.w3.org/2000/svg\" class=\"aie-view\" "
     << "viewBox=\"0 0 " << num(width) << " " << num(height) << "\" width=\""
     << num(width) << "\" height=\"" << num(height) << "\" data-view=\""
     << escape(options.title) << "\">\n";
  os << R"(<style>
.aie-view text{font-family:"DejaVu Sans",Verdana,Arial,sans-serif;font-size:9.5px;fill:#222}
.aie-view .title{font-size:15px;font-weight:bold}
.aie-view .tile-title{font-size:10.5px;fill:#555}
.aie-view .tile{stroke:#c8ccd2;stroke-width:1}
.aie-view .tile.unused{fill:#fafafa;stroke-dasharray:4 3}
.aie-view .tile.core{fill:#eef6ee}.aie-view .tile.mem{fill:#fdf5e6}.aie-view .tile.shim{fill:#edf2fa}
.aie-view .switchbox{fill:#fbfbfb;stroke:#9aa0a6;stroke-width:1}
.aie-view .switchbox.ghost{fill:none;stroke:#c8ccd2;stroke-dasharray:3 2}
.aie-view .lane-tick{stroke:#c8ccd2;stroke-width:1}
.aie-view .edge path.line{fill:none;stroke-linecap:round;stroke-linejoin:round}
.aie-view .edge path.hit{fill:none;stroke:transparent;stroke-width:9}
.aie-view .node rect{stroke-width:1}
.aie-view .node.dma rect{fill:#fff;stroke:#7d848c}
.aie-view .node.buffer rect{fill:#fff;stroke:#7d848c}
.aie-view .node.call rect{fill:#fff7d6;stroke:#a08a3c}
.aie-view .node.arg rect{fill:#fff;stroke:#555;stroke-width:1.3}
.aie-view .node.unused{opacity:.45}
.aie-view .node.port rect{fill:#5f6368;stroke:none}
.aie-view .node.port.junction circle{fill:#333}
.aie-view .legend text{font-size:10.5px}
</style>
)";
  os << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
  os << "<text class=\"title\" x=\"" << num(kMargin) << "\" y=\""
     << num(kMargin + 6) << "\">" << escape(options.title) << "</text>\n";

  // Tiles and switchboxes.
  bool switchboxesShown =
      llvm::any_of(graph.nodes,
                   [](const Node &node) {
                     return node.kind == kind::Port && !node.hidden;
                   }) ||
      llvm::none_of(graph.nodes,
                    [](const Node &node) { return node.kind == kind::Port; });
  if (llvm::is_contained(graph.contracted, kind::Port))
    switchboxesShown = false;
  os << "<g class=\"tiles\">\n";
  for (const auto &[coordinates, cell] : cells) {
    const TileInfo &tile = *cell.tile;
    os << "<rect class=\"tile " << tile.type << (tile.used ? "" : " unused")
       << "\" x=\"" << num(cell.content.x) << "\" y=\"" << num(cell.content.y)
       << "\" width=\"" << num(cell.content.w) << "\" height=\""
       << num(cell.content.h) << "\" rx=\"6\"/>\n";
    os << "<text class=\"tile-title\" x=\"" << num(cell.content.x + 6)
       << "\" y=\"" << num(cell.content.y + 13) << "\">(" << tile.col << ","
       << tile.row << ") " << (tile.type == "core" ? "compute" : tile.type)
       << "</text>\n";
    // Lines routed through contracted switchboxes show where they are.
    bool ghost =
        !switchboxesShown && options.routing == RenderOptions::Routing::MustHit;
    if (!switchboxesShown && !ghost)
      continue;
    const Box &box = cell.switchbox;
    os << "<rect class=\"switchbox" << (ghost ? " ghost" : "") << "\" x=\""
       << num(box.x) << "\" y=\"" << num(box.y) << "\" width=\"" << num(box.w)
       << "\" height=\"" << num(box.h) << "\" rx=\"2\"><title>(" << tile.col
       << "," << tile.row << ") switchbox</title></rect>\n";
  }
  os << "</g>\n";

  // Edges, each a group of the strokes of the physical segments it stands
  // for, bridged across the contracted nodes it passes through.
  double lightest = 0, heaviest = 0;
  for (const Edge &edge : graph.edges)
    if (edge.bytes) {
      lightest = lightest ? std::min<double>(lightest, edge.bytes) : edge.bytes;
      heaviest = std::max<double>(heaviest, edge.bytes);
    }
  auto strokeWidth = [&](const Edge &edge) {
    if (!options.weighEdges || !edge.bytes || heaviest <= lightest)
      return kLineWidth;
    double t = (std::log2(edge.bytes) - std::log2(lightest)) /
               (std::log2(heaviest) - std::log2(lightest));
    return 0.8 + t * 4.2;
  };

  std::map<std::pair<int, int>, std::vector<Box>> drawnBoxes;
  for (const auto &[id, box] : boxes) {
    const Node &node = *nodes.at(id);
    if (node.hasTile() && node.kind != kind::Argument && isVisible(id))
      drawnBoxes[tileOf(node)].push_back(box);
  }

  os << "<g class=\"edges\">\n";
  for (const Edge &edge : graph.edges) {
    std::string color = colors[edge.id()];
    std::string description =
        nodes.at(edge.source)->label + " → " + nodes.at(edge.target)->label;
    if (edge.bytes)
      description += " (" + formatBytes(edge.bytes) + ")";
    os << "<g class=\"edge\" data-id=\"" << escape(edge.id())
       << "\" data-source=\"" << escape(edge.source) << "\" data-target=\""
       << escape(edge.target) << "\" data-bytes=\"" << edge.bytes
       << "\" stroke=\"" << color << "\" fill=\"" << color << "\"><title>"
       << escape(description) << "</title>\n";
    std::string lines, hits, heads;
    double lineWidth = strokeWidth(edge);
    auto addPath = [&](const std::string &d) {
      lines += "<path class=\"line\" stroke-width=\"" + num(lineWidth) +
               "\" d=\"" + d + "\"/>\n";
      hits += "<path class=\"hit\" d=\"" + d + "\"/>\n";
    };
    if (auto routed = routes.find(edge.id()); routed != routes.end()) {
      const std::vector<Point> &points = routed->second;
      addPath(rounded(points));
      if (points.size() > 1 && isVisible(edge.target) &&
          nodes.at(edge.target)->kind != kind::Port)
        heads += "<path class=\"head\" stroke=\"none\" d=\"" +
                 arrowhead(points.back(),
                           unit(points[points.size() - 2], points.back())) +
                 "\"/>\n";
      os << lines << heads << hits << "</g>\n";
      continue;
    }
    // A tile's contracted boxes are one region a stream passes through. It
    // crosses the region in one curve, from where it enters (from a
    // switchbox port, a runtime argument or a box still drawn) to where it
    // leaves, unless that curve would cross a box still drawn; then it
    // follows its physical path through the contracted boxes' empty places.
    auto hiddenBox = [&](const std::string &id) {
      const Node &node = *nodes.at(id);
      return isBox(node) && !isVisible(id) && boxes.count(id);
    };
    auto drawnBox = [&](const std::string &id) {
      const Node &node = *nodes.at(id);
      return isBox(node) && node.kind != kind::Argument && isVisible(id) &&
             boxes.count(id);
    };
    auto drawSegment = [&](const std::string &from, const std::string &to) {
      Stroke stroke = strokeFor(from, to);
      addPath(stroke.d);
      // Port markers and the mid-wire chevrons say which way a stream runs
      // through the switch network; arrowheads mark where it arrives.
      if (isVisible(to) && nodes.at(to)->kind != kind::Port)
        heads += "<path class=\"head\" stroke=\"none\" d=\"" +
                 arrowhead(stroke.end, stroke.heading) + "\"/>\n";
      if (stroke.chevron)
        heads += "<path class=\"head\" stroke=\"none\" d=\"" +
                 arrowhead(stroke.chevron->first, stroke.chevron->second) +
                 "\"/>\n";
      return stroke;
    };
    using Way = std::pair<Point, Point>;
    struct Region {
      std::vector<Way> entries, exits;
      std::vector<bool> exitIsDrawn;
      /// Segments drawn only if the region is crossed on its physical path.
      std::vector<std::pair<std::string, std::string>> held;
    };
    std::map<std::pair<int, int>, Region> regions;
    for (const auto &[from, to] : edge.segments) {
      if (!nodes.count(from) || !nodes.count(to))
        continue;
      size_t index = segmentIndex.at({from, to});
      bool fromHidden = hiddenBox(from), toHidden = hiddenBox(to);
      if (fromHidden && toHidden &&
          tileOf(*nodes.at(from)) == tileOf(*nodes.at(to))) {
        regions[tileOf(*nodes.at(from))].held.push_back({from, to});
        continue;
      }
      if (toHidden && drawnBox(from)) {
        Region &region = regions[tileOf(*nodes.at(to))];
        Point side = inward(sideFacing(*nodes.at(from), *nodes.at(to)));
        region.entries.push_back(
            {anchors.at({index, true}), {-side.x, -side.y}});
        region.held.push_back({from, to});
        continue;
      }
      if (fromHidden && drawnBox(to)) {
        Region &region = regions[tileOf(*nodes.at(from))];
        region.exits.push_back(
            {anchors.at({index, false}),
             inward(sideFacing(*nodes.at(to), *nodes.at(from)))});
        region.exitIsDrawn.push_back(true);
        region.held.push_back({from, to});
        continue;
      }
      Stroke stroke = drawSegment(from, to);
      if (toHidden)
        regions[tileOf(*nodes.at(to))].entries.push_back(
            {stroke.end, stroke.heading});
      if (fromHidden) {
        Region &region = regions[tileOf(*nodes.at(from))];
        region.exits.push_back({stroke.start, stroke.startHeading});
        region.exitIsDrawn.push_back(false);
      }
    }
    for (auto &[tile, region] : regions) {
      std::vector<std::tuple<std::array<Point, 4>, bool>> curves;
      bool clear = true;
      for (auto [out, drawn] : zip_equal(region.exits, region.exitIsDrawn))
        for (const auto &[from, fromHeading] : region.entries) {
          const auto &[to, toHeading] = out;
          bool turnsBack =
              fromHeading.x * toHeading.x + fromHeading.y * toHeading.y < 0;
          double distance = std::hypot(to.x - from.x, to.y - from.y);
          double reach =
              turnsBack ? std::min(8 + std::abs(to.y - from.y) * 0.45, 36.0)
                        : std::max(12.0, distance * 0.45);
          std::array<Point, 4> curve{
              from,
              Point{from.x + fromHeading.x * reach,
                    from.y + fromHeading.y * reach},
              Point{to.x - toHeading.x * reach, to.y - toHeading.y * reach},
              to};
          for (int step = 1; step < 16 && clear; ++step) {
            double t = step / 16.0, u = 1 - t;
            Point p{u * u * u * curve[0].x + 3 * u * u * t * curve[1].x +
                        3 * u * t * t * curve[2].x + t * t * t * curve[3].x,
                    u * u * u * curve[0].y + 3 * u * u * t * curve[1].y +
                        3 * u * t * t * curve[2].y + t * t * t * curve[3].y};
            for (const Box &box : drawnBoxes[tile])
              if (p.x > box.x + 1.5 && p.x < box.right() - 1.5 &&
                  p.y > box.y + 1.5 && p.y < box.bottom() - 1.5)
                clear = false;
          }
          curves.emplace_back(curve, drawn);
        }
      if (clear) {
        for (const auto &[curve, drawn] : curves) {
          addPath(cubic(curve[0], curve[1], curve[2], curve[3]));
          if (drawn)
            heads += "<path class=\"head\" stroke=\"none\" d=\"" +
                     arrowhead(curve[3], unit(curve[2], curve[3])) + "\"/>\n";
        }
        continue;
      }
      // Through each contracted box on the way, join every way in to every
      // way out.
      for (const auto &[from, to] : region.held)
        drawSegment(from, to);
      std::map<std::string, std::vector<Way>> arriving, leaving;
      for (const auto &[from, to] : edge.segments) {
        if (!nodes.count(from) || !nodes.count(to))
          continue;
        size_t index = segmentIndex.at({from, to});
        if (hiddenBox(to) && tileOf(*nodes.at(to)) == tile)
          if (auto anchor = anchors.find({index, false});
              anchor != anchors.end())
            arriving[to].push_back(
                {anchor->second,
                 inward(sideFacing(*nodes.at(to), *nodes.at(from)))});
        if (hiddenBox(from) && tileOf(*nodes.at(from)) == tile)
          if (auto anchor = anchors.find({index, true});
              anchor != anchors.end()) {
            Point side = inward(sideFacing(*nodes.at(from), *nodes.at(to)));
            leaving[from].push_back({anchor->second, {-side.x, -side.y}});
          }
      }
      for (const auto &[id, ins] : arriving) {
        const Box &box = boxes.at(id);
        for (const auto &[from, fromHeading] : ins)
          for (const auto &[to, toHeading] : leaving[id]) {
            bool turnsBack =
                fromHeading.x * toHeading.x + fromHeading.y * toHeading.y < 0;
            double reach =
                turnsBack ? std::min(box.w / 3, 6 + std::abs(to.y - from.y))
                          : box.w * 0.6;
            addPath(cubic(
                from,
                {from.x + fromHeading.x * reach,
                 from.y + fromHeading.y * reach},
                {to.x - toHeading.x * reach, to.y - toHeading.y * reach}, to));
          }
      }
    }
    os << lines << heads << hits << "</g>\n";
  }
  os << "</g>\n";

  // Nodes.
  os << "<g class=\"nodes\">\n";
  std::set<std::string> connected;
  for (const auto &[from, to] : segments) {
    connected.insert(from);
    connected.insert(to);
  }
  for (const Node &node : graph.nodes) {
    if (node.hidden)
      continue;
    std::string tip = node.detail.empty() ? node.label : node.detail;
    if (node.bytes && node.kind != kind::Port)
      tip += "\n" + formatBytes(node.bytes) +
             (node.kind == kind::Buffer ? "" : " per run");
    std::string classes = "node " + node.kind;
    if (!connected.count(node.id))
      classes += " unused";
    if (node.kind == kind::Port) {
      if (!cells.count(tileOf(node)))
        continue;
      Point p = portPoint(node);
      if (!switchboxesShown)
        classes += " junction";
      os << "<g class=\"" << classes << "\" data-id=\"" << escape(node.id)
         << "\"><title>" << escape(tip) << "</title>";
      if (switchboxesShown)
        os << "<rect x=\"" << num(p.x - 2) << "\" y=\"" << num(p.y - 2)
           << "\" width=\"4\" height=\"4\"/>";
      else
        os << "<circle cx=\"" << num(p.x) << "\" cy=\"" << num(p.y)
           << "\" r=\"2.6\"/>";
      os << "</g>\n";
      continue;
    }
    auto found = boxes.find(node.id);
    if (found == boxes.end())
      continue;
    Box box = found->second;
    for (const std::string &member : node.members) {
      auto memberBox = boxes.find(member);
      if (memberBox == boxes.end())
        continue;
      const Box &other = memberBox->second;
      double right = std::max(box.right(), other.right());
      double bottom = std::max(box.bottom(), other.bottom());
      box.x = std::min(box.x, other.x);
      box.y = std::min(box.y, other.y);
      box.w = right - box.x;
      box.h = bottom - box.y;
    }
    os << "<g class=\"" << classes << "\" data-id=\"" << escape(node.id)
       << "\"><title>" << escape(tip) << "</title><rect x=\"" << num(box.x)
       << "\" y=\"" << num(box.y) << "\" width=\"" << num(box.w)
       << "\" height=\"" << num(box.h) << "\" rx=\""
       << (node.kind == kind::Call ? 8 : 2) << "\"/><text x=\""
       << num(box.x + box.w / 2) << "\" y=\"" << num(box.y + box.h / 2 + 3.3)
       << "\" text-anchor=\"middle\">" << escape(fit(node.label, box.w - 6))
       << "</text></g>\n";
  }
  os << "</g>\n";

  os << legendSvg;
  os << "</svg>\n";
  return out;
}

} // namespace

Box Renderer::drawnBoxOf(const Node &node) const {
  Box box = boxes.at(node.id);
  for (const std::string &member : node.members) {
    auto found = boxes.find(member);
    if (found == boxes.end())
      continue;
    const Box &other = found->second;
    double right = std::max(box.right(), other.right());
    double bottom = std::max(box.bottom(), other.bottom());
    box.x = std::min(box.x, other.x);
    box.y = std::min(box.y, other.y);
    box.w = right - box.x;
    box.h = bottom - box.y;
  }
  return box;
}

Error Renderer::routeEdges() {
  // Every box still drawn is an obstacle that lines end on.
  std::vector<routing::Rect> obstacles;
  std::map<std::string, int> obstacleOf;
  for (const Node &node : graph.nodes) {
    if (node.hidden || node.kind == kind::Port || !boxes.count(node.id))
      continue;
    Box box = drawnBoxOf(node);
    obstacleOf[node.id] = obstacles.size();
    obstacles.push_back({box.x, box.y, box.w, box.h});
  }
  auto positionOf = [&](const Node &node) {
    if (node.kind == kind::Port)
      return portPoint(node);
    return drawnBoxOf(node).center();
  };
  auto endOf = [&](const std::string &id) {
    const Node &node = *nodes.at(id);
    routing::End end;
    if (auto found = obstacleOf.find(id); found != obstacleOf.end()) {
      end.obstacle = found->second;
      end.sides = node.kind == kind::Argument
                      ? unsigned(routing::Top)
                      : unsigned(routing::Left | routing::Right);
    } else {
      Point point = positionOf(node);
      end.point = {point.x, point.y};
    }
    return end;
  };

  // The places a line must pass through: the switchbox of each contracted
  // port and the tile of each contracted DMA channel or buffer, in the order
  // its path meets them, each once in a row.
  using Place = std::tuple<bool, int, int>;
  auto placeOf = [&](const Node &node) -> std::optional<Place> {
    if (!node.hasTile() || !cells.count(tileOf(node)))
      return std::nullopt;
    return Place{node.kind == kind::Port, node.col, node.row};
  };
  auto boxOfPlace = [&](const Place &place) {
    const Cell &cell = cells.at({std::get<1>(place), std::get<2>(place)});
    return std::get<0>(place) ? cell.switchbox : cell.content;
  };
  std::vector<routing::Request> requests;
  std::vector<std::vector<Place>> places;
  std::vector<std::string> ids;
  std::vector<std::string> sources, targets;
  for (const Edge &edge : graph.edges) {
    if (!nodes.count(edge.source) || !nodes.count(edge.target))
      continue;
    ids.push_back(edge.id());
    sources.push_back(edge.source);
    targets.push_back(edge.target);
    requests.push_back({endOf(edge.source), endOf(edge.target), {}});
    std::vector<Place> sequence;
    if (options.routing == RenderOptions::Routing::MustHit) {
      std::map<std::string, std::vector<std::string>> next;
      for (const auto &[from, to] : edge.segments)
        next[from].push_back(to);
      std::map<std::string, std::string> parent{{edge.source, ""}};
      std::deque<std::string> queue{edge.source};
      while (!queue.empty() && !parent.count(edge.target)) {
        std::string current = queue.front();
        queue.pop_front();
        for (const std::string &successor : next[current])
          if (parent.try_emplace(successor, current).second)
            queue.push_back(successor);
      }
      std::vector<std::string> path;
      if (parent.count(edge.target))
        for (std::string node = parent[edge.target];
             !node.empty() && node != edge.source; node = parent[node])
          path.push_back(node);
      std::reverse(path.begin(), path.end());
      std::optional<Place> first = placeOf(*nodes.at(edge.source));
      std::optional<Place> last = placeOf(*nodes.at(edge.target));
      for (const std::string &id : path) {
        std::optional<Place> place = placeOf(*nodes.at(id));
        if (!place || (!sequence.empty() && sequence.back() == *place) ||
            (sequence.empty() && first == place))
          continue;
        sequence.push_back(*place);
      }
      while (!sequence.empty() && sequence.back() == last)
        sequence.pop_back();
    }
    places.push_back(std::move(sequence));
  }

  // Each line passes a place where the straight way from its previous stop
  // to its next one comes closest, moved as little as it takes to share
  // neither a row nor a column with another line's point there, so that no
  // two run along each other through it. A point inside a tile keeps clear
  // of the boxes drawn there.
  std::map<Place, std::vector<std::tuple<double, size_t, size_t, Point>>>
      passing;
  for (auto [index, sequence] : enumerate(places)) {
    Point target = positionOf(*nodes.at(targets[index]));
    Point previous = positionOf(*nodes.at(sources[index]));
    for (auto [step, place] : enumerate(sequence)) {
      Box box = boxOfPlace(place);
      Point next = step + 1 < sequence.size()
                       ? boxOfPlace(sequence[step + 1]).center()
                       : target;
      Point ideal{
          std::clamp((previous.x + next.x) / 2, box.x + 4, box.right() - 4),
          std::clamp((previous.y + next.y) / 2, box.y + 4, box.bottom() - 4)};
      passing[place].emplace_back(ideal.x + ideal.y, index, step, ideal);
      previous = ideal;
    }
  }
  std::vector<std::vector<routing::Point>> checkpoints(places.size());
  for (auto [index, sequence] : enumerate(places))
    checkpoints[index].resize(sequence.size());
  for (auto &[place, list] : passing) {
    llvm::sort(list, [](const auto &a, const auto &b) {
      return std::tie(std::get<0>(a), std::get<1>(a)) <
             std::tie(std::get<0>(b), std::get<1>(b));
    });
    Box box = boxOfPlace(place);
    std::vector<Point> taken;
    auto clear = [&](Point candidate) {
      for (const routing::Rect &obstacle : obstacles)
        if (candidate.x > obstacle.x - 6 &&
            candidate.x < obstacle.x + obstacle.w + 6 &&
            candidate.y > obstacle.y - 6 &&
            candidate.y < obstacle.y + obstacle.h + 6)
          return false;
      return llvm::none_of(taken, [&](Point other) {
        return std::abs(other.x - candidate.x) < 5 ||
               std::abs(other.y - candidate.y) < 5;
      });
    };
    for (const auto &[unused, index, step, ideal] : list) {
      Point chosen = ideal;
      double best = std::numeric_limits<double>::max();
      for (double y = box.y + 4; y < box.bottom() - 4; y += 3)
        for (double x = box.x + 4; x < box.right() - 4; x += 3) {
          double distance = std::hypot(x - ideal.x, y - ideal.y);
          if (distance < best && clear({x, y})) {
            best = distance;
            chosen = {x, y};
          }
        }
      taken.push_back(chosen);
      checkpoints[index][step] = {chosen.x, chosen.y};
    }
  }
  for (auto [request, points] : zip_equal(requests, checkpoints))
    request.checkpoints = std::move(points);

  Expected<std::vector<std::vector<routing::Point>>> routed =
      routing::routeOrthogonal(obstacles, requests);
  if (!routed)
    return routed.takeError();
  for (auto [id, route] : zip_equal(ids, *routed)) {
    std::vector<Point> &points = routes[id];
    for (const routing::Point &point : route)
      points.push_back({point.x, point.y});
  }
  return Error::success();
}

Expected<std::string>
xilinx::AIE::viz::renderSvg(const Graph &graph, const RenderOptions &options) {
  return Renderer(graph, options).render();
}
