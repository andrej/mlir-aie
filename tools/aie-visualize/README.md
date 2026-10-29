<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# aie-visualize

Draws where the data of a placed and routed AIE design goes: from runtime
sequence arguments, through shim, memtile and compute DMAs, switchboxes and
buffers, to the kernels that read it, and back.

```bash
aiecc --get-visualization design.mlir       # writes visualization.html
aie-visualize input_physical.mlir -o graph.json
aie-visualize graph.json --view=topology -o topology.svg
aie-visualize graph.json --view=physical -o physical.svg
aie-visualize input_physical.mlir -o view.html
```

An SVG is a drawing you can read without the viewer. It also carries its
graph as JSON, with the source files the graph names, in a `<metadata
class="aie-graph">` element.

`aie-visualize.html`, installed next to the tool, adds interaction to the
SVGs it opens. Each SVG becomes a tab named after its file:

```text
aie-visualize.html?svg=topology.svg&svg=physical.svg
```

Serve the directory over HTTP for that form (`python3 -m http.server`).
Browsers do not let a page opened as a file load other files; open the
viewer as a file and use its Open SVG… button or drop the SVGs on it
instead. `-o view.html` and aiecc write the same viewer with the standard
views inside it, as one file that needs no server.

## Views

`--view` selects which node kinds are contracted, and `--contract` lists
them directly.

- **Physical**: every switchbox hop, DMA channel, buffer and kernel call.
- **Buffers**: switchbox ports and DMA channels contracted.
- **Topology**: ports, DMA channels and buffers contracted. A contracted node
  disappears when it only passes one stream through. Buffers that differ only
  in which half of a ping-pong pair they are merge first. Points where streams
  split (broadcasts), join or end stay.

In the viewer, hover over or click on any element to highlight what feeds it
(upstream) and where its data goes (downstream), and to see the source lines
behind it. These include the DMA programs and switchbox configuration in the
IR the visualizer read, not only the lines in your design.

## Routing

`--routing` sets how an edge that stands for contracted nodes runs:

- `physical` (default): along the exact physical path of the nodes it stands
  for. Views at every level of contraction then match line for line.
- `must-hit`: through the switchbox of each contracted port and the tile of
  each contracted DMA channel or buffer, in order. Between those places, the
  line takes the path with the fewest crossings and bends.
- `free`: any path from source to target, with the fewest crossings and
  bends.

## Layout

`--layout` sets where nodes go:

- `physical` (default): every switchbox port, DMA channel and wire of the
  device in a place of its own.
- `compact`: only the nodes the view draws. A tile lists them in columns as
  wide as their labels. A switchbox spaces its ports along the sides they
  face. A tile or switchbox grows until each line routed through it has a
  row and a column of its own there. Tiles with nothing to draw and no line
  through them take no room. Compact needs `must-hit` (default) or `free`
  routing. It suits contracted views best: the fewer nodes a view draws,
  the smaller the figure.

`must-hit` and `free` route with [libavoid](https://github.com/mjwybrow/adaptagrams)
(LGPL-2.1), an optional dependency. Configure with
`-DAIE_VISUALIZE_LIBAVOID=<prefix>`, where `<prefix>` holds
`include/libavoid/*.h` and `lib/libavoid.so`.

## The design graph

`-o graph.json` writes the graph every view is drawn from. It is the only
interface between reading the IR and drawing, and can be queried directly.

```text
nodes[]   id, kind (port | dma | buffer | call | arg), label, tile [col, row],
          bundle/channel/direction (ports: in or out; DMAs: MM2S or S2MM),
          bytes, sources [{file, lines}], hidden (contracted in this view),
          members / mergedInto (merged interchangeable nodes)
edges[]   source, target, flows (indices into flows), packets, bytes per run,
          sources, segments (the full-graph edges a contracted edge stands for)
flows[]   the routed streams (aie.flow and aie.packet_flow)
tiles[]   every tile in the drawn columns, with its switchbox port counts
files     source file contents, by name
```

For example, to find which kernels the data of runtime argument 0 reaches, run
`aie-visualize graph.json --view=topology -o topology.json` and follow edges
from `arg:0`. Only kernel calls, branch points and joins remain.
