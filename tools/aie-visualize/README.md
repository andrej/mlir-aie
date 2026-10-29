<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# aie-visualize

Draws where the data of a placed and routed AIE design goes: from runtime
sequence arguments, through shim, memtile and compute DMAs, switchboxes and
buffers, to the kernels that read it, and back.

```bash
aiecc --get-visualization design.mlir       # writes visualization.html
aie-visualize input_physical.mlir -o view.html
aie-visualize input_physical.mlir -o graph.json
aie-visualize graph.json --view=topology -o topology.svg
```

Open the `.html` file in a browser. It needs no server and embeds the source
files the design's locations name.

## Views

All views share one fixed layout of the device. Every switchbox port, DMA
channel and wire between tiles has a place of its own, so two streams never
share a line.

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
