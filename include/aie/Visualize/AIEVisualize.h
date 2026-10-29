//===- AIEVisualize.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_VISUALIZE_AIEVISUALIZE_H
#define AIE_VISUALIZE_AIEVISUALIZE_H

#include "aie/Visualize/DesignGraph.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <string>
#include <utility>
#include <vector>

namespace xilinx::AIE {

struct VisualizeOptions {
  /// Select one `aie.device` by symbol when the input holds several.
  std::string deviceName;
  /// Select one `aie.runtime_sequence` by symbol when the device holds several.
  std::string runtimeSequenceName;
  /// Include the control packet routes aie-generate-column-control-overlay
  /// adds.
  bool controlOverlay = false;
  /// Read the source files the design's locations name into the graph.
  bool embedSources = true;
  /// Where to look for source files named by a relative path, besides the
  /// working directory.
  std::string sourceDirectory;
  /// The name under which the design's own IR, as read, is embedded and
  /// referred to, so that each element also points at the ops that implement
  /// it there: DMA programs, switchbox configuration, and so on. Empty leaves
  /// it out.
  std::string irName = "design.mlir";
};

/// Builds the full dataflow graph of a placed design. The design may be
/// routed (switchboxes) or carry flows; it is analyzed on a copy, so
/// `module` is not modified. Reports failures through the module's
/// diagnostic handler.
mlir::FailureOr<viz::Graph> buildDesignGraph(mlir::ModuleOp module,
                                             const VisualizeOptions &options);

/// The views a viewer offers by default: the full physical graph, the graph
/// with switchbox ports contracted, and the topology with ports, DMA channels
/// and buffers contracted.
std::vector<std::pair<std::string, viz::Graph>>
standardViews(const viz::Graph &graph);

} // namespace xilinx::AIE

#endif // AIE_VISUALIZE_AIEVISUALIZE_H
