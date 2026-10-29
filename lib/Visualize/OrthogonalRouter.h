//===- OrthogonalRouter.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_VISUALIZE_ORTHOGONALROUTER_H
#define AIE_VISUALIZE_ORTHOGONALROUTER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace xilinx::AIE::viz::routing {

struct Point {
  double x = 0;
  double y = 0;
};

struct Rect {
  double x = 0;
  double y = 0;
  double w = 0;
  double h = 0;
};

enum Sides : unsigned { Left = 1, Right = 2, Top = 4, Bottom = 8 };

/// A connector end: on one of the given sides of an obstacle, where no other
/// connector ends, or at a fixed point.
struct End {
  int obstacle = -1;
  unsigned sides = 0;
  Point point;
};

struct Request {
  End from;
  End to;
  /// Points the connector passes through, in order.
  std::vector<Point> checkpoints;
};

/// Routes every request as an orthogonal polyline around the obstacles,
/// penalizing bends and crossings. Lines keep `spacing` from obstacles and
/// from each other's parallel runs. Fails if aie-visualize was built without
/// libavoid.
llvm::Expected<std::vector<std::vector<Point>>>
routeOrthogonal(llvm::ArrayRef<Rect> obstacles,
                llvm::ArrayRef<Request> requests, double spacing);

} // namespace xilinx::AIE::viz::routing

#endif // AIE_VISUALIZE_ORTHOGONALROUTER_H
