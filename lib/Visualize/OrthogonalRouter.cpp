//===- OrthogonalRouter.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrthogonalRouter.h"

#ifdef AIE_VISUALIZE_HAS_LIBAVOID
#include "libavoid/libavoid.h"
#endif

#include <map>
#include <memory>

using namespace llvm;
using namespace xilinx::AIE::viz::routing;

#ifndef AIE_VISUALIZE_HAS_LIBAVOID

Expected<std::vector<std::vector<Point>>>
xilinx::AIE::viz::routing::routeOrthogonal(ArrayRef<Rect>, ArrayRef<Request>,
                                           double) {
  return createStringError(
      inconvertibleErrorCode(),
      "aie-visualize was built without libavoid; only the physical routing "
      "is available. Configure with -DAIE_VISUALIZE_LIBAVOID=<prefix>.");
}

#else

Expected<std::vector<std::vector<Point>>>
xilinx::AIE::viz::routing::routeOrthogonal(ArrayRef<Rect> obstacles,
                                           ArrayRef<Request> requests,
                                           double spacing) {
  Avoid::Router router(Avoid::OrthogonalRouting);
  // Penalties from comparing crossings, bends and length over sample
  // designs: a crossing costs about as much as two bends.
  router.setRoutingParameter(Avoid::segmentPenalty, 150);
  router.setRoutingParameter(Avoid::crossingPenalty, 250);
  router.setRoutingParameter(Avoid::shapeBufferDistance, spacing);
  router.setRoutingParameter(Avoid::idealNudgingDistance, spacing);
  router.setRoutingOption(Avoid::nudgeOrthogonalSegmentsConnectedToShapes,
                          true);
  router.setRoutingOption(Avoid::nudgeOrthogonalTouchingColinearSegments, true);
  router.setRoutingOption(Avoid::performUnifyingNudgingPreprocessingStep, true);
  router.setRoutingOption(Avoid::nudgeSharedPathsWithCommonEndPoint, true);

  std::vector<Avoid::ShapeRef *> shapes;
  for (const Rect &rect : obstacles) {
    Avoid::Rectangle outline(Avoid::Point(rect.x, rect.y),
                             Avoid::Point(rect.x + rect.w, rect.y + rect.h));
    shapes.push_back(new Avoid::ShapeRef(&router, outline));
  }

  // Each end on an obstacle takes a pin of its own: a class of pins per set
  // of sides, with one pin per end on each of those sides.
  std::map<std::pair<int, unsigned>, unsigned> ends;
  for (const Request &request : requests)
    for (const End &end : {request.from, request.to})
      if (end.obstacle >= 0)
        ++ends[{end.obstacle, end.sides}];
  for (const auto &[key, count] : ends) {
    auto [obstacle, sides] = key;
    for (unsigned side : {Left, Right, Top, Bottom}) {
      if (!(sides & side))
        continue;
      for (unsigned i = 0; i < count; ++i) {
        double t = (i + 1.0) / (count + 1.0);
        double x = side == Left    ? Avoid::ATTACH_POS_LEFT
                   : side == Right ? Avoid::ATTACH_POS_RIGHT
                                   : t;
        double y = side == Top      ? Avoid::ATTACH_POS_TOP
                   : side == Bottom ? Avoid::ATTACH_POS_BOTTOM
                                    : t;
        Avoid::ConnDirFlags direction = side == Left    ? Avoid::ConnDirLeft
                                        : side == Right ? Avoid::ConnDirRight
                                        : side == Top   ? Avoid::ConnDirUp
                                                        : Avoid::ConnDirDown;
        auto *pin = new Avoid::ShapeConnectionPin(shapes[obstacle], sides, x, y,
                                                  true, 0, direction);
        pin->setExclusive(true);
      }
    }
  }

  // Connectors ending at one point meet at a junction fixed there, so that
  // nudging parallel runs apart does not move their ends.
  std::map<std::pair<double, double>, Avoid::JunctionRef *> junctions;
  auto connEnd = [&](const End &end) {
    if (end.obstacle >= 0)
      return Avoid::ConnEnd(shapes[end.obstacle], end.sides);
    Avoid::JunctionRef *&junction = junctions[{end.point.x, end.point.y}];
    if (!junction) {
      junction = new Avoid::JunctionRef(&router,
                                        Avoid::Point(end.point.x, end.point.y));
      junction->setPositionFixed(true);
    }
    return Avoid::ConnEnd(junction);
  };
  std::vector<Avoid::ConnRef *> connectors;
  for (const Request &request : requests) {
    auto *connector =
        new Avoid::ConnRef(&router, connEnd(request.from), connEnd(request.to));
    std::vector<Avoid::Checkpoint> checkpoints;
    for (const Point &point : request.checkpoints)
      checkpoints.emplace_back(Avoid::Point(point.x, point.y));
    connector->setRoutingCheckpoints(checkpoints);
    connectors.push_back(connector);
  }
  router.processTransaction();

  std::vector<std::vector<Point>> routes;
  for (Avoid::ConnRef *connector : connectors) {
    std::vector<Point> route;
    for (const Avoid::Point &point : connector->displayRoute().ps)
      route.push_back({point.x, point.y});
    routes.push_back(std::move(route));
  }
  return routes;
}

#endif
