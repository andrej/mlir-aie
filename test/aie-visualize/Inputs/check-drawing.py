# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Checks that an aie-visualize drawing can be read: every edge of the graph
is drawn as one connected line from its source to its target, ends in an
arrowhead where it arrives at a box, runs along no other edge, and crosses no
box it does not connect.

Usage: check-drawing.py drawing.svg graph.json
Prints a summary line, then one line per problem; exits 1 on any problem.
"""

import json
import math
import re
import sys
import xml.etree.ElementTree as ET

NS = "{http://www.w3.org/2000/svg}"
EPSILON = 0.6


def parse_path(d):
    """Flattens an SVG path of M, L, C and Q commands into polylines."""
    tokens = re.findall(r"[MLCQZ]|-?[\d.]+", d)
    polylines, current, i = [], [], 0
    pen = None

    def take(n):
        nonlocal i
        values = [float(t) for t in tokens[i : i + n]]
        i += n
        return values

    while i < len(tokens):
        command = tokens[i]
        i += 1
        if command == "M":
            if len(current) > 1:
                polylines.append(current)
            pen = tuple(take(2))
            current = [pen]
        elif command == "L":
            pen = tuple(take(2))
            current.append(pen)
        elif command in "CQ":
            n = 6 if command == "C" else 4
            values = take(n)
            points = [pen] + [tuple(values[k : k + 2]) for k in range(0, n, 2)]
            for step in range(1, 13):
                t = step / 12
                current.append(bezier(points, t))
            pen = points[-1]
        elif command == "Z":
            pass
    if len(current) > 1:
        polylines.append(current)
    return polylines


def bezier(points, t):
    while len(points) > 1:
        points = [
            (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            for a, b in zip(points, points[1:])
        ]
    return points[0]


def touches(box, point, slack=EPSILON):
    x, y, w, h = box
    return (
        x - slack <= point[0] <= x + w + slack
        and y - slack <= point[1] <= y + h + slack
    )


def inside(box, point, margin=1.5):
    x, y, w, h = box
    return (
        x + margin < point[0] < x + w - margin
        and y + margin < point[1] < y + h - margin
    )


def main():
    svg_path, graph_path = sys.argv[1:3]
    graph = json.load(open(graph_path))
    root = ET.parse(svg_path).getroot()
    nodes = {n["id"]: n for n in graph["nodes"]}

    boxes = {}
    for g in root.iter(NS + "g"):
        if "node" not in g.get("class", "").split():
            continue
        rect = g.find(NS + "rect")
        circle = g.find(NS + "circle")
        if rect is not None:
            boxes[g.get("data-id")] = tuple(
                float(rect.get(k)) for k in ("x", "y", "width", "height")
            )
        elif circle is not None:
            cx, cy = float(circle.get("cx")), float(circle.get("cy"))
            boxes[g.get("data-id")] = (cx - 3, cy - 3, 6, 6)
    for node in graph["nodes"]:
        for member in node.get("members", []):
            if node["id"] in boxes:
                boxes[member] = boxes[node["id"]]

    drawn = {}
    heads = {}
    for g in root.iter(NS + "g"):
        if g.get("class") != "edge":
            continue
        lines, tips = [], []
        for path in g.iter(NS + "path"):
            kind = path.get("class")
            if kind == "line":
                lines.extend(parse_path(path.get("d")))
            elif kind == "head":
                tips.append(parse_path(path.get("d") + " Z")[0][0])
        drawn[g.get("data-id")] = lines
        heads[g.get("data-id")] = tips

    problems = []
    edges = {e["id"]: e for e in graph["edges"]}

    def box_of(node_id):
        node = nodes[node_id]
        return boxes.get(node.get("mergedInto") or node_id, boxes.get(node_id))

    for edge_id, edge in edges.items():
        lines = drawn.get(edge_id)
        if not lines:
            problems.append(f"{edge_id}: not drawn")
            continue
        # Connected: join stroke ends that meet, then ask whether one piece
        # touches both the source and the target.
        parent = list(range(len(lines)))

        def find(k):
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k

        ends = [(line[0], line[-1]) for line in lines]
        for a in range(len(lines)):
            for b in range(a + 1, len(lines)):
                if any(
                    math.dist(p, q) < EPSILON for p in ends[a] for q in ends[b]
                ) or any(math.dist(p, q) < EPSILON for p in ends[a] for q in lines[b]):
                    parent[find(a)] = find(b)
        source_box, target_box = box_of(edge["source"]), box_of(edge["target"])
        if source_box is None or target_box is None:
            problems.append(f"{edge_id}: an end is not drawn")
            continue
        sources = {
            find(k)
            for k, e in enumerate(ends)
            if touches(source_box, e[0]) or touches(source_box, e[1])
        }
        targets = {
            find(k)
            for k, e in enumerate(ends)
            if touches(target_box, e[0]) or touches(target_box, e[1])
        }
        if not sources & targets:
            problems.append(f"{edge_id}: line does not join its source and target")
        if nodes[edge["target"]]["kind"] != "port" and not any(
            touches(target_box, tip, 1.0) for tip in heads[edge_id]
        ):
            problems.append(f"{edge_id}: no arrowhead at the target")
        # A kernel's access to a buffer points both ways.
        if (
            edge.get("kernelAccess")
            and nodes[edge["source"]]["kind"] in ("call", "buffer")
            and "call" in (nodes[edge["source"]]["kind"], nodes[edge["target"]]["kind"])
            and not any(touches(source_box, tip, 1.0) for tip in heads[edge_id])
        ):
            problems.append(f"{edge_id}: no arrowhead back into the source")

    # Overlap and box crossings, over short pieces hashed into a grid.
    pieces = []
    for edge_id, lines in drawn.items():
        for line in lines:
            for p, q in zip(line, line[1:]):
                if math.dist(p, q) > 0.05:
                    pieces.append((edge_id, p, q))
    grid = {}
    for index, (_, p, q) in enumerate(pieces):
        for cell in {
            (int(p[0] // 20), int(p[1] // 20)),
            (int(q[0] // 20), int(q[1] // 20)),
        }:
            grid.setdefault(cell, []).append(index)

    def shares_end(a, b):
        ea, eb = edges.get(a), edges.get(b)
        return (
            ea and eb and (ea["source"] == eb["source"] or ea["target"] == eb["target"])
        )

    overlapping = set()
    for cell, members in grid.items():
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                ea, p1, q1 = pieces[a]
                eb, p2, q2 = pieces[b]
                if ea == eb or (ea, eb) in overlapping or shares_end(ea, eb):
                    continue
                if runs_along(p1, q1, p2, q2):
                    overlapping.add((ea, eb))
    for a, b in sorted(overlapping):
        problems.append(f"{a} runs along {b}")

    crossings = set()
    for edge_id, p, q in pieces:
        edge = edges.get(edge_id)
        own = {edge["source"], edge["target"]} if edge else set()
        own |= {nodes[n].get("mergedInto") for n in own if n in nodes}
        middle = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        for node_id, box in boxes.items():
            if node_id in own or nodes.get(node_id, {}).get("kind") == "port":
                continue
            if inside(box, middle) and (edge_id, node_id) not in crossings:
                crossings.add((edge_id, node_id))
    for edge_id, node_id in sorted(crossings):
        problems.append(f"{edge_id} crosses {node_id}")

    print(
        f"edges: {len(edges)} drawn: {len(drawn)} pieces: {len(pieces)} "
        f"problems: {len(problems)}"
    )
    for problem in problems:
        print(problem)
    return 1 if problems else 0


def runs_along(p1, q1, p2, q2):
    """Two pieces run along each other if they are parallel, close, and
    overlap by more than a hair."""
    d1 = (q1[0] - p1[0], q1[1] - p1[1])
    d2 = (q2[0] - p2[0], q2[1] - p2[1])
    l1, l2 = math.hypot(*d1), math.hypot(*d2)
    if l1 < 1.5 or l2 < 1.5:
        return False
    u = (d1[0] / l1, d1[1] / l1)
    if abs(u[0] * d2[1] - u[1] * d2[0]) / l2 > 0.05:
        return False
    # Distance of the second piece from the first's line.
    offset = abs((p2[0] - p1[0]) * u[1] - (p2[1] - p1[1]) * u[0])
    if offset > 1.5:
        return False
    a = sorted([0, l1])
    t2 = sorted(
        [
            (p2[0] - p1[0]) * u[0] + (p2[1] - p1[1]) * u[1],
            (q2[0] - p1[0]) * u[0] + (q2[1] - p1[1]) * u[1],
        ]
    )
    return min(a[1], t2[1]) - max(a[0], t2[0]) > 2.0


if __name__ == "__main__":
    sys.exit(main())
