# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Prints a design graph one element per line, for FileCheck.

  node <id> <kind> "<label>" [hidden] [members=a,b] [sources=file:l1,l2;...]
  edge <id> [bytes=N] [packets=a,b] [segments=N] [sources=...]

Usage: summarize-graph.py graph.json [--sources] [--visible]
"""

import json
import os
import sys


def sources(element):
    return ";".join(
        f"{os.path.basename(s['file'])}:{','.join(map(str, s['lines']))}"
        for s in element.get("sources", [])
    )


def main():
    graph = json.load(open(sys.argv[1]))
    with_sources = "--sources" in sys.argv
    visible_only = "--visible" in sys.argv
    for node in sorted(graph["nodes"], key=lambda n: n["id"]):
        if visible_only and node.get("hidden"):
            continue
        line = f"node {node['id']} {node['kind']} \"{node['label']}\""
        if node.get("hidden"):
            line += " hidden"
        if node.get("members"):
            line += " members=" + ",".join(node["members"])
        if with_sources and node.get("sources"):
            line += " sources=" + sources(node)
        print(line)
    for edge in sorted(graph["edges"], key=lambda e: e["id"]):
        line = f"edge {edge['id']}"
        for key in ("bytes",):
            if key in edge:
                line += f" {key}={edge[key]}"
        if edge.get("packets"):
            line += " packets=" + ",".join(map(str, edge["packets"]))
        if "segments" in edge:
            line += f" segments={len(edge['segments'])}"
        if with_sources and edge.get("sources"):
            line += " sources=" + sources(edge)
        print(line)


if __name__ == "__main__":
    main()
