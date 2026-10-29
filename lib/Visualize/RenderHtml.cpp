//===- RenderHtml.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Visualize/DesignGraph.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace xilinx::AIE::viz;

static constexpr const char kTemplate[] =
#include "ViewerTemplate.inc"
    ;

/// JSON inside a script element may not contain "</", or the HTML parser
/// ends the element there.
static std::string scriptSafe(StringRef json) {
  std::string result;
  result.reserve(json.size());
  for (size_t i = 0; i < json.size(); ++i) {
    if (json[i] == '<' && i + 1 < json.size() && json[i + 1] == '/') {
      result += "<\\/";
      ++i;
      continue;
    }
    result += json[i];
  }
  return result;
}

std::string
xilinx::AIE::viz::renderHtml(ArrayRef<std::pair<std::string, Graph>> views,
                             StringRef title) {
  // Every view reads the same files; embed them once.
  json::Object files;
  json::Array viewData;
  std::string svgs;
  for (const auto &[name, graph] : views) {
    for (const auto &[path, text] : graph.files)
      files[path] = text;
    Graph withoutFiles = graph;
    withoutFiles.files.clear();
    viewData.push_back(
        json::Object{{"name", name}, {"graph", toJSON(withoutFiles)}});
    RenderOptions options;
    options.title = (title + " — " + name).str();
    svgs += "<div class=\"view\">" + cantFail(renderSvg(graph, options)) +
            "</div>\n";
  }
  json::Object root{{"title", title.str()},
                    {"files", std::move(files)},
                    {"views", std::move(viewData)}};
  std::string data = formatv("{0}", json::Value(std::move(root)));

  std::string html = kTemplate;
  auto substitute = [&](StringRef marker, StringRef text) {
    size_t at = html.find(marker.str());
    if (at != std::string::npos)
      html.replace(at, marker.size(), text.str());
  };
  substitute("<!--AIE_VIZ_SVGS-->", svgs);
  substitute("/*AIE_VIZ_DATA*/", scriptSafe(data));
  return html;
}
