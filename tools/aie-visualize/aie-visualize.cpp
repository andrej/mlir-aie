//===- aie-visualize.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

// Without -o, this tool prints the device layout and which tiles a design
// uses. With -o, it writes the design's dataflow graph as JSON, a drawing of
// it as SVG, or an interactive viewer as HTML. The input is an MLIR design,
// or a graph this tool wrote as JSON.

#include "aie/InitialAllDialect.h"
#include "aie/Target/LLVMIR/Dialect/XLLVM/XLLVMToLLVMIRTranslation.h"
#include "aie/Visualize/AIEVisualize.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <iostream>

using namespace llvm;
using namespace mlir;
using namespace xilinx;

static cl::opt<std::string>
    fileName(cl::Positional, cl::desc("<input mlir or json>"), cl::Required);
static cl::opt<std::string>
    outputFilename("o",
                   cl::desc("Write a .json graph, an .svg drawing or an "
                            ".html viewer, as the extension says"),
                   cl::value_desc("filename"));
static cl::opt<std::string>
    view("view",
         cl::desc("For .svg and .json: topology, buffers or physical "
                  "(default physical)"),
         cl::init(""));
static cl::list<std::string>
    contract("contract",
             cl::desc("For .svg and .json: node kinds to contract (port, dma, "
                      "buffer, call, arg)"),
             cl::CommaSeparated);
static cl::opt<std::string> deviceName("device",
                                       cl::desc("aie.device to draw"));
static cl::opt<std::string>
    runtimeSequenceName("runtime-sequence",
                        cl::desc("aie.runtime_sequence to draw"));
static cl::opt<bool>
    controlOverlay("control-overlay",
                   cl::desc("Include control packet overlay routes"));
static cl::opt<bool> weighEdges("weigh-edges",
                                cl::desc("Scale line width by bytes moved"));
static cl::opt<bool> noSources("no-sources",
                               cl::desc("Do not embed source files"));
static cl::opt<bool>
    skipVerify("skip-verify",
               cl::desc("Do not verify the input, for designs whose lifted "
                        "flows claim overlapping routes"));

const std::string bold("\033[0;1m");
const std::string dim("\033[0;2m");
const std::string red("\033[0;31m");
const std::string green("\033[1;32m");
const std::string yellow("\033[1;33m");
const std::string blue("\033[1;34m");
const std::string cyan("\033[0;36m");
const std::string magenta("\033[0;35m");
const std::string bwhite("\033[0;47m");
const std::string reset("\033[0m");
const std::string bgray("\033[48;5;239m");

static int emitGraph(const AIE::viz::Graph &graph) {
  StringRef extension = sys::path::extension(outputFilename);
  if (extension != ".json" && extension != ".svg" && extension != ".html") {
    errs() << "aie-visualize: -o wants a .json, .svg or .html file\n";
    return 1;
  }
  std::string error;
  auto output = openOutputFile(outputFilename, &error);
  if (!output) {
    errs() << error << "\n";
    return 1;
  }
  std::string title = graph.device + " (" + graph.target + ")";
  if (extension == ".html") {
    output->os() << AIE::viz::renderHtml(AIE::standardViews(graph), title);
    output->keep();
    return 0;
  }
  AIE::viz::Graph shown = graph;
  if (!contract.empty()) {
    shown = AIE::viz::contract(graph, contract);
  } else if (!view.empty()) {
    auto views = AIE::standardViews(graph);
    auto found = llvm::find_if(views, [](const auto &entry) {
      return StringRef(entry.first).equals_insensitive(view);
    });
    if (found == views.end()) {
      errs() << "aie-visualize: unknown view '" << view << "'\n";
      return 1;
    }
    shown = found->second;
    title += " — " + found->first;
  }
  if (extension == ".json") {
    output->os() << formatv("{0:2}\n", toJSON(shown));
  } else {
    AIE::viz::RenderOptions options;
    options.weighEdges = weighEdges;
    options.title = title;
    output->os() << AIE::viz::renderSvg(shown, options);
  }
  output->keep();
  return 0;
}

int main(int argc, char *argv[]) {
  cl::ParseCommandLineOptions(argc, argv);

  if (StringRef(fileName).ends_with(".json")) {
    if (outputFilename.empty()) {
      errs() << "aie-visualize: a JSON input needs -o\n";
      return 1;
    }
    auto buffer = MemoryBuffer::getFileOrSTDIN(fileName);
    if (!buffer) {
      errs() << "aie-visualize: cannot read " << fileName << "\n";
      return 1;
    }
    Expected<json::Value> parsed = json::parse((*buffer)->getBuffer());
    if (!parsed) {
      errs() << "aie-visualize: " << toString(parsed.takeError()) << "\n";
      return 1;
    }
    Expected<AIE::viz::Graph> graph = AIE::viz::graphFromJSON(*parsed);
    if (!graph) {
      errs() << "aie-visualize: " << toString(graph.takeError()) << "\n";
      return 1;
    }
    return emitGraph(*graph);
  }

  MLIRContext ctx;
  ParserConfig pcfg(&ctx, /*verifyAfterParse=*/!skipVerify);
  SourceMgr srcMgr;

  DialectRegistry registry;
  registry.insert<arith::ArithDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<vector::VectorDialect>();
  xilinx::registerAllDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  xilinx::xllvm::registerXLLVMDialectTranslation(registry);
  ctx.appendDialectRegistry(registry);

  OwningOpRef<ModuleOp> owning =
      parseSourceFile<ModuleOp>(fileName, srcMgr, pcfg);

  if (!owning)
    return 1;

  if (!outputFilename.empty()) {
    SourceMgrDiagnosticHandler diagnostics(srcMgr, &ctx);
    AIE::VisualizeOptions options;
    options.deviceName = deviceName;
    options.runtimeSequenceName = runtimeSequenceName;
    options.controlOverlay = controlOverlay;
    options.embedSources = !noSources;
    options.sourceDirectory = sys::path::parent_path(fileName).str();
    options.irName = (sys::path::filename(fileName) + " (as read)").str();
    FailureOr<AIE::viz::Graph> graph = AIE::buildDesignGraph(*owning, options);
    if (failed(graph))
      return 1;
    return emitGraph(*graph);
  }

  auto deviceOps = owning->getOps<AIE::DeviceOp>();
  if (!llvm::hasSingleElement(deviceOps))
    return 2;

  AIE::DeviceOp deviceOp = *deviceOps.begin();

  const xilinx::AIE::AIETargetModel &model = deviceOp.getTargetModel();

  model.validate();

  std::vector<bool> used(model.columns() * model.rows());
  for (int col = 0; col < model.columns(); col++) {
    for (int row = 0; row < model.rows(); row++) {
      used[col + model.columns() * row] = false;
    }
  }
  for (auto tile : deviceOp.getOps<AIE::TileOp>()) {
    used[tile.getCol() + model.columns() * tile.getRow()] = true;
  }

  std::cout << model.columns() << " Columns and " << model.rows() << " Rows\n";
  for (int row = model.rows() - 1; row >= 0; row--) {
    std::cout << reset << row % 10 << " ";
    for (int col = 0; col < model.columns(); col++) {
      if (used[col + model.columns() * row])
        std::cout << bgray;
      else
        std::cout << dim;
      std::string v = reset + ".";
      if (model.isCoreTile(col, row))
        v = green + 'C';
      else if (model.isMemTile(col, row))
        v = red + 'M';
      else if (model.isShimNOCTile(col, row))
        v = blue + 'D';
      else if (model.isShimPLTile(col, row))
        v = magenta + 'P';
      std::cout << v << reset;
    }
    std::cout << "\n";
  }

  std::cout << reset << "  ";
  for (int col = 0; col < model.columns(); col++)
    std::cout << col % 10;
  std::cout << "\n";

  std::cout << "  ";
  for (int col = 0; col < model.columns(); col++) {
    int coltens = col / 10;
    if (coltens > 0)
      std::cout << coltens;
    else
      std::cout << " ";
  }
  std::cout << "\n";

  return 0;
}
