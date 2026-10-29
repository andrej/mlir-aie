// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A buffer with an initial value holds data the configuration loads, such as
// weights. Nothing writes it at run time; the graph says where its data
// comes from.

// REQUIRES: aie-visualize
// RUN: aie-visualize %s -o %t.json
// RUN: %PYTHON %S/Inputs/summarize-graph.py %t.json | FileCheck %s
// RUN: aie-visualize %t.json -o %t.svg
// RUN: FileCheck %s --input-file=%t.svg --check-prefix=SVG

// CHECK-DAG: node buffer:scratch buffer "scratch"{{$}}
// CHECK-DAG: node buffer:weights buffer "weights" initialized
// CHECK-DAG: edge buffer:weights->call:0,2#0
// SVG: class="node buffer initialized" data-id="buffer:weights"
// SVG: buffer with an initial value

module {
  aie.device(npu1_1col) {
    func.func private @apply(memref<4xi32>, memref<4xi32>)
    %tile = aie.tile(0, 2)
    %weights = aie.buffer(%tile) {sym_name = "weights"} : memref<4xi32> = dense<[1, 2, 3, 4]>
    %scratch = aie.buffer(%tile) {sym_name = "scratch"} : memref<4xi32>
    %core = aie.core(%tile) {
      func.call @apply(%weights, %scratch) : (memref<4xi32>, memref<4xi32>) -> ()
      aie.end
    }
  }
}
