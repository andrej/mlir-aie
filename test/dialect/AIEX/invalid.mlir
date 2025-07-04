//===- invalid.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu1) {
  aiex.runtime_sequence() {
    // expected-error@+1 {{'aiex.npu.dma_wait' op couldn't find symbol in parent device}}
    aiex.npu.dma_wait {symbol = @out0}
  }
}
