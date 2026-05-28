//===- passthrough.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>

extern "C" {
void passThroughLine(uint8_t *in, uint8_t *out, int32_t n) {
  for (int i = 0; i < n; i++)
    out[i] = in[i];
}
}
