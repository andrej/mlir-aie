//===- kernel.cc (AIE2PS kernel) --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

extern "C" {
void add_one(int *in, int *out);
}

void add_one(int *in, int *out) {
  for (int i = 0; i < 64; i++)
    out[i] = in[i] + 1;
}
