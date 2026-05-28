//===- add_peano.cc ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

extern "C" {

void eltwise_add_bf16_vector(bfloat16 *a, bfloat16 *b, bfloat16 *c) {
  event0();
  for (int i = 0; i < 1024; i++) {
    c[i] = a[i] + b[i];
  }
  event1();
}

} // extern "C"
