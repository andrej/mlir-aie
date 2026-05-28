//===- mm_peano.cc ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif
#ifndef DIM_K
#define DIM_K 32
#endif
#ifndef DIM_N
#define DIM_N 32
#endif

extern "C" {

#ifdef DTYPE_BF16

void matmul_bf16_f32(bfloat16 *a, bfloat16 *b, float *c) {
  event0();
  for (int i = 0; i < DIM_M; i++)
    for (int j = 0; j < DIM_N; j++)
      for (int kk = 0; kk < DIM_K; kk++)
        c[i * DIM_N + j] += (float)a[i * DIM_K + kk] * (float)b[kk * DIM_N + j];
  event1();
}

void zero_f32(float *c) {
  event0();
  for (int i = 0; i < DIM_M * DIM_N; i++)
    c[i] = 0.0f;
  event1();
}

#else // i16×i16→i32 (default)

void matmul_i16_i32(int16_t *a, int16_t *b, int32_t *c) {
  event0();
  for (int i = 0; i < DIM_M; i++)
    for (int j = 0; j < DIM_N; j++)
      for (int kk = 0; kk < DIM_K; kk++)
        c[i * DIM_N + j] +=
            (int32_t)a[i * DIM_K + kk] * (int32_t)b[kk * DIM_N + j];
  event1();
}

void zero_i32(int32_t *c) {
  event0();
  for (int i = 0; i < DIM_M * DIM_N; i++)
    c[i] = 0;
  event1();
}

#endif

} // extern "C"
