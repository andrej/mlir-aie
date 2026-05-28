//===- scale_peano.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#define DTYPE_I32 1
#define DTYPE_I16 2

#ifndef DTYPE
#define DTYPE DTYPE_I32
#endif

#ifndef VECTORIZED
#define VECTORIZED 0
#endif

extern "C" {

#if DTYPE == DTYPE_I16

#if VECTORIZED
void vector_scalar_mul_vector(int16_t *a_in, int16_t *c_out, int32_t *factor,
                              int32_t N) {
  event0();
  int16_t f = (int16_t)*factor;
  v32int16 vf = broadcast_to_v32int16(f);
  for (int i = 0; i < N / 32; i++) {
    v32int16 va = *(v32int16 *)(a_in + i * 32);
    v32acc32 acc = mul_elem_32_32b(va, vf);
    *(v32int16 *)(c_out + i * 32) = lsrs(acc, 0, 1);
  }
  event1();
}
#else
void vector_scalar_mul_vector(int16_t *a_in, int16_t *c_out, int32_t *factor,
                              int32_t N) {
  event0();
  int16_t f = (int16_t)*factor;
  for (int i = 0; i < N; i++) {
    c_out[i] = a_in[i] * f;
  }
  event1();
}
#endif

#else // DTYPE_I32

void vector_scalar_mul_vector(int32_t *a_in, int32_t *c_out, int32_t *factor,
                              int32_t N) {
  event0();
  int32_t f = *factor;
  for (int i = 0; i < N; i++) {
    c_out[i] = f * a_in[i];
  }
  event1();
}

#endif

} // extern "C"
