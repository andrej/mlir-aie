//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

// Scalar scale template
template <typename T>
void scale_scalar(T *a, T *c, T factor, const int32_t N) {
  event0();
  for (int i = 0; i < N; i++) {
    c[i] = factor * a[i];
  }
  event1();
}

// Vectorized scale template (general case)
// Assume N is multiple of 16
template <typename T>
void scale_vectorized(T *__restrict a, T *__restrict c, int32_t factor,
                      const int32_t N) {
  event0();
  constexpr int vec_factor = 32;
  T *__restrict pA1 = a;
  T *__restrict pC1 = c;
  const int F = N / vec_factor;
  T fac = factor;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < F; i++) {
    aie::vector<T, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
    pA1 += vec_factor;
    aie::accum<acc32, vec_factor> cout = aie::mul(A0, fac);
    aie::store_v(pC1, cout.template to_vector<T>(0));
    pC1 += vec_factor;
  }
  event1();
}

// Vectorized scale template (int32_t case, acc64 used)
// Assume N is multiple of 16
template <>
void scale_vectorized<int32_t>(int32_t *__restrict a, int32_t *__restrict c,
                               int32_t factor, const int32_t N) {
  event0();
  constexpr int vec_factor = 16;
  int32_t *__restrict pA1 = a;
  int32_t *__restrict pC1 = c;
  const int F = N / vec_factor;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < F; i++) {
    aie::vector<int32_t, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
    pA1 += vec_factor;
    aie::accum<acc64, vec_factor> cout = aie::mul(A0, factor);
    aie::store_v(pC1, cout.template to_vector<int32_t>(0));
    pC1 += vec_factor;
  }
  event1();
}

extern "C" {

#if BIT_WIDTH == 16

void vector_scalar_mul_scalar(int16_t *a_in, int16_t *c_out, int32_t *factor,
                              int32_t N) {
  scale_scalar<int16_t>(a_in, c_out, *factor, N);
}

void vector_scalar_mul_vector(int16_t *a_in, int16_t *c_out, int32_t *factor,
                              int32_t N) {
  scale_vectorized<int16_t>(a_in, c_out, *factor, N);
}

#else // Defaults to 32-bit

void vector_scalar_mul_scalar(int32_t *a_in, int32_t *c_out, int32_t *factor,
                              int32_t N) {
  scale_scalar<int32_t>(a_in, c_out, *factor, N);
}

void vector_scalar_mul_vector(int32_t *a_in, int32_t *c_out, int32_t *factor,
                              int32_t N) {
  scale_vectorized<int32_t>(a_in, c_out, *factor, N);
}

#endif

} // extern "C"
