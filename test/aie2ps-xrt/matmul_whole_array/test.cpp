//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#ifndef MAT_M
#define MAT_M 512
#endif
#ifndef MAT_K
#define MAT_K 512
#endif
#ifndef MAT_N
#define MAT_N 1152
#endif

#ifdef DTYPE_BF16
using in_t = uint16_t; // raw bf16 bits
using out_t = float;

static uint16_t f2bf(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  return static_cast<uint16_t>(bits >> 16);
}
static float bf2f(uint16_t bf) {
  uint32_t bits = static_cast<uint32_t>(bf) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
#else
using in_t = int16_t;
using out_t = int32_t;
#endif

int main() {
  try {
    xrt::device device{0};
    xrt::elf elf("aie_control_full.elf");
    xrt::hw_context hwctx{device, elf};
    xrt::kernel kernel = xrt::ext::kernel{hwctx, "main:sequence"};

    constexpr size_t M = MAT_M, K = MAT_K, N = MAT_N;

    xrt::bo bo_a(hwctx, M * K * sizeof(in_t), 0);
    xrt::bo bo_b(hwctx, K * N * sizeof(in_t), 0);
    xrt::bo bo_c(hwctx, M * N * sizeof(out_t), 0);

    auto *a = bo_a.map<in_t *>();
    auto *b = bo_b.map<in_t *>();
    auto *c = bo_c.map<out_t *>();

    // Initialize inputs
    for (size_t i = 0; i < M * K; i++) {
#ifdef DTYPE_BF16
      a[i] = f2bf(static_cast<float>((i % 7) - 3)); // small values
#else
      a[i] = static_cast<in_t>((i % 7) - 3);
#endif
    }
    for (size_t i = 0; i < K * N; i++) {
#ifdef DTYPE_BF16
      b[i] = f2bf(static_cast<float>((i % 5) - 2));
#else
      b[i] = static_cast<in_t>((i % 5) - 2);
#endif
    }
    std::memset(c, 0, M * N * sizeof(out_t));

    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::run run{kernel};
    run.set_arg(0, bo_a);
    run.set_arg(1, bo_b);
    run.set_arg(2, bo_c);
    run.start();
    run.wait(600000);

    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Reference computation
    int errors = 0;
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
#ifdef DTYPE_BF16
        float ref = 0.0f;
        for (size_t kk = 0; kk < K; kk++)
          ref += bf2f(a[i * K + kk]) * bf2f(b[kk * N + j]);
        float got = c[i * N + j];
        float rel_err = std::abs(got - ref) / std::max(std::abs(ref), 1.0f);
        if (rel_err > 0.05f) {
#else
        out_t ref = 0;
        for (size_t kk = 0; kk < K; kk++)
          ref += static_cast<out_t>(a[i * K + kk]) *
                 static_cast<out_t>(b[kk * N + j]);
        if (c[i * N + j] != ref) {
#endif
          if (errors < 10)
            std::cout << "error @ (" << i << "," << j << "): got "
                      << c[i * N + j] << " expected " << ref << std::endl;
          errors++;
        }
      }
    }

    std::cout << (errors ? "TEST FAILED" : "TEST PASSED") << std::endl;
    return errors ? 1 : 0;
  } catch (const std::exception &e) {
    std::cout << "Exception: " << e.what() << std::endl;
    std::cout << "TEST FAILED" << std::endl;
    return 1;
  }
}
