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

constexpr size_t N = 1024;

// bfloat16 helpers
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

int main() {
  try {
    xrt::device device{0};
    xrt::elf elf("aie_control_full.elf");
    xrt::hw_context hwctx{device, elf};
    xrt::kernel kernel = xrt::ext::kernel{hwctx, "main:sequence"};

    xrt::bo bo_in1(hwctx, N * sizeof(uint16_t), 0);
    xrt::bo bo_in2(hwctx, N * sizeof(uint16_t), 0);
    xrt::bo bo_out(hwctx, N * sizeof(uint16_t), 0);

    auto *in1 = bo_in1.map<uint16_t *>();
    auto *in2 = bo_in2.map<uint16_t *>();
    auto *out = bo_out.map<uint16_t *>();

    for (size_t i = 0; i < N; i++) {
      in1[i] = f2bf(static_cast<float>(i + 1));
      in2[i] = f2bf(static_cast<float>(i + 1));
      out[i] = 0;
    }

    bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::run run{kernel};
    run.set_arg(0, bo_in1);
    run.set_arg(1, bo_in2);
    run.set_arg(2, bo_out);
    run.start();
    run.wait(600000);

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int errors = 0;
    for (size_t i = 0; i < N; i++) {
      float got = bf2f(out[i]);
      float a_val = static_cast<float>(i + 1);
      float expected = a_val + a_val; // a[i] + b[i], where a==b
      float rel_err =
          std::abs(got - expected) / std::max(std::abs(expected), 1.0f);
      if (rel_err > 0.02f) {
        if (errors < 10)
          std::cout << "error @ " << i << ": got " << got << " expected "
                    << expected << " (rel_err=" << rel_err << ")" << std::endl;
        errors++;
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
