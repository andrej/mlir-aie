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

#include <cstdint>
#include <iostream>

// tile_size * 6 * 4 = 1024 * 12 * 4
constexpr size_t N = 1024 * 3 * 16;

int main() {
  try {
    xrt::device device{0};
    xrt::elf elf("aie_control_full.elf");
    xrt::hw_context hwctx{device, elf};
    xrt::kernel kernel = xrt::ext::kernel{hwctx, "main:sequence"};

    xrt::bo bo_a(hwctx, N * sizeof(int32_t), 0);
    xrt::bo bo_b(hwctx, N * sizeof(int32_t), 0);
    xrt::bo bo_c(hwctx, N * sizeof(int32_t), 0);

    auto *a = bo_a.map<int32_t *>();
    auto *b = bo_b.map<int32_t *>();
    auto *c = bo_c.map<int32_t *>();

    for (size_t i = 0; i < N; i++) {
      a[i] = static_cast<int32_t>(i % 100 + 1);
      b[i] = static_cast<int32_t>(i % 7 + 1);
      c[i] = 0;
    }

    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::run run{kernel};
    run.set_arg(0, bo_a);
    run.set_arg(1, bo_b);
    run.set_arg(2, bo_c);
    run.start();
    run.wait(600000);

    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int errors = 0;
    for (size_t i = 0; i < N; i++) {
      int32_t expected = a[i] * b[i];
      if (c[i] != expected) {
        if (errors < 10)
          std::cout << "error @ " << i << ": got " << c[i] << " expected "
                    << expected << std::endl;
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
