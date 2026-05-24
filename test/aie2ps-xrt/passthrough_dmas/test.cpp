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

constexpr size_t N = 4096;
constexpr size_t BUF_SIZE = N * sizeof(int32_t);

int main() {
  try {
    xrt::device device{0};
    xrt::elf elf("aie_control_full.elf");
    xrt::hw_context hwctx{device, elf};
    xrt::kernel kernel = xrt::ext::kernel{hwctx, "main:sequence"};

    xrt::bo bo_in(hwctx, BUF_SIZE, 0);
    xrt::bo bo_buf(hwctx, BUF_SIZE, 0);
    xrt::bo bo_out(hwctx, BUF_SIZE, 0);

    auto *in = bo_in.map<int32_t *>();
    auto *out = bo_out.map<int32_t *>();
    for (size_t i = 0; i < N; i++) {
      in[i] = static_cast<int32_t>(i + 1);
      out[i] = 0;
    }
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::run run{kernel};
    run.set_arg(0, bo_in);
    run.set_arg(1, bo_buf);
    run.set_arg(2, bo_out);
    run.start();
    auto state = run.wait(600000);

    if (state == ERT_CMD_STATE_TIMEOUT) {
      std::cout << "TEST FAILED" << std::endl;
      return 1;
    }

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int errors = 0;
    for (size_t i = 0; i < N; i++) {
      if (out[i] != static_cast<int32_t>(i + 1)) {
        if (errors < 10)
          std::cout << "error @ " << i << ": got " << out[i] << " expected "
                    << (i + 1) << std::endl;
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
