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
#include <cstring>
#include <fstream>
#include <iostream>

constexpr size_t N = 1024;
constexpr int32_t SCALE_FACTOR = 3;
constexpr size_t TRACE_SIZE = 8192; // in 32-bit words

int main() {
  try {
    xrt::device device{0};
    xrt::elf elf("aie_control_full.elf");
    xrt::hw_context hwctx{device, elf};
    xrt::kernel kernel = xrt::ext::kernel{hwctx, "main:sequence"};

    xrt::bo bo_in1(hwctx, N * sizeof(int32_t), 0);
    xrt::bo bo_in2(hwctx, 1 * sizeof(int32_t), 0);
    xrt::bo bo_out(hwctx, N * sizeof(int32_t), 0);
    xrt::bo bo_trace(hwctx, TRACE_SIZE * sizeof(int32_t), 0);

    auto *in1 = bo_in1.map<int32_t *>();
    auto *in2 = bo_in2.map<int32_t *>();
    auto *out = bo_out.map<int32_t *>();
    auto *trace = bo_trace.map<char *>();

    for (size_t i = 0; i < N; i++) {
      in1[i] = static_cast<int32_t>(i + 1);
      out[i] = 0;
    }
    in2[0] = SCALE_FACTOR;
    memset(trace, 0, TRACE_SIZE * sizeof(int32_t));

    bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::run run{kernel};
    run.set_arg(0, bo_in1);
    run.set_arg(1, bo_in2);
    run.set_arg(2, bo_out);
    run.set_arg(3, bo_trace);
    run.start();
    run.wait(600000);

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Write trace buffer to file
    std::ofstream trace_file("trace.bin", std::ios::binary);
    trace_file.write(trace, TRACE_SIZE * sizeof(int32_t));
    trace_file.close();
    std::cout << "Trace written to trace.bin (" << TRACE_SIZE * sizeof(int32_t)
              << " bytes)" << std::endl;

    // Verify output
    int errors = 0;
    for (size_t i = 0; i < N; i++) {
      int32_t expected = static_cast<int32_t>(i + 1) * SCALE_FACTOR;
      if (out[i] != expected) {
        if (errors < 10)
          std::cout << "error @ " << i << ": got " << out[i] << " expected "
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
