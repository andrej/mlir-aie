//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_ext.h"

#include "cxxopts.hpp"
#include "test_utils.h"

constexpr int IN_SIZE = 1024*1024;
constexpr int OUT_SIZE = 1024*1024;

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("add_one_objFifo_elf");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  int verbosity = vm["verbosity"].as<int>();

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  std::string kernel_name = vm["kernel"].as<std::string>();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  xrt::elf elf(vm["instr"].as<std::string>());
  xrt::module mod{elf};

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernel_name << "\n";
  auto kernel = xrt::ext::kernel(context, mod, kernel_name);

  xrt::bo bo_in = xrt::ext::bo{device, IN_SIZE * sizeof(int32_t)};
  xrt::bo bo_out = xrt::ext::bo{device, OUT_SIZE * sizeof(int32_t)};

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  uint32_t *buf_in = bo_in.map<uint32_t *>();
  std::vector<uint32_t> srcVecA;
  for (int i = 0; i < IN_SIZE; i++)
    srcVecA.push_back(i + 1);
  memcpy(buf_in, srcVecA.data(), (srcVecA.size() * sizeof(uint32_t)));

  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;

  auto t_start = std::chrono::high_resolution_clock::now();
  auto run = kernel(opcode, 0, 0, bo_in, bo_out);
  run.wait2();
  auto t_stop = std::chrono::high_resolution_clock::now();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint32_t *buf_out = bo_out.map<uint32_t *>();

  int errors = 0;

  for (uint32_t i = 0; i < OUT_SIZE; i++) {
    uint32_t ref = i + 42;
    if (*(buf_out + i) != ref) {
      std::cout << "Error in output " << *(buf_out + i) << " != " << ref
                << std::endl;
      errors++;
    }
  }

  float time = std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start).count();
  std::cout << "Elapsed time for kernel execution: " << std::fixed << std::setprecision(0) << std::setw(8) << time << " μs" << std::endl;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nfailed.\n\n";
    return 1;
  }
}
