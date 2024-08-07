//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <math.h>
#include <numeric>
#include <sstream>
#include <string.h>
#include <string>
#include <unistd.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#ifndef XCLBIN
#define XCLBIN "build/final.xclbin"
#endif

#ifndef INSTS_TXT
#define INSTS_TXT "build/insts.txt"
#endif

struct target {
  char name[128];
  char xclbin[128];
  char insts[128];
};

struct target targets[] = {
    {"do nothing", "build/final_do_nothing.xclbin",
     "build/insts_do_nothing.txt"},
    {"config + address patch BD only", "build/final_config.xclbin",
     "build/insts_config.txt"},
    {"config + address patch + submit", "build/final_config_start.xclbin",
     "build/insts_config_start.txt"},
    {"config + address patch + submit + sync",
     "build/final_config_start_sync.xclbin",
     "build/insts_config_start_sync.txt"}};

#ifndef KERNEL_NAME
#define KERNEL_NAME "MLIR_AIE"
#endif

std::vector<uint32_t> load_instr_sequence(std::string instr_path) {
  std::ifstream instr_file(instr_path);
  std::string line;
  std::vector<uint32_t> instr_v;
  while (std::getline(instr_file, line)) {
    std::istringstream iss(line);
    uint32_t a;
    if (!(iss >> std::hex >> a)) {
      throw std::runtime_error("Unable to parse instruction file\n");
    }
    instr_v.push_back(a);
  }
  return instr_v;
}

int main(int argc, const char *argv[]) {
  size_t target = 0;
  if (argc >= 2) {
    target = std::stoi(argv[1]);
    assert(target < sizeof(targets));
  }

  long iters = 1;
  if (argc >= 3) {
    iters = std::stol(argv[2]);
  }

  std::cout << std::endl << "Benchmark: " << targets[target].name << std::endl;

  // Load instruction sequence
  std::vector<uint32_t> instr_v = load_instr_sequence(targets[target].insts);
  std::cout << "Sequence instr count: " << instr_v.size() << std::endl;

  // Get a device handle
  unsigned int device_index = 0;
  xrt::device device = xrt::device(device_index);

  // Load the xclbin
  xrt::xclbin xclbin = xrt::xclbin(targets[target].xclbin);

  // Get the kernel from the xclbin
  std::vector<xrt::xclbin::kernel> xkernels = xclbin.get_kernels();
  xrt::xclbin::kernel xkernel = *std::find_if(
      xkernels.begin(), xkernels.end(), [](xrt::xclbin::kernel &k) {
        return k.get_name().rfind(KERNEL_NAME, 0) == 0;
      });
  std::string kernel_name = xkernel.get_name();
  assert(strcmp(kernel_name.c_str(), KERNEL_NAME) == 0);

  device.register_xclbin(xclbin);

  // get a hardware context
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  auto kernel = xrt::kernel(context, kernel_name);

  // set up the buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Execute the kernel and wait to finish
  unsigned int opcode = 3;

  std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
  double npu_times[iters];
  int i;
  for (i = 0; i < iters; i++) {
    start = std::chrono::high_resolution_clock::now();
    auto run = kernel(opcode, bo_instr, instr_v.size());
    ert_cmd_state r = run.wait();
    stop = std::chrono::high_resolution_clock::now();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << std::endl
                << "Iteration " << i << " errored with code: " << r;
      break;
    }
    npu_times[i] =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
  }

  double sum = std::accumulate(npu_times, npu_times + i, 0.0);
  double mean = sum / i;

  double sq_sum = std::inner_product(npu_times, npu_times + i, npu_times, 0.0);
  double stddev = std::sqrt(sq_sum / i - mean * mean);

  double min = *std::min_element(npu_times, npu_times + i);
  double max = *std::max_element(npu_times, npu_times + i);

  std::cout << std::endl
            << "Runtime stats after " << i << " iterations:" << std::endl;
  std::cout << "Average: " << std::setw(10) << std::fixed
            << std::setprecision(0) << mean << " μs" << std::endl;
  std::cout << "Stddev:  " << std::setw(10) << std::fixed
            << std::setprecision(0) << stddev << " μs" << std::endl;
  std::cout << "Min:     " << std::setw(10) << std::fixed
            << std::setprecision(0) << min << " μs" << std::endl;
  std::cout << "Max:     " << std::setw(10) << std::fixed
            << std::setprecision(0) << max << " μs" << std::endl;
}
