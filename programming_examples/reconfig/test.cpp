//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <vector>
#include <fstream>
#include <cstring>
#include <cmath>
#include <iomanip>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#define DTYPE int32_t

// All matrices are the same square size, 4x4, for 16 elements total.
#define MATRIX_ROWS 8
#define MATRIX_COLS 8
#define SIZE MATRIX_ROWS*MATRIX_COLS

// Point to the correct location and kernel names for these files.
#define KERNEL "MLIR_AIE"
#define CONFIG_INSTS_BIN "build/config_insts.bin"
#define RUN_INSTS_BIN "build/run_insts.bin"
#define FINAL_XCLBIN "build/empty.xclbin"

std::vector<uint32_t> load_instr_binary(std::string instr_path) {
  // Open file in binary mode
  std::ifstream instr_file(instr_path, std::ios::binary);
  if (!instr_file.is_open()) {
    throw std::runtime_error("Unable to open instruction file\n");
  }

  // Get the size of the file
  instr_file.seekg(0, std::ios::end);
  std::streamsize size = instr_file.tellg();
  instr_file.seekg(0, std::ios::beg);

  // Check that the file size is a multiple of 4 bytes (size of uint32_t)
  if (size % 4 != 0) {
    throw std::runtime_error("File size is not a multiple of 4 bytes\n");
  }

  // Allocate vector and read the binary data
  std::vector<uint32_t> instr_v(size / 4);
  if (!instr_file.read(reinterpret_cast<char *>(instr_v.data()), size)) {
    throw std::runtime_error("Failed to read instruction file\n");
  }
  return instr_v;
}

template <typename T>
static inline T get_random();

template <>
int32_t get_random<int32_t>() {
  return (int32_t)rand();
}

void print_matrix(std::vector<DTYPE> matrix) {
  std::cout << std::setfill('0') << std::setw(7) << std::fixed << std::setprecision(3);
  for(int row = 0; row < MATRIX_ROWS; row++) {
    for(int col = 0; col < MATRIX_COLS; col++) {
      std::cout << matrix[row * MATRIX_COLS + col] << "  "; 
    }
    std::cout << std::endl;
  }
}

int main(int argc, const char *argv[]) {

  // Fix the seed to ensure reproducibility.
  srand(1726250518); // srand(time(NULL));

  size_t buf_size = SIZE * sizeof(DTYPE);

  // XRT setup: load instruction sequence and xclbin, find kernel in xclbin, initialize buffers, finally call the kernel.
  std::vector<uint32_t> config_instr_v = load_instr_binary(CONFIG_INSTS_BIN);
  const size_t config_instr_size = config_instr_v.size() * sizeof(uint32_t);
  std::cout << "Config sequence instr count: " << config_instr_v.size() << "\n";
  std::vector<uint32_t> run_instr_v = load_instr_binary(RUN_INSTS_BIN);
  const size_t run_instr_size = run_instr_v.size() * sizeof(uint32_t);
  std::cout << "Run sequence instr count: " << run_instr_v.size() << "\n";
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  std::cout << "Loading xclbin: " << FINAL_XCLBIN << "\n";
  auto xclbin = xrt::xclbin(std::string(FINAL_XCLBIN));
  std::string kernelName = KERNEL;
  std::cout << "Registering xclbin: " << FINAL_XCLBIN << "\n";
  device.register_xclbin(xclbin);
  std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());
  std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  // Initialize A, B, C buffers
  const size_t instr_size = config_instr_size > run_instr_size ? config_instr_size : run_instr_size;
  auto bo_instr = xrt::bo(device, instr_size, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, buf_size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b = xrt::bo(device, buf_size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out = xrt::bo(device, buf_size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));


  // Instruction buffer for DMA configuration
  void *bufInstr = bo_instr.map<void *>();
  const unsigned int opcode = 3;
  ert_cmd_state r;
  xrt::run run;

  // 1.)
  // Configuration

  memcpy(bufInstr, config_instr_v.data(), config_instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  std::cout << "Running Configuration Kernel.\n";
  run = kernel(opcode, bo_instr, config_instr_v.size(), bo_a, bo_out);
  r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }
  std::cout << "Kernel completed.\n";

  // 2.) 
  // Run

  // Set up (random) input matrices.
  std::cout << "Writing data into buffer objects.\n";
  DTYPE *bufA = bo_a.map<DTYPE *>();
  std::vector<DTYPE> AVec(SIZE);
  for (int i = 0; i < SIZE; i++) {
    AVec[i] = get_random<DTYPE>();
  }
  memcpy(bufA, AVec.data(), (AVec.size() * sizeof(DTYPE)));

  DTYPE *bufB = bo_b.map<DTYPE *>();
  std::vector<DTYPE> BVec(SIZE);
  for (int i = 0; i < SIZE; i++) {
    BVec[i] = get_random<DTYPE>();
  }
  memcpy(bufB, BVec.data(), (BVec.size() * sizeof(DTYPE)));

  char *bufOut = bo_out.map<char *>();
  std::vector<DTYPE> CVec(SIZE);
  memset(bufOut, 0, buf_size);

  memcpy(bufInstr, run_instr_v.data(), run_instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Finally run the kernel.
  std::cout << "Running Kernel.\n";
  run = kernel(opcode, bo_instr, run_instr_v.size(), bo_a, bo_out);
  r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }
  std::cout << "Kernel completed.\n";

  // Copy out outputs from kernel run.
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  memcpy(CVec.data(), bufOut, (CVec.size() * sizeof(DTYPE)));

  // Print input, output, reference
  std::cout << std::endl;
  std::cout << "Input" << std::endl;
  std::cout << "##########" << std::endl;
  std::cout << std::endl;
  std::cout << "A:" << std::endl;
  print_matrix(AVec);

  std::cout << "Observed Output" << std::endl;
  std::cout << "###############" << std::endl;
  print_matrix(CVec);
  std::cout << std::endl;

  return 0;
}
