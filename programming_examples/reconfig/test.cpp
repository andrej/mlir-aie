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
#include <chrono>


#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#define DTYPE int32_t

// So logging to stdout doesn't affect measurements, disable it during benchmarking.
#define VERBOSE 0
#if VERBOSE
#define log(X) X
#else
#define log(X) do {} while(0)
#endif

// All matrices are the same square size, 4x4, for 16 elements total.
#define MATRIX_ROWS 64 
#define MATRIX_COLS 64 
#define SIZE MATRIX_ROWS*MATRIX_COLS
#define PRINT_MATRIX_ROWS 8
#define PRINT_MATRIX_COLS 8

// Point to the correct location and kernel names for these files.
#define KERNEL "MLIR_AIE"

constexpr size_t BUF_SIZE = SIZE * sizeof(DTYPE);

struct kernel {
  xrt::xclbin xclbin;
  xrt::kernel xrt_kernel;
  xrt::hw_context hw_context;
  xrt::bo bo_instr;
  xrt::bo bo_inout;
  void *buf_instr;
  DTYPE *buf_inout;
};

struct kernel load_xclbin(xrt::device &device, std::string &path, size_t max_instr_size) {
  log(std::cout << "Loading xclbin: " << path << std::endl);
  auto xclbin = xrt::xclbin(path);
  std::string kernelName = KERNEL;
  log(std::cout << "Registering xclbin: " << path << std::endl);
  device.register_xclbin(xclbin);
  log(std::cout << "Getting hardware context." << std::endl);
  xrt::hw_context context(device, xclbin.get_uuid());

  xrt::kernel kernel = xrt::kernel(context, kernelName);

  xrt::bo bo_instr = xrt::bo(device, max_instr_size * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  xrt::bo bo_inout = xrt::bo(device, BUF_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  void *buf_instr = bo_instr.map<void *>();
  DTYPE *buf_inout = bo_inout.map<DTYPE *>();

  return {
    std::move(xclbin),
    std::move(kernel),
    std::move(context),
    std::move(bo_instr),
    std::move(bo_inout),
    buf_instr,
    buf_inout
  };
}

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
  for(int row = 0; row < PRINT_MATRIX_ROWS; row++) {
    for(int col = 0; col < PRINT_MATRIX_COLS; col++) {
      std::cout << std::setfill(' ') << std::setw(10) << std::fixed << std::setprecision(3)
                << matrix[row * MATRIX_COLS + col] << "  "; 
    }
    if constexpr (PRINT_MATRIX_COLS < MATRIX_COLS) {
      std::cout << "...";
    }
    std::cout << std::endl;
  }
}

int main(int argc, const char *argv[]) {
  // Fix the seed to ensure reproducibility.
  srand(1726250518); // srand(time(NULL));

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " [<xclbin> <insts.bin>] ..\n";
    exit(1);
  }

  // pre-load all insts.bins
  std::map<std::string, std::vector<uint32_t>> instr_vectors;
  size_t max_instr_size = 0;
  for(int i = 2; i < argc; i += 2) {
    std::string instr_path = argv[i];
    instr_vectors[instr_path] = load_instr_binary(instr_path);
    max_instr_size = std::max(max_instr_size, instr_vectors[instr_path].size());
  }
  log(std::cout << "Max sequence instr count: " << max_instr_size << std::endl);

  // XRT setup: load instruction sequence and xclbin, find kernel in xclbin, initialize buffers, finally call the kernel.
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // pre-load all xclbins
  std::map<std::string, struct kernel> kernels;
  for (int i = 1; i < argc; i += 2) {
    std::string xclbin_path = argv[i];
    kernels[xclbin_path] = load_xclbin(device, xclbin_path, max_instr_size);
  }

  const unsigned int opcode = 3;
  ert_cmd_state r;
  xrt::run run;

  // Set up (random) input data for first kernel.
  DTYPE *buf_inout = kernels[argv[1]].buf_inout;
  std::vector<DTYPE> vec_in(SIZE);
  for (int i = 0; i < SIZE; i++) {
    vec_in[i] = get_random<DTYPE>();
  }
  memcpy(buf_inout, vec_in.data(), BUF_SIZE);

  // Run each of the kernels.
  auto t_start = std::chrono::high_resolution_clock::now();
  struct kernel *last_kernel = nullptr;
  for (int i = 1; i < argc; i += 2) {
    std::string xclbin_path = argv[i];
    std::string instr_path = argv[i + 1];
    struct kernel &kernel = kernels[xclbin_path];
    xrt::kernel &xrt_kernel = kernel.xrt_kernel;
    xrt::bo &bo_instr = kernel.bo_instr;
    xrt::bo &bo_inout = kernel.bo_inout;
    void *buf_instr = kernel.buf_instr;

    std::vector<uint32_t> &instr_v = instr_vectors[instr_path];

    memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(int));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    if (&kernel != last_kernel) {
      if (last_kernel != nullptr) {
        log(std::cout << "Copying last kernels results into this kernels." << std::endl);
        memcpy(kernel.buf_inout, last_kernel->buf_inout, BUF_SIZE);
      }
      bo_inout.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    log(std::cout << "Running Kernel." << std::endl);
    run = xrt_kernel(opcode, bo_instr, instr_v.size(), bo_inout);
    r = run.wait();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete. Returned status: " << r << std::endl;
      return 1;
    }
    log(std::cout << "Kernel completed." << std::endl);

    last_kernel = &kernel;
  }
  auto t_stop = std::chrono::high_resolution_clock::now();

  // Copy out outputs from last kernel run.
  std::vector<DTYPE> vec_out(SIZE);

  buf_inout = kernels[argv[argc-2]].buf_inout;
  memcpy(vec_out.data(), buf_inout, BUF_SIZE);

  #if VERBOSE
    // Print input, output, reference
    std::cout << std::endl;
    std::cout << "Input" << std::endl;
    std::cout << "##########" << std::endl;
    print_matrix(vec_in);

    std::cout << "Observed Output" << std::endl;
    std::cout << "#################" << std::endl;
    print_matrix(vec_out);

    std::cout << std::endl;
  #endif

  float time = std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start).count();
  std::cout << "Elapsed time for all kernel executions: " << std::fixed << std::setprecision(0) << std::setw(8) << time << " μs" << std::endl;

  return 0;
}
