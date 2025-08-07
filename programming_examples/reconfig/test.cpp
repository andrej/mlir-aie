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


#if USE_RUNLIST
#include "xrt/experimental/xrt_kernel.h"
#endif
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#define DTYPE int32_t

// So logging to stdout doesn't affect measurements, disable it during benchmarking.
#if VERBOSE
#define log(X) X
#else
#define log(X) do {} while(0)
#endif

// All matrices are the same square size, 4x4, for 16 elements total.
#define MATRIX_ROWS 1024 
#define MATRIX_COLS 1024 
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
};

struct kernel load_xclbin(xrt::device &device, std::string &path) {
  log(std::cout << "Loading xclbin: " << path << std::endl);
  auto xclbin = xrt::xclbin(path);
  std::string kernelName = KERNEL;
  log(std::cout << "Registering xclbin: " << path << std::endl);
  device.register_xclbin(xclbin);
  log(std::cout << "Getting hardware context." << std::endl);
  xrt::hw_context context(device, xclbin.get_uuid());

  xrt::kernel kernel = xrt::kernel(context, kernelName);

  return {
    std::move(xclbin),
    std::move(kernel),
    std::move(context),
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
  xrt::device device = xrt::device(device_index);

  // pre-load all xclbins and copy in insts.bins
  std::map<std::string, struct kernel> kernels;
  for (int i = 1; i < argc; i += 2) {
    std::string xclbin_path = argv[i];
    struct kernel &kernel = kernels[xclbin_path];
    kernel = load_xclbin(device, xclbin_path);
  }

  // set up buffer objects
  // we assume that all kernels use the same group_id(3) for the inout buffer, so we can reuse the same buffer object
  xrt::bo bo_inout = xrt::bo(device, BUF_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernels.begin()->second.xrt_kernel.group_id(3));
  DTYPE *buf_inout = bo_inout.map<DTYPE *>();
  std::vector<xrt::bo> bo_instrs;
  std::vector<void*> buf_instrs;
  for (int i = 1; i < argc; i += 2) {
    std::string xclbin_path = argv[i];
    std::string instr_path = argv[i + 1];
    std::vector<uint32_t> &instr_v = instr_vectors[instr_path];
    xrt::bo &bo_instr = bo_instrs.emplace_back(device, instr_v.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE, kernels[xclbin_path].xrt_kernel.group_id(1));
    void *buf_instr = bo_instr.map<void *>();
    buf_instrs.push_back(buf_instr);
    memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(int));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  const unsigned int opcode = 3;
  ert_cmd_state r;
  xrt::run run;

  // Set up (random) input data for first kernel.
  std::vector<DTYPE> vec_in(SIZE);
  for (int i = 0; i < SIZE; i++) {
    vec_in[i] = get_random<DTYPE>();
  }
  memcpy(buf_inout, vec_in.data(), BUF_SIZE);

#if USE_RUNLIST

  // Add kernels to runlist.
  std::vector<xrt::run> runs;
  std::string first_xclbin = argv[1];
  xrt::runlist runlist = xrt::runlist(kernels[first_xclbin].hw_context);
  for (int i = 1; i < argc; i += 2) {
    std::string xclbin_path = argv[i];
    std::string instr_path = argv[i + 1];
    struct kernel &kernel = kernels[xclbin_path];
    std::vector<uint32_t> &instr_v = instr_vectors[instr_path];
    xrt::bo &bo_instr = bo_instrs[(i - 1) / 2];
    xrt::run &run = runs.emplace_back(kernel.xrt_kernel);
    run.set_arg(0, opcode);
    run.set_arg(1, bo_instr);
    run.set_arg(2, instr_v.size());
    run.set_arg(3, bo_inout);
    run.set_arg(4, 0);
    run.set_arg(5, 0);
    run.set_arg(6, 0);
    run.set_arg(7, 0);
    runlist.add(run);
  }

  // Run them.
  bo_inout.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto t_start = std::chrono::high_resolution_clock::now();
  runlist.execute();
  runlist.wait();
  auto t_stop = std::chrono::high_resolution_clock::now();
  bo_inout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

#else

  // Run each of the kernels.
  auto t_start = std::chrono::high_resolution_clock::now();
  struct kernel *last_kernel = nullptr;
  for (int i = 1; i < argc; i += 2) {
    std::string xclbin_path = argv[i];
    std::string instr_path = argv[i + 1];
    struct kernel &kernel = kernels[xclbin_path];
    xrt::kernel &xrt_kernel = kernel.xrt_kernel;
    xrt::bo &bo_instr = bo_instrs[(i - 1) / 2];

    std::vector<uint32_t> &instr_v = instr_vectors[instr_path];

    if (&kernel != last_kernel) {
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

#endif


  // Copy out outputs from last kernel run.
  std::vector<DTYPE> vec_out(SIZE);

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
