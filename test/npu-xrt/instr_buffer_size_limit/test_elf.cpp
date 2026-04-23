// (c) Copyright 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test for the ELF-flow instruction buffer size limit (or lack thereof).
//
// The ELF flow packages the instruction stream inside an ELF that XRT loads
// directly, bypassing the instruction BO path used by the xclbin flow.
// This test optionally inflates the ELF by appending zero-padding after the
// valid ELF sections, then loads it via the xrt::elf API.
//
// Because the ELF flow does not allocate a separate instruction BO, the
// ~4 MiB firmware limit that affects the xclbin flow should NOT apply here.
// Both the "below" and "above" 4 MiB configurations are expected to pass.

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#define DTYPE int32_t

constexpr size_t DATA_COUNT = 256;
constexpr size_t BUF_SIZE = DATA_COUNT * sizeof(DTYPE);

static std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("Cannot open file: " + path);
  auto size = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> buf(size);
  f.read(reinterpret_cast<char *>(buf.data()), size);
  return buf;
}

static void write_file(const std::string &path,
                       const std::vector<uint8_t> &data) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f)
    throw std::runtime_error("Cannot write file: " + path);
  f.write(reinterpret_cast<const char *>(data.data()), data.size());
}

int main(int argc, const char *argv[]) {
  // Minimal argument parsing (no cxxopts dependency beyond what's needed).
  std::string elf_path;
  size_t pad_to = 0;
  int verbosity = 0;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--elf" && i + 1 < argc)
      elf_path = argv[++i];
    else if (arg == "--pad-to-bytes" && i + 1 < argc)
      pad_to = std::stoull(argv[++i]);
    else if (arg == "-v" && i + 1 < argc)
      verbosity = std::stoi(argv[++i]);
  }

  if (elf_path.empty()) {
    std::cerr << "Usage: " << argv[0]
              << " --elf <path> [--pad-to-bytes N] [-v level]\n";
    return 1;
  }

  // Optionally inflate the ELF by appending zero-padding.
  // ELF loaders ignore trailing data beyond the sections defined in the
  // ELF header, so this should not affect functional behavior.
  if (pad_to > 0) {
    auto elf_data = read_file(elf_path);
    if (verbosity >= 1)
      std::cout << "Original ELF size: " << elf_data.size() << " bytes\n";

    if (elf_data.size() < pad_to) {
      elf_data.resize(pad_to, 0x00);
      elf_path = "padded_aie.elf";
      write_file(elf_path, elf_data);
      if (verbosity >= 1)
        std::cout << "Padded ELF size:   " << elf_data.size() << " bytes\n";
    }
  }

  // Set up input data.
  srand(1726250518);
  std::vector<DTYPE> vec_in(DATA_COUNT);
  for (size_t i = 0; i < vec_in.size(); i++)
    vec_in[i] = DTYPE(rand());

  // XRT setup (ELF flow — no xclbin, no instruction BO).
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  if (verbosity >= 1)
    std::cout << "Loading ELF: " << elf_path << "\n";
  xrt::elf ctx_elf{elf_path};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);

  std::string kernelName = "add_two:sequence";
  auto kernel = xrt::ext::kernel(context, kernelName);
  xrt::bo bo_inout = xrt::ext::bo{device, BUF_SIZE};

  char *buf_inout = bo_inout.map<char *>();
  memcpy(buf_inout, vec_in.data(), BUF_SIZE);
  bo_inout.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = xrt::run(kernel);
  run.set_arg(0, bo_inout);

  if (verbosity >= 1)
    std::cout << "Running kernel.\n";
  auto t_start = std::chrono::high_resolution_clock::now();
  run.start();
  run.wait2();
  auto t_stop = std::chrono::high_resolution_clock::now();
  bo_inout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Validate output (kernel adds 2 to each element).
  std::vector<DTYPE> vec_out(DATA_COUNT);
  std::vector<DTYPE> vec_ref(DATA_COUNT);
  memcpy(vec_out.data(), buf_inout, BUF_SIZE);
  for (size_t i = 0; i < DATA_COUNT; i++)
    vec_ref[i] = vec_in[i] + 2;
  bool outputs_correct = (vec_out == vec_ref);

  float time =
      std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start)
          .count();
  std::cout << "Elapsed time: " << std::fixed << std::setprecision(0)
            << std::setw(8) << time << " us\n";

  if (outputs_correct) {
    std::cout << "\nPASS!\n\n";
  } else {
    for (size_t i = 0; i < DATA_COUNT; i++) {
      if (vec_out[i] != vec_ref[i])
        std::cout << "in: " << std::setw(12) << vec_in[i]
                  << ", out: " << std::setw(12) << vec_out[i]
                  << ", ref: " << std::setw(12) << vec_ref[i] << "\n";
    }
    std::cout << "\nFail.\n\n";
  }

  return (outputs_correct ? 0 : 1);
}
