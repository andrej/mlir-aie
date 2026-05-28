//===- test_utils.cpp ----------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This file contains common helper functions for the generic host code

#include "test_utils.h"
#include "memory_allocator.h"
#include <cassert>
#include <cinttypes>
#include <cstring>
#include <filesystem>

#ifdef TEST_UTILS_USE_XRT
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#endif

// --------------------------------------------------------------------------
// Command Line Argument Handling
// --------------------------------------------------------------------------

void test_utils::print_test_banner(const std::string &title,
                                   const std::string &description) {
  constexpr std::size_t ruleWidth = 72;
  const std::string rule(ruleWidth, '=');

  std::cout << '\n' << rule << '\n';
  std::cout << title << '\n';
  if (!description.empty()) {
    std::cout << description << '\n';
  }
  std::cout << rule << '\n';
}

void test_utils::check_arg_file_exists(const cxxopts::ParseResult &result,
                                       std::string name) {
  if (!result.count(name)) {
    throw std::runtime_error("Missing required argument: " + name);
  }
  std::string path = result[name].as<std::string>();
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("File does not exist: " + path);
  }
}

void test_utils::add_default_options(cxxopts::Options &options) {
  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i",
      "path of file containing userspace instructions sent to the NPU",
      cxxopts::value<std::string>())(
      "verify", "whether to verify the AIE computed output",
      cxxopts::value<bool>()->default_value("true"))(
      "iters", "number of iterations",
      cxxopts::value<int>()->default_value("1"))(
      "warmup", "number of warmup iterations",
      cxxopts::value<int>()->default_value("0"))(
      "trace_sz,t", "trace size", cxxopts::value<int>()->default_value("0"))(
      "trace_file", "where to store trace output",
      cxxopts::value<std::string>()->default_value("trace.txt"));
}

void test_utils::parse_options(int argc, const char *argv[],
                               cxxopts::Options &options,
                               cxxopts::ParseResult &vm) {
  try {
    vm = options.parse(argc, argv);

    if (vm.count("help")) {
      std::cout << options.help() << "\n";
      std::exit(1);
    }
  } catch (const cxxopts::exceptions::parsing &e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Usage:\n" << options.help() << "\n";
    std::exit(1);
  }

  try {
    check_arg_file_exists(vm, "xclbin");
    check_arg_file_exists(vm, "instr");
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n\n";
  }
}

// --------------------------------------------------------------------------
// AIE Specifics
// --------------------------------------------------------------------------

std::vector<uint32_t> test_utils::load_instr_sequence(std::string instr_path) {
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

std::vector<uint32_t> test_utils::load_instr_binary(std::string instr_path) {
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

#ifdef TEST_UTILS_USE_XRT

// --------------------------------------------------------------------------
// XRT
// --------------------------------------------------------------------------
void test_utils::init_xrt_load_kernel(xrt::device &device, xrt::kernel &kernel,
                                      int verbosity, std::string xclbinFileName,
                                      std::string kernelNameInXclbin) {
  // Get a device handle
  unsigned int device_index = 0;
  device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << xclbinFileName << "\n";
  auto xclbin = xrt::xclbin(xclbinFileName);

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << kernelNameInXclbin << "\n";

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel =
      *std::find_if(xkernels.begin(), xkernels.end(),
                    [kernelNameInXclbin, verbosity](xrt::xclbin::kernel &k) {
                      auto name = k.get_name();
                      if (verbosity >= 1) {
                        std::cout << "Name: " << name << std::endl;
                      }
                      return name.rfind(kernelNameInXclbin, 0) == 0;
                    });
  auto kernelName = xkernel.get_name();
  // Register xclbin
  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << xclbinFileName << "\n";

  device.register_xclbin(xclbin);

  // Get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // Get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  kernel = xrt::kernel(context, kernelName);

  return;
}

#endif // TEST_UTILS_USE_XRT

// --------------------------------------------------------------------------
// Matrix / Float / Math
// --------------------------------------------------------------------------

// nearly_equal function adapted from Stack Overflow, License CC BY-SA 4.0
// Original author: P-Gn
// Source: https://stackoverflow.com/a/32334103
bool test_utils::nearly_equal(float a, float b, float epsilon, float abs_th)
// those defaults are arbitrary and could be removed
{
  assert(std::numeric_limits<float>::epsilon() <= epsilon);
  assert(epsilon < 1.f);

  if (a == b)
    return true;

  auto diff = std::abs(a - b);
  auto norm =
      std::min((std::abs(a) + std::abs(b)), std::numeric_limits<float>::max());
  // or even faster: std::min(std::abs(a + b),
  // std::numeric_limits<float>::max()); keeping this commented out until I
  // update figures below
  return diff < std::max(abs_th, epsilon * norm);
}

// --------------------------------------------------------------------------
// Tracing
// --------------------------------------------------------------------------
void test_utils::write_out_trace(char *traceOutPtr, size_t trace_size,
                                 std::string path) {
  std::ofstream fout(path);
  uint32_t *traceOut = (uint32_t *)traceOutPtr;
  for (int i = 0; i < trace_size / sizeof(traceOut[0]); i++) {
    fout << std::setfill('0') << std::setw(8) << std::hex << (int)traceOut[i];
    fout << std::endl;
  }
}

// --------------------------------------------------------------------------
// Transaction Binary for Simulation
// --------------------------------------------------------------------------
#include <cstdio>

// Unpack transaction binary and write it to the aie array simulation
void test_utils::write_transaction_binary(unsigned char *data,
                                          uint64_t nbytes) {
  assert(nbytes >= 12);
  uint32_t major = data[0];
  uint32_t minor = data[1];
  uint32_t num_cols = data[4];

  uint32_t num_ops, txn_size;
  std::memcpy(&num_ops, &data[8], 4);
  std::memcpy(&txn_size, &data[12], 4);

  uint32_t i = 16;

  uint64_t addr = 0;
  uint32_t value = 0;
  uint32_t size = 0;
  uint32_t mask = 0;
  const uint32_t *data_ptr = nullptr;

  if (major == 0 && minor == 1) {
    while (i < nbytes) {
      uint8_t opc = data[i];
      // printf("  at offset %u: opc=0x%02x\n", i, opc);
      if (opc == 0x00) {
        uint32_t addr0, addr1;
        std::memcpy(&addr0, &data[i + 8], 4);
        std::memcpy(&addr1, &data[i + 12], 4);
        std::memcpy(&value, &data[i + 16], 4);
        std::memcpy(&size, &data[i + 20], 4);
        addr = (static_cast<uint64_t>(addr1) << 32) | addr0;
        // printf("    write32 addr=0x%016llx value=0x%08x size=%u\n",
        //        (unsigned long long)addr, (unsigned int)value, size);
        i += size;
        mlir_aie_sim_write32(addr, value);
      } else if (opc == 0x01) {
        std::memcpy(&addr, &data[i + 8], 4);
        std::memcpy(&size, &data[i + 12], 4);
        data_ptr = reinterpret_cast<const uint32_t *>(data + i + 16);
        // printf("    burst write addr=0x%016llx size=%u (payload words=%u)\n",
        //        (unsigned long long)addr, size, (size - 16) / 4);
        i += size;
        size = size - 16;
        for (int j = 0; j < size / 4; j++) {
          // printf("      write32 addr=0x%016llx value=0x%08x\n",
          //        (unsigned long long)(addr + (4 * j)),
          //        (unsigned int)data_ptr[j]);
          mlir_aie_sim_write32(addr + (4 * j), data_ptr[j]);
        }
      } else if (opc == 0x03) {
        uint32_t addr0, addr1;
        std::memcpy(&addr0, &data[i + 8], 4);
        std::memcpy(&addr1, &data[i + 12], 4);
        std::memcpy(&value, &data[i + 16], 4);
        std::memcpy(&mask, &data[i + 20], 4);
        std::memcpy(&size, &data[i + 24], 4);
        addr = (static_cast<uint64_t>(addr1) << 32) | addr0;
        uint32_t r = mlir_aie_sim_read32(addr);
        uint32_t w = (r & ~mask) | (value & mask);
        // printf("    read-modify-write addr=0x%016llx r=0x%08x value=0x%08x "
        //        "mask=0x%08x -> w=0x%08x size=%u\n",
        //        (unsigned long long)addr, (unsigned int)r, (unsigned
        //        int)value, (unsigned int)mask, (unsigned int)w, size);
        mlir_aie_sim_write32(addr, w);
        i += size;
      } else {
        printf("    ERROR: unhandled transaction binary opcode 0x%02x at "
               "offset %u\n",
               opc, i);
        assert(0 && "unhandled transaction binary opcode");
      }
    }
  } else if (major == 1 && minor == 0) {
    while (i < nbytes) {
      uint8_t opc = data[i];
      // printf("  at offset %u: opc=0x%02x\n", i, opc);
      if (opc == 0x00) {
        std::memcpy(&addr, &data[i + 4], 4);
        std::memcpy(&value, &data[i + 8], 4);
        // printf("    write32 addr=0x%016llx value=0x%08x (fixed hdr)\n",
        //        (unsigned long long)addr, (unsigned int)value);
        i += 12;
        mlir_aie_sim_write32(addr, value);
      } else if (opc == 0x01) {
        std::memcpy(&addr, &data[i + 4], 4);
        std::memcpy(&size, &data[i + 8], 4);
        data_ptr = reinterpret_cast<const uint32_t *>(data + i + 12);
        // printf("    burst write addr=0x%016llx size=%u (payload words=%u) "
        //        "(fixed hdr)\n",
        //        (unsigned long long)addr, size, (size - 12) / 4);
        i += size;
        size = size - 12;
        for (int j = 0; j < size / 4; j++) {
          // printf("      write32 addr=0x%016llx value=0x%08x\n",
          //        (unsigned long long)(addr + (4 * j)),
          //        (unsigned int)data_ptr[j]);
          mlir_aie_sim_write32(addr + (4 * j), data_ptr[j]);
        }
      } else if (opc == 0x03) {
        std::memcpy(&addr, &data[i + 4], 4);
        std::memcpy(&value, &data[i + 8], 4);
        std::memcpy(&mask, &data[i + 12], 4);
        uint32_t r = mlir_aie_sim_read32(addr);
        uint32_t w = (r & ~mask) | (value & mask);
        // printf("    read-modify-write addr=0x%016llx r=0x%08x value=0x%08x "
        //        "mask=0x%08x -> w=0x%08x (fixed hdr)\n",
        //        (unsigned long long)addr, (unsigned int)r, (unsigned
        //        int)value, (unsigned int)mask, (unsigned int)w);
        mlir_aie_sim_write32(addr, w);
        i += 16;
      } else {
        printf("    ERROR: unhandled transaction binary opcode 0x%02x at "
               "offset %u\n",
               opc, i);
        assert(0 && "unhandled transaction binary opcode");
      }
    }
  } else {
    printf("ERROR: unhandled transaction binary version %u.%u\n", major, minor);
    assert(0 && "unhandled transaction binary version");
  }
}
