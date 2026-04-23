// (c) Copyright 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test for the xclbin-flow instruction buffer size limit.
//
// The XRT firmware imposes a ~4 MiB limit on instruction buffer objects (bos).
// This test loads a small, valid instruction binary and inflates it by
// inserting NOOP words (TXN opcode 0x05 = XAIE_IO_NOOP) into the TXN stream,
// then updates the TXN header fields (NumOps, TxnSize) to match.  The inflated
// buffer is allocated as a BO and submitted to the device.
//
// With --pad-to-bytes just below 4 MiB the test should pass.
// With --pad-to-bytes just above 4 MiB the test should fail (timeout / error).

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int IN_SIZE = 64;
constexpr int OUT_SIZE = 64;

// TXN v0.1 NOOP opcode (XAIE_IO_NOOP = 5).
// A NOOP is emitted as a single 32-bit word.  The firmware skips it.
static constexpr uint32_t TXN_NOOP_WORD = 0x05;

// Insert NOOP words so total byte size reaches `target_bytes`.
// Updates the TXN header: word[2] = NumOps, word[3] = TxnSize.
static void pad_with_noops(std::vector<uint32_t> &instr, size_t target_bytes) {
  size_t current_bytes = instr.size() * sizeof(uint32_t);
  if (current_bytes >= target_bytes)
    return;

  size_t pad_words = (target_bytes - current_bytes) / sizeof(uint32_t);

  for (size_t i = 0; i < pad_words; ++i)
    instr.push_back(TXN_NOOP_WORD);

  // Update TXN header: add inserted NOOPs to op count, fix total size.
  instr[2] += static_cast<uint32_t>(pad_words);
  instr[3] = static_cast<uint32_t>(instr.size() * sizeof(uint32_t));
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("instr_buffer_size_limit_xclbin");
  test_utils::add_default_options(options);
  options.add_options()("pad-to-bytes", "Pad instruction buffer to this size",
                        cxxopts::value<size_t>()->default_value("0"));

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  int verbosity = vm["verbosity"].as<int>();
  size_t pad_to = vm["pad-to-bytes"].as<size_t>();

  // Load the compiled instruction binary (TXN format).
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Original instruction count: " << instr_v.size() << " ("
              << instr_v.size() * 4 << " bytes)\n";

  if (pad_to > 0) {
    pad_with_noops(instr_v, pad_to);
    if (verbosity >= 1)
      std::cout << "Padded instruction count:   " << instr_v.size() << " ("
                << instr_v.size() * 4 << " bytes)\n";
  }

  // --- XRT setup (xclbin flow) ---
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  std::string Node = vm["kernel"].as<std::string>();
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 return k.get_name().rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_inB = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  // Initialize input: 1, 2, 3, ..., 64.
  uint32_t *bufInA = bo_inA.map<uint32_t *>();
  for (int i = 0; i < IN_SIZE; i++)
    bufInA[i] = i + 1;

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running kernel (instr BO = " << instr_v.size() * 4
              << " bytes).\n";

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  uint32_t *bufOut = bo_out.map<uint32_t *>();

  // Validate: kernel adds 2 to each input element.
  int errors = 0;
  for (uint32_t i = 0; i < OUT_SIZE; i++) {
    uint32_t ref = (i + 1) + 2;
    if (bufOut[i] != ref) {
      std::cout << "Error at " << i << ": " << bufOut[i] << " != " << ref
                << std::endl;
      errors++;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\n" << errors << " errors. FAIL.\n\n";
    return 1;
  }
}
