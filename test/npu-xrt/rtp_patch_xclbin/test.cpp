// (c) Copyright 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Host code for the RTP patching end-to-end test.
// Loads the NPU instruction binary, parses the JSON offsets to find
// write32 (RTP) entries, patches the "add_value" RTP to a chosen value,
// runs the kernel, and verifies output[i] == input[i] + patched_value.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int DATA_SIZE = 512;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("rtp_patch_xclbin");
  test_utils::add_default_options(options);
  options.add_options()("offsets", "Path to JSON offsets file",
                        cxxopts::value<std::string>())(
      "rtp-value", "RTP value to patch (default 7)",
      cxxopts::value<uint32_t>()->default_value("7"));

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  int verbosity = vm["verbosity"].as<int>();
  uint32_t rtp_value = vm["rtp-value"].as<uint32_t>();

  // Load the NPU instruction binary
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Instruction count: " << instr_v.size() << "\n";

  // Parse JSON offsets and patch the RTP value by name
  std::string offsets_path = vm["offsets"].as<std::string>();
  auto write32_infos = test_utils::parse_write32_offsets_json(offsets_path);

  if (verbosity >= 1)
    std::cout << "Found " << write32_infos.size()
              << " write32 (RTP) entry/entries to patch\n";

  test_utils::patch_rtp(instr_v, write32_infos, "add_value", rtp_value);

  if (verbosity >= 1)
    std::cout << "Patched RTP 'add_value' to " << rtp_value << "\n";

  // Initialize XRT
  auto device = xrt::device(0);

  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  std::string Node = vm["kernel"].as<std::string>();

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  // Create buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in = xrt::bo(device, DATA_SIZE * sizeof(int32_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_buf = xrt::bo(device, 32 * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));
  auto bo_out = xrt::bo(device, DATA_SIZE * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  // Initialize input data: 1, 2, 3, ..., DATA_SIZE
  uint32_t *bufIn = bo_in.map<uint32_t *>();
  for (int i = 0; i < DATA_SIZE; i++)
    bufIn[i] = i + 1;

  uint32_t *bufOut = bo_out.map<uint32_t *>();
  memset(bufOut, 0, DATA_SIZE * sizeof(int32_t));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running kernel.\n";
  unsigned int opcode = 3;
  auto run =
      kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_buf, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errors = 0;
  for (uint32_t i = 0; i < DATA_SIZE; i++) {
    uint32_t ref = (i + 1) + rtp_value;
    if (bufOut[i] != ref) {
      std::cout << "Error at output[" << i << "]: " << bufOut[i] << " != "
                << ref << "\n";
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
