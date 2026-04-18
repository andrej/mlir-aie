// (c) Copyright 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Host code for the LOAD_PDI reconfiguration xclbin flow test.
// The MLIR uses `aiex.configure` and `aiex.run` ops (lowered to LOAD_PDI by
// the materialize-runtime-sequences pass).  This test loads two different PDIs
// (add_two and add_three) and verifies that each configuration produces the
// correct output on its buffer region.

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int DATA_SIZE = 512;

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("reconfigure_loadpdi_xclbin");
  test_utils::add_default_options(options);
  options.add_options()("offsets", "Path to JSON offsets file",
                        cxxopts::value<std::string>())(
      "pdi1", "Path to first PDI binary (add_two)",
      cxxopts::value<std::string>())(
      "pdi2", "Path to second PDI binary (add_three)",
      cxxopts::value<std::string>());

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  int verbosity = vm["verbosity"].as<int>();

  // Load the NPU instruction binary
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Instruction count before patching: " << instr_v.size()
              << "\n";

  // Parse JSON offsets
  std::string offsets_path = vm["offsets"].as<std::string>();
  auto patch_infos = test_utils::parse_instr_offsets_json(offsets_path);
  if (verbosity >= 1)
    std::cout << "Found " << patch_infos.size()
              << " LOAD_PDI instruction(s) to patch\n";

  // Map pdi_id -> PDI file path.
  // PDI IDs are auto-assigned in device iteration order (1-based):
  //   @add_two = 1, @add_three = 2, @main = 3
  std::string pdi1_path = vm["pdi1"].as<std::string>();
  std::string pdi2_path = vm["pdi2"].as<std::string>();
  std::map<int, std::string> pdi_map = {{1, pdi1_path}, {2, pdi2_path}};

  // Append each PDI and patch its LOAD_PDI instruction separately
  for (const auto &info : patch_infos) {
    auto it = pdi_map.find(info.pdi_id);
    if (it == pdi_map.end()) {
      std::cerr << "Unknown pdi_id=" << info.pdi_id << "\n";
      return 1;
    }
    test_utils::append_and_patch_pdi(instr_v, info, it->second, verbosity);
  }

  if (verbosity >= 1)
    std::cout << "Instruction count after patching: " << instr_v.size()
              << "\n";

  // Start the XRT test code
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
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

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";
  device.register_xclbin(xclbin);

  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  // Create buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_data = xrt::bo(device, DATA_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  // Fix up LOAD_PDI address fields with absolute device addresses
  uint64_t instr_bo_addr = bo_instr.address();
  if (verbosity >= 1)
    std::cout << "Instruction BO device address: 0x" << std::hex
              << instr_bo_addr << std::dec << "\n";

  test_utils::fixup_load_pdi_addresses(instr_v, patch_infos, instr_bo_addr,
                                       verbosity);

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Initialize input data: 1, 2, 3, ..., 512
  uint32_t *bufData = bo_data.map<uint32_t *>();
  for (int i = 0; i < DATA_SIZE; i++)
    bufData[i] = i + 1;

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_data.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;
  auto run =
      kernel(opcode, bo_instr, instr_v.size(), bo_data, bo_data, bo_data);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_data.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errors = 0;

  // Verify: elements [0..3] should be input + 2 (add_two configuration)
  for (uint32_t i = 0; i < 4; i++) {
    uint32_t ref = (i + 1) + 2;
    if (bufData[i] != ref) {
      std::cout << "Error at [" << i << "]: " << bufData[i] << " != " << ref
                << " (expected add_two: input+2)" << std::endl;
      errors++;
    }
  }

  // Verify: elements [4..7] should be input + 3 (add_three configuration)
  for (uint32_t i = 4; i < 8; i++) {
    uint32_t ref = (i + 1) + 3;
    if (bufData[i] != ref) {
      std::cout << "Error at [" << i << "]: " << bufData[i] << " != " << ref
                << " (expected add_three: input+3)" << std::endl;
      errors++;
    }
  }

  // Verify: elements [8..511] should be unchanged
  for (uint32_t i = 8; i < DATA_SIZE; i++) {
    uint32_t ref = i + 1;
    if (bufData[i] != ref) {
      std::cout << "Error at [" << i << "]: " << bufData[i] << " != " << ref
                << " (expected unchanged)" << std::endl;
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
