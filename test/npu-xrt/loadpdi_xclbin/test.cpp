// (c) Copyright 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Host code for the LOAD_PDI xclbin flow test.
// This test loads the NPU instruction binary, parses the JSON offsets file
// to find LOAD_PDI instructions, patches them with the PDI binary data,
// and runs the kernel via the xclbin flow.

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
  // Program arguments parsing
  cxxopts::Options options("loadpdi_xclbin");
  test_utils::add_default_options(options);
  options.add_options()("offsets", "Path to JSON offsets file",
                        cxxopts::value<std::string>())(
      "pdi", "Path to PDI binary file", cxxopts::value<std::string>());

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  int verbosity = vm["verbosity"].as<int>();

  // Load the NPU instruction binary
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Instruction count before patching: " << instr_v.size()
              << "\n";

  // Parse JSON offsets and patch LOAD_PDI instructions
  std::string offsets_path = vm["offsets"].as<std::string>();
  std::string pdi_path = vm["pdi"].as<std::string>();

  auto patch_infos = test_utils::parse_instr_offsets_json(offsets_path);
  if (verbosity >= 1)
    std::cout << "Found " << patch_infos.size()
              << " LOAD_PDI instruction(s) to patch\n";

  // Append PDI to instruction buffer and patch address/size fields
  test_utils::patch_load_pdi(instr_v, patch_infos, pdi_path);

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
  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << Node << "\n";

  // Get the kernel from the xclbin
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

  // Get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // Get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  // Create buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  // The runtime_sequence has a single memref<512xi32> argument (in-place).
  auto bo_data = xrt::bo(device, DATA_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  // Fix up the LOAD_PDI address fields with absolute device addresses.
  // patch_load_pdi stored relative byte offsets; the firmware needs absolute
  // device addresses.
  uint64_t instr_bo_addr = bo_instr.address();
  if (verbosity >= 1)
    std::cout << "Instruction BO device address: 0x" << std::hex
              << instr_bo_addr << std::dec << "\n";

  for (const auto &info : patch_infos) {
    uint32_t rel_offset = instr_v[info.address_field_offset_bytes / 4];
    uint64_t abs_addr = instr_bo_addr + rel_offset;
    instr_v[info.address_field_offset_bytes / 4] =
        static_cast<uint32_t>(abs_addr & 0xFFFFFFFF);
    instr_v[info.address_field_offset_bytes / 4 + 1] =
        static_cast<uint32_t>(abs_addr >> 32);
    if (verbosity >= 1)
      std::cout << "Patched LOAD_PDI address: 0x" << std::hex << abs_addr
                << std::dec << "\n";
  }

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
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_data, bo_data, bo_data);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_data.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errors = 0;

  for (uint32_t i = 0; i < DATA_SIZE; i++) {
    uint32_t ref = (i + 1) + 2; // input is i+1, kernel adds 2
    if (bufData[i] != ref) {
      std::cout << "Error in output " << i << ": " << bufData[i] << " != "
                << ref << std::endl;
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
