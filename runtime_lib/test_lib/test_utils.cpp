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
#include <cassert>
#include <filesystem>

#ifdef TEST_UTILS_USE_XRT
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#endif

// --------------------------------------------------------------------------
// Command Line Argument Handling
// --------------------------------------------------------------------------

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
// LOAD_PDI Patching
// --------------------------------------------------------------------------

// Minimal JSON parser for the instruction offsets file.
// Extracts load_pdi entries from the "instructions" array.
static int64_t extract_json_int(const std::string &obj,
                                const std::string &key) {
  std::string needle = "\"" + key + "\"";
  auto pos = obj.find(needle);
  if (pos == std::string::npos)
    throw std::runtime_error("JSON key not found: " + key);
  pos = obj.find(':', pos + needle.size());
  if (pos == std::string::npos)
    throw std::runtime_error("JSON malformed after key: " + key);
  ++pos;
  // skip whitespace
  while (pos < obj.size() && (obj[pos] == ' ' || obj[pos] == '\t'))
    ++pos;
  char *end = nullptr;
  int64_t val = std::strtoll(obj.c_str() + pos, &end, 10);
  if (end == obj.c_str() + pos)
    throw std::runtime_error("JSON could not parse int for key: " + key);
  return val;
}

// Extract a JSON string value for the given key.  Returns empty string if the
// key is not present (optional field).
static std::string extract_json_string(const std::string &obj,
                                       const std::string &key) {
  std::string needle = "\"" + key + "\"";
  auto pos = obj.find(needle);
  if (pos == std::string::npos)
    return {};
  pos = obj.find(':', pos + needle.size());
  if (pos == std::string::npos)
    return {};
  ++pos;
  // skip whitespace
  while (pos < obj.size() && (obj[pos] == ' ' || obj[pos] == '\t'))
    ++pos;
  if (pos >= obj.size() || obj[pos] != '"')
    return {};
  ++pos; // skip opening quote
  auto end = obj.find('"', pos);
  if (end == std::string::npos)
    return {};
  return obj.substr(pos, end - pos);
}

/// Read a JSON file and find all objects whose "type" field matches \p
/// type_value.  Returns the substring for each matching `{ ... }` block.
static std::vector<std::string>
find_json_objects_by_type(const std::string &json_path,
                          const std::string &type_value) {
  std::ifstream ifs(json_path);
  if (!ifs.is_open())
    throw std::runtime_error("Cannot open JSON offsets file: " + json_path);

  std::string content((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());

  std::vector<std::string> objects;
  std::string marker = "\"" + type_value + "\"";
  size_t search_pos = 0;
  while (true) {
    auto pos = content.find(marker, search_pos);
    if (pos == std::string::npos)
      break;

    auto obj_start = content.rfind('{', pos);
    if (obj_start == std::string::npos) {
      search_pos = pos + marker.size();
      continue;
    }
    auto obj_end = content.find('}', pos);
    if (obj_end == std::string::npos) {
      search_pos = pos + marker.size();
      continue;
    }

    objects.push_back(content.substr(obj_start, obj_end - obj_start + 1));
    search_pos = obj_end + 1;
  }

  return objects;
}

std::vector<test_utils::LoadPdiPatchInfo>
test_utils::parse_instr_offsets_json(const std::string &json_path) {
  auto objs = find_json_objects_by_type(json_path, "load_pdi");

  std::vector<LoadPdiPatchInfo> results;
  for (const auto &obj : objs) {
    LoadPdiPatchInfo info;
    info.load_pdi_offset_bytes =
        static_cast<size_t>(extract_json_int(obj, "offset_bytes"));
    info.pdi_id = static_cast<int>(extract_json_int(obj, "pdi_id"));
    info.address_field_offset_bytes =
        static_cast<size_t>(extract_json_int(obj, "address_field_offset_bytes"));
    info.size_field_offset_bytes =
        static_cast<size_t>(extract_json_int(obj, "size_field_offset_bytes"));
    results.push_back(info);
  }

  return results;
}

void test_utils::patch_load_pdi(
    std::vector<uint32_t> &instr_v,
    const std::vector<LoadPdiPatchInfo> &patch_infos,
    const std::string &pdi_path) {
  // Read PDI binary file
  std::ifstream pdi_file(pdi_path, std::ios::binary);
  if (!pdi_file.is_open())
    throw std::runtime_error("Cannot open PDI file: " + pdi_path);

  pdi_file.seekg(0, std::ios::end);
  size_t pdi_size = static_cast<size_t>(pdi_file.tellg());
  pdi_file.seekg(0, std::ios::beg);

  std::vector<uint8_t> pdi_data(pdi_size);
  if (!pdi_file.read(reinterpret_cast<char *>(pdi_data.data()), pdi_size))
    throw std::runtime_error("Failed to read PDI file: " + pdi_path);

  // Record where we will append the PDI data (byte offset)
  uint32_t pdi_append_offset =
      static_cast<uint32_t>(instr_v.size() * sizeof(uint32_t));

  // Pad PDI data to 4-byte alignment
  size_t padded_size = (pdi_size + 3) & ~static_cast<size_t>(3);
  pdi_data.resize(padded_size, 0);

  // Append PDI data as uint32_t words
  const uint32_t *pdi_words =
      reinterpret_cast<const uint32_t *>(pdi_data.data());
  instr_v.insert(instr_v.end(), pdi_words, pdi_words + padded_size / 4);

  // Patch each LOAD_PDI instruction
  uint32_t pdi_size_bytes = static_cast<uint32_t>(pdi_size);
  for (const auto &info : patch_infos) {
    // Patch address field (lower 32 bits)
    instr_v[info.address_field_offset_bytes / 4] = pdi_append_offset;
    // Upper 32 bits = 0
    instr_v[info.address_field_offset_bytes / 4 + 1] = 0;
    // Patch size field
    instr_v[info.size_field_offset_bytes / 4] = pdi_size_bytes;
  }
}

// --------------------------------------------------------------------------
// Write32 / RTP Patching
// --------------------------------------------------------------------------

std::vector<test_utils::Write32PatchInfo>
test_utils::parse_write32_offsets_json(const std::string &json_path) {
  auto objs = find_json_objects_by_type(json_path, "write32");

  std::vector<Write32PatchInfo> results;
  for (const auto &obj : objs) {
    Write32PatchInfo info;
    info.name = extract_json_string(obj, "name");
    info.offset_bytes =
        static_cast<size_t>(extract_json_int(obj, "offset_bytes"));
    info.value_field_offset_bytes =
        static_cast<size_t>(extract_json_int(obj, "value_field_offset_bytes"));
    results.push_back(std::move(info));
  }

  return results;
}

void test_utils::patch_rtp(std::vector<uint32_t> &instr_v,
                           const std::vector<Write32PatchInfo> &infos,
                           const std::string &name, uint32_t value) {
  for (const auto &info : infos) {
    if (info.name == name) {
      instr_v[info.value_field_offset_bytes / 4] = value;
      return;
    }
  }
  throw std::runtime_error("RTP entry not found: " + name);
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
