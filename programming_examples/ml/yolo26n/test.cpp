//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// End-to-end person/no-person classifier on the NPU, driven through the
// XRT coreutils full-ELF API.
//
// Build (after `source ~/setup_buildenv.sh`): `make test`
//
// Run:
//   ./build/test.exe -e build/final_chain.elf --image /path/to/img.jpg
//
// Class order is alphabetical, matching the calibration notebook:
// index 0 = no_person, index 1 = person.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "cxxopts.hpp"

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace {

constexpr int IMGSZ = 512;
constexpr int IN_C_DECL = 3;
constexpr int IN_C_PAD = 8;   // m0 expects 8-channel input; 3->8 zero-pad
constexpr int OUT_C = 2;      // head emits 2 class probs (no_person, person)
constexpr int OUT_PAD = 4;    // head output padded to 4 bytes for shim alignment
constexpr int PERSON_IDX = 1; // alphabetical: ["no_person", "person"]
static_assert(PERSON_IDX < OUT_C, "PERSON_IDX out of range");

// Kernel name format for full-ELF flow is "<device_sym_name>:<sequence_name>".
// device_sym_name is set in aie2_yolo_iron_partial.py (DEVICE_NAME) and baked
// into the ELF via Program(...).resolve_program(device_name=...).
constexpr const char *KERNEL_NAME = "yolo26n_chain:sequence";

// Input QuantizeLinear scale from phase1_25k_xint8_acc0.8968.onnx (ZP=0).
// Verified via:
//   python3 -c "import onnx, onnx.numpy_helper as nph; \
//     m=onnx.load('models/phase1_25k_xint8_acc0.8968.onnx'); \
//     ql=next(n for n in m.graph.node if n.op_type=='QuantizeLinear' \
//             and n.input[0]==m.graph.input[0].name); \
//     inits={t.name:t for t in m.graph.initializer}; \
//     print(nph.to_array(inits[ql.input[1]]).item())"
// -> 0.0078125  (= 2^-7 = 1/128)
constexpr float INPUT_SCALE = 1.0f / 128.0f;

// Head emits int8 probs at Q = 2^-7 (multiplier 128). Dequant with this.
constexpr float HEAD_SCALE = 1.0f / 128.0f;

// Preprocess: Ultralytics-cls pipeline matching preprocess_for_onnx() in
// notebooks/quark_quantization.ipynb:
//   PIL RGB -> center-crop to square -> bilinear resize to 512 -> /255 -> CHW
// We additionally int8-quantize at INPUT_SCALE and pad channels 3->8 (HWC).
// Output buffer must be IMGSZ*IMGSZ*IN_C_PAD bytes, pre-zeroed.
void preprocess(const std::string &img_path, int8_t *out_hwc_padded) {
  cv::Mat bgr = cv::imread(img_path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("failed to read image: " + img_path);
  }
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  // Center-crop to square.
  int W = rgb.cols, H = rgb.rows;
  int side = std::min(W, H);
  int left = (W - side) / 2;
  int top = (H - side) / 2;
  cv::Mat cropped = rgb(cv::Rect(left, top, side, side));

  // Bilinear resize to IMGSZ.
  cv::Mat resized;
  cv::resize(cropped, resized, cv::Size(IMGSZ, IMGSZ), 0, 0, cv::INTER_LINEAR);

  // /255 -> /scale (= *128) -> round/clip to int8 -> pad C 3->8.
  // Output buffer is HWC with 8 channels; caller has zeroed it so channels
  // 3..7 stay zero.
  const float k = (1.0f / 255.0f) / INPUT_SCALE; // = 128/255
  for (int y = 0; y < IMGSZ; ++y) {
    const uint8_t *src_row = resized.ptr<uint8_t>(y);
    int8_t *dst_row = out_hwc_padded + y * IMGSZ * IN_C_PAD;
    for (int x = 0; x < IMGSZ; ++x) {
      const uint8_t *src_px = src_row + x * IN_C_DECL;
      int8_t *dst_px = dst_row + x * IN_C_PAD;
      for (int c = 0; c < IN_C_DECL; ++c) {
        float q = std::round(static_cast<float>(src_px[c]) * k);
        if (q > 127.0f)
          q = 127.0f;
        if (q < -128.0f)
          q = -128.0f;
        dst_px[c] = static_cast<int8_t>(q);
      }
    }
  }
}

} // namespace

int main(int argc, const char *argv[]) {
  cxxopts::Options options("yolo26n_cls");
  options.add_options()                                                    //
      ("e,elf", "path to full-ELF (.elf) produced by aiecc",               //
       cxxopts::value<std::string>())                                      //
      ("image", "path to input image (jpg/png/...)",                       //
       cxxopts::value<std::string>())                                      //
      ("threshold", "P(person) decision threshold (default 0.5)",          //
       cxxopts::value<float>()->default_value("0.5"))                      //
      ("v,verbosity", "verbosity level (0|1|2)",                           //
       cxxopts::value<int>()->default_value("0"))                          //
      ("h,help", "print help");

  cxxopts::ParseResult vm;
  try {
    vm = options.parse(argc, argv);
  } catch (const cxxopts::exceptions::exception &e) {
    std::cerr << e.what() << "\n" << options.help() << "\n";
    return 2;
  }
  if (vm.count("help")) {
    std::cout << options.help() << "\n";
    return 0;
  }
  if (!vm.count("elf")) {
    std::cerr << "--elf is required\n";
    return 2;
  }
  if (!vm.count("image")) {
    std::cerr << "--image is required\n";
    return 2;
  }
  std::string elf_path = vm["elf"].as<std::string>();
  std::string image_path = vm["image"].as<std::string>();
  float threshold = vm["threshold"].as<float>();
  int verbosity = vm["verbosity"].as<int>();

  // Bring up XRT in full-ELF mode: open device, load ELF, build hw_context
  // with the ELF attached so XRT can patch load_pdi at dispatch.
  auto device = xrt::device(0);
  xrt::elf ctx_elf{elf_path};
  xrt::hw_context context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, KERNEL_NAME);
  if (verbosity >= 1)
    std::cout << "Loaded ELF " << elf_path << " kernel=" << KERNEL_NAME
              << "\n";

  // Buffer objects: input (HWC padded to 8 channels) + output (4 B).
  constexpr size_t IN_BYTES = static_cast<size_t>(IMGSZ) * IMGSZ * IN_C_PAD;
  constexpr size_t OUT_BYTES = OUT_PAD;
  xrt::bo bo_in = xrt::ext::bo{device, IN_BYTES};
  xrt::bo bo_out = xrt::ext::bo{device, OUT_BYTES};

  // Stage the input into a pre-zeroed buffer (zeros the C=3..7 padding).
  int8_t *buf_in = bo_in.map<int8_t *>();
  std::memset(buf_in, 0, IN_BYTES);
  try {
    preprocess(image_path, buf_in);
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 2;
  }
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Dispatch. The IRON runtime sequence signature is (in, out); load_pdi
  // is emitted as an inline op at the start of the sequence and does not
  // consume an arg slot.
  auto run = xrt::run(kernel);
  run.set_arg(0, bo_in);
  run.set_arg(1, bo_out);

  auto t0 = std::chrono::steady_clock::now();
  run.start();
  run.wait2();
  auto t1 = std::chrono::steady_clock::now();

  // Pull output and decode.
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int8_t *buf_out = bo_out.map<int8_t *>();
  float p_no_person = static_cast<float>(buf_out[0]) * HEAD_SCALE;
  float p_person = static_cast<float>(buf_out[PERSON_IDX]) * HEAD_SCALE;
  const char *label = (p_person >= threshold) ? "person" : "no_person";

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "image:        " << image_path << "\n"
            << "P(no_person): " << p_no_person << "\n"
            << "P(person):    " << p_person << "\n"
            << "prediction:   " << label << "  (threshold=" << threshold
            << ")\n"
            << "wall time:    " << ms << " ms\n";
  return 0;
}
