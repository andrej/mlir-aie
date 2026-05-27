#!/usr/bin/env python3
"""End-to-end person/no-person classifier on the NPU (full-ELF flow).

Loads an image from disk, preprocesses it to match the calibration pipeline
(center-crop -> 512x512 -> /255 -> CHW), int8-quantizes at the model's
input scale, runs the full m0..m10 chain ELF on the NPU, and prints
the predicted class + probabilities.

The ELF must have been built with CHAIN_N_SAMPLES=1 (default), i.e.
plain `make chain`. Class order is alphabetical, matching the calibration
notebook: index 0 = no_person, index 1 = person.

Run:
    source ~/setup_buildenv.sh
    python3 test.py -e build/final_chain.elf --image /path/to/img.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from PIL import Image

import aie.iron as iron
from aie.utils import DefaultNPURuntime, NPUKernel

HERE = Path(__file__).resolve().parent
ONNX_PATH = HERE / "models" / "phase1_25k_xint8_acc0.8968.onnx"

IMGSZ = 512
IN_C_DECL = 3
IN_C_PAD = 8  # m0 expects 8-channel input; 3->8 zero-pad
OUT_C = 2  # head emits 2 class probs (no_person, person)
OUT_PAD = 4  # head output padded to 4 bytes for shim alignment
HEAD_SCALE = 2**-7  # NPU emits int8 probs at Q=2^-7 (multiplier 128)
CLASS_NAMES = ["no_person", "person"]
PERSON_IDX = 1


def get_input_scale(onnx_path: Path) -> float:
    """Pull the input QuantizeLinear scale from the ONNX graph (ZP must be 0)."""
    m = onnx.load(str(onnx_path))
    inits = {t.name: t for t in m.graph.initializer}
    in_name = m.graph.input[0].name
    ql = next(
        n for n in m.graph.node if n.op_type == "QuantizeLinear" and n.input[0] == in_name
    )
    scale = float(onnx.numpy_helper.to_array(inits[ql.input[1]]).item())
    if len(ql.input) >= 3:
        zp = int(onnx.numpy_helper.to_array(inits[ql.input[2]]).item())
        assert zp == 0, f"expected input ZP=0, got {zp}"
    return scale


def preprocess(img_path: Path, in_scale: float) -> np.ndarray:
    """Image -> int8 HWC with channels padded 3->8, flat. Matches the
    Ultralytics-cls preprocessing used for calibration:
    PIL RGB -> center-crop to square -> bilinear resize to 512 -> /255."""
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    side = min(W, H)
    left = (W - side) // 2
    top = (H - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize(
        (IMGSZ, IMGSZ), Image.BILINEAR
    )
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0, 1]
    q = np.clip(np.rint(arr / in_scale), -128, 127).astype(np.int8)
    padded = np.zeros((IMGSZ, IMGSZ, IN_C_PAD), dtype=np.int8)
    padded[:, :, :IN_C_DECL] = q
    return padded.reshape(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="path to an input image (jpg/png/...)")
    p.add_argument(
        "-e", "--elf", default="build/final_chain.elf",
        help="path to the chain full-ELF (default: build/final_chain.elf)",
    )
    p.add_argument(
        "-k", "--kernel", default="yolo26n:sequence",
        help="kernel name = '<device>:<sequence>' baked into the ELF (default: "
        "yolo26n:sequence; matches DEVICE_NAME in aie2_yolo_per_block.py)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="P(person) threshold for the binary decision (default 0.5)",
    )
    opts = p.parse_args(sys.argv[1:])

    img_path = Path(opts.image)
    if not img_path.is_file():
        print(f"image not found: {img_path}", file=sys.stderr)
        return 2

    in_scale = get_input_scale(ONNX_PATH)
    in_flat = preprocess(img_path, in_scale)

    in_tensor = iron.tensor(in_flat, dtype=np.int8)
    out_tensor = iron.zeros([OUT_PAD], dtype=np.int8)

    npu_kernel = NPUKernel(elf_path=opts.elf, kernel_name=opts.kernel)
    rt = DefaultNPURuntime
    handle = rt.load(npu_kernel)
    result = rt.run(handle, [in_tensor, out_tensor])
    if result.ret.name != "ERT_CMD_STATE_COMPLETED":
        print(f"NPU run failed: {result.ret.name}", file=sys.stderr)
        return 1

    out_tensor.to("cpu")
    out_i8 = out_tensor.numpy()[:OUT_C]
    probs = out_i8.astype(np.float32) * HEAD_SCALE  # 2^-7 quantization
    p_person = float(probs[PERSON_IDX])
    label = "person" if p_person >= opts.threshold else "no_person"

    print(f"image:       {img_path}")
    print(f"P(no_person): {float(probs[0]):.4f}")
    print(f"P(person):    {p_person:.4f}")
    print(f"prediction:   {label}  (threshold={opts.threshold})")
    print(f"NPU time:     {result.npu_time / 1e6:.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
