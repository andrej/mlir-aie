"""Quick HW timing for a built per-block ELF.

Usage:
  python3 scripts/time_block.py --block m8 -e build/aie_m8.elf
"""

import argparse
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import aie.iron as iron
from aie.utils import DefaultNPURuntime, NPUKernel
from aie.utils.trace import TraceConfig

import yolo_spec  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--block", required=True)
    p.add_argument(
        "-e", "--elf", required=True,
        help="path to the per-block full-ELF (e.g. build/aie_m8.elf)",
    )
    p.add_argument(
        "-k", "--kernel", default="yolo26n:sequence",
        help="kernel name = '<device>:<sequence>' baked into the ELF (default: "
        "yolo26n:sequence; matches DEVICE_NAME in aie2_yolo_per_block.py)",
    )
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-iters", type=int, default=20)
    # Trace pass-through (used by scripts/trace_m8.sh). When --trace-sz > 0
    # the host reads back a trace BO of that size per dispatch; --ddr-id
    # selects the dedicated BO slot (-1 = append after last tensor).
    p.add_argument("--trace-sz", dest="trace_size", type=int, default=0)
    p.add_argument("--trace-file", default="trace.txt")
    p.add_argument("--ddr-id", dest="ddr_id", type=int, default=4)
    opts = p.parse_args()

    blk = yolo_spec.block(opts.block)
    in_w, in_h, in_c = blk.layers[0].in_shape
    last_out_shape = blk.layers[-1].out_shape

    in_bytes = in_w * in_h * (8 if opts.block == "m0" else in_c)
    out_bytes = int(np.prod(last_out_shape))

    rng = np.random.default_rng(seed=0)
    in_data = rng.integers(-128, 128, size=(in_bytes,), dtype=np.int8)

    in_tensor = iron.tensor(in_data, dtype=np.int8)
    out_tensor = iron.zeros([out_bytes], dtype=np.int8)

    trace_config = None
    if opts.trace_size > 0:
        trace_config = TraceConfig(
            trace_size=opts.trace_size,
            trace_file=opts.trace_file,
            ddr_id=opts.ddr_id,
        )
    npu_kernel = NPUKernel(
        elf_path=opts.elf, kernel_name=opts.kernel, trace_config=trace_config
    )
    rt = DefaultNPURuntime

    print(f"{opts.block}: warmup x{opts.n_warmup}, time x{opts.n_iters}")
    for _ in range(opts.n_warmup):
        rt.load_and_run(npu_kernel, [in_tensor, out_tensor])

    times_ms = []
    for _ in range(opts.n_iters):
        _h, result = rt.load_and_run(npu_kernel, [in_tensor, out_tensor])
        times_ms.append(result.npu_time / 1e6)

    arr = np.array(times_ms)
    print(
        f"{opts.block}: n={opts.n_iters} mean={arr.mean():.2f} ms "
        f"min={arr.min():.2f} ms median={float(np.median(arr)):.2f} ms "
        f"max={arr.max():.2f} ms std={arr.std():.2f} ms"
    )
    print(
        f"{opts.block}: throughput @ median = {1000.0 / float(np.median(arr)):.2f} fps"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
