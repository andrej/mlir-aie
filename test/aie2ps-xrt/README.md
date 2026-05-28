# AIE2PS XRT Hardware Tests

End-to-end tests for AIE2PS (xcve3858) on VEK385, using the full-ELF CERT flow
with XRT runtime.

## Board

- **Device:** xc2ve3858 (36 columns, 7 rows: 1 shim + 2 memtile + 4 core)
- **Board:** VEK385 running AMD EDF Linux
- **Driver:** amdxdna
- **Runtime:** XRT with `xrt::elf` / `xrt::hw_context` API

## Tested Versions

| Component | Version |
|-----------|---------|
| EDF Linux | 25.11+development (AMD Embedded Development Framework) |
| XRT | 2.22.0 |
| amdxdna driver | 2.21.0_20260330 |
| EDF SDK (cross-compile) | 25.11+development-027ae3f30fdccfa8b8043955821a11c6286cd66e |
| Vitis (Chess compiler) | 2025.1 |
| Peano (llvm-aie) | nightly 20.0.0.2026012801+201faf7c (LLVM 21) |
| Boot images source | Vitis-AI-Telluride vai_6.1 |

### Boot images

The board was set up using EDF boot images from Vitis-AI-Telluride. The key
artifacts are:

| Artifact | Description |
|----------|-------------|
| `BOOT.bin` | Versal bootloader (PMC + PLM) |
| `edf-ospi-versal2-vek385-sdt-full.bin` | EDF Linux OSPI image |
| `Image` + `minimal.rootfs.cpio.gz.u-boot` | Kernel + minimal rootfs for initial boot |
| `rootfs.wic.xz` | Full EDF rootfs with XRT + amdxdna driver |
| `overlay/vpl_gen_fixed_pld.pdi` | FPGA bitstream for AIE2PS partition |
| `overlay/pl_aiarm.dtbo` | Device tree overlay |
| `sdk.sh` | EDF SDK installer (aarch64 cross-compiler + sysroot) |

### Cross-compilation

The EDF SDK (`sdk.sh`) provides a matched `aarch64-amd-linux-g++` cross-compiler
and sysroot with XRT headers (`xrt_device.h`, `xrt_elf.h`, etc.) and libraries
(`libxrt_coreutil.so`). Use `setup_edf_sdk.sh` to configure the environment:

```bash
source test/aie2ps-xrt/setup_edf_sdk.sh [/path/to/sdk]
```

## Compiler Selection

Tests support both **Chess** (xchesscc from Vitis aietools) and **Peano** (llvm-aie)
compilers:

- **Chess:** `xchesscc_wrapper aie2ps`, kernels may use `aie_api/aie.hpp` (Vitis header library).
- **Peano:** `$PEANO_INSTALL_DIR/bin/clang++ --target=aie2ps-none-unknown-elf`.

Each test has a `Makefile` with `CHESS ?= false` (Peano default). All build artifacts go
into `build/`.

**Note:** Peano kernels (`*_peano.cc`) are rewritten in each test folder because `third_party/aie_api` does not have `aie2ps` support yet.
Existing kernels in shared `aie_kernels/` causes issue due to `aie_api` include line.


## Tests

| Test | Compiler | Data types | Vectorized | Kernel type | What it tests |
|------|----------|-----------|------------|-------------|---------------|
| `passthrough_dmas` | Peano | int32 | N/A | None (DMA only) | CERT ELF generation, MemTile DMA passthrough via `objectfifo.link` |
| `passthrough_kernel` | Chess, Peano | uint8 | No | External plain C (`passthrough.cc`) | Core memcpy, lock acquire/release |
| `vector_scalar_mul` | Chess, Peano | int32, int16 | Peano: `VECTORIZED=1` | Chess: `aie_api` (`scale.cc`), Peano: native builtins (`scale_peano.cc`) | Tests i32, i16, vectorized Peano kernel|
| `vec_vec_mul` | Chess, Peano | int32 | No (inline MLIR scalar) | Inline MLIR (`arith.muli`) | 3x3 core tile design |
| `eltwise_add_iron` | Chess, Peano | bf16 | No (scalar) | Chess: `aie_api` (`add.cc`), Peano: native builtins (`add_peano.cc`) | Uses IRON API |

## Building

### Prerequisites

1. **Environment**:
   ```bash
   cd /path/to/mlir-aie
   source ironenv/bin/activate
   source utils/env_setup.sh install
   ```

2. **aarch64 cross-compilation** (EDF SDK):
   ```bash
   source test/aie2ps-xrt/setup_edf_sdk.sh [/path/to/sdk]
   ```

3. **Vitis** (only needed for Chess builds):
   ```bash
   source /path/to/vitis-setup.sh
   ```

### Build a test

```bash
# Build with Peano (default):
cd test/aie2ps-xrt/passthrough_kernel
make

# Build with Chess:
make CHESS=true

# Build only the ELF or host separately:
make elf
make host

# Clean:
make clean
```

### Test-specific options

| Test | Extra options |
|------|--------------|
| `vector_scalar_mul` | `DTYPE=i16`, `VECTORIZED=1` |
| `matmul_whole_array` | `M=`, `K=`, `N=`, `m=`, `k=`, `n=`, `DTYPE=bf16` |
| `event_trace` | `make parse_trace` (post-run trace analysis) |

### Artifacts

After `make`, artifacts are at:
```
test/aie2ps-xrt/<test>/build/
  aie_control_full.elf   # CERT control ELF
  host.exe               # aarch64 XRT host program
```

Copy both files to the VEK385 board and run:
```bash
./host.exe
# Expected: TEST PASSED
```

## Board setup

### One-time: FPGA overlay and driver

After booting EDF Linux on the VEK385, load the FPGA overlay and driver:

```bash
# Load FPGA bitstream and device tree overlay
sudo fpgautil -b /path/to/vpl_gen_fixed_pld.pdi -o /path/to/pl_aiarm.dtbo

# Verify device is visible
xrt-smi examine
```

### CERT firmware and driver reload

The CERT firmware (`cert_ve2.elf`) must be installed at `/lib/firmware/app.elf`.
This is typically already present from the EDF image. If updating firmware, 
reload the driver:

```bash
sudo mkdir -p /lib/firmware
sudo cp /path/to/cert_ve2.elf /lib/firmware/app.elf
sudo modprobe -r amdxdna
sudo modprobe amdxdna
```

### Running a test

```bash
cd /path/to/test_files
./host.exe
# Expected: TEST PASSED
```

If the test fails, check:
```bash
dmesg | tail -20          # kernel/driver errors
xrt-smi examine           # device status
```

**NOTE:** Core tiles on xcve3858 start at **row 3** (rows 0=shim, 1-2=memtile, 3-6=core).
