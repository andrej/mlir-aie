# Xclbin Decompiler - Quick Start Guide

A 2-minute guide to using the MLIR-AIE xclbin decompiler.

## What It Does

Converts compiled `.xclbin` binaries → readable MLIR code

## Basic Usage

```bash
# Setup environment
source /opt/xilinx/xrt/setup.sh
source buildenv/bin/activate
export PEANO_INSTALL_DIR="$(pip show llvm-aie 2>/dev/null | grep ^Location: | awk '{print $2}')/llvm-aie"
source mlir-aie/utils/env_setup.sh mlir-aie/install ${PEANO_INSTALL_DIR}

# Decompile (lifted mode - recommended)
aie-translate --xclbin-to-mlir --emit-lifted design.xclbin > output.mlir

# Decompile (raw mode - low-level registers)
aie-translate --xclbin-to-mlir design.xclbin > output_raw.mlir
```

## Example

```bash
# Decompile a test example
cd mlir-aie
aie-translate --xclbin-to-mlir --emit-lifted \
  test/npu-xrt/add_blockwrite/aie.xclbin > decompiled.mlir

# View the output
less decompiled.mlir
```

## Output - What You'll See

**Lifted Mode** (readable, semantic operations):
```mlir
module {
  aie.device(npu1_1col) @xclbin_device {
    %tile_0_2 = aie.tile(0, 2)
    %buffer = aie.buffer(%tile_0_2) : memref<1024xi32>
    %lock = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %mem = aie.mem(%tile_0_2) {
      aie.dma_bd(%buffer : memref<1024xi32>, 0, 1024)
      aie.end
    }
    aie.runtime_sequence @configure() {
      aiex.npu.write32 {address = ... : ui32, value = ... : ui32}
      aie.end
    }
  }
}
```

**Raw Mode** (low-level register writes):
```mlir
module {
  aie.device(npu1_1col) @xclbin_device {
    aie.runtime_sequence @configure() {
      aiex.npu.write32 {address = 2228224 : ui32, value = 42 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 0 : ui32}
      // ... hundreds more register writes ...
      aie.end
    }
  }
}
```

## When to Use Each Mode

| Mode | Use When | Output |
|------|----------|--------|
| **Lifted** (`--emit-lifted`) | Understanding design structure | `aie.tile`, `aie.buffer`, `aie.mem`, `aie.dma_bd` |
| **Raw** (default) | Debugging hardware registers | `aiex.npu.write32`, `aiex.npu.maskwrite32` |

## Common Use Cases

✅ **Debugging**: "Why isn't my DMA working?"
```bash
aie-translate --xclbin-to-mlir --emit-lifted my_design.xclbin | grep -A 10 "aie.dma_bd"
```

✅ **Learning**: "How does MLIR map to hardware?"
```bash
# Compare high-level and low-level
aie-translate --xclbin-to-mlir --emit-lifted design.xclbin > lifted.mlir
aie-translate --xclbin-to-mlir design.xclbin > raw.mlir
diff lifted.mlir raw.mlir
```

✅ **Verification**: "Did the compiler do what I expected?"
```bash
aie-translate --xclbin-to-mlir --emit-lifted design.xclbin | grep "aie.buffer"
```

## What Gets Recovered

| Component | Lifted Mode | Raw Mode |
|-----------|-------------|----------|
| Tiles | ✅ `aie.tile(x,y)` | ✅ Addresses |
| Buffers | ✅ `aie.buffer` | ✅ BD addresses |
| DMA BDs | ✅ `aie.dma_bd` | ✅ Register writes |
| Locks | ✅ `aie.lock` | ✅ Register writes |
| Memory Ops | ✅ `aie.mem` | ✅ Register writes |
| Runtime Config | ✅ `aiex.npu.*` | ✅ `aiex.npu.*` |
| Switchboxes | ❌ Not in xclbin | ❌ Not in xclbin |
| Core Programs | ❌ Not in xclbin | ❌ Not in xclbin |

## Known Limitations

⚠️ **Switchbox routing**: Not stored in NPU xclbin format (architectural limitation, not a bug)
⚠️ **Core programs**: Compiled separately, not in xclbin
⚠️ **Shim tiles**: Stay as raw operations even in lifted mode (they lack local memory)

See `docs/XclbinDecompiler.md` for full details.

## Validation

Check if decompiled MLIR is valid:
```bash
# Extract just the MLIR (skip debug output)
aie-translate --xclbin-to-mlir --emit-lifted design.xclbin 2>&1 | \
  grep -A 10000 "^module {" > clean.mlir

# Validate
aie-opt clean.mlir --verify-diagnostics
```

## Need Help?

- **Full Documentation**: `docs/XclbinDecompiler.md` (comprehensive, 43KB)
- **Examples**: `test/xclbin2mlir/roundtrip/`
- **Test Xclbins**: `test/npu-xrt/*/aie.xclbin`
- **Final Assessment**: `FINAL_ASSESSMENT.md` (project status and capabilities)

## Quick Troubleshooting

**Problem**: "xclbin-to-mlir: unknown option"
- **Solution**: Make sure `aie-translate` is in your PATH (run env setup script)

**Problem**: "CDO decoding not available"
- **Solution**: Rebuild with OpenSSL support (`libssl-dev`)

**Problem**: "Warning: Emitting N BDs with inferred channel"
- **Solution**: This is normal - the decompiler inferred DMA channel from context

**Problem**: Output has debug messages mixed with MLIR
- **Solution**: Use `grep -A 10000 "^module {"` to extract just the MLIR part

## Quick Reference Commands

```bash
# Most common: decompile with high-level operations
aie-translate --xclbin-to-mlir --emit-lifted input.xclbin > output.mlir

# See low-level registers
aie-translate --xclbin-to-mlir input.xclbin > output_raw.mlir

# Validate output
tail -n +40 output.mlir > clean.mlir  # Skip debug output
aie-opt clean.mlir --verify-diagnostics

# Extract just buffer descriptors
aie-translate --xclbin-to-mlir --emit-lifted input.xclbin | grep -A 3 "aie.dma_bd"

# See all tiles
aie-translate --xclbin-to-mlir --emit-lifted input.xclbin | grep "aie.tile"
```

---

**Status**: ✅ Production Ready | **Version**: Iteration 23 | **Date**: March 2026
