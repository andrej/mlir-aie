# xclbin2mlir Round-trip Verification Tests

This directory contains round-trip verification tests for the xclbin decompiler (`aie-translate --xclbin-to-mlir`).

## Overview

These tests verify that the xclbin decompiler correctly decompiles binary xclbin files back to valid MLIR. The tests focus on **raw mode** (default mode), which emits low-level register write operations rather than high-level AIE constructs.

## Test Files

### add_blockwrite_raw.mlir
Tests decompilation of `/workspace/mlir-aie/test/npu-xrt/add_blockwrite/aie.xclbin` in raw mode.

**Verifies:**
- Module structure is present
- `aie.device(npu1_1col)` declaration exists
- `aie.runtime_sequence` block is created
- `aiex.npu.write32` operations are generated for register writes
- `aiex.npu.maskwrite32` operations are generated for masked writes
- Proper `aie.end` terminator is present

### ctrl_packet_reconfig_raw.mlir
Tests decompilation of `/workspace/mlir-aie/test/npu-xrt/ctrl_packet_reconfig/aie.xclbin` in raw mode.

**Verifies:**
- Same checks as add_blockwrite_raw.mlir
- Ensures control packet designs are correctly decompiled

## Running the Tests

### Option 1: Using lit (recommended - fully integrated)
```bash
# Setup environment
source /opt/xilinx/xrt/setup.sh
source /workspace/buildenv/bin/activate
export PEANO_INSTALL_DIR="$(pip show llvm-aie 2>/dev/null | grep ^Location: | awk '{print $2}')/llvm-aie"
source /workspace/mlir-aie/utils/env_setup.sh /workspace/mlir-aie/install ${PEANO_INSTALL_DIR}

# Run all roundtrip tests
cd /workspace/mlir-aie/build
lit -v test/xclbin2mlir/roundtrip/

# Or run a specific test
lit -v test/xclbin2mlir/roundtrip/add_blockwrite_raw.mlir
```

### Option 2: Using the test script
```bash
cd /workspace/mlir-aie/test/xclbin2mlir/roundtrip
./run_tests.sh
```

### Option 3: Manual testing with FileCheck
```bash
# Setup environment
source /opt/xilinx/xrt/setup.sh
source /workspace/buildenv/bin/activate
export PEANO_INSTALL_DIR="$(pip show llvm-aie 2>/dev/null | grep ^Location: | awk '{print $2}')/llvm-aie"
source /workspace/mlir-aie/utils/env_setup.sh /workspace/mlir-aie/install ${PEANO_INSTALL_DIR}

# Run test
aie-translate --xclbin-to-mlir test/npu-xrt/add_blockwrite/aie.xclbin | \
  /workspace/mlir-aie/my_install/mlir/bin/FileCheck test/xclbin2mlir/roundtrip/add_blockwrite_raw.mlir
```

## Test Design

These tests use the **lit** test framework with **FileCheck** patterns to validate the decompiler output. Each test:

1. Runs `aie-translate --xclbin-to-mlir` on a pre-compiled xclbin file
2. Uses FileCheck patterns to verify key features are present in the output
3. Focuses on structural correctness rather than exact value matching

## Future Work

- Add lifted mode tests (using `--emit-lifted` flag)
- Add tests for additional xclbin files from the test suite
- Add tests that compare specific values (e.g., buffer sizes, lock IDs)
- Add tests for error cases (malformed xclbin files)

## Status

✅ Raw mode tests implemented and passing (March 2026)
✅ Fully integrated with lit test infrastructure (March 2026)
✅ Tests discoverable and executable via `lit test/xclbin2mlir/roundtrip/`
⏳ Lifted mode tests - pending
