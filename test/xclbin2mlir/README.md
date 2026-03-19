# xclbin Decompiler Tests

This directory contains tests for the xclbin-to-MLIR decompiler (`aie-translate --xclbin-to-mlir`).

## What it does

The decompiler extracts AIE configuration from xclbin binary files and lifts it to MLIR operations. It:

1. Parses the AXLF xclbin format using XRT headers
2. Extracts PDI (Programmable Device Image) sections
3. Extracts CDO (Configuration Data Object) binaries from PDIs
4. Decodes CDO commands using bootgen's decoder
5. Emits MLIR operations: `aiex.npu.write32`, `aiex.npu.maskwrite32`, `aiex.npu.blockwrite`

## Usage

```bash
aie-translate --xclbin-to-mlir input.xclbin
```

## Semantic Lifting Mode

The `--emit-lifted` flag enables semantic lifting of register writes to high-level AIE operations:

```bash
aie-translate --xclbin-to-mlir --emit-lifted input.xclbin
```

This mode produces:
- `aie.tile` - tile references
- `aie.buffer` - buffer allocations for BDs
- `aie.lock` - lock declarations
- `aie.dma_bd` - DMA buffer descriptors with dimensions, locks, iteration
- `aie.switchbox` - switchbox routing configurations
- `aie.connect` - stream connections

## Tests

### Basic Tests
- `basic_xclbin.mlir`: Decompiles a real xclbin and verifies both raw and lifted output modes

### Lifted Mode Integration Tests
- `lifted_bd_output.mlir`: Tests BD lifting (buffers, locks, dma_bd operations)
- `lifted_switchbox_output.mlir`: Tests switchbox routing lifting
- `lifted_complete_output.mlir`: Comprehensive test for all lifted operations
- `lifted_bd_attributes.mlir`: Tests BD operation attributes (dimensions, locks, chaining)
- `roundtrip_verification.mlir`: Round-trip verification test that validates decompiled output is semantically equivalent to the original MLIR (tests all lifting features together)

### C++ Unit Tests (in test/CppTests/)
- `bd_lifting.cpp`: Unit tests for BDFieldExtractor, BDAddressParser, BDAccumulator
- `switchbox_lifting.cpp`: Unit tests for SwitchAddressParser, SwitchboxAccumulator

## Running Tests

```bash
# Run all xclbin2mlir tests
lit test/xclbin2mlir/

# Run a specific test
lit test/xclbin2mlir/basic_xclbin.mlir

# Run C++ unit tests
ninja check-aie-cpp
```
