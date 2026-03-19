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

## Current Limitations

- Phase 1 implementation: basic register write lifting only
- Scans PDI for CDO magic bytes (assumes standard bootgen PDI format)
- Only handles single CDO section
- Only handles embedded PDI (not external PDI file references)
- Outputs low-level register writes, not semantic AIE operations

## Future Enhancements

- Parse bootgen PDI format properly
- Support multiple CDO sections (init, elfs, enable)
- Semantic lifting to `aie.lock`, `aie.dma_bd`, `aie.switchbox` operations
- Register name resolution using aie_registers_aie2.json

## Tests

- `basic_xclbin.mlir`: Decompiles a real xclbin and verifies basic MLIR generation
