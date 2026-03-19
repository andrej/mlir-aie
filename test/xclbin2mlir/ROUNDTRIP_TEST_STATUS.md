# Round-Trip Verification Test Status

## Test Created

The round-trip verification test has been successfully created at:
- `test/xclbin2mlir/roundtrip_verification.mlir`

## What This Test Does

This test validates the complete semantic lifting pipeline by:

1. **Decompiling** a real xclbin file (`test/npu-xrt/add_blockwrite/aie.xclbin`) using `--emit-lifted`
2. **Verifying** the decompiled output contains all expected semantic elements from the original design:
   - **Tiles**: (0,0) shim tile and (0,2) compute tile
   - **Buffers**: 5 buffers on tile (0,2) with type `memref<8xi32>`
   - **Locks**: 4 locks on tile (0,2) with IDs 0-3
   - **DMA BDs**: 4 buffer descriptor operations with offset=0, length=8
   - **Switchbox routing**: Switchbox and connection operations
   - **Runtime sequence**: Preserved for non-lifted operations

## Test Structure

The test uses FileCheck with the following approach:
- Uses `CHECK-DAG` for operations that may appear in any order
- Checks specific tile coordinates: (0,0) and (0,2)
- Verifies exact counts of buffers (5) and locks (4) by having that many CHECK-DAG patterns
- Uses regex patterns to match operation structure while being flexible about SSA value names
- Includes comprehensive comments explaining what each check validates

## Running the Test

Once the `aie-translate` binary is rebuilt with the `--emit-lifted` flag, run:

```bash
# Run just this test
lit test/xclbin2mlir/roundtrip_verification.mlir

# Run all xclbin2mlir tests
lit test/xclbin2mlir/

# Manual verification (for debugging)
aie-translate --xclbin-to-mlir --emit-lifted test/npu-xrt/add_blockwrite/aie.xclbin | \
  FileCheck test/xclbin2mlir/roundtrip_verification.mlir
```

## Building aie-translate

The test requires a rebuild of aie-translate to include recent lifting features:

```bash
cd mlir-aie/build
ninja aie-translate  # or cmake --build . --target aie-translate
```

## Original Design Reference

The test validates against the original design at `test/npu-xrt/add_blockwrite/aie.mlir`:

| Component | Original Design | Expected in Decompiled Output |
|-----------|-----------------|-------------------------------|
| Tiles | `aie.tile(0, 0)`, `aie.tile(0, 2)` | Same tile operations |
| Buffers | 5 buffers: `objFifo_in1_cons_buff_0/1`, `objFifo_out1_buff_0/1`, `constant_buffer` | 5 `aie.buffer` ops with `memref<8xi32>` |
| Locks | 4 locks with IDs 0-3, init values 2,0,2,0 | 4 `aie.lock` ops with IDs 0-3 |
| DMA BDs | 4 BDs at lines 88, 93, 100, 105 | 4 `aie.dma_bd` ops with offset=0, length=8 |
| Routing | `aie.flow` (0,0)→(0,2) and (0,2)→(0,0) | `aie.switchbox` with `aie.connect` ops |

## Why This Test Is Important

Round-trip verification tests are critical for validating decompilers because they:

1. **Test the complete pipeline**: From source MLIR → xclbin → decompiled MLIR
2. **Verify semantic correctness**: Ensures lifted operations match the original design intent
3. **Catch regressions**: Any changes that break the lifting will fail this test
4. **Validate integration**: Tests that all lifting components (BD, locks, switchbox) work together

This test complements the unit tests by validating the end-to-end behavior with a real xclbin.
