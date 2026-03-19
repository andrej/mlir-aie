# Fix for Missing Tile Definitions

## Problem

The `aie.shim_dma_allocation` operation syntax was updated to require a tile reference as the first parameter. Many test files were failing because they referenced tiles (e.g., `%tile_0_0`) that were not defined.

**Error Example:**
```
error: use of undeclared SSA value name
    aie.shim_dma_allocation @fromMem (%tile_0_0, MM2S, 0)
                                      ^
```

## Solution

Created `add_missing_tile_defs.py` - a Python script that:

1. Scans MLIR files for `aie.shim_dma_allocation` operations
2. Identifies undefined tile references
3. Automatically adds the missing tile definitions with proper syntax

## Syntax

The tile definition follows this pattern:
```mlir
%tile_<col>_<row> = aie.tile(<col>, <row>)
```

For example:
```mlir
%tile_0_0 = aie.tile(0, 0)
aie.shim_dma_allocation @fromMem (%tile_0_0, MM2S, 0)
```

## Usage

### Single File
```bash
python3 add_missing_tile_defs.py test/Conversion/DmaToNpu/dma_to_npu.mlir
```

### Entire Directory
```bash
python3 add_missing_tile_defs.py test/Conversion/DmaToNpu/
```

### All Test Files
```bash
python3 add_missing_tile_defs.py test/
```

## Features

- **Module-aware**: Correctly handles MLIR split-file tests (separated by `// -----`)
- **Scope-aware**: Only adds tiles inside `aie.device` blocks
- **Duplicate detection**: Skips tiles that are already defined in the same scope
- **Comment-aware**: Ignores commented-out code
- **Batch processing**: Can process entire directories recursively

## Results

Applied to `test/Conversion/DmaToNpu/`:
- Modified 8 out of 12 test files
- Added 12 tile definitions total
- All DmaToNpu tests now pass

### Files Modified
- `aiert_insts.mlir` (+1 tile)
- `bad_dma_to_npu.mlir` (+1 tile)
- `bad_dma_to_npu_datatype.mlir` (+1 tile)
- `dma_to_npu.mlir` (+4 tiles)
- `dma_to_npu_burst_length.mlir` (+2 tiles)
- `dma_to_npu_burst_length_invalid.mlir` (+2 tiles)
- `dma_to_npu_issue_token.mlir` (+1 tile)
- `dma_to_npu_width_conversion.mlir` (+1 tile, +CHECK pattern fix)

## Example Change

**Before:**
```mlir
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[...]) { metadata = @toMem, id = 1 : i64 } : memref<16xi32>
    }
    aie.shim_dma_allocation @toMem (%tile_0_0, S2MM, 0)
  }
}
```

**After:**
```mlir
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[...]) { metadata = @toMem, id = 1 : i64 } : memref<16xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @toMem (%tile_0_0, S2MM, 0)
  }
}
```
