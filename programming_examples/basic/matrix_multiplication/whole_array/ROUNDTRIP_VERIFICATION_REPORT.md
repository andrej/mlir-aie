# Xclbin Decompiler Roundtrip Verification Report

**Date:** 2026-03-24
**Test Example:** Matrix Multiplication Whole Array (512x512x512, 32x32x32 tiles, 4 columns)
**Original MLIR Device:** npu2
**Decompiled MLIR Device:** npu1

## Executive Summary

**ROUNDTRIP VERIFICATION: FAILED**

The decompiler produces human-readable MLIR with zero raw register writes, but the roundtrip compilation does NOT produce a binary-equivalent xclbin. Critical information is lost during decompilation that prevents reconstruction of the original binary.

## Test Procedure

### Step 1: Generate Original Xclbin
```bash
cd /workspace/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
rm -rf build
make build/final_512x512x512_32x32x32_4c.xclbin
```

**Result:** Generated `build/final_512x512x512_32x32x32_4c.xclbin` (82,400 bytes)

### Step 2: Decompile Xclbin to MLIR
```bash
aie-translate --xclbin-to-mlir --emit-lifted build/final_512x512x512_32x32x32_4c.xclbin > roundtrip_test/decompiled.mlir 2>/dev/null
```

**Result:** Generated `roundtrip_test/decompiled.mlir` (1,034 lines)

**Warnings during decompilation:**
- Max column 3 exceeds npu1_3col, using generic npu1 device
- Multiple warnings about emitting BDs with inferred S2MM_0 channel

### Step 3: Verify Decompiled MLIR Parseable
```bash
aie-opt roundtrip_test/decompiled.mlir
```

**Result:** MLIR parses successfully with high-level constructs

### Step 4: Recompile Decompiled MLIR
```bash
cd roundtrip_test && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=roundtrip.xclbin decompiled.mlir
```

**Result:** Generated `roundtrip.xclbin` (10,746 bytes) - SUCCESS

### Step 5: Binary Comparison

| Metric | Original | Roundtrip | Match |
|--------|----------|-----------|-------|
| Xclbin file size | 82,400 bytes | 10,746 bytes | NO |
| Column width | 8 | 4 | NO |
| CDO init size | 25,044 bytes | 4,692 bytes | NO |
| CDO elfs size | 54,544 bytes | 24 bytes | NO |
| ELF files count | 16 | 0 | NO |

## Detailed Gap Analysis

### 1. Device Type Mismatch
- **Original:** `aie.device(npu2)`
- **Decompiled:** `aie.device(npu1)`
- **Impact:** Different register address maps, different column layout assumptions

### 2. Missing Core Operations (Critical)
- **Original:** 16 `aie.core` blocks containing computational kernels
- **Decompiled:** 0 `aie.core` blocks
- **Impact:** No executable code in the roundtrip xclbin

### 3. Missing ELF Files (Critical)
- **Original:** 16 ELF files (main_core_X_Y.elf) totaling ~80KB
- **Decompiled:** No ELF file references or data
- **Impact:** Device cannot execute any computation

### 4. Lock Operations Not Used in DMA BDs (Critical)
- **Original:** Extensive `aie.use_lock` operations (acquire/release) in DMA buffer descriptors
- **Decompiled:** Locks are declared but NOT used in DMA operations
- **Impact:** DMA synchronization logic lost, data flow will not work correctly

### 5. Buffer Sizes Incorrect (Critical)
- **Original:** Properly sized buffers (e.g., `memref<32x32xi32>`, `memref<4096xi32>`)
- **Decompiled:** All buffers incorrectly shown as `memref<1xi32>`
- **Impact:** Memory layout completely wrong

### 6. Missing Data Layout Transformations
- **Original:** Contains dimension transformations for DMA operations
- **Decompiled:** No dimension transformation information
- **Impact:** Data movement patterns lost

### 7. Missing Function Declarations
- **Original:** `func.func private @matmul_i16_i32(...)`, `func.func private @zero_i32(...)`
- **Decompiled:** No function declarations
- **Impact:** Cannot link with kernel implementations

### 8. Column Width Mismatch
- **Original:** `column_width: 8`
- **Decompiled/Roundtrip:** `column_width: 4`
- **Impact:** Incorrect partition configuration

## What Works

1. Switchbox routing is extracted and can be recompiled
2. Shim mux configuration is extracted
3. Basic tile declarations are preserved
4. Lock declarations (but not their use) are extracted
5. DMA buffer descriptor structure (but not complete semantics) is extracted
6. MLIR syntax is valid and parseable

## What Does NOT Work

1. Core executable code (ELFs) not recovered
2. Lock synchronization in DMAs not recovered
3. Buffer sizes not correctly inferred
4. Data layout transformations not recovered
5. Device type misidentified
6. Function declarations not recovered
7. Complete DMA BD semantics not recovered

## Root Cause Analysis

The decompiler has fundamental limitations:

1. **ELF Recovery:** The xclbin contains compiled AIE core ELF binaries. These are machine code and cannot be decompiled back to MLIR without a sophisticated machine code decompiler. The decompiler simply skips them.

2. **Buffer Size Inference:** The CDO binary contains only base addresses and offsets for DMA operations. The actual buffer sizes must be inferred from transfer lengths and strides, which appears to not be working correctly.

3. **Lock Usage:** Lock acquire/release operations are encoded in DMA BD control registers. The decompiler appears to extract lock declarations but not their usage in DMA operations.

4. **Device Detection:** The decompiler falls back to npu1 when it cannot identify the exact device type, leading to potential register mapping issues.

## Recommendations for Achieving Binary Equivalence

### High Priority (Required for Functional Roundtrip)

1. **ELF File Preservation**: Extract ELF files from xclbin and include them as external references in decompiled MLIR with proper linkage during recompilation

2. **Lock Usage Recovery**: Parse DMA BD control registers to recover lock acquire/release operations and emit proper `aie.use_lock` in `aie.dma_bd` operations

3. **Buffer Size Inference**: Infer buffer sizes from DMA transfer lengths, or preserve buffer metadata from CDO

4. **Device Type Detection**: Properly detect npu2 vs npu1 from the xclbin metadata

### Medium Priority (Semantic Completeness)

5. **Data Layout Transformations**: Parse BD stride/wrap configurations to recover dimension transformations

6. **BD Chain Recovery**: Properly reconstruct BD chains including all next_bd links

### Lower Priority (Quality Improvements)

7. **Meaningful Names**: Recover buffer/lock names from xclbin metadata if available

8. **Column Width**: Preserve partition configuration metadata

## Conclusion

The current decompiler **DOES NOT** achieve the goal of binary-equivalent roundtrip. While it produces valid, parseable MLIR with high-level constructs, the decompiled output is missing critical semantic information required for functional equivalence:

- Executable code (ELFs)
- Lock synchronization semantics
- Correct buffer sizes
- Proper device type

**The roundtrip xclbin is 87% smaller than the original and lacks all computational capability.**

For the decompiler to be useful for its intended purpose (modify and recompile xclbins), significant work is needed to:
1. Preserve or export ELF files
2. Recover complete DMA BD semantics including locks
3. Properly infer buffer sizes
4. Support the correct device type

---

**Test Files:**
- Original xclbin: `build/final_512x512x512_32x32x32_4c.xclbin`
- Decompiled MLIR: `roundtrip_test/decompiled.mlir`
- Roundtrip xclbin: `roundtrip_test/roundtrip.xclbin`
