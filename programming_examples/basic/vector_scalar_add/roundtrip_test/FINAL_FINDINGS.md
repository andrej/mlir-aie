# Roundtrip Test - Final Findings and Recommendations

## Executive Summary

Testing the current decompiler state with ELF preservation on `vector_scalar_add` reveals that **ELF preservation alone is NOT sufficient for binary-equivalent roundtrips**. The decompiler successfully preserves the core ELF file but **compute tile BD configurations are missing** because they exist only in the core program, not in the CDO.

## Test Results

### What's Working ✅

1. **ELF File Preservation**
   - The decompiled MLIR correctly includes `{elf_file = "core_0_2.elf"}`
   - The ELF file can be referenced during recompilation

2. **High-Level Infrastructure**
   - Switchbox connections (3 switchboxes extracted correctly)
   - Shim mux configuration
   - Lock declarations with correct init values (4 locks for compute tile)

### What's Broken ❌

1. **Buffer Sizes**: All `memref<1xi32>` instead of `memref<32xi32>` / `memref<64xi32>`
2. **DMA Transfer Lengths**: All zero instead of 32/64
3. **Lock Operations**: Missing from DMA BDs (no acquire/release)
4. **BD Chaining**: Broken (all self-loops instead of ping-pong)
5. **Runtime Sequence**: Empty (this xclbin has no embedded NPU instructions)

### Binary Comparison

From previous test (see build/roundtrip_comparison.txt):
- Original xclbin: 9,721 bytes
- Recompiled xclbin: 7,830 bytes
- **Preservation: 80.5%** (1,891 bytes lost)
- **PDI Preservation: 42.2%** (1,888 bytes lost)
- **Result: ❌ NOT binary-equivalent**

## Root Cause Analysis

### The Fundamental Problem

**Compute Tile BDs Are Runtime-Configured, Not CDO-Configured**

Evidence:
```
[DEBUG] BD write #1: addr=0x0021D000 value=0x00000000 -> tile(0,2) BD0 reg0
[DEBUG] BD write #2: addr=0x0021D040 value=0x00000000 -> tile(0,2) BD2 reg0
```

All BD register writes in the CDO have **value=0x00000000** (initialization only).

### Why This Happens

In MLIR-AIE compilation:

1. **aie.mem operations** (compute tile DMA config) are lowered to **core program code**
2. **aie.memtile_dma operations** (memtile DMA config) can be in CDO OR runtime
3. **Shim tile BDs** are configured via **NPU instructions** (runtime_sequence)

The compilation flow:
```
aie.mem(%tile_0_2) { ... }
        ↓ (lowering)
Core program that writes to DMA BD registers at 0x1D000-0x1D1FF
        ↓ (compilation)
core_0_2.elf contains the BD configuration code
```

The CDO only **initializes** (zeros out) the BD region. It does NOT configure the BDs.

### What This Means for Decompilation

To achieve binary-equivalent roundtrips, the decompiler must:

1. **Extract BD configuration from core ELF files**, not just preserve them
2. **Parse the core program** to identify DMA BD setup code
3. **Reconstruct aie.mem operations** with proper:
   - Buffer references and sizes
   - Transfer lengths
   - Lock acquire/release operations
   - BD chaining (next_bd relationships)

Simply preserving the ELF and linking it back in will NOT work because:
- The ELF expects specific buffer addresses that won't match
- The MLIR won't have correct buffer sizes for allocation
- The lock synchronization logic won't be visible in MLIR

## Architecture Understanding

### Three Types of BDs in the Design

| BD Type | Configuration Method | Located In | Decompiler Status |
|---------|---------------------|------------|-------------------|
| Shim Tile BDs | NPU Instructions (aiex.npu.writebd) | runtime_sequence | ❌ Empty (xclbin has no embedded insts) |
| MemTile BDs | Mixed: can be CDO or runtime | CDO (this example) | ❌ Zero values extracted |
| Compute Tile BDs | Core program at runtime | core_0_2.elf | ❌ Not extracted |

### Ground Truth Comparison

**What the compiler generated** (input_with_addresses.mlir line 158-169):
```mlir
%mem_0_2 = aie.mem(%tile_0_2) {
  %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
^bb1:
  aie.use_lock(%in_fwd_cons_prod_lock_0, AcquireGreaterEqual, 1)
  aie.dma_bd(%in_fwd_cons_buff_0 : memref<32xi32>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
  aie.use_lock(%in_fwd_cons_cons_lock_0, Release, 1)
  aie.next_bd ^bb2
^bb2:
  aie.use_lock(%in_fwd_cons_prod_lock_0, AcquireGreaterEqual, 1)
  aie.dma_bd(%in_fwd_cons_buff_1 : memref<32xi32>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
  aie.use_lock(%in_fwd_cons_cons_lock_0, Release, 1)
  aie.next_bd ^bb1
...
}
```

**What the decompiler extracted** (decompiled_lifted.mlir line 69-82):
```mlir
%mem_0_2 = aie.mem(%tile_0_2) {
  %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
^bb1:  // 5 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4
  aie.dma_bd(%bd_buf_0_2_0 : memref<1xi32>, 0, 0)
  aie.next_bd ^bb1
^bb2:  // no predecessors
  aie.dma_bd(%bd_buf_0_2_1 : memref<1xi32>, 0, 0)
  aie.next_bd ^bb1
...
}
```

**Differences**:
- ❌ No lock operations
- ❌ Wrong buffer types (memref<1xi32> vs memref<32xi32>)
- ❌ Zero transfer lengths (0 vs 32)
- ❌ Broken chaining (all -> bb1 vs bb1 <-> bb2 ping-pong)
- ❌ Missing bd_id/next_bd_id attributes

## Verification: Can Recompilation Work With Just ELF Preservation?

**NO, for several reasons:**

1. **Buffer allocation mismatch**:
   - Original: allocates memref<32xi32> at specific addresses
   - Recompiled: would allocate memref<1xi32> (32x too small)
   - ELF expects 128 bytes but only gets 4 bytes

2. **Address references broken**:
   - Core program has hardcoded addresses for buffers
   - Recompiled MLIR may allocate at different addresses
   - BD configuration in ELF writes to wrong memory

3. **Synchronization missing**:
   - No lock operations in decompiled DMA ops
   - Core expects lock behavior that won't be set up

4. **No NPU instructions**:
   - Shim BDs won't be configured (runtime_sequence empty)
   - Host can't start DMA transfers

## Recommended Next Steps

### Option A: Extract BD Configuration from ELF (COMPLEX)

**Approach**:
1. Implement ELF parser in decompiler
2. Disassemble core program to identify BD register writes
3. Reverse-engineer BD configuration from assembly
4. Reconstruct aie.mem operations with extracted config

**Challenges**:
- Requires AIE instruction set knowledge
- Program analysis (control flow, data flow)
- Distinguishing BD config from other DMA register accesses
- Handling dynamic/computed BD values

**Benefits**:
- Complete BD reconstruction
- Accurate buffer sizes, transfer lengths, locks
- True binary-equivalent roundtrips possible

### Option B: Hybrid Approach - Preserve ELF + Infer from Metadata (MODERATE)

**Approach**:
1. Keep ELF preservation as-is
2. Add metadata extraction to infer missing info:
   - Parse ELF symbol table for buffer names/addresses
   - Use buffer addresses to infer sizes from memory layout
   - Extract NPU instructions from xclbin for runtime_sequence
3. Generate "good enough" MLIR that compiles correctly

**Challenges**:
- May not achieve binary equivalence
- Requires assumptions about buffer layout
- NPU instructions may not be embedded in xclbin

**Benefits**:
- Easier to implement than full disassembly
- Better than current state
- May work for many common cases

### Option C: Focus on MemTile and Shim BDs First (INCREMENTAL)

**Approach**:
1. Fix MemTile BD extraction (currently shows zero values)
2. Extract NPU instructions from xclbin for shim BDs
3. Document compute tile BD limitation
4. Provide tool to compare recompiled vs original

**Challenges**:
- Doesn't solve compute tile BD problem
- Roundtrips still won't be binary-equivalent

**Benefits**:
- Quick wins on low-hanging fruit
- Builds foundation for full solution
- Shows incremental progress

## Conclusion

**The ELF preservation feature is necessary but NOT sufficient for binary-equivalent roundtrips.**

The decompiler correctly preserves ELF files, but:
- Compute tile BD configurations exist only in the ELF (runtime)
- CDO only initializes BDs to zero
- Without extracting BD config from ELF, the decompiled MLIR is incomplete

**Recommendation**: Pursue Option A (ELF BD extraction) as the complete solution, with Option C as an incremental step to validate the infrastructure while developing the ELF parser.

**Current Blocker**: Compute tile BD configuration extraction from core ELF files is REQUIRED for binary-equivalent roundtrips.
