# Roundtrip Test Analysis - Vector Scalar Add

## Test Setup

- **Original xclbin**: `original.xclbin` (9,721 bytes)
- **Decompiled MLIR**: `decompiled_lifted.mlir` (with --emit-lifted flag)
- **ELF file**: `core_0_2.elf` (2,796 bytes)

## Key Findings

### 1. ELF Preservation ✅ WORKING

The decompiled MLIR correctly preserves the ELF file reference:
```mlir
%core_0_2 = aie.core(%tile_0_2) {
  aie.end
} {elf_file = "core_0_2.elf"}
```

### 2. High-Level Hardware Structures ✅ MOSTLY WORKING

The following are correctly extracted:
- **Switchbox connections**: Shim, MemTile, and Compute tile switchboxes (lines 8-11, 13-18, 57-60)
- **Shim mux configuration**: DMA connections (lines 4-7)
- **Lock initialization**: 4 locks for tile(0,2) with correct init values (lines 65-68)
  - Lock 0: init=2
  - Lock 1: init=0
  - Lock 2: init=2
  - Lock 3: init=0

### 3. Buffer Descriptor (BD) Configuration ❌ BROKEN

#### Observations from Debug Output:
```
[DEBUG] BD write #1: addr=0x0021D000 value=0x00000000 -> tile(0,2) BD0 reg0
[DEBUG] BD write #2: addr=0x0021D040 value=0x00000000 -> tile(0,2) BD2 reg0
...
[DEBUG] BD write #10: addr=0x001A0320 value=0x00000000 -> tile(0,1) BD25 reg0
[DEBUG] BD extraction from decoded commands: 12 BD register writes, 0 completed BDs
```

**Critical Finding**: All BD register writes have **VALUE=0x00000000**

This means:
- The CDO contains BD register writes, but they are **initialization writes** (clearing to zero)
- The CDO does NOT contain the actual BD configuration (buffer addresses, lengths, lock operations, etc.)
- The decompiler correctly identifies BD writes but has no configuration data to extract

#### Resulting Defects:

**Buffer Sizes** (Lines 19-26, 61-64):
- Ground truth: `memref<32xi32>` and `memref<64xi32>`
- Decompiled: ALL buffers are `memref<1xi32>`
- Impact: 32x-64x smaller buffers in recompiled design

**DMA BD Transfer Lengths** (Lines 30, 33, 36, 72, 75, 78):
- Ground truth: `aie.dma_bd(%in_cons_buff_0 : memref<64xi32>, 0, 64)`
- Decompiled: `aie.dma_bd(%bd_buf_0_1_0 : memref<1xi32>, 0, 0)`
- Impact: Zero data transfer

**Lock Operations** (Lines 70-82):
- Ground truth: `aie.use_lock(%in_fwd_cons_prod_lock_0, AcquireGreaterEqual, 1)` before each BD
- Decompiled: NO lock operations in DMA BDs
- Impact: Race conditions, no synchronization

**BD Chaining** (Lines 31, 34, 37, 73, 76, 79):
- Ground truth: Proper ping-pong: BD0 -> BD1 -> BD0
- Decompiled: ALL BDs point back to `^bb1` (self-loop)
- Impact: No double buffering

### 4. Runtime Sequence ❌ EMPTY

Lines 86-88 show the runtime sequence is completely empty:
```mlir
aie.runtime_sequence @configure() {
  aie.end
}
```

Ground truth should contain aiex.npu.writebd operations for shim tile BDs.

## Root Cause Analysis

### Where is the BD Configuration?

Based on the evidence:

1. **CDO contains BD address writes with ZERO values** → Only initialization, not configuration
2. **Compute tile BDs are NOT in the CDO** → Must be configured elsewhere
3. **Two possibilities**:
   a. **Runtime configuration**: Compute tile BDs are configured by the core ELF at runtime
   b. **Missing extraction**: The CDO has the data but decompiler isn't finding it

### Test: Are Compute Tile BDs Runtime-Configured?

The ground truth (`input_with_addresses.mlir`) shows:
```mlir
%mem_0_2 = aie.mem(%tile_0_2) {
  %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
^bb1:
  aie.use_lock(%in_fwd_cons_prod_lock_0, AcquireGreaterEqual, 1)
  aie.dma_bd(%in_fwd_cons_buff_0 : memref<32xi32>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
  aie.use_lock(%in_fwd_cons_cons_lock_0, Release, 1)
  aie.next_bd ^bb2
...
}
```

This is an `aie.mem` operation which gets **lowered to runtime code** in the core ELF, NOT CDO commands.

### Shim Tile BDs vs Compute Tile BDs

- **Shim tile BDs**: Configured via NPU instructions (aiex.npu.writebd) → Should be in runtime_sequence
- **Compute tile BDs**: Configured via core program at runtime → Must be extracted from ELF

## Verification Needed

### Does the Shim BD Configuration Exist?

The previous iteration 13 report mentioned that shim BD buffer_length=128 was correct. But the current decompiled_lifted.mlir shows an EMPTY runtime_sequence. This suggests:

**Either**:
1. The decompiler is not emitting the runtime sequence properly
2. The original xclbin doesn't have NPU instructions embedded

Let me check if NPU instructions were used to generate this xclbin.

## Next Steps

1. **Extract NPU instructions from xclbin** (if they exist)
2. **Re-decompile with --npu-insts flag** to populate runtime_sequence
3. **Analyze core ELF** to understand BD configuration code
4. **Determine if compute tile BD extraction from ELF is necessary** or if preservation is sufficient

## Current State Summary

| Component | Status | Notes |
|-----------|--------|-------|
| ELF Preservation | ✅ | `elf_file` attribute present |
| Switchbox | ✅ | All connections extracted |
| Locks | ✅ | Correct count and init values |
| Buffer Sizes | ❌ | All memref<1xi32> |
| BD Lengths | ❌ | All zero |
| BD Lock Ops | ❌ | Missing |
| BD Chaining | ❌ | All self-loops |
| Runtime Seq | ❌ | Empty (should have shim BDs) |
| Roundtrip | ❌ | Will fail due to above issues |
