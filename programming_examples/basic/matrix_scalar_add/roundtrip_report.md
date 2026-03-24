# ROUNDTRIP COMPILATION TEST REPORT

## Test Date
March 24, 2026

## Test Example
- **Example**: matrix_scalar_add
- **Original xclbin**: build/final.xclbin  
- **Original NPU instructions**: build/aie.mlir.prj/insts.bin (300 bytes)
- **Decompiled MLIR**: lifted_clean.mlir (61 lines)

## Test 1: Full xclbin Compilation

**Objective**: Recompile the decompiled MLIR back to xclbin using aiecc.py

**Command**:
```bash
aiecc.py --aie-generate-xclbin --xclbin-name=roundtrip.xclbin lifted_clean.mlir
```

**Result**: ❌ **FAILED**

**Error**:
```
<unknown>:0: error: 'aie.connect' op targets same destination DMA: 0 as another connect operation
Error: Routing pipeline failed
```

**Root Cause**:
The decompiled MLIR contains **redundant routing information**:
1. High-level `aie.flow` operations (abstract flow declarations)
2. Low-level `aie.switchbox` connections (explicit port-to-port connections)  
3. `aie.shim_mux` connections

During compilation, the routing pipeline attempts to convert `aie.flow` operations into explicit switchbox connections, but the switchbox connections already exist, causing conflicts. The compiler expects EITHER high-level flows OR low-level switchbox connections, not both.

**Analysis**: The decompiler correctly reconstructs both the abstract flows and the concrete switchbox configurations from the xclbin. However, this creates an over-specified MLIR that cannot be recompiled without modification.

## Test 2: NPU Instructions Generation

**Objective**: Generate just the NPU instructions binary from the runtime_sequence

**Command**:
```bash
aie-translate --aie-npu-to-binary --aie-output-binary lifted_main.mlir -o roundtrip_insts.bin
```

**Result**: ✅ **SUCCEEDED** (with differences)

**File Comparison**:
- Original: build/aie.mlir.prj/insts.bin = 300 bytes
- Roundtrip: roundtrip_insts.bin = 128 bytes  
- **Difference**: 172 bytes missing (57% smaller)

**Hex Comparison**:
Both files start with the same header (0001 0306 0101 0000), indicating the same instruction format. However:
- Original contains significantly more instruction data
- Roundtrip only has the basic writebd, address_patch, push_queue, and sync operations

## Root Cause Analysis

### Missing High-Level Operations

**Original runtime_sequence** (build/aie.mlir):
```mlir
aie.runtime_sequence(%arg0: memref<16x128xi32>, ...) {
  %0 = aiex.dma_configure_task_for @in0 { ... }
  aiex.dma_start_task(%0)
  %1 = aiex.dma_configure_task_for @out0 { ... } {issue_token = true}
  aiex.dma_start_task(%1)
  aiex.dma_await_task(%1)      // TASK SYNCHRONIZATION
  aiex.dma_free_task(%0)        // TASK CLEANUP
}
```

**Decompiled runtime_sequence** (lifted_clean.mlir):
```mlir
aie.runtime_sequence @configure() {
  aiex.npu.writebd {...}
  aiex.npu.address_patch {...}
  aiex.npu.push_queue(0, 0, MM2S : 0) {...}
  aiex.npu.writebd {...}
  aiex.npu.address_patch {...}
  aiex.npu.push_queue(0, 0, S2MM : 0) {...}
  aiex.npu.sync {...}           // ONLY BASIC SYNC
  // MISSING: aiex.dma_await_task
  // MISSING: aiex.dma_free_task
}
```

### Abstraction Level Mismatch

The decompiler produces **NPU-level operations** (aiex.npu.*) but the original source uses **DMA task-level operations** (aiex.dma_*). The task-level operations:
1. Manage task lifecycle (configure, start, await, free)
2. Reference objectfifos (@in0, @out0)
3. Contain rich multi-dimensional stride information
4. Include task dependencies and synchronization

When these are compiled down to NPU instructions, they are expanded into:
- Buffer descriptor writes (aiex.npu.writebd)
- Runtime address patching (aiex.npu.address_patch)  
- Queue push operations (aiex.npu.push_queue)
- Synchronization (aiex.npu.sync)
- **Additional control flow and cleanup instructions**

The decompiler successfully lifts the basic NPU operations but **cannot reconstruct** the higher-level task management operations and their full instruction sequences.

## Detailed Differences

### Missing from Decompiled MLIR:

1. **Runtime sequence function signature**:
   - Original: `(%arg0: memref<16x128xi32>, %arg1: ..., %arg2: ...)`
   - Decompiled: `@configure()` (no arguments)

2. **ObjectFifo references**:
   - Original: References `@in0` and `@out0` objectfifos
   - Decompiled: No objectfifo references (they're lowered away)

3. **Task management operations**:
   - Missing: `aiex.dma_configure_task_for`
   - Missing: `aiex.dma_await_task`
   - Missing: `aiex.dma_free_task`

4. **Additional NPU instructions**:
   - Original binary has 300 bytes of instructions
   - Roundtrip binary has only 128 bytes
   - Missing: ~172 bytes of control/cleanup instructions

### Redundant in Decompiled MLIR:

1. **Dual routing representation**:
   - Has both: `aie.flow` operations
   - And: Explicit `aie.switchbox` connections
   - Should have: Only one or the other

2. **Unused global constants**:
   - `@config_blockwrite_data_0` and `@config_blockwrite_data_1`
   - Not referenced in the runtime sequence

## Conclusions

### ✅ Successes

1. **Static configuration is complete**:
   - All tiles, switchboxes, buffers, locks are correctly decompiled
   - Flow connections are accurately reconstructed
   - DMA buffer descriptors in aie.mem are correct

2. **Basic NPU instructions are valid**:
   - The decompiled NPU instructions can be compiled to binary
   - The instruction format is correct
   - Basic operations (writebd, push_queue, sync) are present

3. **Human-readable output**:
   - No raw register writes
   - Meaningful operation names
   - Structured MLIR

### ❌ Failures

1. **Cannot roundtrip to xclbin**:
   - Compilation fails due to redundant routing information
   - The combination of aie.flow + explicit switchbox connections is invalid

2. **Incomplete NPU instruction sequence**:
   - Missing 57% of the original instruction bytes
   - Missing higher-level task management operations
   - Missing task synchronization and cleanup

3. **Lost abstraction information**:
   - Cannot reconstruct objectfifos from lowered code
   - Cannot recover task-level operations from NPU instructions
   - Function signatures and argument information lost

## Recommendations

### Critical Fixes Needed

1. **Fix routing redundancy**:
   - Decompiler should emit EITHER aie.flow operations OR explicit switchbox connections
   - If emitting flows, omit switchbox connection details
   - If emitting switchboxes, omit aie.flow operations

2. **Improve NPU instruction decompilation**:
   - Identify and lift all NPU instruction sequences from the binary
   - Current decompiler misses significant portions of the instruction stream
   - Need to analyze why 172 bytes are missing from the roundtrip

3. **Attempt higher-level reconstruction** (stretch goal):
   - Try to reconstruct aiex.dma_await_task from sync patterns
   - Try to infer task cleanup operations
   - Consider pattern-matching common task sequences

### Testing Strategy

1. **Immediate**: Test if removing aie.flow operations allows compilation to succeed
2. **Immediate**: Compare full instruction hex dumps to identify missing operations
3. **Next**: Test on simpler examples with fewer operations
4. **Next**: Test on examples without objectfifos (using explicit buffers)

## Summary

The decompiler **successfully produces human-readable MLIR** with zero raw register writes, which was the primary goal. However, it **cannot yet achieve full roundtrip compilation** due to:
1. Redundant routing information causing compilation conflicts
2. Incomplete NPU instruction sequence reconstruction
3. Lost high-level abstraction information

The roundtrip capability requires:
- Fixing the routing redundancy issue (straightforward)
- Completing NPU instruction sequence decompilation (significant work needed)
- Potentially accepting that full high-level reconstruction may not be possible from binaries alone

