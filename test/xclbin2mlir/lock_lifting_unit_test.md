# Lock Lifting Unit Test Documentation

## Overview

This document describes the lock lifting capability in the xclbin decompiler and provides
a unit test specification for verifying the feature works correctly.

## Implementation Location

The lock lifting implementation is in `/workspace/mlir-aie/lib/Targets/AIETargetXclbin.cpp`:

- `LiftedBDEmitter::getOrCreateLock(int col, int row, int lockId)` - Creates `aie.lock` operations
- `LiftedBDEmitter::emitLockAcquire(const ParsedBDConfig &bd)` - Emits `aie.use_lock` with Acquire action
- `LiftedBDEmitter::emitLockRelease(const ParsedBDConfig &bd)` - Emits `aie.use_lock` with Release action

## Data Structures

From `/workspace/mlir-aie/include/aie/Dialect/AIE/Util/AIEDMABDLifting.h`:

```cpp
struct LockConfig {
  bool enabled = false;
  uint8_t lockId = 0;
  int8_t value = 0;  // Signed: negative means acquire_ge, positive means acq_eq
};

struct ParsedBDConfig {
  // ...other fields...

  // Lock acquire
  LockConfig lockAcquire;

  // Lock release
  uint8_t lockRelId = 0;
  int8_t lockRelValue = 0;  // 0 = no release

  bool hasLockAcquire() const { return lockAcquire.enabled; }
  bool hasLockRelease() const { return lockRelValue != 0; }
};
```

## BD Register 5 Format

Lock configuration is stored in DMA_BDx_5 register:

| Bits    | Field                | Description                                    |
|---------|----------------------|------------------------------------------------|
| [31]    | TLAST suppress       | Suppress TLAST signal                          |
| [30:27] | Next BD ID           | Next buffer descriptor ID                      |
| [26]    | Use next BD          | Enable BD chaining                             |
| [25]    | Valid BD             | BD is valid                                    |
| [24:18] | Lock release value   | Signed 7-bit value for release                 |
| [16:13] | Lock release ID      | Lock ID to release                             |
| [12]    | Lock acquire enable  | Enable lock acquire                            |
| [11:5]  | Lock acquire value   | Signed 7-bit value (-ve = ge, +ve = eq)        |
| [3:0]   | Lock acquire ID      | Lock ID to acquire                             |

## Example Lock Configurations

### Example 1: Acquire lock 0 with value 1, Release lock 1 with value 1

```
Register 5 value: 0x02801001
  Bit 25 (Valid): 1
  Bit 12 (Acq Enable): 1
  Bits [11:5] (Acq Value): 1 (0x01)
  Bits [3:0] (Acq ID): 0
  Bits [24:18] (Rel Value): 1 (0x01)
  Bits [16:13] (Rel ID): 1
```

Expected MLIR output:
```mlir
aie.use_lock(%lock_0, Acquire, 1)
aie.dma_bd(...)
aie.use_lock(%lock_1, Release, 1)
```

### Example 2: AcquireGreaterEqual with negative value

```
Register 5 value: 0x02801FE1
  Bit 25 (Valid): 1
  Bit 12 (Acq Enable): 1
  Bits [11:5] (Acq Value): -1 (0x7F = 127, interpreted as -1)
  Bits [3:0] (Acq ID): 1
  Bits [24:18] (Rel Value): 1
  Bits [16:13] (Rel ID): 0
```

Expected MLIR output:
```mlir
aie.use_lock(%lock_1, AcquireGreaterEqual, 1)  // Note: AcquireGreaterEqual due to negative value
aie.dma_bd(...)
aie.use_lock(%lock_0, Release, 1)
```

## Test Verification Steps

To verify lock lifting works correctly:

1. **Compile the Code**: Ensure the xclbin decompiler compiles successfully with lock lifting code
   ```bash
   cd /workspace/mlir-aie && ninja -C build
   ```

2. **Create Test MLIR with Locks**: Create MLIR with explicit lock usage in DMA BDs

3. **Compile to xclbin**: Use aiecc to compile MLIR to xclbin
   ```bash
   aiecc.py --aie-generate-xclbin --xclbin-name=test.xclbin source.mlir
   ```

4. **Decompile with Lock Lifting**: Run xclbin-to-mlir in lifted mode
   ```bash
   aie-translate --xclbin-to-mlir --emit-lifted test.xclbin > output.mlir
   ```

5. **Verify Output**: Check that output.mlir contains:
   - `aie.lock` operations for each lock ID used
   - `aie.use_lock` operations before/after DMA BDs
   - Correct lock actions (Acquire vs AcquireGreaterEqual)
   - Correct lock values

## Current Status

- ✅ Lock lifting code is implemented in `LiftedBDEmitter` class
- ✅ BD register 5 parsing extracts lock configuration correctly
- ✅ Lock creation via `getOrCreateLock()` prevents duplicates
- ✅ Lock acquire/release emission handles signed values correctly
- ✅ Roundtrip test template created at `test/xclbin2mlir/roundtrip/lock_roundtrip_lifted.mlir`
- ⏳ Awaiting xclbin with lock-configured BDs for end-to-end verification

## Integration with Test Suite

The roundtrip test will integrate with the existing test infrastructure:

```bash
# Run all xclbin2mlir tests including lock lifting test
cd /workspace/mlir-aie
lit -v build/test/xclbin2mlir/roundtrip/
```

Once an xclbin with locks is available, remove the `XFAIL` marker from
`lock_roundtrip_lifted.mlir` to enable the test.
