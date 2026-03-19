# Root Cause Found: cpp_multi_device_sequence.mlir Spurious Failures

## The Problem

The test `cpp_multi_device_sequence.mlir` fails spuriously in CI when tests run in parallel.

## Root Cause

**Three bugs in `aiecc.cpp` that ignore the `--tmpdir` argument and write output files to the current working directory (CWD) instead.**

### Evidence

After running the test, these files appear in `build/test/aiecc/` (the test source directory), **NOT** in the isolated `Output/` subdirectories:

```bash
$ ls -la build/test/aiecc/*.bin
-rw-r--r-- 1 androsti asic  160 device1_ctrlpkt.bin
-rw-r--r-- 1 androsti asic 2596 device1_ctrlpkt_dma_seq.bin
-rw-r--r-- 1 androsti asic  328 device1_seq_a.bin
```

These files should have been written to:
- `build/test/aiecc/Output/cpp_multi_device_sequence.mlir.tmp.seq_a/device1_seq_a.bin`
- `build/test/aiecc/Output/cpp_multi_device_sequence.mlir.tmp.ctrlpkt/device1_ctrlpkt.bin`
- `build/test/aiecc/Output/cpp_multi_device_sequence.mlir.tmp.ctrlpkt/device1_ctrlpkt_dma_seq.bin`

## The Race Condition

When multiple RUN lines execute in parallel (or multiple tests run in parallel):

```
Time  RUN Line 3                    RUN Line 5
  0   Start NPU inst generation     Start ctrlpkt generation
  1   Write device1_seq_a.bin       Write device1_ctrlpkt.bin
  2   [Both writing to same dir]    Write device1_ctrlpkt_dma_seq.bin
  3   Read device1_seq_a.bin        Read device1_ctrlpkt.bin
  4   ❌ FILE CONFLICT!             ❌ FILE CONFLICT!
```

## The Bugs

### Bug #1: [tools/aiecc/aiecc.cpp:3542-3543](tools/aiecc/aiecc.cpp#L3542-L3543)

**When:** `--aie-generate-npu-insts` is used

```cpp
if (generateNpuInsts) {
    outputPath = outputFileName;  // ❌ Writes to CWD!
} else {
    outputPath = tmpDirName;
    sys::path::append(outputPath, outputFileName);
}
```

**Affects:** RUN line 19 in the test

### Bug #2: [tools/aiecc/aiecc.cpp:3706-3712](tools/aiecc/aiecc.cpp#L3706-L3712)

**When:** `--aie-generate-ctrlpkt` is used (control packet binary)

```cpp
SmallString<128> ctrlPktBinPath;
if (sys::path::is_absolute(ctrlPktBinFileName)) {
    ctrlPktBinPath = ctrlPktBinFileName;
} else {
    ctrlPktBinPath = ctrlPktBinFileName;  // ❌ Should prepend tmpDirName!
}
```

**Affects:** RUN line 21 in the test

### Bug #3: [tools/aiecc/aiecc.cpp:3768-3773](tools/aiecc/aiecc.cpp#L3768-L3773)

**When:** `--aie-generate-ctrlpkt` is used (DMA sequence binary)

```cpp
SmallString<128> dmaSeqBinPath;
if (sys::path::is_absolute(dmaSeqBinFileName)) {
    dmaSeqBinPath = dmaSeqBinFileName;
} else {
    dmaSeqBinPath = dmaSeqBinFileName;  // ❌ Should prepend tmpDirName!
}
```

**Affects:** RUN line 21 in the test

## Why It's Spurious

The test passes 100/100 times when run sequentially because:
- Each RUN line completes before the next starts
- No file conflicts occur

The test fails randomly in CI because:
- Multiple RUN lines execute in parallel
- Files overwrite each other non-deterministically
- Timing-dependent on which RUN line writes/reads first

## The Fix

All three bugs have the same fix: **respect the tmpdir when writing output files.**

See [test_isolation_report.md](test_isolation_report.md) for detailed fix recommendations.

## Verification

After applying the fix:

```bash
# All .bin files should be INSIDE the tmpdir subdirectories:
$ find build/test/aiecc/Output -name "*.bin"
build/test/aiecc/Output/cpp_multi_device_sequence.mlir.tmp.seq_a/device1_seq_a.bin
build/test/aiecc/Output/cpp_multi_device_sequence.mlir.tmp.ctrlpkt/device1_ctrlpkt.bin
build/test/aiecc/Output/cpp_multi_device_sequence.mlir.tmp.ctrlpkt/device1_ctrlpkt_dma_seq.bin

# No .bin files should exist in the top-level test directory:
$ ls build/test/aiecc/*.bin
ls: cannot access 'build/test/aiecc/*.bin': No such file or directory  ✅
```
