# Test Isolation Analysis: cpp_multi_device_sequence.mlir

## Executive Summary

The `cpp_multi_device_sequence.mlir` test is properly isolated from other tests via lit's `%t` substitution mechanism, which creates unique temporary directories per test. However, there is a **potential race condition within the test itself** when multiple RUN lines execute in parallel.

## Key Findings

### 1. tmpdir Isolation Mechanism

In [tools/aiecc/aiecc.cpp:5219-5227](tools/aiecc/aiecc.cpp#L5219-L5227), the tmpdir is determined as follows:

```cpp
SmallString<128> actualTmpDir;
if (!tmpDirName.empty()) {
    actualTmpDir = tmpDirName;
} else {
    // Create a project directory based on input filename
    StringRef baseName = sys::path::filename(inputFile);
    actualTmpDir = baseName;
    actualTmpDir += ".prj";
}
```

**When `--tmpdir` is specified:** Uses the provided directory (properly isolated in tests via `%t.dev1`, `%t.dev2`, etc.)
**When `--tmpdir` is NOT specified:** Creates `<filename>.prj` in the CWD

### 2. File Generation Analysis

The `cpp_multi_device_sequence.mlir` test has 5 RUN lines that execute aiecc:

| RUN Line | Arguments | tmpdir | Output Directory |
|----------|-----------|--------|------------------|
| 1 | `--device-name=device1` | `%t.dev1` | Isolated |
| 2 | `--device-name=device2` | `%t.dev2` | Isolated |
| 3 | `--device-name=device1 --sequence-name=seq_a --aie-generate-npu-insts` | `%t.seq_a` | Isolated |
| 4 | (no device filter) | `%t.all` | Isolated |
| 5 | `--device-name=device1 --aie-generate-ctrlpkt` (with overlay input) | `%t.ctrlpkt` | Isolated |

#### Files Generated per RUN:

**Inside tmpdir (all isolated):**
- `input_with_addresses.mlir` - Intermediate MLIR after address assignment

**Potential issue with --aie-generate-npu-insts (RUN line 3):**

When `--aie-generate-npu-insts` is used WITHOUT an explicit output path, the code at [tools/aiecc/aiecc.cpp:3542-3547](tools/aiecc/aiecc.cpp#L3542-L3547) writes to CWD:

```cpp
SmallString<128> outputPath;
if (generateNpuInsts) {
    outputPath = outputFileName;  // Written to CWD!
} else {
    outputPath = tmpDirName;
    sys::path::append(outputPath, outputFileName);
}
```

However, in this test, since `--tmpdir=%t.seq_a` is specified, the test infrastructure should handle this properly. The issue is that the default filename is `{deviceName}_{sequenceName}.bin` which becomes `device1_seq_a.bin`.

### 3. Tests Using --aie-generate-npu-insts

The following tests use `--aie-generate-npu-insts` and could potentially conflict:

1. **[test/aiecc/cpp_basic.mlir](test/aiecc/cpp_basic.mlir)**
   - Device: unnamed (defaults to "main")
   - Sequence: unnamed (defaults to "main")
   - Expected output: `main_main.bin`
   - Uses `%s` (no tmpdir specified)

2. **[test/aiecc/cpp_multi_device_sequence.mlir](test/aiecc/cpp_multi_device_sequence.mlir)**
   - Device: device1, Sequence: seq_a
   - Expected output: `device1_seq_a.bin`
   - Uses `--tmpdir=%t.seq_a` (properly isolated)

3. **[test/aiecc/cpp_npu_and_xclbin.mlir](test/aiecc/cpp_npu_and_xclbin.mlir)** (needs verification)

4. **[test/aiecc/only_insts.mlir](test/aiecc/only_insts.mlir)**
   - Uses `--npu-insts-name=my_insts.bin`
   - Custom filename (properly isolated)

5. **[test/aiecc/simple_xclbin.mlir](test/aiecc/simple_xclbin.mlir)** (needs verification)

### 4. Potential Race Conditions

#### Within cpp_multi_device_sequence.mlir

While each RUN line uses different tmpdir values, lit may execute these RUN lines in parallel. The current setup appears safe because:
- Each RUN uses a unique tmpdir (`%t.dev1`, `%t.dev2`, `%t.seq_a`, `%t.all`, `%t.ctrlpkt`)
- No files are written outside these directories

#### Between Different Tests

Tests that don't specify `--tmpdir` could theoretically conflict if:
1. They have the same base filename AND
2. They run in parallel AND
3. They're executed from the same working directory

However, lit typically runs each test in its own `Output/` subdirectory, so this is unlikely.

## Conclusions

**CRITICAL BUG FOUND: Output files are written to CWD instead of tmpdir!**

Even though `cpp_multi_device_sequence.mlir` specifies `--tmpdir=%t.*` for isolation, the compiler has bugs that write output `.bin` files to the current working directory instead of the tmpdir:

### Bug #1: NPU Instructions ([aiecc.cpp:3542-3543](tools/aiecc/aiecc.cpp#L3542-L3543))
```cpp
if (generateNpuInsts) {
    outputPath = outputFileName;  // BUG: Writes to CWD!
```
When `--aie-generate-npu-insts` is used, the output path is set to just the filename, writing to CWD instead of tmpdir.

**Affected RUN line:** Line 19 (`--device-name=device1 --sequence-name=seq_a --aie-generate-npu-insts --tmpdir=%t.seq_a`)
**File written to CWD:** `device1_seq_a.bin`

### Bug #2: Control Packet Binary ([aiecc.cpp:3706-3712](tools/aiecc/aiecc.cpp#L3706-L3712))
```cpp
std::string ctrlPktBinFileName = formatString(ctrlPktName, devName);
SmallString<128> ctrlPktBinPath;
if (sys::path::is_absolute(ctrlPktBinFileName)) {
    ctrlPktBinPath = ctrlPktBinFileName;
} else {
    ctrlPktBinPath = ctrlPktBinFileName;  // BUG: Should prepend tmpDirName!
}
```

**Affected RUN line:** Line 21 (`--device-name=device1 --aie-generate-ctrlpkt --tmpdir=%t.ctrlpkt`)
**File written to CWD:** `device1_ctrlpkt.bin`

### Bug #3: DMA Sequence Binary ([aiecc.cpp:3768-3773](tools/aiecc/aiecc.cpp#L3768-L3773))
```cpp
SmallString<128> dmaSeqBinPath;
if (sys::path::is_absolute(dmaSeqBinFileName)) {
    dmaSeqBinPath = dmaSeqBinFileName;
} else {
    dmaSeqBinPath = dmaSeqBinFileName;  // BUG: Should prepend tmpDirName!
}
```

**Affected RUN line:** Line 21 (same as Bug #2)
**File written to CWD:** `device1_ctrlpkt_dma_seq.bin`

### Evidence

In `build/test/aiecc/` (NOT in `Output/`), we find:
```
-rw-r--r-- 1 androsti asic  160 Mar 12 12:26 device1_ctrlpkt.bin
-rw-r--r-- 1 androsti asic 2596 Mar 12 12:26 device1_ctrlpkt_dma_seq.bin
-rw-r--r-- 1 androsti asic  328 Mar 12 12:26 device1_seq_a.bin
```

These files are written to the test's CWD (which lit sets to the source directory), creating a **race condition when multiple tests run in parallel**.

**Potential conflicts:**
- ❌ RUN lines within `cpp_multi_device_sequence.mlir` that use the same device name will overwrite each other's files
- ❌ Different tests that generate files with the same device/sequence names will conflict
- ❌ Tests without `--tmpdir` that use `--aie-generate-npu-insts` write output files to CWD

## Recommendations

### IMMEDIATE FIX REQUIRED

**Fix the bugs in aiecc.cpp to respect tmpdir for all output files:**

1. **Fix NPU instructions output** ([aiecc.cpp:3542-3547](tools/aiecc/aiecc.cpp#L3542-L3547)):
   ```cpp
   SmallString<128> outputPath;
   if (generateNpuInsts) {
       // FIXED: Write to tmpdir, not CWD
       if (sys::path::is_absolute(outputFileName)) {
           outputPath = outputFileName;
       } else {
           outputPath = tmpDirName;
           sys::path::append(outputPath, outputFileName);
       }
   } else {
       outputPath = tmpDirName;
       sys::path::append(outputPath, outputFileName);
   }
   ```

2. **Fix control packet binary output** ([aiecc.cpp:3706-3712](tools/aiecc/aiecc.cpp#L3706-L3712)):
   ```cpp
   SmallString<128> ctrlPktBinPath;
   if (sys::path::is_absolute(ctrlPktBinFileName)) {
       ctrlPktBinPath = ctrlPktBinFileName;
   } else {
       ctrlPktBinPath = tmpDirName;  // FIXED: Prepend tmpDirName
       sys::path::append(ctrlPktBinPath, ctrlPktBinFileName);
   }
   ```

3. **Fix DMA sequence binary output** ([aiecc.cpp:3768-3773](tools/aiecc/aiecc.cpp#L3768-L3773)):
   ```cpp
   SmallString<128> dmaSeqBinPath;
   if (sys::path::is_absolute(dmaSeqBinFileName)) {
       dmaSeqBinPath = dmaSeqBinFileName;
   } else {
       dmaSeqBinPath = tmpDirName;  // FIXED: Prepend tmpDirName
       sys::path::append(dmaSeqBinPath, dmaSeqBinFileName);
   }
   ```

### Additional Recommendations

4. **Verify the fix:** After applying the fixes, confirm that all `.bin` files are created inside `Output/cpp_multi_device_sequence.mlir.tmp.*/` directories

5. **Run parallel tests:** Test with `ninja check-aie -j8` to verify the spurious failures are eliminated

6. **Add regression test:** Consider adding a test that verifies output files are written to tmpdir when specified

## Next Steps

1. Run a single iteration of the test with verbose lit output to see actual paths
2. Check if the spurious failure still occurs when running the test in isolation
3. If the failure only occurs during parallel test execution, investigate compiler thread-safety or resource contention issues
