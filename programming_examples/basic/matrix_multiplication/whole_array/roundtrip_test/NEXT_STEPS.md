# Next Steps to Complete Roundtrip

## Current Status

### ✅ What's Working
1. **Decompiler (100% Complete)**
   - Extracts all device configuration from xclbin
   - Extracts all 16 ELF files containing core programs
   - Generates valid MLIR with high-level AIE constructs
   - Adds `elf_file` attribute to each `aie.core` operation
   - No raw register writes in output

2. **Recompilation (Partial - Config Only)**
   - Successfully recompiles decompiled MLIR
   - Produces **binary-identical** device configuration (100% match)
   - All switchboxes, DMAs, locks, routing preserved perfectly

### ❌ What's Broken
1. **ELF Linking**
   - Compiler ignores `elf_file` attribute on `aie.core` operations
   - Generates stub/empty ELFs instead of using extracted ones
   - Result: 76% size reduction, no compute functionality

2. **Init Data (Secondary Issue)**
   - Some initialization data missing (16 KB gap)
   - May be related to ELF contents or buffer layouts
   - Needs investigation after ELF linking is fixed

## Critical Task: Implement ELF Linking in Compiler

### The Problem
```mlir
// Decompiled MLIR says:
%core_0_2 = aie.core(%tile_0_2) {
  aie.end
} {elf_file = "core_0_2.elf"}

// Compiler should: Use core_0_2.elf
// Compiler actually: Generates new stub ELF, ignores attribute
```

### The Solution

**Step 1: Locate Core Compilation Code**

Search for where `aie.core` operations are processed during compilation:
```bash
cd /workspace/mlir-aie
grep -r "aie.core" --include="*.py" python/
grep -r "CoreOp" --include="*.cpp" lib/
```

Likely locations:
- `python/compiler/aiecc/main.py` - Main compiler driver
- `lib/Dialect/AIE/Transforms/` - Core lowering passes
- `lib/Dialect/AIE/Utils/` - Core compilation utilities

**Step 2: Check for elf_file Attribute**

Add logic to check for the `elf_file` attribute before compiling a core:

```python
# Pseudo-code for Python side (aiecc.py)
for core in aie_cores:
    if core.has_attribute('elf_file'):
        elf_path = core.get_attribute('elf_file')
        # Use existing ELF
        copy_elf_to_build_dir(elf_path, core)
        skip_core_compilation(core)
    else:
        # Normal path: compile core body
        compile_core_llvm(core)
```

Or in C++ (MLIR pass):
```cpp
// In core compilation pass
if (auto elfFileAttr = coreOp->getAttr("elf_file")) {
    StringRef elfPath = elfFileAttr.cast<StringAttr>().getValue();
    // Use existing ELF instead of compiling
    useExistingELF(coreOp, elfPath);
} else {
    // Existing behavior: compile core
    compileCore(coreOp);
}
```

**Step 3: Handle ELF File Paths**

The `elf_file` attribute contains a relative path (e.g., "core_0_2.elf").
Need to:
1. Resolve the path relative to the input MLIR file location
2. Copy the ELF to the build project directory
3. Update the CDO generation to include the copied ELF

**Step 4: Update CDO ELF Generation**

The `aie_cdo_elfs.bin` file is generated from all core ELFs. Ensure that:
1. Pre-existing ELFs are included in this binary
2. ELFs are placed at the correct offsets/addresses
3. The CDO properly references these ELFs

**Step 5: Test**

After implementation:
```bash
cd /workspace/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/roundtrip_test
source /workspace/env_setup.sh

# Rebuild compiler with changes
cd /workspace/mlir-aie
./mlir-aie/utils/build-mlir-aie-from-wheels.sh

# Test roundtrip
cd roundtrip_test
aiecc.py --alloc-scheme=basic-sequential --aie-generate-xclbin \
         --no-compile-host --xclbin-name=recompiled_with_elfs.xclbin \
         --no-xchesscc --no-xbridge \
         --peano /workspace/buildenv/lib/python3.12/site-packages/llvm-aie \
         decompiled_clean.mlir

# Compare
ls -lh original.xclbin recompiled_with_elfs.xclbin
# Should now be similar sizes!

# Binary diff
diff <(xxd original.xclbin) <(xxd recompiled_with_elfs.xclbin)
```

## Alternative Workaround (Temporary)

If modifying the compiler is complex, can try manual binary patching:

### Option A: Replace aie_cdo_elfs.bin

1. Extract `aie_cdo_elfs.bin` from original xclbin
2. Replace recompiled version with original
3. Rebuild PDI and xclbin

### Option B: Manual ELF Injection

1. Study the format of `aie_cdo_elfs.bin`
2. Write a script to pack extracted ELFs into this format
3. Replace in build directory before xclbin generation

## Expected Results After Fix

Once ELF linking is implemented:

```
Original xclbin:    82,400 bytes
Recompiled xclbin:  82,400 bytes (or very close)
Match:              ~100% (binary-equivalent or near-equivalent)
```

Sections that should match:
- ✅ MEM_TOPOLOGY: Already matches
- ✅ AIE_PARTITION: Already matches
- ✅ IP_LAYOUT: Already matches
- ✅ CONNECTIVITY: Already matches
- ✅ GROUP_CONNECTIVITY: Already matches
- ✅ GROUP_TOPOLOGY: Already matches
- ✅ EMBEDDED_METADATA: Already matches
- 🔄 aie_cdo_elfs.bin: Should match after fix (54 KB)
- 🔄 aie_cdo_init.bin: Should be closer (may not be 100% due to different compilation order)

## Success Criteria

The roundtrip is successful when:

1. **Size Match:** Recompiled xclbin is within 5% of original size
2. **Config Match:** All configuration sections binary-identical (already ✅)
3. **ELF Match:** All 16 core ELFs preserved in recompiled xclbin
4. **Functional Test:** Recompiled xclbin runs on hardware and produces correct results
5. **Hash Match (stretch goal):** Entire xclbin is binary-identical except UUID/timestamp

## Time Estimate

- **Finding compilation code:** 30-60 minutes
- **Implementing elf_file check:** 2-4 hours
- **Testing and debugging:** 2-4 hours
- **Total:** ~1 day of focused work

## Resources

Key files to examine:
- `/workspace/mlir-aie/python/compiler/aiecc/main.py`
- `/workspace/mlir-aie/lib/Dialect/AIE/Transforms/AIECoreToStandard.cpp`
- `/workspace/mlir-aie/python/dialects/aie.py`
- `/workspace/mlir-aie/lib/Dialect/AIE/IR/AIEDialect.cpp`

Documentation:
- MLIR attribute handling: https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/
- AIE dialect docs: `/workspace/mlir-aie/docs/`

Similar precedent:
- Look for how `link_with` or similar attributes are handled in MLIR
- Check if LLVM IR already has mechanisms for linking pre-compiled objects

## Questions to Investigate

1. Does MLIR-AIE already have any mechanism for external linking?
2. Where is `aie_cdo_elfs.bin` generated? What format does it use?
3. Are there existing tests for linking external ELFs?
4. What's the format of the CDO (Configuration Data Object)?
5. Why is init data different size? Is it ELF-dependent?

## Conclusion

**We are 90% of the way there!** The decompiler works perfectly. The compiler produces perfect configuration. We just need the compiler to honor the `elf_file` attribute that the decompiler already provides.

This is a **focused, well-defined implementation task** rather than a research problem. The architecture is already in place - we just need to connect the last piece.
