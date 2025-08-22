// Next steps
// 1. Merge new syntax for configure/run ops and devices with names
// 2. Allow runtime sequence argument "slices"
// 3. Allow parametrizable designs
// 4. Allow specifying designs by reference to an xclbin
// 5. Python (IRON) bindings

module {
  // Parametrizable designs
  aie.device(npu2) @mm(%M: ui32, %K: ui32, %N: ui32) {
    // ...
  }

  aie.device(npu2) @main {
    aiex.runtime_sequence @sequence (%buf_inout: memref<4x4096x4096xi32>) {
      // Argument slices
      %a = memref.subview %buf_inout[0][1024x1024][1]
      %b = memref.subview %buf_ionut[1024x1024][1024x1024][1]
      %c = memref.subview %buf_ionut[2x1024x1024][1024x1024][1]
      
      // Parametrizable design
      %config_mm_1024_1024_1024 = aiex.configure @mm(1024, 1024, 1024) : (ui32, ui32, ui32)
      aiex.run %config_mm_1024_1024_1024 -> @sequence (%a, %b, %c) : (memref<1024x1024xi32>, memref<1024x1024x1024xi32>, memref<1024x1024x1024xi32>)
      %config_mm_4096_4096_4096 = aiex.configure @mm(4096, 4096, 4096) : (ui32, ui32, ui32)
      aiex.run %config_mm_4096_4096_4096 -> @sequence (%a, %b, %c) : (memref<4096x4096xi32>, memref<4096x4096x4096xi32>, memref<4096x4096x4096xi32>)
    }
  }

  // Allow specifying devices by reference to external file
  aie.device(npu2) @mm {} {xclbin = "mm.xclbin"}
}

