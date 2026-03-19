// RUN: aie-opt --aie-expand-load-pdi %s | FileCheck %s

// This test demonstrates the aie-expand-load-pdi pass

module {
  // CHECK: aie.device(npu1_1col) @empty
  // CHECK-NEXT: }

  // A simple device with some configuration
  aie.device(npu1_1col) @my_device {
    %tile00 = aie.tile(0, 0)
    %tile01 = aie.tile(0, 1)
    %tile02 = aie.tile(0, 2)
    
    %core02 = aie.core(%tile02) {
      aie.end
    }
  }

  // Device with runtime sequence that loads the device configuration
  aie.device(npu1_1col) @my_host {
    // Runtime sequence that loads the device configuration
    aiex.runtime_sequence @test_seq() {
      // CHECK: aiex.runtime_sequence @test_seq()
      // CHECK: aiex.npu.load_pdi {device_ref = @empty}
      // CHECK: aiex.npu.write32
      aiex.npu.load_pdi {device_ref = @my_device}
    }
  }
}
