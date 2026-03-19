// Test file to verify SSA value copying during aiex.run inlining
// This tests that tiles referenced in the callee runtime sequence are correctly
// copied to the caller device

module {
    aie.device(npu2) @callee_device {
        %tile_0_2 = aie.tile(0, 2)
        
        aiex.runtime_sequence @callee_sequence(%t : index) {
            aiex.npu.write32 { column = 0 : i32, row = 2 : i32, address = 0x1234 : ui32, value = 42 : ui32 }
        }
    }
    
    aie.device(npu2) @caller_device {
        aiex.runtime_sequence @caller_sequence(%t : index) {
            aiex.configure @callee_device {
                aiex.run @callee_sequence(%t) : (index)
            }
        }
    }
}
