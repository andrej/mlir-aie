module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @objFifo_in0(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @objFifo_out0(%tile_0_0, S2MM, 0)
    
    memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[64, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
    
    aie.runtime_sequence(%arg0: memref<64xi32>, %arg1: memref<32xi32>, %arg2: memref<64xi32>) {
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119316 : ui32, value = 2147483648 : ui32}
      
      %1 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%1) {address = 118816 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119300 : ui32, value = 2147483649 : ui32}
      
      aie.end
    }
  }
}
