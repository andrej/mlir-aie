module {
  aie.device(npu1_1col) {
    memref.global "private" constant @data1 : memref<2xi32> = dense<[10, 20]>
    memref.global "private" constant @data2 : memref<2xi32> = dense<[30, 40]>
    
    aiex.runtime_sequence @test_seq() {
      %0 = memref.get_global @data1 : memref<2xi32>
      %1 = memref.get_global @data2 : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 2000 : ui32} : memref<2xi32>
      aiex.npu.blockwrite(%1) {address = 2008 : ui32} : memref<2xi32>
    }
  }
}
