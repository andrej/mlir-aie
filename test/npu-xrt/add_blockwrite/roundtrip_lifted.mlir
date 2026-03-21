module {
  aie.device(npu1_1col) {
    memref.global "private" constant @config_blockwrite_data_0 : memref<8xi32> = dense<[1, 2, 3, 4, 5, 6, 7, 8]>
    memref.global "private" constant @config_blockwrite_data_1 : memref<8xi32> = dense<[64, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_2 : memref<8xi32> = dense<[64, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
    %0 = memref.get_global @config_blockwrite_data_0 : memref<8xi32>
    aiex.npu.blockwrite(%0) {address = 2098304 : ui32} : memref<8xi32>
    aiex.npu.write32 {address = 2098320 : ui32, value = 42 : ui32}
    %1 = memref.get_global @config_blockwrite_data_1 : memref<8xi32>
    aiex.npu.blockwrite(%1) {address = 118784 : ui32} : memref<8xi32>
    aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
    aiex.npu.maskwrite32 {address = 119312 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
    aiex.npu.write32 {address = 119316 : ui32, value = 2147483648 : ui32}
    %2 = memref.get_global @config_blockwrite_data_2 : memref<8xi32>
    aiex.npu.blockwrite(%2) {address = 118816 : ui32} : memref<8xi32>
    aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
    aiex.npu.maskwrite32 {address = 119296 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
    aiex.npu.write32 {address = 119300 : ui32, value = 2147483649 : ui32}
    aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
    aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
  }
}

