module {
  aie.device(npu1_1col) @xclbin_device {
    memref.global "private" constant @config_blockwrite_data_2 : memref<8xi32> = dense<[64, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_1 : memref<8xi32> = dense<[64, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
    memref.global "private" constant @config_blockwrite_data_0 : memref<8xi32> = dense<[1, 2, 3, 4, 5, 6, 7, 8]>
    aie.runtime_sequence @configure() {
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219536 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 2219544 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 2219520 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 2219528 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.write32 {address = 2098336 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2228224 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2229120 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2229152 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2229184 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2229360 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219536 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219544 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219520 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219528 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224128 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224144 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224160 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224176 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224192 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224208 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224224 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224240 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224256 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224272 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224288 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224304 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224320 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224336 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224352 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224368 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224128 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224144 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224160 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224176 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2215936 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2216000 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2216032 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2215968 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2219524 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219520 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 2219540 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219536 : ui32, mask = 0 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 258100 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258324 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258064 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258364 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258056 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258304 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258560 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769520 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769760 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769504 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769784 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2355204 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2355480 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2355224 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2355460 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 126976 : ui32, mask = 3072 : ui32, value = 1024 : ui32}
      aiex.npu.maskwrite32 {address = 126980 : ui32, mask = 48 : ui32, value = 16 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 1 : ui32}
      %0 = memref.get_global @config_blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 2098304 : ui32} : memref<8xi32>
      aiex.npu.write32 {address = 2098320 : ui32, value = 42 : ui32}
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 64 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(0, 0, MM2S : 0) {bd_id = 0 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 64 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(0, 0, S2MM : 0) {bd_id = 1 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aie.end
    }
  }
}
