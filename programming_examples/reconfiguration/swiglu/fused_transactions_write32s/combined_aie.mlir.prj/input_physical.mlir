module {
  aie.device(npu2) @empty_xx {
  }
  aie.device(npu2) {
    aie.runtime_sequence(%arg0: memref<50368512xbf16>) {
      aiex.npu.load_pdi {ref = @empty_xx}
      aiex.configure @gemv_1 {
        %subview = memref.subview %arg0[2048] [16777216] [1] : memref<50368512xbf16> to memref<16777216xbf16, strided<[1], offset: 2048>>
        %reinterpret_cast = memref.reinterpret_cast %subview to offset: [0], sizes: [16777216], strides: [1] : memref<16777216xbf16, strided<[1], offset: 2048>> to memref<16777216xbf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [2048], strides: [1] : memref<50368512xbf16> to memref<2048xbf16>
        %subview_1 = memref.subview %arg0[50333696] [8192] [1] : memref<50368512xbf16> to memref<8192xbf16, strided<[1], offset: 50333696>>
        %reinterpret_cast_2 = memref.reinterpret_cast %subview_1 to offset: [0], sizes: [8192], strides: [1] : memref<8192xbf16, strided<[1], offset: 50333696>> to memref<8192xbf16>
        aiex.run @sequence(%reinterpret_cast, %reinterpret_cast_0, %reinterpret_cast_2) : (memref<16777216xbf16>, memref<2048xbf16>, memref<8192xbf16>)
        %subview_3 = memref.subview %arg0[16779264] [16777216] [1] : memref<50368512xbf16> to memref<16777216xbf16, strided<[1], offset: 16779264>>
        %reinterpret_cast_4 = memref.reinterpret_cast %subview_3 to offset: [0], sizes: [16777216], strides: [1] : memref<16777216xbf16, strided<[1], offset: 16779264>> to memref<16777216xbf16>
        %subview_5 = memref.subview %arg0[50350080] [8192] [1] : memref<50368512xbf16> to memref<8192xbf16, strided<[1], offset: 50350080>>
        %reinterpret_cast_6 = memref.reinterpret_cast %subview_5 to offset: [0], sizes: [8192], strides: [1] : memref<8192xbf16, strided<[1], offset: 50350080>> to memref<8192xbf16>
        aiex.run @sequence(%reinterpret_cast_4, %reinterpret_cast_0, %reinterpret_cast_6) : (memref<16777216xbf16>, memref<2048xbf16>, memref<8192xbf16>)
      }
      aiex.configure @silu {
        %subview = memref.subview %arg0[50333696] [8192] [1] : memref<50368512xbf16> to memref<8192xbf16, strided<[1], offset: 50333696>>
        %reinterpret_cast = memref.reinterpret_cast %subview to offset: [0], sizes: [8192], strides: [1] : memref<8192xbf16, strided<[1], offset: 50333696>> to memref<8192xbf16>
        %subview_0 = memref.subview %arg0[50341888] [8192] [1] : memref<50368512xbf16> to memref<8192xbf16, strided<[1], offset: 50341888>>
        %reinterpret_cast_1 = memref.reinterpret_cast %subview_0 to offset: [0], sizes: [8192], strides: [1] : memref<8192xbf16, strided<[1], offset: 50341888>> to memref<8192xbf16>
        aiex.run @sequence(%reinterpret_cast, %reinterpret_cast_1) : (memref<8192xbf16>, memref<8192xbf16>)
      }
      aiex.configure @eltwise_mul {
        %subview = memref.subview %arg0[50341888] [8192] [1] : memref<50368512xbf16> to memref<8192xbf16, strided<[1], offset: 50341888>>
        %reinterpret_cast = memref.reinterpret_cast %subview to offset: [0], sizes: [8192], strides: [1] : memref<8192xbf16, strided<[1], offset: 50341888>> to memref<8192xbf16>
        %subview_0 = memref.subview %arg0[50350080] [8192] [1] : memref<50368512xbf16> to memref<8192xbf16, strided<[1], offset: 50350080>>
        %reinterpret_cast_1 = memref.reinterpret_cast %subview_0 to offset: [0], sizes: [8192], strides: [1] : memref<8192xbf16, strided<[1], offset: 50350080>> to memref<8192xbf16>
        %subview_2 = memref.subview %arg0[50358272] [8192] [1] : memref<50368512xbf16> to memref<8192xbf16, strided<[1], offset: 50358272>>
        %reinterpret_cast_3 = memref.reinterpret_cast %subview_2 to offset: [0], sizes: [8192], strides: [1] : memref<8192xbf16, strided<[1], offset: 50358272>> to memref<8192xbf16>
        aiex.run @sequence(%reinterpret_cast, %reinterpret_cast_1, %reinterpret_cast_3) : (memref<8192xbf16>, memref<8192xbf16>, memref<8192xbf16>)
      }
      aiex.configure @gemv_2 {
        %subview = memref.subview %arg0[33556480] [16777216] [1] : memref<50368512xbf16> to memref<16777216xbf16, strided<[1], offset: 33556480>>
        %reinterpret_cast = memref.reinterpret_cast %subview to offset: [0], sizes: [16777216], strides: [1] : memref<16777216xbf16, strided<[1], offset: 33556480>> to memref<16777216xbf16>
        %subview_0 = memref.subview %arg0[50358272] [8192] [1] : memref<50368512xbf16> to memref<8192xbf16, strided<[1], offset: 50358272>>
        %reinterpret_cast_1 = memref.reinterpret_cast %subview_0 to offset: [0], sizes: [8192], strides: [1] : memref<8192xbf16, strided<[1], offset: 50358272>> to memref<8192xbf16>
        %subview_2 = memref.subview %arg0[50366464] [2048] [1] : memref<50368512xbf16> to memref<2048xbf16, strided<[1], offset: 50366464>>
        %reinterpret_cast_3 = memref.reinterpret_cast %subview_2 to offset: [0], sizes: [2048], strides: [1] : memref<2048xbf16, strided<[1], offset: 50366464>> to memref<2048xbf16>
        aiex.run @sequence(%reinterpret_cast, %reinterpret_cast_1, %reinterpret_cast_3) : (memref<16777216xbf16>, memref<8192xbf16>, memref<2048xbf16>)
      }
    }
  }
  aie.device(npu2) @gemv_1 {
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_4_0 = aie.tile(4, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_5_0 = aie.tile(5, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_6_0 = aie.tile(6, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_7_0 = aie.tile(7, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %C_L1L3_7_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 6) {init = 0 : i32, sym_name = "C_L1L3_7_cons_prod_lock_0"}
    %C_L1L3_7_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 7) {init = 0 : i32, sym_name = "C_L1L3_7_cons_cons_lock_0"}
    %C_L1L3_7_buff_0 = aie.buffer(%tile_1_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "C_L1L3_7_buff_0"} : memref<1024xbf16> 
    %C_L1L3_7_prod_lock_0 = aie.lock(%tile_1_5, 4) {init = 1 : i32, sym_name = "C_L1L3_7_prod_lock_0"}
    %C_L1L3_7_cons_lock_0 = aie.lock(%tile_1_5, 5) {init = 0 : i32, sym_name = "C_L1L3_7_cons_lock_0"}
    %C_L1L3_6_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 4) {init = 0 : i32, sym_name = "C_L1L3_6_cons_prod_lock_0"}
    %C_L1L3_6_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 5) {init = 0 : i32, sym_name = "C_L1L3_6_cons_cons_lock_0"}
    %C_L1L3_6_buff_0 = aie.buffer(%tile_1_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "C_L1L3_6_buff_0"} : memref<1024xbf16> 
    %C_L1L3_6_prod_lock_0 = aie.lock(%tile_1_4, 4) {init = 1 : i32, sym_name = "C_L1L3_6_prod_lock_0"}
    %C_L1L3_6_cons_lock_0 = aie.lock(%tile_1_4, 5) {init = 0 : i32, sym_name = "C_L1L3_6_cons_lock_0"}
    %C_L1L3_5_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 6) {init = 0 : i32, sym_name = "C_L1L3_5_cons_prod_lock_0"}
    %C_L1L3_5_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 7) {init = 0 : i32, sym_name = "C_L1L3_5_cons_cons_lock_0"}
    %C_L1L3_5_buff_0 = aie.buffer(%tile_1_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "C_L1L3_5_buff_0"} : memref<1024xbf16> 
    %C_L1L3_5_prod_lock_0 = aie.lock(%tile_1_3, 4) {init = 1 : i32, sym_name = "C_L1L3_5_prod_lock_0"}
    %C_L1L3_5_cons_lock_0 = aie.lock(%tile_1_3, 5) {init = 0 : i32, sym_name = "C_L1L3_5_cons_lock_0"}
    %C_L1L3_4_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 4) {init = 0 : i32, sym_name = "C_L1L3_4_cons_prod_lock_0"}
    %C_L1L3_4_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 5) {init = 0 : i32, sym_name = "C_L1L3_4_cons_cons_lock_0"}
    %C_L1L3_4_buff_0 = aie.buffer(%tile_1_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "C_L1L3_4_buff_0"} : memref<1024xbf16> 
    %C_L1L3_4_prod_lock_0 = aie.lock(%tile_1_2, 4) {init = 1 : i32, sym_name = "C_L1L3_4_prod_lock_0"}
    %C_L1L3_4_cons_lock_0 = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "C_L1L3_4_cons_lock_0"}
    %C_L1L3_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 6) {init = 0 : i32, sym_name = "C_L1L3_3_cons_prod_lock_0"}
    %C_L1L3_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 7) {init = 0 : i32, sym_name = "C_L1L3_3_cons_cons_lock_0"}
    %C_L1L3_3_buff_0 = aie.buffer(%tile_0_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "C_L1L3_3_buff_0"} : memref<1024xbf16> 
    %C_L1L3_3_prod_lock_0 = aie.lock(%tile_0_5, 4) {init = 1 : i32, sym_name = "C_L1L3_3_prod_lock_0"}
    %C_L1L3_3_cons_lock_0 = aie.lock(%tile_0_5, 5) {init = 0 : i32, sym_name = "C_L1L3_3_cons_lock_0"}
    %C_L1L3_2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 4) {init = 0 : i32, sym_name = "C_L1L3_2_cons_prod_lock_0"}
    %C_L1L3_2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 5) {init = 0 : i32, sym_name = "C_L1L3_2_cons_cons_lock_0"}
    %C_L1L3_2_buff_0 = aie.buffer(%tile_0_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "C_L1L3_2_buff_0"} : memref<1024xbf16> 
    %C_L1L3_2_prod_lock_0 = aie.lock(%tile_0_4, 4) {init = 1 : i32, sym_name = "C_L1L3_2_prod_lock_0"}
    %C_L1L3_2_cons_lock_0 = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "C_L1L3_2_cons_lock_0"}
    %C_L1L3_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 6) {init = 0 : i32, sym_name = "C_L1L3_1_cons_prod_lock_0"}
    %C_L1L3_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 7) {init = 0 : i32, sym_name = "C_L1L3_1_cons_cons_lock_0"}
    %C_L1L3_1_buff_0 = aie.buffer(%tile_0_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "C_L1L3_1_buff_0"} : memref<1024xbf16> 
    %C_L1L3_1_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 1 : i32, sym_name = "C_L1L3_1_prod_lock_0"}
    %C_L1L3_1_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "C_L1L3_1_cons_lock_0"}
    %C_L1L3_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "C_L1L3_0_cons_prod_lock_0"}
    %C_L1L3_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "C_L1L3_0_cons_cons_lock_0"}
    %C_L1L3_0_buff_0 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "C_L1L3_0_buff_0"} : memref<1024xbf16> 
    %C_L1L3_0_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 1 : i32, sym_name = "C_L1L3_0_prod_lock_0"}
    %C_L1L3_0_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "C_L1L3_0_cons_lock_0"}
    %B_L3L1_7_cons_buff_0 = aie.buffer(%tile_1_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "B_L3L1_7_cons_buff_0"} : memref<2048xbf16> 
    %B_L3L1_7_cons_prod_lock_0 = aie.lock(%tile_1_5, 2) {init = 1 : i32, sym_name = "B_L3L1_7_cons_prod_lock_0"}
    %B_L3L1_7_cons_cons_lock_0 = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "B_L3L1_7_cons_cons_lock_0"}
    %B_L3L1_7_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 2) {init = 0 : i32, sym_name = "B_L3L1_7_prod_lock_0"}
    %B_L3L1_7_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 3) {init = 0 : i32, sym_name = "B_L3L1_7_cons_lock_0"}
    %B_L3L1_6_cons_buff_0 = aie.buffer(%tile_1_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "B_L3L1_6_cons_buff_0"} : memref<2048xbf16> 
    %B_L3L1_6_cons_prod_lock_0 = aie.lock(%tile_1_4, 2) {init = 1 : i32, sym_name = "B_L3L1_6_cons_prod_lock_0"}
    %B_L3L1_6_cons_cons_lock_0 = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "B_L3L1_6_cons_cons_lock_0"}
    %B_L3L1_6_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 0) {init = 0 : i32, sym_name = "B_L3L1_6_prod_lock_0"}
    %B_L3L1_6_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 1) {init = 0 : i32, sym_name = "B_L3L1_6_cons_lock_0"}
    %B_L3L1_5_cons_buff_0 = aie.buffer(%tile_1_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "B_L3L1_5_cons_buff_0"} : memref<2048xbf16> 
    %B_L3L1_5_cons_prod_lock_0 = aie.lock(%tile_1_3, 2) {init = 1 : i32, sym_name = "B_L3L1_5_cons_prod_lock_0"}
    %B_L3L1_5_cons_cons_lock_0 = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "B_L3L1_5_cons_cons_lock_0"}
    %B_L3L1_5_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 2) {init = 0 : i32, sym_name = "B_L3L1_5_prod_lock_0"}
    %B_L3L1_5_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 3) {init = 0 : i32, sym_name = "B_L3L1_5_cons_lock_0"}
    %B_L3L1_4_cons_buff_0 = aie.buffer(%tile_1_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "B_L3L1_4_cons_buff_0"} : memref<2048xbf16> 
    %B_L3L1_4_cons_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 1 : i32, sym_name = "B_L3L1_4_cons_prod_lock_0"}
    %B_L3L1_4_cons_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "B_L3L1_4_cons_cons_lock_0"}
    %B_L3L1_4_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 0) {init = 0 : i32, sym_name = "B_L3L1_4_prod_lock_0"}
    %B_L3L1_4_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 1) {init = 0 : i32, sym_name = "B_L3L1_4_cons_lock_0"}
    %B_L3L1_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "B_L3L1_3_cons_buff_0"} : memref<2048xbf16> 
    %B_L3L1_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 1 : i32, sym_name = "B_L3L1_3_cons_prod_lock_0"}
    %B_L3L1_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "B_L3L1_3_cons_cons_lock_0"}
    %B_L3L1_3_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 2) {init = 0 : i32, sym_name = "B_L3L1_3_prod_lock_0"}
    %B_L3L1_3_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 3) {init = 0 : i32, sym_name = "B_L3L1_3_cons_lock_0"}
    %B_L3L1_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "B_L3L1_2_cons_buff_0"} : memref<2048xbf16> 
    %B_L3L1_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 1 : i32, sym_name = "B_L3L1_2_cons_prod_lock_0"}
    %B_L3L1_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "B_L3L1_2_cons_cons_lock_0"}
    %B_L3L1_2_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 0) {init = 0 : i32, sym_name = "B_L3L1_2_prod_lock_0"}
    %B_L3L1_2_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 1) {init = 0 : i32, sym_name = "B_L3L1_2_cons_lock_0"}
    %B_L3L1_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "B_L3L1_1_cons_buff_0"} : memref<2048xbf16> 
    %B_L3L1_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 1 : i32, sym_name = "B_L3L1_1_cons_prod_lock_0"}
    %B_L3L1_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "B_L3L1_1_cons_cons_lock_0"}
    %B_L3L1_1_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 2) {init = 0 : i32, sym_name = "B_L3L1_1_prod_lock_0"}
    %B_L3L1_1_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 3) {init = 0 : i32, sym_name = "B_L3L1_1_cons_lock_0"}
    %B_L3L1_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "B_L3L1_0_cons_buff_0"} : memref<2048xbf16> 
    %B_L3L1_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "B_L3L1_0_cons_prod_lock_0"}
    %B_L3L1_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "B_L3L1_0_cons_cons_lock_0"}
    %B_L3L1_0_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 0) {init = 0 : i32, sym_name = "B_L3L1_0_prod_lock_0"}
    %B_L3L1_0_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 1) {init = 0 : i32, sym_name = "B_L3L1_0_cons_lock_0"}
    %A_L3L1_7_cons_buff_0 = aie.buffer(%tile_1_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "A_L3L1_7_cons_buff_0"} : memref<2048xbf16> 
    %A_L3L1_7_cons_buff_1 = aie.buffer(%tile_1_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "A_L3L1_7_cons_buff_1"} : memref<2048xbf16> 
    %A_L3L1_7_cons_prod_lock_0 = aie.lock(%tile_1_5, 0) {init = 2 : i32, sym_name = "A_L3L1_7_cons_prod_lock_0"}
    %A_L3L1_7_cons_cons_lock_0 = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "A_L3L1_7_cons_cons_lock_0"}
    %A_L3L1_7_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "A_L3L1_7_prod_lock_0"}
    %A_L3L1_7_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "A_L3L1_7_cons_lock_0"}
    %A_L3L1_6_cons_buff_0 = aie.buffer(%tile_1_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "A_L3L1_6_cons_buff_0"} : memref<2048xbf16> 
    %A_L3L1_6_cons_buff_1 = aie.buffer(%tile_1_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "A_L3L1_6_cons_buff_1"} : memref<2048xbf16> 
    %A_L3L1_6_cons_prod_lock_0 = aie.lock(%tile_1_4, 0) {init = 2 : i32, sym_name = "A_L3L1_6_cons_prod_lock_0"}
    %A_L3L1_6_cons_cons_lock_0 = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "A_L3L1_6_cons_cons_lock_0"}
    %A_L3L1_6_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "A_L3L1_6_prod_lock_0"}
    %A_L3L1_6_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "A_L3L1_6_cons_lock_0"}
    %A_L3L1_5_cons_buff_0 = aie.buffer(%tile_1_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "A_L3L1_5_cons_buff_0"} : memref<2048xbf16> 
    %A_L3L1_5_cons_buff_1 = aie.buffer(%tile_1_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "A_L3L1_5_cons_buff_1"} : memref<2048xbf16> 
    %A_L3L1_5_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "A_L3L1_5_cons_prod_lock_0"}
    %A_L3L1_5_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "A_L3L1_5_cons_cons_lock_0"}
    %A_L3L1_5_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "A_L3L1_5_prod_lock_0"}
    %A_L3L1_5_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "A_L3L1_5_cons_lock_0"}
    %A_L3L1_4_cons_buff_0 = aie.buffer(%tile_1_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "A_L3L1_4_cons_buff_0"} : memref<2048xbf16> 
    %A_L3L1_4_cons_buff_1 = aie.buffer(%tile_1_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "A_L3L1_4_cons_buff_1"} : memref<2048xbf16> 
    %A_L3L1_4_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "A_L3L1_4_cons_prod_lock_0"}
    %A_L3L1_4_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "A_L3L1_4_cons_cons_lock_0"}
    %A_L3L1_4_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "A_L3L1_4_prod_lock_0"}
    %A_L3L1_4_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "A_L3L1_4_cons_lock_0"}
    %A_L3L1_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "A_L3L1_3_cons_buff_0"} : memref<2048xbf16> 
    %A_L3L1_3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "A_L3L1_3_cons_buff_1"} : memref<2048xbf16> 
    %A_L3L1_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "A_L3L1_3_cons_prod_lock_0"}
    %A_L3L1_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "A_L3L1_3_cons_cons_lock_0"}
    %A_L3L1_3_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "A_L3L1_3_prod_lock_0"}
    %A_L3L1_3_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "A_L3L1_3_cons_lock_0"}
    %A_L3L1_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "A_L3L1_2_cons_buff_0"} : memref<2048xbf16> 
    %A_L3L1_2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "A_L3L1_2_cons_buff_1"} : memref<2048xbf16> 
    %A_L3L1_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "A_L3L1_2_cons_prod_lock_0"}
    %A_L3L1_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "A_L3L1_2_cons_cons_lock_0"}
    %A_L3L1_2_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "A_L3L1_2_prod_lock_0"}
    %A_L3L1_2_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "A_L3L1_2_cons_lock_0"}
    %A_L3L1_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "A_L3L1_1_cons_buff_0"} : memref<2048xbf16> 
    %A_L3L1_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "A_L3L1_1_cons_buff_1"} : memref<2048xbf16> 
    %A_L3L1_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "A_L3L1_1_cons_prod_lock_0"}
    %A_L3L1_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "A_L3L1_1_cons_cons_lock_0"}
    %A_L3L1_1_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "A_L3L1_1_prod_lock_0"}
    %A_L3L1_1_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "A_L3L1_1_cons_lock_0"}
    %A_L3L1_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "A_L3L1_0_cons_buff_0"} : memref<2048xbf16> 
    %A_L3L1_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "A_L3L1_0_cons_buff_1"} : memref<2048xbf16> 
    %A_L3L1_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "A_L3L1_0_cons_prod_lock_0"}
    %A_L3L1_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "A_L3L1_0_cons_cons_lock_0"}
    %A_L3L1_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "A_L3L1_0_prod_lock_0"}
    %A_L3L1_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "A_L3L1_0_cons_lock_0"}
    func.func private @matvec_vectorized_bf16_bf16(i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>)
    %core_0_2 = aie.core(%tile_0_2) {
      %c2 = arith.constant 2 : index
      %c2048_i32 = arith.constant 2048 : i32
      %c1_i32 = arith.constant 1 : i32
      %c1024 = arith.constant 1024 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c1024 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = index.casts %4 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c2048_i32, %6, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %7 = arith.addi %4, %c1 : index
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = index.casts %7 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c2048_i32, %8, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %4, %c2 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      %10 = arith.addi %2, %c1 : index
      cf.br ^bb3(%10 : index)
    ^bb8:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "mv.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c2 = arith.constant 2 : index
      %c2048_i32 = arith.constant 2048 : i32
      %c1_i32 = arith.constant 1 : i32
      %c1024 = arith.constant 1024 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_1_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c1024 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L3L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = index.casts %4 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c2048_i32, %6, %A_L3L1_1_cons_buff_0, %B_L3L1_1_cons_buff_0, %C_L1L3_1_buff_0) : (i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>) -> ()
      aie.use_lock(%A_L3L1_1_cons_prod_lock_0, Release, 1)
      %7 = arith.addi %4, %c1 : index
      aie.use_lock(%A_L3L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = index.casts %7 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c2048_i32, %8, %A_L3L1_1_cons_buff_1, %B_L3L1_1_cons_buff_0, %C_L1L3_1_buff_0) : (i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>) -> ()
      aie.use_lock(%A_L3L1_1_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %4, %c2 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_1_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_1_cons_prod_lock_0, Release, 1)
      %10 = arith.addi %2, %c1 : index
      cf.br ^bb3(%10 : index)
    ^bb8:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "mv.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c2 = arith.constant 2 : index
      %c2048_i32 = arith.constant 2048 : i32
      %c1_i32 = arith.constant 1 : i32
      %c1024 = arith.constant 1024 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_2_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c1024 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L3L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = index.casts %4 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c2048_i32, %6, %A_L3L1_2_cons_buff_0, %B_L3L1_2_cons_buff_0, %C_L1L3_2_buff_0) : (i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>) -> ()
      aie.use_lock(%A_L3L1_2_cons_prod_lock_0, Release, 1)
      %7 = arith.addi %4, %c1 : index
      aie.use_lock(%A_L3L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = index.casts %7 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c2048_i32, %8, %A_L3L1_2_cons_buff_1, %B_L3L1_2_cons_buff_0, %C_L1L3_2_buff_0) : (i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>) -> ()
      aie.use_lock(%A_L3L1_2_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %4, %c2 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_2_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_2_cons_prod_lock_0, Release, 1)
      %10 = arith.addi %2, %c1 : index
      cf.br ^bb3(%10 : index)
    ^bb8:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "mv.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c2 = arith.constant 2 : index
      %c2048_i32 = arith.constant 2048 : i32
      %c1_i32 = arith.constant 1 : i32
      %c1024 = arith.constant 1024 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_3_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c1024 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L3L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = index.casts %4 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c2048_i32, %6, %A_L3L1_3_cons_buff_0, %B_L3L1_3_cons_buff_0, %C_L1L3_3_buff_0) : (i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>) -> ()
      aie.use_lock(%A_L3L1_3_cons_prod_lock_0, Release, 1)
      %7 = arith.addi %4, %c1 : index
      aie.use_lock(%A_L3L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = index.casts %7 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c2048_i32, %8, %A_L3L1_3_cons_buff_1, %B_L3L1_3_cons_buff_0, %C_L1L3_3_buff_0) : (i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>) -> ()
      aie.use_lock(%A_L3L1_3_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %4, %c2 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_3_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_3_cons_prod_lock_0, Release, 1)
      %10 = arith.addi %2, %c1 : index
      cf.br ^bb3(%10 : index)
    ^bb8:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "mv.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c2 = arith.constant 2 : index
      %c2048_i32 = arith.constant 2048 : i32
      %c1_i32 = arith.constant 1 : i32
      %c1024 = arith.constant 1024 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_4_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c1024 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L3L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = index.casts %4 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c2048_i32, %6, %A_L3L1_4_cons_buff_0, %B_L3L1_4_cons_buff_0, %C_L1L3_4_buff_0) : (i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>) -> ()
      aie.use_lock(%A_L3L1_4_cons_prod_lock_0, Release, 1)
      %7 = arith.addi %4, %c1 : index
      aie.use_lock(%A_L3L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = index.casts %7 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c2048_i32, %8, %A_L3L1_4_cons_buff_1, %B_L3L1_4_cons_buff_0, %C_L1L3_4_buff_0) : (i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>) -> ()
      aie.use_lock(%A_L3L1_4_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %4, %c2 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_4_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_4_cons_prod_lock_0, Release, 1)
      %10 = arith.addi %2, %c1 : index
      cf.br ^bb3(%10 : index)
    ^bb8:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "mv.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c2 = arith.constant 2 : index
      %c2048_i32 = arith.constant 2048 : i32
      %c1_i32 = arith.constant 1 : i32
      %c1024 = arith.constant 1024 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_5_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c1024 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L3L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = index.casts %4 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c2048_i32, %6, %A_L3L1_5_cons_buff_0, %B_L3L1_5_cons_buff_0, %C_L1L3_5_buff_0) : (i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>) -> ()
      aie.use_lock(%A_L3L1_5_cons_prod_lock_0, Release, 1)
      %7 = arith.addi %4, %c1 : index
      aie.use_lock(%A_L3L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = index.casts %7 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c2048_i32, %8, %A_L3L1_5_cons_buff_1, %B_L3L1_5_cons_buff_0, %C_L1L3_5_buff_0) : (i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>) -> ()
      aie.use_lock(%A_L3L1_5_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %4, %c2 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_5_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_5_cons_prod_lock_0, Release, 1)
      %10 = arith.addi %2, %c1 : index
      cf.br ^bb3(%10 : index)
    ^bb8:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "mv.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c2 = arith.constant 2 : index
      %c2048_i32 = arith.constant 2048 : i32
      %c1_i32 = arith.constant 1 : i32
      %c1024 = arith.constant 1024 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_6_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c1024 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L3L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = index.casts %4 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c2048_i32, %6, %A_L3L1_6_cons_buff_0, %B_L3L1_6_cons_buff_0, %C_L1L3_6_buff_0) : (i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>) -> ()
      aie.use_lock(%A_L3L1_6_cons_prod_lock_0, Release, 1)
      %7 = arith.addi %4, %c1 : index
      aie.use_lock(%A_L3L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = index.casts %7 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c2048_i32, %8, %A_L3L1_6_cons_buff_1, %B_L3L1_6_cons_buff_0, %C_L1L3_6_buff_0) : (i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>) -> ()
      aie.use_lock(%A_L3L1_6_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %4, %c2 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_6_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_6_cons_prod_lock_0, Release, 1)
      %10 = arith.addi %2, %c1 : index
      cf.br ^bb3(%10 : index)
    ^bb8:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "mv.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c2 = arith.constant 2 : index
      %c2048_i32 = arith.constant 2048 : i32
      %c1_i32 = arith.constant 1 : i32
      %c1024 = arith.constant 1024 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_7_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c1024 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L3L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = index.casts %4 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c2048_i32, %6, %A_L3L1_7_cons_buff_0, %B_L3L1_7_cons_buff_0, %C_L1L3_7_buff_0) : (i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>) -> ()
      aie.use_lock(%A_L3L1_7_cons_prod_lock_0, Release, 1)
      %7 = arith.addi %4, %c1 : index
      aie.use_lock(%A_L3L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = index.casts %7 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c2048_i32, %8, %A_L3L1_7_cons_buff_1, %B_L3L1_7_cons_buff_0, %C_L1L3_7_buff_0) : (i32, i32, i32, memref<2048xbf16>, memref<2048xbf16>, memref<1024xbf16>) -> ()
      aie.use_lock(%A_L3L1_7_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %4, %c2 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_7_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_7_cons_prod_lock_0, Release, 1)
      %10 = arith.addi %2, %c1 : index
      cf.br ^bb3(%10 : index)
    ^bb8:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "mv.o"}
    aie.runtime_sequence(%arg0: memref<16777216xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<8192xbf16>) {
      %0 = aiex.dma_configure_task_for @A_L3L1_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<16777216xbf16>, 0, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L1_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @A_L3L1_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<16777216xbf16>, 2097152, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @B_L3L1_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @A_L3L1_2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16777216xbf16>, 4194304, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L1_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @A_L3L1_3_shim_alloc {
        aie.dma_bd(%arg0 : memref<16777216xbf16>, 6291456, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L1_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @A_L3L1_4_shim_alloc {
        aie.dma_bd(%arg0 : memref<16777216xbf16>, 8388608, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @B_L3L1_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @A_L3L1_5_shim_alloc {
        aie.dma_bd(%arg0 : memref<16777216xbf16>, 10485760, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @B_L3L1_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @A_L3L1_6_shim_alloc {
        aie.dma_bd(%arg0 : memref<16777216xbf16>, 12582912, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @B_L3L1_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @A_L3L1_7_shim_alloc {
        aie.dma_bd(%arg0 : memref<16777216xbf16>, 14680064, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @B_L3L1_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @C_L1L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 0, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @C_L1L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 1024, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @C_L1L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 2048, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @C_L1L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 3072, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @C_L1L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 4096, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @C_L1L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 5120, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @C_L1L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 6144, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @C_L1L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 7168, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%17)
      aiex.dma_await_task(%18)
      aiex.dma_await_task(%19)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%23)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%12)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%15)
    }
    aie.shim_dma_allocation @A_L3L1_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_0_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_0_cons_buff_1 : memref<2048xbf16>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_0_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%C_L1L3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_0_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @A_L3L1_1_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_1_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_1_cons_buff_1 : memref<2048xbf16>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_1_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%C_L1L3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_1_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @A_L3L1_2_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_2_cons_buff_1 : memref<2048xbf16>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_2_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%C_L1L3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_2_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @A_L3L1_3_shim_alloc(%shim_noc_tile_1_0, MM2S, 1)
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_3_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_3_cons_buff_1 : memref<2048xbf16>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_3_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%C_L1L3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_3_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @A_L3L1_4_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_4_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_4_cons_buff_1 : memref<2048xbf16>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_4_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%C_L1L3_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_4_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @A_L3L1_5_shim_alloc(%shim_noc_tile_2_0, MM2S, 1)
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_5_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_5_cons_buff_1 : memref<2048xbf16>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_5_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%C_L1L3_5_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_5_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_5_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @A_L3L1_6_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_6_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_6_cons_buff_1 : memref<2048xbf16>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_6_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%C_L1L3_6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_6_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_6_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @A_L3L1_7_shim_alloc(%shim_noc_tile_3_0, MM2S, 1)
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_7_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_7_cons_buff_1 : memref<2048xbf16>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_7_cons_buff_0 : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%C_L1L3_7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_7_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_7_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @B_L3L1_0_shim_alloc(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @B_L3L1_1_shim_alloc(%shim_noc_tile_4_0, MM2S, 1)
    aie.shim_dma_allocation @B_L3L1_2_shim_alloc(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @B_L3L1_3_shim_alloc(%shim_noc_tile_5_0, MM2S, 1)
    aie.shim_dma_allocation @B_L3L1_4_shim_alloc(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @B_L3L1_5_shim_alloc(%shim_noc_tile_6_0, MM2S, 1)
    aie.shim_dma_allocation @B_L3L1_6_shim_alloc(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @B_L3L1_7_shim_alloc(%shim_noc_tile_7_0, MM2S, 1)
    aie.shim_dma_allocation @C_L1L3_0_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @C_L1L3_1_shim_alloc(%shim_noc_tile_0_0, S2MM, 1)
    aie.shim_dma_allocation @C_L1L3_2_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @C_L1L3_3_shim_alloc(%shim_noc_tile_1_0, S2MM, 1)
    aie.shim_dma_allocation @C_L1L3_4_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @C_L1L3_5_shim_alloc(%shim_noc_tile_2_0, S2MM, 1)
    aie.shim_dma_allocation @C_L1L3_6_shim_alloc(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @C_L1L3_7_shim_alloc(%shim_noc_tile_3_0, S2MM, 1)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_4_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_4_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_5_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_5_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_6_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_6_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_7_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_7_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<South : 7, North : 2>
      aie.connect<North : 1, South : 2>
      aie.connect<North : 3, South : 3>
      aie.connect<North : 0, East : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 2, North : 2>
      aie.connect<North : 1, South : 1>
      aie.connect<North : 3, South : 3>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 2, North : 0>
      aie.connect<East : 0, North : 5>
      aie.connect<East : 3, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 3>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 5, North : 4>
      aie.connect<East : 0, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<South : 4, DMA : 0>
      aie.connect<East : 3, DMA : 1>
      aie.connect<DMA : 0, East : 0>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<South : 7, North : 5>
      aie.connect<East : 2, North : 3>
      aie.connect<East : 1, North : 2>
      aie.connect<East : 0, North : 0>
      aie.connect<North : 0, South : 2>
      aie.connect<West : 3, South : 3>
      aie.connect<North : 2, East : 3>
      aie.connect<North : 1, East : 0>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 5, North : 5>
      aie.connect<South : 3, North : 3>
      aie.connect<South : 2, North : 2>
      aie.connect<South : 0, North : 0>
      aie.connect<North : 0, South : 0>
      aie.connect<North : 2, South : 2>
      aie.connect<North : 1, South : 1>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<South : 1, West : 0>
      aie.connect<South : 5, North : 3>
      aie.connect<East : 0, DMA : 0>
      aie.connect<South : 3, North : 2>
      aie.connect<South : 2, North : 4>
      aie.connect<South : 0, West : 3>
      aie.connect<East : 2, North : 5>
      aie.connect<East : 3, DMA : 1>
      aie.connect<East : 1, North : 0>
      aie.connect<North : 3, South : 0>
      aie.connect<DMA : 0, East : 0>
      aie.connect<North : 2, South : 2>
      aie.connect<North : 0, South : 1>
    }
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
      aie.connect<East : 0, DMA : 0>
      aie.connect<East : 3, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<South : 3, North : 5>
      aie.connect<South : 2, DMA : 0>
      aie.connect<East : 0, North : 1>
      aie.connect<South : 4, North : 2>
      aie.connect<South : 5, West : 0>
      aie.connect<East : 3, North : 4>
      aie.connect<South : 0, DMA : 1>
      aie.connect<East : 2, North : 3>
      aie.connect<North : 1, South : 3>
      aie.connect<DMA : 0, South : 2>
      aie.connect<North : 3, South : 0>
    }
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<South : 5, North : 5>
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 2, North : 2>
      aie.connect<South : 4, West : 3>
      aie.connect<South : 3, DMA : 1>
      aie.connect<West : 0, South : 1>
      aie.connect<DMA : 0, South : 3>
    }
    %switchbox_1_5 = aie.switchbox(%tile_1_5) {
      aie.connect<South : 5, West : 0>
      aie.connect<South : 2, DMA : 0>
      aie.connect<East : 3, West : 3>
      aie.connect<East : 2, DMA : 1>
      aie.connect<DMA : 0, East : 3>
    }
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<South : 7, West : 2>
      aie.connect<East : 2, North : 5>
      aie.connect<East : 1, West : 1>
      aie.connect<East : 0, West : 0>
      aie.connect<East : 3, North : 4>
      aie.connect<North : 3, South : 2>
      aie.connect<West : 3, South : 3>
      aie.connect<West : 0, East : 0>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_2_0 = aie.shim_mux(%shim_noc_tile_2_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %mem_tile_2_1 = aie.tile(2, 1)
    %switchbox_2_1 = aie.switchbox(%mem_tile_2_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 5, North : 5>
      aie.connect<South : 4, North : 4>
      aie.connect<North : 3, South : 3>
    }
    %tile_2_2 = aie.tile(2, 2)
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<South : 1, West : 0>
      aie.connect<South : 5, North : 5>
      aie.connect<South : 4, West : 2>
      aie.connect<East : 0, West : 3>
      aie.connect<East : 1, West : 1>
      aie.connect<East : 2, North : 0>
      aie.connect<West : 0, South : 3>
    }
    %tile_2_3 = aie.tile(2, 3)
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
      aie.connect<South : 5, West : 0>
      aie.connect<East : 1, West : 3>
      aie.connect<East : 2, North : 5>
      aie.connect<South : 0, West : 2>
      aie.connect<East : 3, North : 0>
    }
    %switchbox_3_0 = aie.switchbox(%shim_noc_tile_3_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 2, West : 0>
      aie.connect<East : 1, West : 3>
      aie.connect<East : 0, North : 1>
      aie.connect<East : 3, North : 5>
      aie.connect<West : 0, South : 2>
      aie.connect<North : 3, South : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_3_0 = aie.shim_mux(%shim_noc_tile_3_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %switchbox_4_0 = aie.switchbox(%shim_noc_tile_4_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 2, North : 5>
      aie.connect<East : 1, West : 0>
      aie.connect<East : 0, West : 3>
      aie.connect<East : 3, North : 0>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_4_0 = aie.shim_mux(%shim_noc_tile_4_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %tile_3_2 = aie.tile(3, 2)
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<East : 3, North : 0>
      aie.connect<South : 1, North : 3>
      aie.connect<South : 5, West : 0>
      aie.connect<East : 2, West : 1>
      aie.connect<East : 1, West : 2>
      aie.connect<East : 0, North : 4>
      aie.connect<North : 2, South : 3>
    }
    %tile_3_3 = aie.tile(3, 3)
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
      aie.connect<South : 0, West : 1>
      aie.connect<South : 3, West : 2>
      aie.connect<South : 4, West : 3>
      aie.connect<North : 3, South : 2>
    }
    %mem_tile_4_1 = aie.tile(4, 1)
    %switchbox_4_1 = aie.switchbox(%mem_tile_4_1) {
      aie.connect<South : 5, North : 5>
      aie.connect<South : 0, North : 0>
    }
    %tile_4_2 = aie.tile(4, 2)
    %switchbox_4_2 = aie.switchbox(%tile_4_2) {
      aie.connect<South : 5, West : 3>
      aie.connect<East : 1, West : 2>
      aie.connect<East : 3, West : 1>
      aie.connect<South : 0, West : 0>
    }
    %switchbox_5_0 = aie.switchbox(%shim_noc_tile_5_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 2, West : 0>
      aie.connect<East : 1, North : 5>
      aie.connect<East : 0, West : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_5_0 = aie.shim_mux(%shim_noc_tile_5_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %tile_2_4 = aie.tile(2, 4)
    %switchbox_2_4 = aie.switchbox(%tile_2_4) {
      aie.connect<South : 5, North : 5>
      aie.connect<South : 0, North : 3>
      aie.connect<North : 0, East : 3>
    }
    %tile_2_5 = aie.tile(2, 5)
    %switchbox_2_5 = aie.switchbox(%tile_2_5) {
      aie.connect<South : 5, West : 3>
      aie.connect<South : 3, West : 2>
      aie.connect<West : 3, South : 0>
    }
    %mem_tile_3_1 = aie.tile(3, 1)
    %switchbox_3_1 = aie.switchbox(%mem_tile_3_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 5, North : 5>
      aie.connect<North : 3, South : 3>
    }
    %switchbox_6_0 = aie.switchbox(%shim_noc_tile_6_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 3, West : 0>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_6_0 = aie.shim_mux(%shim_noc_tile_6_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %mem_tile_5_1 = aie.tile(5, 1)
    %switchbox_5_1 = aie.switchbox(%mem_tile_5_1) {
      aie.connect<South : 5, North : 5>
    }
    %tile_5_2 = aie.tile(5, 2)
    %switchbox_5_2 = aie.switchbox(%tile_5_2) {
      aie.connect<South : 5, West : 1>
      aie.connect<East : 3, West : 3>
    }
    %tile_6_2 = aie.tile(6, 2)
    %switchbox_6_2 = aie.switchbox(%tile_6_2) {
      aie.connect<East : 1, West : 3>
    }
    %switchbox_7_0 = aie.switchbox(%shim_noc_tile_7_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, West : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_7_0 = aie.shim_mux(%shim_noc_tile_7_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %mem_tile_7_1 = aie.tile(7, 1)
    %switchbox_7_1 = aie.switchbox(%mem_tile_7_1) {
      aie.connect<South : 0, North : 0>
    }
    %tile_7_2 = aie.tile(7, 2)
    %switchbox_7_2 = aie.switchbox(%tile_7_2) {
      aie.connect<South : 0, West : 1>
    }
    %tile_3_4 = aie.tile(3, 4)
    %switchbox_3_4 = aie.switchbox(%tile_3_4) {
      aie.connect<West : 3, South : 3>
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%tile_0_4 : Core, %switchbox_0_4 : Core)
    aie.wire(%tile_0_4 : DMA, %switchbox_0_4 : DMA)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
    aie.wire(%tile_0_5 : Core, %switchbox_0_5 : Core)
    aie.wire(%tile_0_5 : DMA, %switchbox_0_5 : DMA)
    aie.wire(%switchbox_0_4 : North, %switchbox_0_5 : South)
    aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%shim_noc_tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
    aie.wire(%switchbox_0_3 : East, %switchbox_1_3 : West)
    aie.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
    aie.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
    aie.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
    aie.wire(%switchbox_0_4 : East, %switchbox_1_4 : West)
    aie.wire(%tile_1_4 : Core, %switchbox_1_4 : Core)
    aie.wire(%tile_1_4 : DMA, %switchbox_1_4 : DMA)
    aie.wire(%switchbox_1_3 : North, %switchbox_1_4 : South)
    aie.wire(%switchbox_0_5 : East, %switchbox_1_5 : West)
    aie.wire(%tile_1_5 : Core, %switchbox_1_5 : Core)
    aie.wire(%tile_1_5 : DMA, %switchbox_1_5 : DMA)
    aie.wire(%switchbox_1_4 : North, %switchbox_1_5 : South)
    aie.wire(%switchbox_1_0 : East, %switchbox_2_0 : West)
    aie.wire(%shim_mux_2_0 : North, %switchbox_2_0 : South)
    aie.wire(%shim_noc_tile_2_0 : DMA, %shim_mux_2_0 : DMA)
    aie.wire(%switchbox_1_1 : East, %switchbox_2_1 : West)
    aie.wire(%mem_tile_2_1 : Core, %switchbox_2_1 : Core)
    aie.wire(%mem_tile_2_1 : DMA, %switchbox_2_1 : DMA)
    aie.wire(%switchbox_2_0 : North, %switchbox_2_1 : South)
    aie.wire(%switchbox_1_2 : East, %switchbox_2_2 : West)
    aie.wire(%tile_2_2 : Core, %switchbox_2_2 : Core)
    aie.wire(%tile_2_2 : DMA, %switchbox_2_2 : DMA)
    aie.wire(%switchbox_2_1 : North, %switchbox_2_2 : South)
    aie.wire(%switchbox_1_3 : East, %switchbox_2_3 : West)
    aie.wire(%tile_2_3 : Core, %switchbox_2_3 : Core)
    aie.wire(%tile_2_3 : DMA, %switchbox_2_3 : DMA)
    aie.wire(%switchbox_2_2 : North, %switchbox_2_3 : South)
    aie.wire(%switchbox_1_4 : East, %switchbox_2_4 : West)
    aie.wire(%tile_2_4 : Core, %switchbox_2_4 : Core)
    aie.wire(%tile_2_4 : DMA, %switchbox_2_4 : DMA)
    aie.wire(%switchbox_2_3 : North, %switchbox_2_4 : South)
    aie.wire(%switchbox_1_5 : East, %switchbox_2_5 : West)
    aie.wire(%tile_2_5 : Core, %switchbox_2_5 : Core)
    aie.wire(%tile_2_5 : DMA, %switchbox_2_5 : DMA)
    aie.wire(%switchbox_2_4 : North, %switchbox_2_5 : South)
    aie.wire(%switchbox_2_0 : East, %switchbox_3_0 : West)
    aie.wire(%shim_mux_3_0 : North, %switchbox_3_0 : South)
    aie.wire(%shim_noc_tile_3_0 : DMA, %shim_mux_3_0 : DMA)
    aie.wire(%switchbox_2_1 : East, %switchbox_3_1 : West)
    aie.wire(%mem_tile_3_1 : Core, %switchbox_3_1 : Core)
    aie.wire(%mem_tile_3_1 : DMA, %switchbox_3_1 : DMA)
    aie.wire(%switchbox_3_0 : North, %switchbox_3_1 : South)
    aie.wire(%switchbox_2_2 : East, %switchbox_3_2 : West)
    aie.wire(%tile_3_2 : Core, %switchbox_3_2 : Core)
    aie.wire(%tile_3_2 : DMA, %switchbox_3_2 : DMA)
    aie.wire(%switchbox_3_1 : North, %switchbox_3_2 : South)
    aie.wire(%switchbox_2_3 : East, %switchbox_3_3 : West)
    aie.wire(%tile_3_3 : Core, %switchbox_3_3 : Core)
    aie.wire(%tile_3_3 : DMA, %switchbox_3_3 : DMA)
    aie.wire(%switchbox_3_2 : North, %switchbox_3_3 : South)
    aie.wire(%switchbox_2_4 : East, %switchbox_3_4 : West)
    aie.wire(%tile_3_4 : Core, %switchbox_3_4 : Core)
    aie.wire(%tile_3_4 : DMA, %switchbox_3_4 : DMA)
    aie.wire(%switchbox_3_3 : North, %switchbox_3_4 : South)
    aie.wire(%switchbox_3_0 : East, %switchbox_4_0 : West)
    aie.wire(%shim_mux_4_0 : North, %switchbox_4_0 : South)
    aie.wire(%shim_noc_tile_4_0 : DMA, %shim_mux_4_0 : DMA)
    aie.wire(%switchbox_3_1 : East, %switchbox_4_1 : West)
    aie.wire(%mem_tile_4_1 : Core, %switchbox_4_1 : Core)
    aie.wire(%mem_tile_4_1 : DMA, %switchbox_4_1 : DMA)
    aie.wire(%switchbox_4_0 : North, %switchbox_4_1 : South)
    aie.wire(%switchbox_3_2 : East, %switchbox_4_2 : West)
    aie.wire(%tile_4_2 : Core, %switchbox_4_2 : Core)
    aie.wire(%tile_4_2 : DMA, %switchbox_4_2 : DMA)
    aie.wire(%switchbox_4_1 : North, %switchbox_4_2 : South)
    aie.wire(%switchbox_4_0 : East, %switchbox_5_0 : West)
    aie.wire(%shim_mux_5_0 : North, %switchbox_5_0 : South)
    aie.wire(%shim_noc_tile_5_0 : DMA, %shim_mux_5_0 : DMA)
    aie.wire(%switchbox_4_1 : East, %switchbox_5_1 : West)
    aie.wire(%mem_tile_5_1 : Core, %switchbox_5_1 : Core)
    aie.wire(%mem_tile_5_1 : DMA, %switchbox_5_1 : DMA)
    aie.wire(%switchbox_5_0 : North, %switchbox_5_1 : South)
    aie.wire(%switchbox_4_2 : East, %switchbox_5_2 : West)
    aie.wire(%tile_5_2 : Core, %switchbox_5_2 : Core)
    aie.wire(%tile_5_2 : DMA, %switchbox_5_2 : DMA)
    aie.wire(%switchbox_5_1 : North, %switchbox_5_2 : South)
    aie.wire(%switchbox_5_0 : East, %switchbox_6_0 : West)
    aie.wire(%shim_mux_6_0 : North, %switchbox_6_0 : South)
    aie.wire(%shim_noc_tile_6_0 : DMA, %shim_mux_6_0 : DMA)
    aie.wire(%switchbox_5_2 : East, %switchbox_6_2 : West)
    aie.wire(%tile_6_2 : Core, %switchbox_6_2 : Core)
    aie.wire(%tile_6_2 : DMA, %switchbox_6_2 : DMA)
    aie.wire(%switchbox_6_0 : East, %switchbox_7_0 : West)
    aie.wire(%shim_mux_7_0 : North, %switchbox_7_0 : South)
    aie.wire(%shim_noc_tile_7_0 : DMA, %shim_mux_7_0 : DMA)
    aie.wire(%mem_tile_7_1 : Core, %switchbox_7_1 : Core)
    aie.wire(%mem_tile_7_1 : DMA, %switchbox_7_1 : DMA)
    aie.wire(%switchbox_7_0 : North, %switchbox_7_1 : South)
    aie.wire(%switchbox_6_2 : East, %switchbox_7_2 : West)
    aie.wire(%tile_7_2 : Core, %switchbox_7_2 : Core)
    aie.wire(%tile_7_2 : DMA, %switchbox_7_2 : DMA)
    aie.wire(%switchbox_7_1 : North, %switchbox_7_2 : South)
  }
  aie.device(npu2) @silu {
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_2_2 = aie.tile(2, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_2_3 = aie.tile(2, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_2_4 = aie.tile(2, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_2_5 = aie.tile(2, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_3_2 = aie.tile(3, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_3_3 = aie.tile(3, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_3_4 = aie.tile(3, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_3_5 = aie.tile(3, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_4_0 = aie.tile(4, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_5_0 = aie.tile(5, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_6_0 = aie.tile(6, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_7_0 = aie.tile(7, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %out7_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 6) {init = 0 : i32, sym_name = "out7_1_cons_prod_lock_0"}
    %out7_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 7) {init = 0 : i32, sym_name = "out7_1_cons_cons_lock_0"}
    %out7_1_buff_0 = aie.buffer(%tile_3_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out7_1_buff_0"} : memref<512xbf16> 
    %out7_1_buff_1 = aie.buffer(%tile_3_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out7_1_buff_1"} : memref<512xbf16> 
    %out7_1_prod_lock_0 = aie.lock(%tile_3_5, 2) {init = 2 : i32, sym_name = "out7_1_prod_lock_0"}
    %out7_1_cons_lock_0 = aie.lock(%tile_3_5, 3) {init = 0 : i32, sym_name = "out7_1_cons_lock_0"}
    %out7_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 4) {init = 0 : i32, sym_name = "out7_0_cons_prod_lock_0"}
    %out7_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 5) {init = 0 : i32, sym_name = "out7_0_cons_cons_lock_0"}
    %out7_0_buff_0 = aie.buffer(%tile_3_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out7_0_buff_0"} : memref<512xbf16> 
    %out7_0_buff_1 = aie.buffer(%tile_3_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out7_0_buff_1"} : memref<512xbf16> 
    %out7_0_prod_lock_0 = aie.lock(%tile_3_4, 2) {init = 2 : i32, sym_name = "out7_0_prod_lock_0"}
    %out7_0_cons_lock_0 = aie.lock(%tile_3_4, 3) {init = 0 : i32, sym_name = "out7_0_cons_lock_0"}
    %out6_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 6) {init = 0 : i32, sym_name = "out6_1_cons_prod_lock_0"}
    %out6_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 7) {init = 0 : i32, sym_name = "out6_1_cons_cons_lock_0"}
    %out6_1_buff_0 = aie.buffer(%tile_3_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out6_1_buff_0"} : memref<512xbf16> 
    %out6_1_buff_1 = aie.buffer(%tile_3_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out6_1_buff_1"} : memref<512xbf16> 
    %out6_1_prod_lock_0 = aie.lock(%tile_3_3, 2) {init = 2 : i32, sym_name = "out6_1_prod_lock_0"}
    %out6_1_cons_lock_0 = aie.lock(%tile_3_3, 3) {init = 0 : i32, sym_name = "out6_1_cons_lock_0"}
    %out6_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 4) {init = 0 : i32, sym_name = "out6_0_cons_prod_lock_0"}
    %out6_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 5) {init = 0 : i32, sym_name = "out6_0_cons_cons_lock_0"}
    %out6_0_buff_0 = aie.buffer(%tile_3_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out6_0_buff_0"} : memref<512xbf16> 
    %out6_0_buff_1 = aie.buffer(%tile_3_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out6_0_buff_1"} : memref<512xbf16> 
    %out6_0_prod_lock_0 = aie.lock(%tile_3_2, 2) {init = 2 : i32, sym_name = "out6_0_prod_lock_0"}
    %out6_0_cons_lock_0 = aie.lock(%tile_3_2, 3) {init = 0 : i32, sym_name = "out6_0_cons_lock_0"}
    %out5_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 6) {init = 0 : i32, sym_name = "out5_1_cons_prod_lock_0"}
    %out5_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 7) {init = 0 : i32, sym_name = "out5_1_cons_cons_lock_0"}
    %out5_1_buff_0 = aie.buffer(%tile_2_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out5_1_buff_0"} : memref<512xbf16> 
    %out5_1_buff_1 = aie.buffer(%tile_2_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out5_1_buff_1"} : memref<512xbf16> 
    %out5_1_prod_lock_0 = aie.lock(%tile_2_5, 2) {init = 2 : i32, sym_name = "out5_1_prod_lock_0"}
    %out5_1_cons_lock_0 = aie.lock(%tile_2_5, 3) {init = 0 : i32, sym_name = "out5_1_cons_lock_0"}
    %out5_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 4) {init = 0 : i32, sym_name = "out5_0_cons_prod_lock_0"}
    %out5_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 5) {init = 0 : i32, sym_name = "out5_0_cons_cons_lock_0"}
    %out5_0_buff_0 = aie.buffer(%tile_2_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out5_0_buff_0"} : memref<512xbf16> 
    %out5_0_buff_1 = aie.buffer(%tile_2_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out5_0_buff_1"} : memref<512xbf16> 
    %out5_0_prod_lock_0 = aie.lock(%tile_2_4, 2) {init = 2 : i32, sym_name = "out5_0_prod_lock_0"}
    %out5_0_cons_lock_0 = aie.lock(%tile_2_4, 3) {init = 0 : i32, sym_name = "out5_0_cons_lock_0"}
    %out4_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 6) {init = 0 : i32, sym_name = "out4_1_cons_prod_lock_0"}
    %out4_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 7) {init = 0 : i32, sym_name = "out4_1_cons_cons_lock_0"}
    %out4_1_buff_0 = aie.buffer(%tile_2_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out4_1_buff_0"} : memref<512xbf16> 
    %out4_1_buff_1 = aie.buffer(%tile_2_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out4_1_buff_1"} : memref<512xbf16> 
    %out4_1_prod_lock_0 = aie.lock(%tile_2_3, 2) {init = 2 : i32, sym_name = "out4_1_prod_lock_0"}
    %out4_1_cons_lock_0 = aie.lock(%tile_2_3, 3) {init = 0 : i32, sym_name = "out4_1_cons_lock_0"}
    %out4_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 4) {init = 0 : i32, sym_name = "out4_0_cons_prod_lock_0"}
    %out4_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 5) {init = 0 : i32, sym_name = "out4_0_cons_cons_lock_0"}
    %out4_0_buff_0 = aie.buffer(%tile_2_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out4_0_buff_0"} : memref<512xbf16> 
    %out4_0_buff_1 = aie.buffer(%tile_2_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out4_0_buff_1"} : memref<512xbf16> 
    %out4_0_prod_lock_0 = aie.lock(%tile_2_2, 2) {init = 2 : i32, sym_name = "out4_0_prod_lock_0"}
    %out4_0_cons_lock_0 = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "out4_0_cons_lock_0"}
    %out3_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 6) {init = 0 : i32, sym_name = "out3_1_cons_prod_lock_0"}
    %out3_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 7) {init = 0 : i32, sym_name = "out3_1_cons_cons_lock_0"}
    %out3_1_buff_0 = aie.buffer(%tile_1_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out3_1_buff_0"} : memref<512xbf16> 
    %out3_1_buff_1 = aie.buffer(%tile_1_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out3_1_buff_1"} : memref<512xbf16> 
    %out3_1_prod_lock_0 = aie.lock(%tile_1_5, 2) {init = 2 : i32, sym_name = "out3_1_prod_lock_0"}
    %out3_1_cons_lock_0 = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "out3_1_cons_lock_0"}
    %out3_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 4) {init = 0 : i32, sym_name = "out3_0_cons_prod_lock_0"}
    %out3_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 5) {init = 0 : i32, sym_name = "out3_0_cons_cons_lock_0"}
    %out3_0_buff_0 = aie.buffer(%tile_1_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out3_0_buff_0"} : memref<512xbf16> 
    %out3_0_buff_1 = aie.buffer(%tile_1_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out3_0_buff_1"} : memref<512xbf16> 
    %out3_0_prod_lock_0 = aie.lock(%tile_1_4, 2) {init = 2 : i32, sym_name = "out3_0_prod_lock_0"}
    %out3_0_cons_lock_0 = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "out3_0_cons_lock_0"}
    %out2_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 6) {init = 0 : i32, sym_name = "out2_1_cons_prod_lock_0"}
    %out2_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 7) {init = 0 : i32, sym_name = "out2_1_cons_cons_lock_0"}
    %out2_1_buff_0 = aie.buffer(%tile_1_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out2_1_buff_0"} : memref<512xbf16> 
    %out2_1_buff_1 = aie.buffer(%tile_1_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out2_1_buff_1"} : memref<512xbf16> 
    %out2_1_prod_lock_0 = aie.lock(%tile_1_3, 2) {init = 2 : i32, sym_name = "out2_1_prod_lock_0"}
    %out2_1_cons_lock_0 = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "out2_1_cons_lock_0"}
    %out2_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 4) {init = 0 : i32, sym_name = "out2_0_cons_prod_lock_0"}
    %out2_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 5) {init = 0 : i32, sym_name = "out2_0_cons_cons_lock_0"}
    %out2_0_buff_0 = aie.buffer(%tile_1_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out2_0_buff_0"} : memref<512xbf16> 
    %out2_0_buff_1 = aie.buffer(%tile_1_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out2_0_buff_1"} : memref<512xbf16> 
    %out2_0_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "out2_0_prod_lock_0"}
    %out2_0_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "out2_0_cons_lock_0"}
    %out1_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 6) {init = 0 : i32, sym_name = "out1_1_cons_prod_lock_0"}
    %out1_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 7) {init = 0 : i32, sym_name = "out1_1_cons_cons_lock_0"}
    %out1_1_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out1_1_buff_0"} : memref<512xbf16> 
    %out1_1_buff_1 = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out1_1_buff_1"} : memref<512xbf16> 
    %out1_1_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 2 : i32, sym_name = "out1_1_prod_lock_0"}
    %out1_1_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "out1_1_cons_lock_0"}
    %out1_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 4) {init = 0 : i32, sym_name = "out1_0_cons_prod_lock_0"}
    %out1_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 5) {init = 0 : i32, sym_name = "out1_0_cons_cons_lock_0"}
    %out1_0_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out1_0_buff_0"} : memref<512xbf16> 
    %out1_0_buff_1 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out1_0_buff_1"} : memref<512xbf16> 
    %out1_0_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "out1_0_prod_lock_0"}
    %out1_0_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "out1_0_cons_lock_0"}
    %out0_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 6) {init = 0 : i32, sym_name = "out0_1_cons_prod_lock_0"}
    %out0_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 7) {init = 0 : i32, sym_name = "out0_1_cons_cons_lock_0"}
    %out0_1_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out0_1_buff_0"} : memref<512xbf16> 
    %out0_1_buff_1 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out0_1_buff_1"} : memref<512xbf16> 
    %out0_1_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "out0_1_prod_lock_0"}
    %out0_1_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "out0_1_cons_lock_0"}
    %out0_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "out0_0_cons_prod_lock_0"}
    %out0_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "out0_0_cons_cons_lock_0"}
    %out0_0_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out0_0_buff_0"} : memref<512xbf16> 
    %out0_0_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out0_0_buff_1"} : memref<512xbf16> 
    %out0_0_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "out0_0_prod_lock_0"}
    %out0_0_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out0_0_cons_lock_0"}
    %in7_1_cons_buff_0 = aie.buffer(%tile_3_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in7_1_cons_buff_0"} : memref<512xbf16> 
    %in7_1_cons_buff_1 = aie.buffer(%tile_3_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in7_1_cons_buff_1"} : memref<512xbf16> 
    %in7_1_cons_prod_lock_0 = aie.lock(%tile_3_5, 0) {init = 2 : i32, sym_name = "in7_1_cons_prod_lock_0"}
    %in7_1_cons_cons_lock_0 = aie.lock(%tile_3_5, 1) {init = 0 : i32, sym_name = "in7_1_cons_cons_lock_0"}
    %in7_1_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 2) {init = 0 : i32, sym_name = "in7_1_prod_lock_0"}
    %in7_1_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 3) {init = 0 : i32, sym_name = "in7_1_cons_lock_0"}
    %in7_0_cons_buff_0 = aie.buffer(%tile_3_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in7_0_cons_buff_0"} : memref<512xbf16> 
    %in7_0_cons_buff_1 = aie.buffer(%tile_3_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in7_0_cons_buff_1"} : memref<512xbf16> 
    %in7_0_cons_prod_lock_0 = aie.lock(%tile_3_4, 0) {init = 2 : i32, sym_name = "in7_0_cons_prod_lock_0"}
    %in7_0_cons_cons_lock_0 = aie.lock(%tile_3_4, 1) {init = 0 : i32, sym_name = "in7_0_cons_cons_lock_0"}
    %in7_0_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 0) {init = 0 : i32, sym_name = "in7_0_prod_lock_0"}
    %in7_0_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 1) {init = 0 : i32, sym_name = "in7_0_cons_lock_0"}
    %in6_1_cons_buff_0 = aie.buffer(%tile_3_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in6_1_cons_buff_0"} : memref<512xbf16> 
    %in6_1_cons_buff_1 = aie.buffer(%tile_3_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in6_1_cons_buff_1"} : memref<512xbf16> 
    %in6_1_cons_prod_lock_0 = aie.lock(%tile_3_3, 0) {init = 2 : i32, sym_name = "in6_1_cons_prod_lock_0"}
    %in6_1_cons_cons_lock_0 = aie.lock(%tile_3_3, 1) {init = 0 : i32, sym_name = "in6_1_cons_cons_lock_0"}
    %in6_1_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 2) {init = 0 : i32, sym_name = "in6_1_prod_lock_0"}
    %in6_1_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 3) {init = 0 : i32, sym_name = "in6_1_cons_lock_0"}
    %in6_0_cons_buff_0 = aie.buffer(%tile_3_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in6_0_cons_buff_0"} : memref<512xbf16> 
    %in6_0_cons_buff_1 = aie.buffer(%tile_3_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in6_0_cons_buff_1"} : memref<512xbf16> 
    %in6_0_cons_prod_lock_0 = aie.lock(%tile_3_2, 0) {init = 2 : i32, sym_name = "in6_0_cons_prod_lock_0"}
    %in6_0_cons_cons_lock_0 = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "in6_0_cons_cons_lock_0"}
    %in6_0_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 0) {init = 0 : i32, sym_name = "in6_0_prod_lock_0"}
    %in6_0_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 1) {init = 0 : i32, sym_name = "in6_0_cons_lock_0"}
    %in5_1_cons_buff_0 = aie.buffer(%tile_2_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in5_1_cons_buff_0"} : memref<512xbf16> 
    %in5_1_cons_buff_1 = aie.buffer(%tile_2_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in5_1_cons_buff_1"} : memref<512xbf16> 
    %in5_1_cons_prod_lock_0 = aie.lock(%tile_2_5, 0) {init = 2 : i32, sym_name = "in5_1_cons_prod_lock_0"}
    %in5_1_cons_cons_lock_0 = aie.lock(%tile_2_5, 1) {init = 0 : i32, sym_name = "in5_1_cons_cons_lock_0"}
    %in5_1_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 2) {init = 0 : i32, sym_name = "in5_1_prod_lock_0"}
    %in5_1_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 3) {init = 0 : i32, sym_name = "in5_1_cons_lock_0"}
    %in5_0_cons_buff_0 = aie.buffer(%tile_2_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in5_0_cons_buff_0"} : memref<512xbf16> 
    %in5_0_cons_buff_1 = aie.buffer(%tile_2_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in5_0_cons_buff_1"} : memref<512xbf16> 
    %in5_0_cons_prod_lock_0 = aie.lock(%tile_2_4, 0) {init = 2 : i32, sym_name = "in5_0_cons_prod_lock_0"}
    %in5_0_cons_cons_lock_0 = aie.lock(%tile_2_4, 1) {init = 0 : i32, sym_name = "in5_0_cons_cons_lock_0"}
    %in5_0_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 0) {init = 0 : i32, sym_name = "in5_0_prod_lock_0"}
    %in5_0_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 1) {init = 0 : i32, sym_name = "in5_0_cons_lock_0"}
    %in4_1_cons_buff_0 = aie.buffer(%tile_2_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in4_1_cons_buff_0"} : memref<512xbf16> 
    %in4_1_cons_buff_1 = aie.buffer(%tile_2_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in4_1_cons_buff_1"} : memref<512xbf16> 
    %in4_1_cons_prod_lock_0 = aie.lock(%tile_2_3, 0) {init = 2 : i32, sym_name = "in4_1_cons_prod_lock_0"}
    %in4_1_cons_cons_lock_0 = aie.lock(%tile_2_3, 1) {init = 0 : i32, sym_name = "in4_1_cons_cons_lock_0"}
    %in4_1_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 2) {init = 0 : i32, sym_name = "in4_1_prod_lock_0"}
    %in4_1_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 3) {init = 0 : i32, sym_name = "in4_1_cons_lock_0"}
    %in4_0_cons_buff_0 = aie.buffer(%tile_2_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in4_0_cons_buff_0"} : memref<512xbf16> 
    %in4_0_cons_buff_1 = aie.buffer(%tile_2_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in4_0_cons_buff_1"} : memref<512xbf16> 
    %in4_0_cons_prod_lock_0 = aie.lock(%tile_2_2, 0) {init = 2 : i32, sym_name = "in4_0_cons_prod_lock_0"}
    %in4_0_cons_cons_lock_0 = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "in4_0_cons_cons_lock_0"}
    %in4_0_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 0) {init = 0 : i32, sym_name = "in4_0_prod_lock_0"}
    %in4_0_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 1) {init = 0 : i32, sym_name = "in4_0_cons_lock_0"}
    %in3_1_cons_buff_0 = aie.buffer(%tile_1_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in3_1_cons_buff_0"} : memref<512xbf16> 
    %in3_1_cons_buff_1 = aie.buffer(%tile_1_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in3_1_cons_buff_1"} : memref<512xbf16> 
    %in3_1_cons_prod_lock_0 = aie.lock(%tile_1_5, 0) {init = 2 : i32, sym_name = "in3_1_cons_prod_lock_0"}
    %in3_1_cons_cons_lock_0 = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "in3_1_cons_cons_lock_0"}
    %in3_1_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "in3_1_prod_lock_0"}
    %in3_1_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "in3_1_cons_lock_0"}
    %in3_0_cons_buff_0 = aie.buffer(%tile_1_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in3_0_cons_buff_0"} : memref<512xbf16> 
    %in3_0_cons_buff_1 = aie.buffer(%tile_1_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in3_0_cons_buff_1"} : memref<512xbf16> 
    %in3_0_cons_prod_lock_0 = aie.lock(%tile_1_4, 0) {init = 2 : i32, sym_name = "in3_0_cons_prod_lock_0"}
    %in3_0_cons_cons_lock_0 = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "in3_0_cons_cons_lock_0"}
    %in3_0_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "in3_0_prod_lock_0"}
    %in3_0_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "in3_0_cons_lock_0"}
    %in2_1_cons_buff_0 = aie.buffer(%tile_1_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_1_cons_buff_0"} : memref<512xbf16> 
    %in2_1_cons_buff_1 = aie.buffer(%tile_1_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_1_cons_buff_1"} : memref<512xbf16> 
    %in2_1_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "in2_1_cons_prod_lock_0"}
    %in2_1_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "in2_1_cons_cons_lock_0"}
    %in2_1_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "in2_1_prod_lock_0"}
    %in2_1_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "in2_1_cons_lock_0"}
    %in2_0_cons_buff_0 = aie.buffer(%tile_1_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_0_cons_buff_0"} : memref<512xbf16> 
    %in2_0_cons_buff_1 = aie.buffer(%tile_1_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_0_cons_buff_1"} : memref<512xbf16> 
    %in2_0_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "in2_0_cons_prod_lock_0"}
    %in2_0_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "in2_0_cons_cons_lock_0"}
    %in2_0_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "in2_0_prod_lock_0"}
    %in2_0_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "in2_0_cons_lock_0"}
    %in1_1_cons_buff_0 = aie.buffer(%tile_0_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in1_1_cons_buff_0"} : memref<512xbf16> 
    %in1_1_cons_buff_1 = aie.buffer(%tile_0_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in1_1_cons_buff_1"} : memref<512xbf16> 
    %in1_1_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "in1_1_cons_prod_lock_0"}
    %in1_1_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "in1_1_cons_cons_lock_0"}
    %in1_1_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "in1_1_prod_lock_0"}
    %in1_1_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "in1_1_cons_lock_0"}
    %in1_0_cons_buff_0 = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in1_0_cons_buff_0"} : memref<512xbf16> 
    %in1_0_cons_buff_1 = aie.buffer(%tile_0_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in1_0_cons_buff_1"} : memref<512xbf16> 
    %in1_0_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "in1_0_cons_prod_lock_0"}
    %in1_0_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "in1_0_cons_cons_lock_0"}
    %in1_0_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "in1_0_prod_lock_0"}
    %in1_0_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "in1_0_cons_lock_0"}
    %in0_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in0_1_cons_buff_0"} : memref<512xbf16> 
    %in0_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in0_1_cons_buff_1"} : memref<512xbf16> 
    %in0_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "in0_1_cons_prod_lock_0"}
    %in0_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "in0_1_cons_cons_lock_0"}
    %in0_1_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "in0_1_prod_lock_0"}
    %in0_1_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "in0_1_cons_lock_0"}
    %in0_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in0_0_cons_buff_0"} : memref<512xbf16> 
    %in0_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in0_0_cons_buff_1"} : memref<512xbf16> 
    %in0_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "in0_0_cons_prod_lock_0"}
    %in0_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in0_0_cons_cons_lock_0"}
    %in0_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "in0_0_prod_lock_0"}
    %in0_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "in0_0_cons_lock_0"}
    func.func private @silu_bf16(memref<512xbf16>, memref<512xbf16>, i32)
    %core_0_2 = aie.core(%tile_0_2) {
      %c512_i32 = arith.constant 512 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out0_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in0_0_cons_buff_0, %out0_0_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out0_0_cons_lock_0, Release, 1)
      aie.use_lock(%out0_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in0_0_cons_buff_1, %out0_0_buff_1, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out0_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%out0_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in0_0_cons_buff_0, %out0_0_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out0_0_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "silu.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c512_i32 = arith.constant 512 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out0_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in0_1_cons_buff_0, %out0_1_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out0_1_cons_lock_0, Release, 1)
      aie.use_lock(%out0_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in0_1_cons_buff_1, %out0_1_buff_1, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out0_1_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%out0_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in0_1_cons_buff_0, %out0_1_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out0_1_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "silu.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c512_i32 = arith.constant 512 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out1_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in1_0_cons_buff_0, %out1_0_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_0_cons_lock_0, Release, 1)
      aie.use_lock(%out1_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in1_0_cons_buff_1, %out1_0_buff_1, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%out1_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in1_0_cons_buff_0, %out1_0_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_0_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "silu.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c512_i32 = arith.constant 512 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out1_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in1_1_cons_buff_0, %out1_1_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_1_cons_lock_0, Release, 1)
      aie.use_lock(%out1_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in1_1_cons_buff_1, %out1_1_buff_1, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_1_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%out1_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in1_1_cons_buff_0, %out1_1_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out1_1_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "silu.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c512_i32 = arith.constant 512 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out2_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in2_0_cons_buff_0, %out2_0_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_0_cons_lock_0, Release, 1)
      aie.use_lock(%out2_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in2_0_cons_buff_1, %out2_0_buff_1, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%out2_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in2_0_cons_buff_0, %out2_0_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_0_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "silu.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c512_i32 = arith.constant 512 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out2_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in2_1_cons_buff_0, %out2_1_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_1_cons_lock_0, Release, 1)
      aie.use_lock(%out2_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in2_1_cons_buff_1, %out2_1_buff_1, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_1_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%out2_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in2_1_cons_buff_0, %out2_1_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out2_1_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "silu.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c512_i32 = arith.constant 512 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out3_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in3_0_cons_buff_0, %out3_0_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out3_0_cons_lock_0, Release, 1)
      aie.use_lock(%out3_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in3_0_cons_buff_1, %out3_0_buff_1, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out3_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%out3_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in3_0_cons_buff_0, %out3_0_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in3_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out3_0_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "silu.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c512_i32 = arith.constant 512 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out3_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in3_1_cons_buff_0, %out3_1_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in3_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out3_1_cons_lock_0, Release, 1)
      aie.use_lock(%out3_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in3_1_cons_buff_1, %out3_1_buff_1, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in3_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out3_1_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%out3_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in3_1_cons_buff_0, %out3_1_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in3_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out3_1_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "silu.o"}
    %core_2_2 = aie.core(%tile_2_2) {
      %c512_i32 = arith.constant 512 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out4_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in4_0_cons_buff_0, %out4_0_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in4_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out4_0_cons_lock_0, Release, 1)
      aie.use_lock(%out4_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in4_0_cons_buff_1, %out4_0_buff_1, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in4_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out4_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%out4_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in4_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in4_0_cons_buff_0, %out4_0_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in4_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out4_0_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "silu.o"}
    %core_2_3 = aie.core(%tile_2_3) {
      %c512_i32 = arith.constant 512 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out4_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in4_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in4_1_cons_buff_0, %out4_1_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in4_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out4_1_cons_lock_0, Release, 1)
      aie.use_lock(%out4_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in4_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in4_1_cons_buff_1, %out4_1_buff_1, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in4_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out4_1_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%out4_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in4_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in4_1_cons_buff_0, %out4_1_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in4_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out4_1_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "silu.o"}
    %core_2_4 = aie.core(%tile_2_4) {
      %c512_i32 = arith.constant 512 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out5_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in5_0_cons_buff_0, %out5_0_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in5_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out5_0_cons_lock_0, Release, 1)
      aie.use_lock(%out5_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in5_0_cons_buff_1, %out5_0_buff_1, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in5_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out5_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%out5_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in5_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in5_0_cons_buff_0, %out5_0_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in5_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out5_0_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "silu.o"}
    %core_2_5 = aie.core(%tile_2_5) {
      %c512_i32 = arith.constant 512 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out5_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in5_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in5_1_cons_buff_0, %out5_1_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in5_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out5_1_cons_lock_0, Release, 1)
      aie.use_lock(%out5_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in5_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in5_1_cons_buff_1, %out5_1_buff_1, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in5_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out5_1_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%out5_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in5_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in5_1_cons_buff_0, %out5_1_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in5_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out5_1_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "silu.o"}
    %core_3_2 = aie.core(%tile_3_2) {
      %c512_i32 = arith.constant 512 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out6_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in6_0_cons_buff_0, %out6_0_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in6_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out6_0_cons_lock_0, Release, 1)
      aie.use_lock(%out6_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in6_0_cons_buff_1, %out6_0_buff_1, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in6_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out6_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%out6_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in6_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in6_0_cons_buff_0, %out6_0_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in6_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out6_0_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "silu.o"}
    %core_3_3 = aie.core(%tile_3_3) {
      %c512_i32 = arith.constant 512 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out6_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in6_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in6_1_cons_buff_0, %out6_1_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in6_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out6_1_cons_lock_0, Release, 1)
      aie.use_lock(%out6_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in6_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in6_1_cons_buff_1, %out6_1_buff_1, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in6_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out6_1_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%out6_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in6_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in6_1_cons_buff_0, %out6_1_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in6_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out6_1_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "silu.o"}
    %core_3_4 = aie.core(%tile_3_4) {
      %c512_i32 = arith.constant 512 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out7_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in7_0_cons_buff_0, %out7_0_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in7_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out7_0_cons_lock_0, Release, 1)
      aie.use_lock(%out7_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in7_0_cons_buff_1, %out7_0_buff_1, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in7_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out7_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%out7_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in7_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in7_0_cons_buff_0, %out7_0_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in7_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out7_0_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "silu.o"}
    %core_3_5 = aie.core(%tile_3_5) {
      %c512_i32 = arith.constant 512 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out7_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in7_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in7_1_cons_buff_0, %out7_1_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in7_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out7_1_cons_lock_0, Release, 1)
      aie.use_lock(%out7_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in7_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in7_1_cons_buff_1, %out7_1_buff_1, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in7_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out7_1_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%out7_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in7_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @silu_bf16(%in7_1_cons_buff_0, %out7_1_buff_0, %c512_i32) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
      aie.use_lock(%in7_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out7_1_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "silu.o"}
    aie.runtime_sequence(%arg0: memref<8192xbf16>, %arg1: memref<8192xbf16>) {
      %0 = aiex.dma_configure_task_for @in0_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 0, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @in0_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 512, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @in1_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 1024, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @in1_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 1536, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @in2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 2048, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @in2_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 2560, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @in3_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 3072, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @in3_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 3584, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @in4_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 4096, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @in4_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 4608, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @in5_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 5120, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @in5_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 5632, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @in6_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 6144, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @in6_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 6656, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @in7_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 7168, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @in7_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 7680, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @out0_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 0, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @out0_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 512, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @out1_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 1024, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @out1_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 1536, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @out2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 2048, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @out2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 2560, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @out3_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 3072, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @out3_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 3584, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @out4_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 4096, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @out4_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 4608, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @out5_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 5120, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @out5_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 5632, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @out6_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 6144, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @out6_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 6656, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @out7_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 7168, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @out7_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 7680, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%17)
      aiex.dma_await_task(%18)
      aiex.dma_await_task(%19)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%23)
      aiex.dma_await_task(%24)
      aiex.dma_await_task(%25)
      aiex.dma_await_task(%26)
      aiex.dma_await_task(%27)
      aiex.dma_await_task(%28)
      aiex.dma_await_task(%29)
      aiex.dma_await_task(%30)
      aiex.dma_await_task(%31)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%12)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%15)
    }
    aie.shim_dma_allocation @in0_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in0_0_cons_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in0_0_cons_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out0_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out0_0_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out0_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out0_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out0_0_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out0_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in0_1_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in0_1_cons_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in0_1_cons_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out0_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out0_1_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out0_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out0_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out0_1_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out0_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in1_0_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_0_cons_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_0_cons_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_0_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_0_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in1_1_shim_alloc(%shim_noc_tile_1_0, MM2S, 1)
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_1_cons_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_1_cons_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_1_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out1_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_1_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out1_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in2_0_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_0_cons_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_0_cons_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out2_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_0_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out2_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out2_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_0_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out2_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in2_1_shim_alloc(%shim_noc_tile_2_0, MM2S, 1)
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_1_cons_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_1_cons_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out2_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_1_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out2_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out2_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_1_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out2_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in3_0_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in3_0_cons_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in3_0_cons_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out3_0_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out3_0_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in3_1_shim_alloc(%shim_noc_tile_3_0, MM2S, 1)
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in3_1_cons_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in3_1_cons_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out3_1_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out3_1_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in4_0_shim_alloc(%shim_noc_tile_4_0, MM2S, 0)
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in4_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in4_0_cons_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in4_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in4_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in4_0_cons_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in4_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out4_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out4_0_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out4_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out4_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out4_0_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out4_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in4_1_shim_alloc(%shim_noc_tile_4_0, MM2S, 1)
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in4_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in4_1_cons_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in4_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in4_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in4_1_cons_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in4_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out4_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out4_1_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out4_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out4_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out4_1_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out4_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in5_0_shim_alloc(%shim_noc_tile_5_0, MM2S, 0)
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in5_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in5_0_cons_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in5_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in5_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in5_0_cons_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in5_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out5_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out5_0_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out5_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out5_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out5_0_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out5_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in5_1_shim_alloc(%shim_noc_tile_5_0, MM2S, 1)
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in5_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in5_1_cons_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in5_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in5_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in5_1_cons_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in5_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out5_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out5_1_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out5_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out5_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out5_1_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out5_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in6_0_shim_alloc(%shim_noc_tile_6_0, MM2S, 0)
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in6_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in6_0_cons_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in6_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in6_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in6_0_cons_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in6_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out6_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out6_0_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out6_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out6_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out6_0_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out6_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in6_1_shim_alloc(%shim_noc_tile_6_0, MM2S, 1)
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in6_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in6_1_cons_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in6_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in6_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in6_1_cons_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in6_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out6_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out6_1_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out6_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out6_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out6_1_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out6_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in7_0_shim_alloc(%shim_noc_tile_7_0, MM2S, 0)
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in7_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in7_0_cons_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in7_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in7_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in7_0_cons_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in7_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out7_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out7_0_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out7_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out7_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out7_0_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out7_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in7_1_shim_alloc(%shim_noc_tile_7_0, MM2S, 1)
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in7_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in7_1_cons_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in7_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in7_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in7_1_cons_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in7_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out7_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out7_1_buff_0 : memref<512xbf16>, 0, 512) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out7_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out7_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out7_1_buff_1 : memref<512xbf16>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out7_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @out0_0_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @out0_1_shim_alloc(%shim_noc_tile_0_0, S2MM, 1)
    aie.shim_dma_allocation @out1_0_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @out1_1_shim_alloc(%shim_noc_tile_1_0, S2MM, 1)
    aie.shim_dma_allocation @out2_0_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @out2_1_shim_alloc(%shim_noc_tile_2_0, S2MM, 1)
    aie.shim_dma_allocation @out3_0_shim_alloc(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @out3_1_shim_alloc(%shim_noc_tile_3_0, S2MM, 1)
    aie.shim_dma_allocation @out4_0_shim_alloc(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @out4_1_shim_alloc(%shim_noc_tile_4_0, S2MM, 1)
    aie.shim_dma_allocation @out5_0_shim_alloc(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @out5_1_shim_alloc(%shim_noc_tile_5_0, S2MM, 1)
    aie.shim_dma_allocation @out6_0_shim_alloc(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @out6_1_shim_alloc(%shim_noc_tile_6_0, S2MM, 1)
    aie.shim_dma_allocation @out7_0_shim_alloc(%shim_noc_tile_7_0, S2MM, 0)
    aie.shim_dma_allocation @out7_1_shim_alloc(%shim_noc_tile_7_0, S2MM, 1)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_4_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_4_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_5_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_5_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_6_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_6_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_7_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_7_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<South : 7, North : 2>
      aie.connect<North : 1, South : 2>
      aie.connect<North : 3, South : 3>
      aie.connect<North : 0, East : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 2, North : 2>
      aie.connect<North : 1, South : 1>
      aie.connect<North : 3, South : 3>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 2, North : 0>
      aie.connect<East : 0, North : 5>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 3>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 5, North : 4>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<South : 4, DMA : 0>
      aie.connect<DMA : 0, East : 0>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<South : 7, North : 5>
      aie.connect<East : 2, North : 3>
      aie.connect<East : 1, North : 2>
      aie.connect<North : 1, South : 2>
      aie.connect<West : 3, South : 3>
      aie.connect<North : 0, East : 2>
      aie.connect<North : 3, East : 0>
      aie.connect<North : 2, East : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 5, North : 5>
      aie.connect<South : 3, North : 3>
      aie.connect<South : 2, North : 2>
      aie.connect<North : 1, South : 1>
      aie.connect<North : 0, South : 0>
      aie.connect<North : 3, South : 3>
      aie.connect<North : 2, South : 2>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<South : 1, West : 0>
      aie.connect<South : 5, North : 3>
      aie.connect<East : 0, DMA : 0>
      aie.connect<South : 3, North : 2>
      aie.connect<South : 2, North : 4>
      aie.connect<North : 3, South : 1>
      aie.connect<DMA : 0, East : 0>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 0, South : 3>
      aie.connect<North : 2, South : 2>
    }
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
      aie.connect<East : 0, DMA : 0>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<South : 3, North : 5>
      aie.connect<South : 2, DMA : 0>
      aie.connect<East : 0, North : 1>
      aie.connect<South : 4, North : 2>
      aie.connect<North : 1, South : 3>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 3, South : 0>
      aie.connect<North : 2, South : 2>
    }
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<South : 5, North : 5>
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 2, North : 2>
      aie.connect<West : 0, South : 1>
      aie.connect<DMA : 0, South : 3>
      aie.connect<North : 0, South : 2>
    }
    %switchbox_1_5 = aie.switchbox(%tile_1_5) {
      aie.connect<South : 5, West : 0>
      aie.connect<South : 2, DMA : 0>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<South : 7, West : 2>
      aie.connect<East : 2, North : 5>
      aie.connect<East : 1, West : 1>
      aie.connect<East : 0, North : 2>
      aie.connect<North : 2, South : 2>
      aie.connect<West : 2, South : 3>
      aie.connect<West : 0, East : 2>
      aie.connect<West : 3, East : 0>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_2_0 = aie.shim_mux(%shim_noc_tile_2_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %mem_tile_2_1 = aie.tile(2, 1)
    %switchbox_2_1 = aie.switchbox(%mem_tile_2_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 5, North : 5>
      aie.connect<South : 2, North : 2>
      aie.connect<North : 2, South : 2>
    }
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<South : 1, West : 0>
      aie.connect<South : 5, North : 5>
      aie.connect<South : 2, DMA : 0>
      aie.connect<East : 2, North : 0>
      aie.connect<West : 0, South : 2>
      aie.connect<DMA : 0, East : 0>
      aie.connect<North : 1, East : 3>
      aie.connect<North : 0, East : 2>
      aie.connect<North : 2, East : 1>
    }
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
      aie.connect<South : 5, West : 0>
      aie.connect<South : 0, DMA : 0>
      aie.connect<East : 3, North : 0>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 3, South : 2>
    }
    %switchbox_3_0 = aie.switchbox(%shim_noc_tile_3_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 2, West : 0>
      aie.connect<East : 1, North : 2>
      aie.connect<East : 0, North : 5>
      aie.connect<East : 3, North : 4>
      aie.connect<West : 2, South : 2>
      aie.connect<West : 0, South : 3>
      aie.connect<North : 2, East : 3>
      aie.connect<North : 0, East : 1>
      aie.connect<North : 1, East : 0>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_3_0 = aie.shim_mux(%shim_noc_tile_3_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %switchbox_4_0 = aie.switchbox(%shim_noc_tile_4_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 2, West : 0>
      aie.connect<East : 1, North : 2>
      aie.connect<East : 0, North : 4>
      aie.connect<East : 3, West : 3>
      aie.connect<West : 3, South : 2>
      aie.connect<North : 1, South : 3>
      aie.connect<North : 3, East : 1>
      aie.connect<West : 1, East : 3>
      aie.connect<West : 0, East : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_4_0 = aie.shim_mux(%shim_noc_tile_4_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %mem_tile_3_1 = aie.tile(3, 1)
    %switchbox_3_1 = aie.switchbox(%mem_tile_3_1) {
      aie.connect<South : 2, North : 2>
      aie.connect<South : 5, North : 5>
      aie.connect<South : 4, North : 4>
      aie.connect<North : 2, South : 2>
      aie.connect<North : 0, South : 0>
      aie.connect<North : 1, South : 1>
    }
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<South : 2, West : 2>
      aie.connect<South : 5, North : 3>
      aie.connect<East : 3, DMA : 0>
      aie.connect<South : 4, North : 0>
      aie.connect<East : 2, North : 1>
      aie.connect<West : 0, South : 2>
      aie.connect<West : 3, East : 3>
      aie.connect<West : 2, East : 1>
      aie.connect<West : 1, South : 0>
      aie.connect<DMA : 0, East : 0>
      aie.connect<North : 1, South : 1>
      aie.connect<North : 0, East : 2>
    }
    %switchbox_2_4 = aie.switchbox(%tile_2_4) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 3>
    }
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
      aie.connect<South : 3, West : 3>
      aie.connect<East : 2, North : 2>
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, North : 5>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 0>
    }
    %switchbox_5_0 = aie.switchbox(%shim_noc_tile_5_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 2, West : 0>
      aie.connect<East : 1, West : 3>
      aie.connect<East : 0, North : 3>
      aie.connect<West : 1, South : 2>
      aie.connect<West : 3, South : 3>
      aie.connect<West : 2, East : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_5_0 = aie.shim_mux(%shim_noc_tile_5_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %switchbox_2_5 = aie.switchbox(%tile_2_5) {
      aie.connect<East : 2, DMA : 0>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_3_4 = aie.switchbox(%tile_3_4) {
      aie.connect<South : 2, North : 1>
      aie.connect<East : 2, DMA : 0>
      aie.connect<South : 5, North : 5>
      aie.connect<DMA : 0, South : 1>
    }
    %switchbox_3_5 = aie.switchbox(%tile_3_5) {
      aie.connect<South : 1, West : 2>
      aie.connect<South : 5, DMA : 0>
      aie.connect<DMA : 0, East : 3>
    }
    %mem_tile_4_1 = aie.tile(4, 1)
    %switchbox_4_1 = aie.switchbox(%mem_tile_4_1) {
      aie.connect<South : 2, North : 2>
      aie.connect<South : 4, North : 4>
      aie.connect<North : 1, South : 1>
      aie.connect<North : 3, South : 3>
    }
    %tile_4_2 = aie.tile(4, 2)
    %switchbox_4_2 = aie.switchbox(%tile_4_2) {
      aie.connect<South : 2, North : 5>
      aie.connect<South : 4, West : 3>
      aie.connect<East : 3, West : 2>
      aie.connect<West : 3, South : 1>
      aie.connect<West : 1, South : 3>
      aie.connect<West : 0, East : 3>
      aie.connect<West : 2, East : 0>
    }
    %tile_4_3 = aie.tile(4, 3)
    %switchbox_4_3 = aie.switchbox(%tile_4_3) {
      aie.connect<South : 5, West : 2>
      aie.connect<North : 3, East : 2>
    }
    %switchbox_6_0 = aie.switchbox(%shim_noc_tile_6_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 3, West : 0>
      aie.connect<North : 0, South : 2>
      aie.connect<West : 3, South : 3>
      aie.connect<North : 1, East : 0>
      aie.connect<North : 3, East : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_6_0 = aie.shim_mux(%shim_noc_tile_6_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %tile_4_4 = aie.tile(4, 4)
    %switchbox_4_4 = aie.switchbox(%tile_4_4) {
      aie.connect<East : 1, West : 2>
      aie.connect<North : 3, South : 3>
    }
    %tile_5_2 = aie.tile(5, 2)
    %switchbox_5_2 = aie.switchbox(%tile_5_2) {
      aie.connect<East : 3, North : 5>
      aie.connect<South : 3, West : 3>
      aie.connect<West : 3, East : 3>
      aie.connect<West : 0, East : 2>
    }
    %tile_5_3 = aie.tile(5, 3)
    %switchbox_5_3 = aie.switchbox(%tile_5_3) {
      aie.connect<South : 5, North : 1>
      aie.connect<West : 2, East : 3>
    }
    %tile_5_4 = aie.tile(5, 4)
    %switchbox_5_4 = aie.switchbox(%tile_5_4) {
      aie.connect<South : 1, West : 1>
    }
    %tile_6_2 = aie.tile(6, 2)
    %switchbox_6_2 = aie.switchbox(%tile_6_2) {
      aie.connect<East : 1, West : 3>
      aie.connect<West : 3, South : 0>
      aie.connect<West : 2, South : 1>
      aie.connect<North : 1, South : 3>
    }
    %switchbox_7_0 = aie.switchbox(%shim_noc_tile_7_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, West : 3>
      aie.connect<West : 0, South : 2>
      aie.connect<West : 2, South : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_7_0 = aie.shim_mux(%shim_noc_tile_7_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %mem_tile_7_1 = aie.tile(7, 1)
    %switchbox_7_1 = aie.switchbox(%mem_tile_7_1) {
      aie.connect<South : 0, North : 0>
    }
    %tile_7_2 = aie.tile(7, 2)
    %switchbox_7_2 = aie.switchbox(%tile_7_2) {
      aie.connect<South : 0, West : 1>
    }
    %mem_tile_5_1 = aie.tile(5, 1)
    %switchbox_5_1 = aie.switchbox(%mem_tile_5_1) {
      aie.connect<South : 3, North : 3>
    }
    %mem_tile_6_1 = aie.tile(6, 1)
    %switchbox_6_1 = aie.switchbox(%mem_tile_6_1) {
      aie.connect<North : 0, South : 0>
      aie.connect<North : 1, South : 1>
      aie.connect<North : 3, South : 3>
    }
    %tile_4_5 = aie.tile(4, 5)
    %switchbox_4_5 = aie.switchbox(%tile_4_5) {
      aie.connect<West : 3, South : 3>
    }
    %tile_6_3 = aie.tile(6, 3)
    %switchbox_6_3 = aie.switchbox(%tile_6_3) {
      aie.connect<West : 3, South : 1>
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%tile_0_4 : Core, %switchbox_0_4 : Core)
    aie.wire(%tile_0_4 : DMA, %switchbox_0_4 : DMA)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
    aie.wire(%tile_0_5 : Core, %switchbox_0_5 : Core)
    aie.wire(%tile_0_5 : DMA, %switchbox_0_5 : DMA)
    aie.wire(%switchbox_0_4 : North, %switchbox_0_5 : South)
    aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%shim_noc_tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
    aie.wire(%switchbox_0_3 : East, %switchbox_1_3 : West)
    aie.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
    aie.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
    aie.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
    aie.wire(%switchbox_0_4 : East, %switchbox_1_4 : West)
    aie.wire(%tile_1_4 : Core, %switchbox_1_4 : Core)
    aie.wire(%tile_1_4 : DMA, %switchbox_1_4 : DMA)
    aie.wire(%switchbox_1_3 : North, %switchbox_1_4 : South)
    aie.wire(%switchbox_0_5 : East, %switchbox_1_5 : West)
    aie.wire(%tile_1_5 : Core, %switchbox_1_5 : Core)
    aie.wire(%tile_1_5 : DMA, %switchbox_1_5 : DMA)
    aie.wire(%switchbox_1_4 : North, %switchbox_1_5 : South)
    aie.wire(%switchbox_1_0 : East, %switchbox_2_0 : West)
    aie.wire(%shim_mux_2_0 : North, %switchbox_2_0 : South)
    aie.wire(%shim_noc_tile_2_0 : DMA, %shim_mux_2_0 : DMA)
    aie.wire(%switchbox_1_1 : East, %switchbox_2_1 : West)
    aie.wire(%mem_tile_2_1 : Core, %switchbox_2_1 : Core)
    aie.wire(%mem_tile_2_1 : DMA, %switchbox_2_1 : DMA)
    aie.wire(%switchbox_2_0 : North, %switchbox_2_1 : South)
    aie.wire(%switchbox_1_2 : East, %switchbox_2_2 : West)
    aie.wire(%tile_2_2 : Core, %switchbox_2_2 : Core)
    aie.wire(%tile_2_2 : DMA, %switchbox_2_2 : DMA)
    aie.wire(%switchbox_2_1 : North, %switchbox_2_2 : South)
    aie.wire(%switchbox_1_3 : East, %switchbox_2_3 : West)
    aie.wire(%tile_2_3 : Core, %switchbox_2_3 : Core)
    aie.wire(%tile_2_3 : DMA, %switchbox_2_3 : DMA)
    aie.wire(%switchbox_2_2 : North, %switchbox_2_3 : South)
    aie.wire(%switchbox_1_4 : East, %switchbox_2_4 : West)
    aie.wire(%tile_2_4 : Core, %switchbox_2_4 : Core)
    aie.wire(%tile_2_4 : DMA, %switchbox_2_4 : DMA)
    aie.wire(%switchbox_2_3 : North, %switchbox_2_4 : South)
    aie.wire(%switchbox_1_5 : East, %switchbox_2_5 : West)
    aie.wire(%tile_2_5 : Core, %switchbox_2_5 : Core)
    aie.wire(%tile_2_5 : DMA, %switchbox_2_5 : DMA)
    aie.wire(%switchbox_2_4 : North, %switchbox_2_5 : South)
    aie.wire(%switchbox_2_0 : East, %switchbox_3_0 : West)
    aie.wire(%shim_mux_3_0 : North, %switchbox_3_0 : South)
    aie.wire(%shim_noc_tile_3_0 : DMA, %shim_mux_3_0 : DMA)
    aie.wire(%switchbox_2_1 : East, %switchbox_3_1 : West)
    aie.wire(%mem_tile_3_1 : Core, %switchbox_3_1 : Core)
    aie.wire(%mem_tile_3_1 : DMA, %switchbox_3_1 : DMA)
    aie.wire(%switchbox_3_0 : North, %switchbox_3_1 : South)
    aie.wire(%switchbox_2_2 : East, %switchbox_3_2 : West)
    aie.wire(%tile_3_2 : Core, %switchbox_3_2 : Core)
    aie.wire(%tile_3_2 : DMA, %switchbox_3_2 : DMA)
    aie.wire(%switchbox_3_1 : North, %switchbox_3_2 : South)
    aie.wire(%switchbox_2_3 : East, %switchbox_3_3 : West)
    aie.wire(%tile_3_3 : Core, %switchbox_3_3 : Core)
    aie.wire(%tile_3_3 : DMA, %switchbox_3_3 : DMA)
    aie.wire(%switchbox_3_2 : North, %switchbox_3_3 : South)
    aie.wire(%switchbox_2_4 : East, %switchbox_3_4 : West)
    aie.wire(%tile_3_4 : Core, %switchbox_3_4 : Core)
    aie.wire(%tile_3_4 : DMA, %switchbox_3_4 : DMA)
    aie.wire(%switchbox_3_3 : North, %switchbox_3_4 : South)
    aie.wire(%switchbox_2_5 : East, %switchbox_3_5 : West)
    aie.wire(%tile_3_5 : Core, %switchbox_3_5 : Core)
    aie.wire(%tile_3_5 : DMA, %switchbox_3_5 : DMA)
    aie.wire(%switchbox_3_4 : North, %switchbox_3_5 : South)
    aie.wire(%switchbox_3_0 : East, %switchbox_4_0 : West)
    aie.wire(%shim_mux_4_0 : North, %switchbox_4_0 : South)
    aie.wire(%shim_noc_tile_4_0 : DMA, %shim_mux_4_0 : DMA)
    aie.wire(%switchbox_3_1 : East, %switchbox_4_1 : West)
    aie.wire(%mem_tile_4_1 : Core, %switchbox_4_1 : Core)
    aie.wire(%mem_tile_4_1 : DMA, %switchbox_4_1 : DMA)
    aie.wire(%switchbox_4_0 : North, %switchbox_4_1 : South)
    aie.wire(%switchbox_3_2 : East, %switchbox_4_2 : West)
    aie.wire(%tile_4_2 : Core, %switchbox_4_2 : Core)
    aie.wire(%tile_4_2 : DMA, %switchbox_4_2 : DMA)
    aie.wire(%switchbox_4_1 : North, %switchbox_4_2 : South)
    aie.wire(%switchbox_3_3 : East, %switchbox_4_3 : West)
    aie.wire(%tile_4_3 : Core, %switchbox_4_3 : Core)
    aie.wire(%tile_4_3 : DMA, %switchbox_4_3 : DMA)
    aie.wire(%switchbox_4_2 : North, %switchbox_4_3 : South)
    aie.wire(%switchbox_3_4 : East, %switchbox_4_4 : West)
    aie.wire(%tile_4_4 : Core, %switchbox_4_4 : Core)
    aie.wire(%tile_4_4 : DMA, %switchbox_4_4 : DMA)
    aie.wire(%switchbox_4_3 : North, %switchbox_4_4 : South)
    aie.wire(%switchbox_3_5 : East, %switchbox_4_5 : West)
    aie.wire(%tile_4_5 : Core, %switchbox_4_5 : Core)
    aie.wire(%tile_4_5 : DMA, %switchbox_4_5 : DMA)
    aie.wire(%switchbox_4_4 : North, %switchbox_4_5 : South)
    aie.wire(%switchbox_4_0 : East, %switchbox_5_0 : West)
    aie.wire(%shim_mux_5_0 : North, %switchbox_5_0 : South)
    aie.wire(%shim_noc_tile_5_0 : DMA, %shim_mux_5_0 : DMA)
    aie.wire(%switchbox_4_1 : East, %switchbox_5_1 : West)
    aie.wire(%mem_tile_5_1 : Core, %switchbox_5_1 : Core)
    aie.wire(%mem_tile_5_1 : DMA, %switchbox_5_1 : DMA)
    aie.wire(%switchbox_5_0 : North, %switchbox_5_1 : South)
    aie.wire(%switchbox_4_2 : East, %switchbox_5_2 : West)
    aie.wire(%tile_5_2 : Core, %switchbox_5_2 : Core)
    aie.wire(%tile_5_2 : DMA, %switchbox_5_2 : DMA)
    aie.wire(%switchbox_5_1 : North, %switchbox_5_2 : South)
    aie.wire(%switchbox_4_3 : East, %switchbox_5_3 : West)
    aie.wire(%tile_5_3 : Core, %switchbox_5_3 : Core)
    aie.wire(%tile_5_3 : DMA, %switchbox_5_3 : DMA)
    aie.wire(%switchbox_5_2 : North, %switchbox_5_3 : South)
    aie.wire(%switchbox_4_4 : East, %switchbox_5_4 : West)
    aie.wire(%tile_5_4 : Core, %switchbox_5_4 : Core)
    aie.wire(%tile_5_4 : DMA, %switchbox_5_4 : DMA)
    aie.wire(%switchbox_5_3 : North, %switchbox_5_4 : South)
    aie.wire(%switchbox_5_0 : East, %switchbox_6_0 : West)
    aie.wire(%shim_mux_6_0 : North, %switchbox_6_0 : South)
    aie.wire(%shim_noc_tile_6_0 : DMA, %shim_mux_6_0 : DMA)
    aie.wire(%switchbox_5_1 : East, %switchbox_6_1 : West)
    aie.wire(%mem_tile_6_1 : Core, %switchbox_6_1 : Core)
    aie.wire(%mem_tile_6_1 : DMA, %switchbox_6_1 : DMA)
    aie.wire(%switchbox_6_0 : North, %switchbox_6_1 : South)
    aie.wire(%switchbox_5_2 : East, %switchbox_6_2 : West)
    aie.wire(%tile_6_2 : Core, %switchbox_6_2 : Core)
    aie.wire(%tile_6_2 : DMA, %switchbox_6_2 : DMA)
    aie.wire(%switchbox_6_1 : North, %switchbox_6_2 : South)
    aie.wire(%switchbox_5_3 : East, %switchbox_6_3 : West)
    aie.wire(%tile_6_3 : Core, %switchbox_6_3 : Core)
    aie.wire(%tile_6_3 : DMA, %switchbox_6_3 : DMA)
    aie.wire(%switchbox_6_2 : North, %switchbox_6_3 : South)
    aie.wire(%switchbox_6_0 : East, %switchbox_7_0 : West)
    aie.wire(%shim_mux_7_0 : North, %switchbox_7_0 : South)
    aie.wire(%shim_noc_tile_7_0 : DMA, %shim_mux_7_0 : DMA)
    aie.wire(%switchbox_6_1 : East, %switchbox_7_1 : West)
    aie.wire(%mem_tile_7_1 : Core, %switchbox_7_1 : Core)
    aie.wire(%mem_tile_7_1 : DMA, %switchbox_7_1 : DMA)
    aie.wire(%switchbox_7_0 : North, %switchbox_7_1 : South)
    aie.wire(%switchbox_6_2 : East, %switchbox_7_2 : West)
    aie.wire(%tile_7_2 : Core, %switchbox_7_2 : Core)
    aie.wire(%tile_7_2 : DMA, %switchbox_7_2 : DMA)
    aie.wire(%switchbox_7_1 : North, %switchbox_7_2 : South)
  }
  aie.device(npu2) @eltwise_mul {
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_4_0 = aie.tile(4, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_5_0 = aie.tile(5, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_6_0 = aie.tile(6, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_7_0 = aie.tile(7, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %out_7_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 6) {init = 0 : i32, sym_name = "out_7_cons_prod_lock_0"}
    %out_7_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 7) {init = 0 : i32, sym_name = "out_7_cons_cons_lock_0"}
    %out_7_buff_0 = aie.buffer(%tile_1_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_7_buff_0"} : memref<1024xbf16> 
    %out_7_buff_1 = aie.buffer(%tile_1_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_7_buff_1"} : memref<1024xbf16> 
    %out_7_prod_lock_0 = aie.lock(%tile_1_5, 4) {init = 2 : i32, sym_name = "out_7_prod_lock_0"}
    %out_7_cons_lock_0 = aie.lock(%tile_1_5, 5) {init = 0 : i32, sym_name = "out_7_cons_lock_0"}
    %out_6_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 4) {init = 0 : i32, sym_name = "out_6_cons_prod_lock_0"}
    %out_6_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 5) {init = 0 : i32, sym_name = "out_6_cons_cons_lock_0"}
    %out_6_buff_0 = aie.buffer(%tile_1_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_6_buff_0"} : memref<1024xbf16> 
    %out_6_buff_1 = aie.buffer(%tile_1_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_6_buff_1"} : memref<1024xbf16> 
    %out_6_prod_lock_0 = aie.lock(%tile_1_4, 4) {init = 2 : i32, sym_name = "out_6_prod_lock_0"}
    %out_6_cons_lock_0 = aie.lock(%tile_1_4, 5) {init = 0 : i32, sym_name = "out_6_cons_lock_0"}
    %out_5_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 6) {init = 0 : i32, sym_name = "out_5_cons_prod_lock_0"}
    %out_5_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 7) {init = 0 : i32, sym_name = "out_5_cons_cons_lock_0"}
    %out_5_buff_0 = aie.buffer(%tile_1_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_5_buff_0"} : memref<1024xbf16> 
    %out_5_buff_1 = aie.buffer(%tile_1_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_5_buff_1"} : memref<1024xbf16> 
    %out_5_prod_lock_0 = aie.lock(%tile_1_3, 4) {init = 2 : i32, sym_name = "out_5_prod_lock_0"}
    %out_5_cons_lock_0 = aie.lock(%tile_1_3, 5) {init = 0 : i32, sym_name = "out_5_cons_lock_0"}
    %out_4_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 4) {init = 0 : i32, sym_name = "out_4_cons_prod_lock_0"}
    %out_4_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 5) {init = 0 : i32, sym_name = "out_4_cons_cons_lock_0"}
    %out_4_buff_0 = aie.buffer(%tile_1_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_4_buff_0"} : memref<1024xbf16> 
    %out_4_buff_1 = aie.buffer(%tile_1_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_4_buff_1"} : memref<1024xbf16> 
    %out_4_prod_lock_0 = aie.lock(%tile_1_2, 4) {init = 2 : i32, sym_name = "out_4_prod_lock_0"}
    %out_4_cons_lock_0 = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "out_4_cons_lock_0"}
    %out_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 6) {init = 0 : i32, sym_name = "out_3_cons_prod_lock_0"}
    %out_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 7) {init = 0 : i32, sym_name = "out_3_cons_cons_lock_0"}
    %out_3_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_3_buff_0"} : memref<1024xbf16> 
    %out_3_buff_1 = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_3_buff_1"} : memref<1024xbf16> 
    %out_3_prod_lock_0 = aie.lock(%tile_0_5, 4) {init = 2 : i32, sym_name = "out_3_prod_lock_0"}
    %out_3_cons_lock_0 = aie.lock(%tile_0_5, 5) {init = 0 : i32, sym_name = "out_3_cons_lock_0"}
    %out_2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 4) {init = 0 : i32, sym_name = "out_2_cons_prod_lock_0"}
    %out_2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 5) {init = 0 : i32, sym_name = "out_2_cons_cons_lock_0"}
    %out_2_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_2_buff_0"} : memref<1024xbf16> 
    %out_2_buff_1 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_2_buff_1"} : memref<1024xbf16> 
    %out_2_prod_lock_0 = aie.lock(%tile_0_4, 4) {init = 2 : i32, sym_name = "out_2_prod_lock_0"}
    %out_2_cons_lock_0 = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "out_2_cons_lock_0"}
    %out_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 6) {init = 0 : i32, sym_name = "out_1_cons_prod_lock_0"}
    %out_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 7) {init = 0 : i32, sym_name = "out_1_cons_cons_lock_0"}
    %out_1_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_1_buff_0"} : memref<1024xbf16> 
    %out_1_buff_1 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_1_buff_1"} : memref<1024xbf16> 
    %out_1_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 2 : i32, sym_name = "out_1_prod_lock_0"}
    %out_1_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "out_1_cons_lock_0"}
    %out_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "out_0_cons_prod_lock_0"}
    %out_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "out_0_cons_cons_lock_0"}
    %out_0_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_0_buff_0"} : memref<1024xbf16> 
    %out_0_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_0_buff_1"} : memref<1024xbf16> 
    %out_0_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "out_0_prod_lock_0"}
    %out_0_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "out_0_cons_lock_0"}
    %in2_7_cons_buff_0 = aie.buffer(%tile_1_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_7_cons_buff_0"} : memref<1024xbf16> 
    %in2_7_cons_buff_1 = aie.buffer(%tile_1_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_7_cons_buff_1"} : memref<1024xbf16> 
    %in2_7_cons_prod_lock_0 = aie.lock(%tile_1_5, 2) {init = 2 : i32, sym_name = "in2_7_cons_prod_lock_0"}
    %in2_7_cons_cons_lock_0 = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "in2_7_cons_cons_lock_0"}
    %in2_7_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 2) {init = 0 : i32, sym_name = "in2_7_prod_lock_0"}
    %in2_7_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 3) {init = 0 : i32, sym_name = "in2_7_cons_lock_0"}
    %in2_6_cons_buff_0 = aie.buffer(%tile_1_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_6_cons_buff_0"} : memref<1024xbf16> 
    %in2_6_cons_buff_1 = aie.buffer(%tile_1_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_6_cons_buff_1"} : memref<1024xbf16> 
    %in2_6_cons_prod_lock_0 = aie.lock(%tile_1_4, 2) {init = 2 : i32, sym_name = "in2_6_cons_prod_lock_0"}
    %in2_6_cons_cons_lock_0 = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "in2_6_cons_cons_lock_0"}
    %in2_6_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 0) {init = 0 : i32, sym_name = "in2_6_prod_lock_0"}
    %in2_6_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 1) {init = 0 : i32, sym_name = "in2_6_cons_lock_0"}
    %in2_5_cons_buff_0 = aie.buffer(%tile_1_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_5_cons_buff_0"} : memref<1024xbf16> 
    %in2_5_cons_buff_1 = aie.buffer(%tile_1_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_5_cons_buff_1"} : memref<1024xbf16> 
    %in2_5_cons_prod_lock_0 = aie.lock(%tile_1_3, 2) {init = 2 : i32, sym_name = "in2_5_cons_prod_lock_0"}
    %in2_5_cons_cons_lock_0 = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "in2_5_cons_cons_lock_0"}
    %in2_5_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 2) {init = 0 : i32, sym_name = "in2_5_prod_lock_0"}
    %in2_5_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 3) {init = 0 : i32, sym_name = "in2_5_cons_lock_0"}
    %in2_4_cons_buff_0 = aie.buffer(%tile_1_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_4_cons_buff_0"} : memref<1024xbf16> 
    %in2_4_cons_buff_1 = aie.buffer(%tile_1_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_4_cons_buff_1"} : memref<1024xbf16> 
    %in2_4_cons_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "in2_4_cons_prod_lock_0"}
    %in2_4_cons_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "in2_4_cons_cons_lock_0"}
    %in2_4_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 0) {init = 0 : i32, sym_name = "in2_4_prod_lock_0"}
    %in2_4_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 1) {init = 0 : i32, sym_name = "in2_4_cons_lock_0"}
    %in2_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_3_cons_buff_0"} : memref<1024xbf16> 
    %in2_3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_3_cons_buff_1"} : memref<1024xbf16> 
    %in2_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 2 : i32, sym_name = "in2_3_cons_prod_lock_0"}
    %in2_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "in2_3_cons_cons_lock_0"}
    %in2_3_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 2) {init = 0 : i32, sym_name = "in2_3_prod_lock_0"}
    %in2_3_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 3) {init = 0 : i32, sym_name = "in2_3_cons_lock_0"}
    %in2_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_2_cons_buff_0"} : memref<1024xbf16> 
    %in2_2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_2_cons_buff_1"} : memref<1024xbf16> 
    %in2_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "in2_2_cons_prod_lock_0"}
    %in2_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "in2_2_cons_cons_lock_0"}
    %in2_2_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 0) {init = 0 : i32, sym_name = "in2_2_prod_lock_0"}
    %in2_2_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 1) {init = 0 : i32, sym_name = "in2_2_cons_lock_0"}
    %in2_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_1_cons_buff_0"} : memref<1024xbf16> 
    %in2_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_1_cons_buff_1"} : memref<1024xbf16> 
    %in2_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "in2_1_cons_prod_lock_0"}
    %in2_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "in2_1_cons_cons_lock_0"}
    %in2_1_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 2) {init = 0 : i32, sym_name = "in2_1_prod_lock_0"}
    %in2_1_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 3) {init = 0 : i32, sym_name = "in2_1_cons_lock_0"}
    %in2_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_0_cons_buff_0"} : memref<1024xbf16> 
    %in2_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_0_cons_buff_1"} : memref<1024xbf16> 
    %in2_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "in2_0_cons_prod_lock_0"}
    %in2_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "in2_0_cons_cons_lock_0"}
    %in2_0_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 0) {init = 0 : i32, sym_name = "in2_0_prod_lock_0"}
    %in2_0_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 1) {init = 0 : i32, sym_name = "in2_0_cons_lock_0"}
    %in1_7_cons_buff_0 = aie.buffer(%tile_1_5) {address = 3072 : i32, mem_bank = 0 : i32, sym_name = "in1_7_cons_buff_0"} : memref<1024xbf16> 
    %in1_7_cons_buff_1 = aie.buffer(%tile_1_5) {address = 18432 : i32, mem_bank = 1 : i32, sym_name = "in1_7_cons_buff_1"} : memref<1024xbf16> 
    %in1_7_cons_prod_lock_0 = aie.lock(%tile_1_5, 0) {init = 2 : i32, sym_name = "in1_7_cons_prod_lock_0"}
    %in1_7_cons_cons_lock_0 = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "in1_7_cons_cons_lock_0"}
    %in1_7_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "in1_7_prod_lock_0"}
    %in1_7_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "in1_7_cons_lock_0"}
    %in1_6_cons_buff_0 = aie.buffer(%tile_1_4) {address = 3072 : i32, mem_bank = 0 : i32, sym_name = "in1_6_cons_buff_0"} : memref<1024xbf16> 
    %in1_6_cons_buff_1 = aie.buffer(%tile_1_4) {address = 18432 : i32, mem_bank = 1 : i32, sym_name = "in1_6_cons_buff_1"} : memref<1024xbf16> 
    %in1_6_cons_prod_lock_0 = aie.lock(%tile_1_4, 0) {init = 2 : i32, sym_name = "in1_6_cons_prod_lock_0"}
    %in1_6_cons_cons_lock_0 = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "in1_6_cons_cons_lock_0"}
    %in1_6_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "in1_6_prod_lock_0"}
    %in1_6_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "in1_6_cons_lock_0"}
    %in1_5_cons_buff_0 = aie.buffer(%tile_1_3) {address = 3072 : i32, mem_bank = 0 : i32, sym_name = "in1_5_cons_buff_0"} : memref<1024xbf16> 
    %in1_5_cons_buff_1 = aie.buffer(%tile_1_3) {address = 18432 : i32, mem_bank = 1 : i32, sym_name = "in1_5_cons_buff_1"} : memref<1024xbf16> 
    %in1_5_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "in1_5_cons_prod_lock_0"}
    %in1_5_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "in1_5_cons_cons_lock_0"}
    %in1_5_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "in1_5_prod_lock_0"}
    %in1_5_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "in1_5_cons_lock_0"}
    %in1_4_cons_buff_0 = aie.buffer(%tile_1_2) {address = 3072 : i32, mem_bank = 0 : i32, sym_name = "in1_4_cons_buff_0"} : memref<1024xbf16> 
    %in1_4_cons_buff_1 = aie.buffer(%tile_1_2) {address = 18432 : i32, mem_bank = 1 : i32, sym_name = "in1_4_cons_buff_1"} : memref<1024xbf16> 
    %in1_4_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "in1_4_cons_prod_lock_0"}
    %in1_4_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "in1_4_cons_cons_lock_0"}
    %in1_4_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "in1_4_prod_lock_0"}
    %in1_4_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "in1_4_cons_lock_0"}
    %in1_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 3072 : i32, mem_bank = 0 : i32, sym_name = "in1_3_cons_buff_0"} : memref<1024xbf16> 
    %in1_3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 18432 : i32, mem_bank = 1 : i32, sym_name = "in1_3_cons_buff_1"} : memref<1024xbf16> 
    %in1_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "in1_3_cons_prod_lock_0"}
    %in1_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "in1_3_cons_cons_lock_0"}
    %in1_3_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "in1_3_prod_lock_0"}
    %in1_3_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "in1_3_cons_lock_0"}
    %in1_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 3072 : i32, mem_bank = 0 : i32, sym_name = "in1_2_cons_buff_0"} : memref<1024xbf16> 
    %in1_2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 18432 : i32, mem_bank = 1 : i32, sym_name = "in1_2_cons_buff_1"} : memref<1024xbf16> 
    %in1_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "in1_2_cons_prod_lock_0"}
    %in1_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "in1_2_cons_cons_lock_0"}
    %in1_2_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "in1_2_prod_lock_0"}
    %in1_2_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "in1_2_cons_lock_0"}
    %in1_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 3072 : i32, mem_bank = 0 : i32, sym_name = "in1_1_cons_buff_0"} : memref<1024xbf16> 
    %in1_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 18432 : i32, mem_bank = 1 : i32, sym_name = "in1_1_cons_buff_1"} : memref<1024xbf16> 
    %in1_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "in1_1_cons_prod_lock_0"}
    %in1_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "in1_1_cons_cons_lock_0"}
    %in1_1_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "in1_1_prod_lock_0"}
    %in1_1_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "in1_1_cons_lock_0"}
    %in1_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 3072 : i32, mem_bank = 0 : i32, sym_name = "in1_0_cons_buff_0"} : memref<1024xbf16> 
    %in1_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 18432 : i32, mem_bank = 1 : i32, sym_name = "in1_0_cons_buff_1"} : memref<1024xbf16> 
    %in1_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "in1_0_cons_prod_lock_0"}
    %in1_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in1_0_cons_cons_lock_0"}
    %in1_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "in1_0_prod_lock_0"}
    %in1_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "in1_0_cons_lock_0"}
    func.func private @eltwise_mul_bf16_vector(memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32)
    %core_0_2 = aie.core(%tile_0_2) {
      %c1024_i32 = arith.constant 1024 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_0_cons_buff_0, %in2_0_cons_buff_0, %out_0_buff_0, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_0_cons_lock_0, Release, 1)
      aie.use_lock(%in1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_0_cons_buff_1, %in2_0_cons_buff_1, %out_0_buff_1, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_0_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_0_cons_buff_0, %in2_0_cons_buff_0, %out_0_buff_0, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_0_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "mul.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c1024_i32 = arith.constant 1024 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_1_cons_buff_0, %in2_1_cons_buff_0, %out_1_buff_0, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_1_cons_lock_0, Release, 1)
      aie.use_lock(%in1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_1_cons_buff_1, %in2_1_cons_buff_1, %out_1_buff_1, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_1_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_1_cons_buff_0, %in2_1_cons_buff_0, %out_1_buff_0, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_1_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "mul.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c1024_i32 = arith.constant 1024 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_2_cons_buff_0, %in2_2_cons_buff_0, %out_2_buff_0, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_2_cons_lock_0, Release, 1)
      aie.use_lock(%in1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_2_cons_buff_1, %in2_2_cons_buff_1, %out_2_buff_1, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_2_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_2_cons_buff_0, %in2_2_cons_buff_0, %out_2_buff_0, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_2_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "mul.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c1024_i32 = arith.constant 1024 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_3_cons_buff_0, %in2_3_cons_buff_0, %out_3_buff_0, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_3_cons_lock_0, Release, 1)
      aie.use_lock(%in1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_3_cons_buff_1, %in2_3_cons_buff_1, %out_3_buff_1, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_3_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_3_cons_buff_0, %in2_3_cons_buff_0, %out_3_buff_0, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_3_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "mul.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c1024_i32 = arith.constant 1024 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_4_cons_buff_0, %in2_4_cons_buff_0, %out_4_buff_0, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_4_cons_lock_0, Release, 1)
      aie.use_lock(%in1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_4_cons_buff_1, %in2_4_cons_buff_1, %out_4_buff_1, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_4_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_4_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_4_cons_buff_0, %in2_4_cons_buff_0, %out_4_buff_0, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_4_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "mul.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c1024_i32 = arith.constant 1024 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_5_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_5_cons_buff_0, %in2_5_cons_buff_0, %out_5_buff_0, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_5_cons_lock_0, Release, 1)
      aie.use_lock(%in1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_5_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_5_cons_buff_1, %in2_5_cons_buff_1, %out_5_buff_1, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_5_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_5_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_5_cons_buff_0, %in2_5_cons_buff_0, %out_5_buff_0, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_5_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "mul.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c1024_i32 = arith.constant 1024 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_6_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_6_cons_buff_0, %in2_6_cons_buff_0, %out_6_buff_0, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_6_cons_lock_0, Release, 1)
      aie.use_lock(%in1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_6_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_6_cons_buff_1, %in2_6_cons_buff_1, %out_6_buff_1, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_6_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_6_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_6_cons_buff_0, %in2_6_cons_buff_0, %out_6_buff_0, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_6_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_6_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "mul.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c1024_i32 = arith.constant 1024 : i32
      %c0 = arith.constant 0 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_7_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_7_cons_buff_0, %in2_7_cons_buff_0, %out_7_buff_0, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_7_cons_lock_0, Release, 1)
      aie.use_lock(%in1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_7_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_7_cons_buff_1, %in2_7_cons_buff_1, %out_7_buff_1, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_7_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%in1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_7_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @eltwise_mul_bf16_vector(%in1_7_cons_buff_0, %in2_7_cons_buff_0, %out_7_buff_0, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%in1_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%in2_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%out_7_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "mul.o"}
    aie.runtime_sequence(%arg0: memref<8192xbf16>, %arg1: memref<8192xbf16>, %arg2: memref<8192xbf16>) {
      %0 = aiex.dma_configure_task_for @in1_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 0, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @in2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 0, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @in1_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 1024, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @in2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 1024, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @in1_2_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 2048, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @in2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 2048, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @in1_3_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 3072, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @in2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 3072, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @in1_4_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 4096, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @in2_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 4096, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @in1_5_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 5120, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @in2_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 5120, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @in1_6_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 6144, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @in2_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 6144, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @in1_7_shim_alloc {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 7168, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @in2_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 7168, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @out_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 0, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @out_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 1024, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @out_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 2048, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @out_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 3072, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @out_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 4096, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @out_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 5120, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @out_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 6144, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @out_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 7168, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%17)
      aiex.dma_await_task(%18)
      aiex.dma_await_task(%19)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%23)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%12)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%15)
    }
    aie.shim_dma_allocation @in1_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_0_cons_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_0_cons_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_0_cons_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_0_cons_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_0_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_0_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_1_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_1_cons_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_1_cons_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_1_cons_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_1_cons_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_1_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_1_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_2_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_2_cons_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_2_cons_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_2_cons_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_2_cons_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_2_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_2_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_3_shim_alloc(%shim_noc_tile_1_0, MM2S, 1)
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_3_cons_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_3_cons_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_3_cons_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_3_cons_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_3_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_3_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_4_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_4_cons_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_4_cons_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_4_cons_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_4_cons_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_4_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_4_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_5_shim_alloc(%shim_noc_tile_2_0, MM2S, 1)
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_5_cons_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_5_cons_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_5_cons_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_5_cons_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_5_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_5_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_5_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_5_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_5_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_5_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_6_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_6_cons_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_6_cons_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_6_cons_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_6_cons_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_6_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_6_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_6_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_6_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in1_7_shim_alloc(%shim_noc_tile_3_0, MM2S, 1)
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_7_cons_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_7_cons_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_7_cons_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_7_cons_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_7_buff_0 : memref<1024xbf16>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_7_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_7_buff_1 : memref<1024xbf16>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_7_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @in2_0_shim_alloc(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @in2_1_shim_alloc(%shim_noc_tile_4_0, MM2S, 1)
    aie.shim_dma_allocation @in2_2_shim_alloc(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @in2_3_shim_alloc(%shim_noc_tile_5_0, MM2S, 1)
    aie.shim_dma_allocation @in2_4_shim_alloc(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @in2_5_shim_alloc(%shim_noc_tile_6_0, MM2S, 1)
    aie.shim_dma_allocation @in2_6_shim_alloc(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @in2_7_shim_alloc(%shim_noc_tile_7_0, MM2S, 1)
    aie.shim_dma_allocation @out_0_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @out_1_shim_alloc(%shim_noc_tile_0_0, S2MM, 1)
    aie.shim_dma_allocation @out_2_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @out_3_shim_alloc(%shim_noc_tile_1_0, S2MM, 1)
    aie.shim_dma_allocation @out_4_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @out_5_shim_alloc(%shim_noc_tile_2_0, S2MM, 1)
    aie.shim_dma_allocation @out_6_shim_alloc(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @out_7_shim_alloc(%shim_noc_tile_3_0, S2MM, 1)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_4_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_4_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_5_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_5_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_6_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_6_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_7_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_7_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<South : 7, North : 2>
      aie.connect<North : 1, South : 2>
      aie.connect<North : 3, South : 3>
      aie.connect<North : 0, East : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 2, North : 2>
      aie.connect<North : 1, South : 1>
      aie.connect<North : 3, South : 3>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 2, North : 0>
      aie.connect<East : 0, North : 5>
      aie.connect<East : 3, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 3>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 5, North : 4>
      aie.connect<East : 0, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<South : 4, DMA : 0>
      aie.connect<East : 3, DMA : 1>
      aie.connect<DMA : 0, East : 0>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<South : 7, North : 5>
      aie.connect<East : 2, North : 3>
      aie.connect<East : 1, North : 2>
      aie.connect<East : 0, North : 0>
      aie.connect<North : 0, South : 2>
      aie.connect<West : 3, South : 3>
      aie.connect<North : 2, East : 3>
      aie.connect<North : 1, East : 0>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 5, North : 5>
      aie.connect<South : 3, North : 3>
      aie.connect<South : 2, North : 2>
      aie.connect<South : 0, North : 0>
      aie.connect<North : 0, South : 0>
      aie.connect<North : 2, South : 2>
      aie.connect<North : 1, South : 1>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<South : 1, West : 0>
      aie.connect<South : 5, North : 3>
      aie.connect<East : 0, DMA : 0>
      aie.connect<South : 3, North : 2>
      aie.connect<South : 2, North : 4>
      aie.connect<South : 0, West : 3>
      aie.connect<East : 2, North : 5>
      aie.connect<East : 3, DMA : 1>
      aie.connect<East : 1, North : 0>
      aie.connect<North : 3, South : 0>
      aie.connect<DMA : 0, East : 0>
      aie.connect<North : 2, South : 2>
      aie.connect<North : 0, South : 1>
    }
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
      aie.connect<East : 0, DMA : 0>
      aie.connect<East : 3, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<South : 3, North : 5>
      aie.connect<South : 2, DMA : 0>
      aie.connect<East : 0, North : 1>
      aie.connect<South : 4, North : 2>
      aie.connect<South : 5, West : 0>
      aie.connect<East : 3, North : 4>
      aie.connect<South : 0, DMA : 1>
      aie.connect<East : 2, North : 3>
      aie.connect<North : 1, South : 3>
      aie.connect<DMA : 0, South : 2>
      aie.connect<North : 3, South : 0>
    }
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<South : 5, North : 5>
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 2, North : 2>
      aie.connect<South : 4, West : 3>
      aie.connect<South : 3, DMA : 1>
      aie.connect<West : 0, South : 1>
      aie.connect<DMA : 0, South : 3>
    }
    %switchbox_1_5 = aie.switchbox(%tile_1_5) {
      aie.connect<South : 5, West : 0>
      aie.connect<South : 2, DMA : 0>
      aie.connect<East : 3, West : 3>
      aie.connect<East : 2, DMA : 1>
      aie.connect<DMA : 0, East : 3>
    }
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<South : 7, West : 2>
      aie.connect<East : 2, North : 5>
      aie.connect<East : 1, West : 1>
      aie.connect<East : 0, West : 0>
      aie.connect<East : 3, North : 4>
      aie.connect<North : 3, South : 2>
      aie.connect<West : 3, South : 3>
      aie.connect<West : 0, East : 0>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_2_0 = aie.shim_mux(%shim_noc_tile_2_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %mem_tile_2_1 = aie.tile(2, 1)
    %switchbox_2_1 = aie.switchbox(%mem_tile_2_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 5, North : 5>
      aie.connect<South : 4, North : 4>
      aie.connect<North : 3, South : 3>
    }
    %tile_2_2 = aie.tile(2, 2)
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<South : 1, West : 0>
      aie.connect<South : 5, North : 5>
      aie.connect<South : 4, West : 2>
      aie.connect<East : 0, West : 3>
      aie.connect<East : 1, West : 1>
      aie.connect<East : 2, North : 0>
      aie.connect<West : 0, South : 3>
    }
    %tile_2_3 = aie.tile(2, 3)
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
      aie.connect<South : 5, West : 0>
      aie.connect<East : 1, West : 3>
      aie.connect<East : 2, North : 5>
      aie.connect<South : 0, West : 2>
      aie.connect<East : 3, North : 0>
    }
    %switchbox_3_0 = aie.switchbox(%shim_noc_tile_3_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 2, West : 0>
      aie.connect<East : 1, West : 3>
      aie.connect<East : 0, North : 1>
      aie.connect<East : 3, North : 5>
      aie.connect<West : 0, South : 2>
      aie.connect<North : 3, South : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_3_0 = aie.shim_mux(%shim_noc_tile_3_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %switchbox_4_0 = aie.switchbox(%shim_noc_tile_4_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 2, North : 5>
      aie.connect<East : 1, West : 0>
      aie.connect<East : 0, West : 3>
      aie.connect<East : 3, North : 0>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_4_0 = aie.shim_mux(%shim_noc_tile_4_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %tile_3_2 = aie.tile(3, 2)
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<East : 3, North : 0>
      aie.connect<South : 1, North : 3>
      aie.connect<South : 5, West : 0>
      aie.connect<East : 2, West : 1>
      aie.connect<East : 1, West : 2>
      aie.connect<East : 0, North : 4>
      aie.connect<North : 2, South : 3>
    }
    %tile_3_3 = aie.tile(3, 3)
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
      aie.connect<South : 0, West : 1>
      aie.connect<South : 3, West : 2>
      aie.connect<South : 4, West : 3>
      aie.connect<North : 3, South : 2>
    }
    %mem_tile_4_1 = aie.tile(4, 1)
    %switchbox_4_1 = aie.switchbox(%mem_tile_4_1) {
      aie.connect<South : 5, North : 5>
      aie.connect<South : 0, North : 0>
    }
    %tile_4_2 = aie.tile(4, 2)
    %switchbox_4_2 = aie.switchbox(%tile_4_2) {
      aie.connect<South : 5, West : 3>
      aie.connect<East : 1, West : 2>
      aie.connect<East : 3, West : 1>
      aie.connect<South : 0, West : 0>
    }
    %switchbox_5_0 = aie.switchbox(%shim_noc_tile_5_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 2, West : 0>
      aie.connect<East : 1, North : 5>
      aie.connect<East : 0, West : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_5_0 = aie.shim_mux(%shim_noc_tile_5_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %tile_2_4 = aie.tile(2, 4)
    %switchbox_2_4 = aie.switchbox(%tile_2_4) {
      aie.connect<South : 5, North : 5>
      aie.connect<South : 0, North : 3>
      aie.connect<North : 0, East : 3>
    }
    %tile_2_5 = aie.tile(2, 5)
    %switchbox_2_5 = aie.switchbox(%tile_2_5) {
      aie.connect<South : 5, West : 3>
      aie.connect<South : 3, West : 2>
      aie.connect<West : 3, South : 0>
    }
    %mem_tile_3_1 = aie.tile(3, 1)
    %switchbox_3_1 = aie.switchbox(%mem_tile_3_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 5, North : 5>
      aie.connect<North : 3, South : 3>
    }
    %switchbox_6_0 = aie.switchbox(%shim_noc_tile_6_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 3, West : 0>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_6_0 = aie.shim_mux(%shim_noc_tile_6_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %mem_tile_5_1 = aie.tile(5, 1)
    %switchbox_5_1 = aie.switchbox(%mem_tile_5_1) {
      aie.connect<South : 5, North : 5>
    }
    %tile_5_2 = aie.tile(5, 2)
    %switchbox_5_2 = aie.switchbox(%tile_5_2) {
      aie.connect<South : 5, West : 1>
      aie.connect<East : 3, West : 3>
    }
    %tile_6_2 = aie.tile(6, 2)
    %switchbox_6_2 = aie.switchbox(%tile_6_2) {
      aie.connect<East : 1, West : 3>
    }
    %switchbox_7_0 = aie.switchbox(%shim_noc_tile_7_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, West : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_7_0 = aie.shim_mux(%shim_noc_tile_7_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %mem_tile_7_1 = aie.tile(7, 1)
    %switchbox_7_1 = aie.switchbox(%mem_tile_7_1) {
      aie.connect<South : 0, North : 0>
    }
    %tile_7_2 = aie.tile(7, 2)
    %switchbox_7_2 = aie.switchbox(%tile_7_2) {
      aie.connect<South : 0, West : 1>
    }
    %tile_3_4 = aie.tile(3, 4)
    %switchbox_3_4 = aie.switchbox(%tile_3_4) {
      aie.connect<West : 3, South : 3>
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%tile_0_4 : Core, %switchbox_0_4 : Core)
    aie.wire(%tile_0_4 : DMA, %switchbox_0_4 : DMA)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
    aie.wire(%tile_0_5 : Core, %switchbox_0_5 : Core)
    aie.wire(%tile_0_5 : DMA, %switchbox_0_5 : DMA)
    aie.wire(%switchbox_0_4 : North, %switchbox_0_5 : South)
    aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%shim_noc_tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
    aie.wire(%switchbox_0_3 : East, %switchbox_1_3 : West)
    aie.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
    aie.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
    aie.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
    aie.wire(%switchbox_0_4 : East, %switchbox_1_4 : West)
    aie.wire(%tile_1_4 : Core, %switchbox_1_4 : Core)
    aie.wire(%tile_1_4 : DMA, %switchbox_1_4 : DMA)
    aie.wire(%switchbox_1_3 : North, %switchbox_1_4 : South)
    aie.wire(%switchbox_0_5 : East, %switchbox_1_5 : West)
    aie.wire(%tile_1_5 : Core, %switchbox_1_5 : Core)
    aie.wire(%tile_1_5 : DMA, %switchbox_1_5 : DMA)
    aie.wire(%switchbox_1_4 : North, %switchbox_1_5 : South)
    aie.wire(%switchbox_1_0 : East, %switchbox_2_0 : West)
    aie.wire(%shim_mux_2_0 : North, %switchbox_2_0 : South)
    aie.wire(%shim_noc_tile_2_0 : DMA, %shim_mux_2_0 : DMA)
    aie.wire(%switchbox_1_1 : East, %switchbox_2_1 : West)
    aie.wire(%mem_tile_2_1 : Core, %switchbox_2_1 : Core)
    aie.wire(%mem_tile_2_1 : DMA, %switchbox_2_1 : DMA)
    aie.wire(%switchbox_2_0 : North, %switchbox_2_1 : South)
    aie.wire(%switchbox_1_2 : East, %switchbox_2_2 : West)
    aie.wire(%tile_2_2 : Core, %switchbox_2_2 : Core)
    aie.wire(%tile_2_2 : DMA, %switchbox_2_2 : DMA)
    aie.wire(%switchbox_2_1 : North, %switchbox_2_2 : South)
    aie.wire(%switchbox_1_3 : East, %switchbox_2_3 : West)
    aie.wire(%tile_2_3 : Core, %switchbox_2_3 : Core)
    aie.wire(%tile_2_3 : DMA, %switchbox_2_3 : DMA)
    aie.wire(%switchbox_2_2 : North, %switchbox_2_3 : South)
    aie.wire(%switchbox_1_4 : East, %switchbox_2_4 : West)
    aie.wire(%tile_2_4 : Core, %switchbox_2_4 : Core)
    aie.wire(%tile_2_4 : DMA, %switchbox_2_4 : DMA)
    aie.wire(%switchbox_2_3 : North, %switchbox_2_4 : South)
    aie.wire(%switchbox_1_5 : East, %switchbox_2_5 : West)
    aie.wire(%tile_2_5 : Core, %switchbox_2_5 : Core)
    aie.wire(%tile_2_5 : DMA, %switchbox_2_5 : DMA)
    aie.wire(%switchbox_2_4 : North, %switchbox_2_5 : South)
    aie.wire(%switchbox_2_0 : East, %switchbox_3_0 : West)
    aie.wire(%shim_mux_3_0 : North, %switchbox_3_0 : South)
    aie.wire(%shim_noc_tile_3_0 : DMA, %shim_mux_3_0 : DMA)
    aie.wire(%switchbox_2_1 : East, %switchbox_3_1 : West)
    aie.wire(%mem_tile_3_1 : Core, %switchbox_3_1 : Core)
    aie.wire(%mem_tile_3_1 : DMA, %switchbox_3_1 : DMA)
    aie.wire(%switchbox_3_0 : North, %switchbox_3_1 : South)
    aie.wire(%switchbox_2_2 : East, %switchbox_3_2 : West)
    aie.wire(%tile_3_2 : Core, %switchbox_3_2 : Core)
    aie.wire(%tile_3_2 : DMA, %switchbox_3_2 : DMA)
    aie.wire(%switchbox_3_1 : North, %switchbox_3_2 : South)
    aie.wire(%switchbox_2_3 : East, %switchbox_3_3 : West)
    aie.wire(%tile_3_3 : Core, %switchbox_3_3 : Core)
    aie.wire(%tile_3_3 : DMA, %switchbox_3_3 : DMA)
    aie.wire(%switchbox_3_2 : North, %switchbox_3_3 : South)
    aie.wire(%switchbox_2_4 : East, %switchbox_3_4 : West)
    aie.wire(%tile_3_4 : Core, %switchbox_3_4 : Core)
    aie.wire(%tile_3_4 : DMA, %switchbox_3_4 : DMA)
    aie.wire(%switchbox_3_3 : North, %switchbox_3_4 : South)
    aie.wire(%switchbox_3_0 : East, %switchbox_4_0 : West)
    aie.wire(%shim_mux_4_0 : North, %switchbox_4_0 : South)
    aie.wire(%shim_noc_tile_4_0 : DMA, %shim_mux_4_0 : DMA)
    aie.wire(%switchbox_3_1 : East, %switchbox_4_1 : West)
    aie.wire(%mem_tile_4_1 : Core, %switchbox_4_1 : Core)
    aie.wire(%mem_tile_4_1 : DMA, %switchbox_4_1 : DMA)
    aie.wire(%switchbox_4_0 : North, %switchbox_4_1 : South)
    aie.wire(%switchbox_3_2 : East, %switchbox_4_2 : West)
    aie.wire(%tile_4_2 : Core, %switchbox_4_2 : Core)
    aie.wire(%tile_4_2 : DMA, %switchbox_4_2 : DMA)
    aie.wire(%switchbox_4_1 : North, %switchbox_4_2 : South)
    aie.wire(%switchbox_4_0 : East, %switchbox_5_0 : West)
    aie.wire(%shim_mux_5_0 : North, %switchbox_5_0 : South)
    aie.wire(%shim_noc_tile_5_0 : DMA, %shim_mux_5_0 : DMA)
    aie.wire(%switchbox_4_1 : East, %switchbox_5_1 : West)
    aie.wire(%mem_tile_5_1 : Core, %switchbox_5_1 : Core)
    aie.wire(%mem_tile_5_1 : DMA, %switchbox_5_1 : DMA)
    aie.wire(%switchbox_5_0 : North, %switchbox_5_1 : South)
    aie.wire(%switchbox_4_2 : East, %switchbox_5_2 : West)
    aie.wire(%tile_5_2 : Core, %switchbox_5_2 : Core)
    aie.wire(%tile_5_2 : DMA, %switchbox_5_2 : DMA)
    aie.wire(%switchbox_5_1 : North, %switchbox_5_2 : South)
    aie.wire(%switchbox_5_0 : East, %switchbox_6_0 : West)
    aie.wire(%shim_mux_6_0 : North, %switchbox_6_0 : South)
    aie.wire(%shim_noc_tile_6_0 : DMA, %shim_mux_6_0 : DMA)
    aie.wire(%switchbox_5_2 : East, %switchbox_6_2 : West)
    aie.wire(%tile_6_2 : Core, %switchbox_6_2 : Core)
    aie.wire(%tile_6_2 : DMA, %switchbox_6_2 : DMA)
    aie.wire(%switchbox_6_0 : East, %switchbox_7_0 : West)
    aie.wire(%shim_mux_7_0 : North, %switchbox_7_0 : South)
    aie.wire(%shim_noc_tile_7_0 : DMA, %shim_mux_7_0 : DMA)
    aie.wire(%mem_tile_7_1 : Core, %switchbox_7_1 : Core)
    aie.wire(%mem_tile_7_1 : DMA, %switchbox_7_1 : DMA)
    aie.wire(%switchbox_7_0 : North, %switchbox_7_1 : South)
    aie.wire(%switchbox_6_2 : East, %switchbox_7_2 : West)
    aie.wire(%tile_7_2 : Core, %switchbox_7_2 : Core)
    aie.wire(%tile_7_2 : DMA, %switchbox_7_2 : DMA)
    aie.wire(%switchbox_7_1 : North, %switchbox_7_2 : South)
  }
  aie.device(npu2) @gemv_2 {
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_4_0 = aie.tile(4, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_5_0 = aie.tile(5, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_6_0 = aie.tile(6, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_7_0 = aie.tile(7, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %C_L1L3_7_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 6) {init = 0 : i32, sym_name = "C_L1L3_7_cons_prod_lock_0"}
    %C_L1L3_7_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 7) {init = 0 : i32, sym_name = "C_L1L3_7_cons_cons_lock_0"}
    %C_L1L3_7_buff_0 = aie.buffer(%tile_1_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "C_L1L3_7_buff_0"} : memref<256xbf16> 
    %C_L1L3_7_prod_lock_0 = aie.lock(%tile_1_5, 4) {init = 1 : i32, sym_name = "C_L1L3_7_prod_lock_0"}
    %C_L1L3_7_cons_lock_0 = aie.lock(%tile_1_5, 5) {init = 0 : i32, sym_name = "C_L1L3_7_cons_lock_0"}
    %C_L1L3_6_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 4) {init = 0 : i32, sym_name = "C_L1L3_6_cons_prod_lock_0"}
    %C_L1L3_6_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 5) {init = 0 : i32, sym_name = "C_L1L3_6_cons_cons_lock_0"}
    %C_L1L3_6_buff_0 = aie.buffer(%tile_1_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "C_L1L3_6_buff_0"} : memref<256xbf16> 
    %C_L1L3_6_prod_lock_0 = aie.lock(%tile_1_4, 4) {init = 1 : i32, sym_name = "C_L1L3_6_prod_lock_0"}
    %C_L1L3_6_cons_lock_0 = aie.lock(%tile_1_4, 5) {init = 0 : i32, sym_name = "C_L1L3_6_cons_lock_0"}
    %C_L1L3_5_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 6) {init = 0 : i32, sym_name = "C_L1L3_5_cons_prod_lock_0"}
    %C_L1L3_5_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 7) {init = 0 : i32, sym_name = "C_L1L3_5_cons_cons_lock_0"}
    %C_L1L3_5_buff_0 = aie.buffer(%tile_1_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "C_L1L3_5_buff_0"} : memref<256xbf16> 
    %C_L1L3_5_prod_lock_0 = aie.lock(%tile_1_3, 4) {init = 1 : i32, sym_name = "C_L1L3_5_prod_lock_0"}
    %C_L1L3_5_cons_lock_0 = aie.lock(%tile_1_3, 5) {init = 0 : i32, sym_name = "C_L1L3_5_cons_lock_0"}
    %C_L1L3_4_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 4) {init = 0 : i32, sym_name = "C_L1L3_4_cons_prod_lock_0"}
    %C_L1L3_4_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 5) {init = 0 : i32, sym_name = "C_L1L3_4_cons_cons_lock_0"}
    %C_L1L3_4_buff_0 = aie.buffer(%tile_1_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "C_L1L3_4_buff_0"} : memref<256xbf16> 
    %C_L1L3_4_prod_lock_0 = aie.lock(%tile_1_2, 4) {init = 1 : i32, sym_name = "C_L1L3_4_prod_lock_0"}
    %C_L1L3_4_cons_lock_0 = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "C_L1L3_4_cons_lock_0"}
    %C_L1L3_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 6) {init = 0 : i32, sym_name = "C_L1L3_3_cons_prod_lock_0"}
    %C_L1L3_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 7) {init = 0 : i32, sym_name = "C_L1L3_3_cons_cons_lock_0"}
    %C_L1L3_3_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "C_L1L3_3_buff_0"} : memref<256xbf16> 
    %C_L1L3_3_prod_lock_0 = aie.lock(%tile_0_5, 4) {init = 1 : i32, sym_name = "C_L1L3_3_prod_lock_0"}
    %C_L1L3_3_cons_lock_0 = aie.lock(%tile_0_5, 5) {init = 0 : i32, sym_name = "C_L1L3_3_cons_lock_0"}
    %C_L1L3_2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 4) {init = 0 : i32, sym_name = "C_L1L3_2_cons_prod_lock_0"}
    %C_L1L3_2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 5) {init = 0 : i32, sym_name = "C_L1L3_2_cons_cons_lock_0"}
    %C_L1L3_2_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "C_L1L3_2_buff_0"} : memref<256xbf16> 
    %C_L1L3_2_prod_lock_0 = aie.lock(%tile_0_4, 4) {init = 1 : i32, sym_name = "C_L1L3_2_prod_lock_0"}
    %C_L1L3_2_cons_lock_0 = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "C_L1L3_2_cons_lock_0"}
    %C_L1L3_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 6) {init = 0 : i32, sym_name = "C_L1L3_1_cons_prod_lock_0"}
    %C_L1L3_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 7) {init = 0 : i32, sym_name = "C_L1L3_1_cons_cons_lock_0"}
    %C_L1L3_1_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "C_L1L3_1_buff_0"} : memref<256xbf16> 
    %C_L1L3_1_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 1 : i32, sym_name = "C_L1L3_1_prod_lock_0"}
    %C_L1L3_1_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "C_L1L3_1_cons_lock_0"}
    %C_L1L3_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "C_L1L3_0_cons_prod_lock_0"}
    %C_L1L3_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "C_L1L3_0_cons_cons_lock_0"}
    %C_L1L3_0_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "C_L1L3_0_buff_0"} : memref<256xbf16> 
    %C_L1L3_0_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 1 : i32, sym_name = "C_L1L3_0_prod_lock_0"}
    %C_L1L3_0_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "C_L1L3_0_cons_lock_0"}
    %B_L3L1_7_cons_buff_0 = aie.buffer(%tile_1_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "B_L3L1_7_cons_buff_0"} : memref<8192xbf16> 
    %B_L3L1_7_cons_prod_lock_0 = aie.lock(%tile_1_5, 2) {init = 1 : i32, sym_name = "B_L3L1_7_cons_prod_lock_0"}
    %B_L3L1_7_cons_cons_lock_0 = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "B_L3L1_7_cons_cons_lock_0"}
    %B_L3L1_7_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 2) {init = 0 : i32, sym_name = "B_L3L1_7_prod_lock_0"}
    %B_L3L1_7_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 3) {init = 0 : i32, sym_name = "B_L3L1_7_cons_lock_0"}
    %B_L3L1_6_cons_buff_0 = aie.buffer(%tile_1_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "B_L3L1_6_cons_buff_0"} : memref<8192xbf16> 
    %B_L3L1_6_cons_prod_lock_0 = aie.lock(%tile_1_4, 2) {init = 1 : i32, sym_name = "B_L3L1_6_cons_prod_lock_0"}
    %B_L3L1_6_cons_cons_lock_0 = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "B_L3L1_6_cons_cons_lock_0"}
    %B_L3L1_6_prod_lock_0 = aie.lock(%shim_noc_tile_7_0, 0) {init = 0 : i32, sym_name = "B_L3L1_6_prod_lock_0"}
    %B_L3L1_6_cons_lock_0 = aie.lock(%shim_noc_tile_7_0, 1) {init = 0 : i32, sym_name = "B_L3L1_6_cons_lock_0"}
    %B_L3L1_5_cons_buff_0 = aie.buffer(%tile_1_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "B_L3L1_5_cons_buff_0"} : memref<8192xbf16> 
    %B_L3L1_5_cons_prod_lock_0 = aie.lock(%tile_1_3, 2) {init = 1 : i32, sym_name = "B_L3L1_5_cons_prod_lock_0"}
    %B_L3L1_5_cons_cons_lock_0 = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "B_L3L1_5_cons_cons_lock_0"}
    %B_L3L1_5_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 2) {init = 0 : i32, sym_name = "B_L3L1_5_prod_lock_0"}
    %B_L3L1_5_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 3) {init = 0 : i32, sym_name = "B_L3L1_5_cons_lock_0"}
    %B_L3L1_4_cons_buff_0 = aie.buffer(%tile_1_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "B_L3L1_4_cons_buff_0"} : memref<8192xbf16> 
    %B_L3L1_4_cons_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 1 : i32, sym_name = "B_L3L1_4_cons_prod_lock_0"}
    %B_L3L1_4_cons_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "B_L3L1_4_cons_cons_lock_0"}
    %B_L3L1_4_prod_lock_0 = aie.lock(%shim_noc_tile_6_0, 0) {init = 0 : i32, sym_name = "B_L3L1_4_prod_lock_0"}
    %B_L3L1_4_cons_lock_0 = aie.lock(%shim_noc_tile_6_0, 1) {init = 0 : i32, sym_name = "B_L3L1_4_cons_lock_0"}
    %B_L3L1_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "B_L3L1_3_cons_buff_0"} : memref<8192xbf16> 
    %B_L3L1_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 1 : i32, sym_name = "B_L3L1_3_cons_prod_lock_0"}
    %B_L3L1_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "B_L3L1_3_cons_cons_lock_0"}
    %B_L3L1_3_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 2) {init = 0 : i32, sym_name = "B_L3L1_3_prod_lock_0"}
    %B_L3L1_3_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 3) {init = 0 : i32, sym_name = "B_L3L1_3_cons_lock_0"}
    %B_L3L1_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "B_L3L1_2_cons_buff_0"} : memref<8192xbf16> 
    %B_L3L1_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 1 : i32, sym_name = "B_L3L1_2_cons_prod_lock_0"}
    %B_L3L1_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "B_L3L1_2_cons_cons_lock_0"}
    %B_L3L1_2_prod_lock_0 = aie.lock(%shim_noc_tile_5_0, 0) {init = 0 : i32, sym_name = "B_L3L1_2_prod_lock_0"}
    %B_L3L1_2_cons_lock_0 = aie.lock(%shim_noc_tile_5_0, 1) {init = 0 : i32, sym_name = "B_L3L1_2_cons_lock_0"}
    %B_L3L1_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "B_L3L1_1_cons_buff_0"} : memref<8192xbf16> 
    %B_L3L1_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 1 : i32, sym_name = "B_L3L1_1_cons_prod_lock_0"}
    %B_L3L1_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "B_L3L1_1_cons_cons_lock_0"}
    %B_L3L1_1_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 2) {init = 0 : i32, sym_name = "B_L3L1_1_prod_lock_0"}
    %B_L3L1_1_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 3) {init = 0 : i32, sym_name = "B_L3L1_1_cons_lock_0"}
    %B_L3L1_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "B_L3L1_0_cons_buff_0"} : memref<8192xbf16> 
    %B_L3L1_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "B_L3L1_0_cons_prod_lock_0"}
    %B_L3L1_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "B_L3L1_0_cons_cons_lock_0"}
    %B_L3L1_0_prod_lock_0 = aie.lock(%shim_noc_tile_4_0, 0) {init = 0 : i32, sym_name = "B_L3L1_0_prod_lock_0"}
    %B_L3L1_0_cons_lock_0 = aie.lock(%shim_noc_tile_4_0, 1) {init = 0 : i32, sym_name = "B_L3L1_0_cons_lock_0"}
    %A_L3L1_7_cons_buff_0 = aie.buffer(%tile_1_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "A_L3L1_7_cons_buff_0"} : memref<8192xbf16> 
    %A_L3L1_7_cons_buff_1 = aie.buffer(%tile_1_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "A_L3L1_7_cons_buff_1"} : memref<8192xbf16> 
    %A_L3L1_7_cons_prod_lock_0 = aie.lock(%tile_1_5, 0) {init = 2 : i32, sym_name = "A_L3L1_7_cons_prod_lock_0"}
    %A_L3L1_7_cons_cons_lock_0 = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "A_L3L1_7_cons_cons_lock_0"}
    %A_L3L1_7_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "A_L3L1_7_prod_lock_0"}
    %A_L3L1_7_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "A_L3L1_7_cons_lock_0"}
    %A_L3L1_6_cons_buff_0 = aie.buffer(%tile_1_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "A_L3L1_6_cons_buff_0"} : memref<8192xbf16> 
    %A_L3L1_6_cons_buff_1 = aie.buffer(%tile_1_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "A_L3L1_6_cons_buff_1"} : memref<8192xbf16> 
    %A_L3L1_6_cons_prod_lock_0 = aie.lock(%tile_1_4, 0) {init = 2 : i32, sym_name = "A_L3L1_6_cons_prod_lock_0"}
    %A_L3L1_6_cons_cons_lock_0 = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "A_L3L1_6_cons_cons_lock_0"}
    %A_L3L1_6_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "A_L3L1_6_prod_lock_0"}
    %A_L3L1_6_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "A_L3L1_6_cons_lock_0"}
    %A_L3L1_5_cons_buff_0 = aie.buffer(%tile_1_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "A_L3L1_5_cons_buff_0"} : memref<8192xbf16> 
    %A_L3L1_5_cons_buff_1 = aie.buffer(%tile_1_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "A_L3L1_5_cons_buff_1"} : memref<8192xbf16> 
    %A_L3L1_5_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "A_L3L1_5_cons_prod_lock_0"}
    %A_L3L1_5_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "A_L3L1_5_cons_cons_lock_0"}
    %A_L3L1_5_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "A_L3L1_5_prod_lock_0"}
    %A_L3L1_5_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "A_L3L1_5_cons_lock_0"}
    %A_L3L1_4_cons_buff_0 = aie.buffer(%tile_1_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "A_L3L1_4_cons_buff_0"} : memref<8192xbf16> 
    %A_L3L1_4_cons_buff_1 = aie.buffer(%tile_1_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "A_L3L1_4_cons_buff_1"} : memref<8192xbf16> 
    %A_L3L1_4_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "A_L3L1_4_cons_prod_lock_0"}
    %A_L3L1_4_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "A_L3L1_4_cons_cons_lock_0"}
    %A_L3L1_4_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "A_L3L1_4_prod_lock_0"}
    %A_L3L1_4_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "A_L3L1_4_cons_lock_0"}
    %A_L3L1_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "A_L3L1_3_cons_buff_0"} : memref<8192xbf16> 
    %A_L3L1_3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "A_L3L1_3_cons_buff_1"} : memref<8192xbf16> 
    %A_L3L1_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "A_L3L1_3_cons_prod_lock_0"}
    %A_L3L1_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "A_L3L1_3_cons_cons_lock_0"}
    %A_L3L1_3_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "A_L3L1_3_prod_lock_0"}
    %A_L3L1_3_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "A_L3L1_3_cons_lock_0"}
    %A_L3L1_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "A_L3L1_2_cons_buff_0"} : memref<8192xbf16> 
    %A_L3L1_2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "A_L3L1_2_cons_buff_1"} : memref<8192xbf16> 
    %A_L3L1_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "A_L3L1_2_cons_prod_lock_0"}
    %A_L3L1_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "A_L3L1_2_cons_cons_lock_0"}
    %A_L3L1_2_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "A_L3L1_2_prod_lock_0"}
    %A_L3L1_2_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "A_L3L1_2_cons_lock_0"}
    %A_L3L1_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "A_L3L1_1_cons_buff_0"} : memref<8192xbf16> 
    %A_L3L1_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "A_L3L1_1_cons_buff_1"} : memref<8192xbf16> 
    %A_L3L1_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "A_L3L1_1_cons_prod_lock_0"}
    %A_L3L1_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "A_L3L1_1_cons_cons_lock_0"}
    %A_L3L1_1_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "A_L3L1_1_prod_lock_0"}
    %A_L3L1_1_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "A_L3L1_1_cons_lock_0"}
    %A_L3L1_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "A_L3L1_0_cons_buff_0"} : memref<8192xbf16> 
    %A_L3L1_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "A_L3L1_0_cons_buff_1"} : memref<8192xbf16> 
    %A_L3L1_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "A_L3L1_0_cons_prod_lock_0"}
    %A_L3L1_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "A_L3L1_0_cons_cons_lock_0"}
    %A_L3L1_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "A_L3L1_0_prod_lock_0"}
    %A_L3L1_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "A_L3L1_0_cons_lock_0"}
    func.func private @matvec_vectorized_bf16_bf16(i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>)
    %core_0_2 = aie.core(%tile_0_2) {
      %c2 = arith.constant 2 : index
      %c8192_i32 = arith.constant 8192 : i32
      %c1_i32 = arith.constant 1 : i32
      %c256 = arith.constant 256 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = index.casts %4 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %6, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %7 = arith.addi %4, %c1 : index
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = index.casts %7 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %8, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %4, %c2 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      %10 = arith.addi %2, %c1 : index
      cf.br ^bb3(%10 : index)
    ^bb8:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "mv.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c2 = arith.constant 2 : index
      %c8192_i32 = arith.constant 8192 : i32
      %c1_i32 = arith.constant 1 : i32
      %c256 = arith.constant 256 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_1_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L3L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = index.casts %4 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %6, %A_L3L1_1_cons_buff_0, %B_L3L1_1_cons_buff_0, %C_L1L3_1_buff_0) : (i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>) -> ()
      aie.use_lock(%A_L3L1_1_cons_prod_lock_0, Release, 1)
      %7 = arith.addi %4, %c1 : index
      aie.use_lock(%A_L3L1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = index.casts %7 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %8, %A_L3L1_1_cons_buff_1, %B_L3L1_1_cons_buff_0, %C_L1L3_1_buff_0) : (i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>) -> ()
      aie.use_lock(%A_L3L1_1_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %4, %c2 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_1_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_1_cons_prod_lock_0, Release, 1)
      %10 = arith.addi %2, %c1 : index
      cf.br ^bb3(%10 : index)
    ^bb8:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "mv.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c2 = arith.constant 2 : index
      %c8192_i32 = arith.constant 8192 : i32
      %c1_i32 = arith.constant 1 : i32
      %c256 = arith.constant 256 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_2_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L3L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = index.casts %4 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %6, %A_L3L1_2_cons_buff_0, %B_L3L1_2_cons_buff_0, %C_L1L3_2_buff_0) : (i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>) -> ()
      aie.use_lock(%A_L3L1_2_cons_prod_lock_0, Release, 1)
      %7 = arith.addi %4, %c1 : index
      aie.use_lock(%A_L3L1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = index.casts %7 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %8, %A_L3L1_2_cons_buff_1, %B_L3L1_2_cons_buff_0, %C_L1L3_2_buff_0) : (i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>) -> ()
      aie.use_lock(%A_L3L1_2_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %4, %c2 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_2_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_2_cons_prod_lock_0, Release, 1)
      %10 = arith.addi %2, %c1 : index
      cf.br ^bb3(%10 : index)
    ^bb8:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "mv.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c2 = arith.constant 2 : index
      %c8192_i32 = arith.constant 8192 : i32
      %c1_i32 = arith.constant 1 : i32
      %c256 = arith.constant 256 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_3_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L3L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = index.casts %4 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %6, %A_L3L1_3_cons_buff_0, %B_L3L1_3_cons_buff_0, %C_L1L3_3_buff_0) : (i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>) -> ()
      aie.use_lock(%A_L3L1_3_cons_prod_lock_0, Release, 1)
      %7 = arith.addi %4, %c1 : index
      aie.use_lock(%A_L3L1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = index.casts %7 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %8, %A_L3L1_3_cons_buff_1, %B_L3L1_3_cons_buff_0, %C_L1L3_3_buff_0) : (i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>) -> ()
      aie.use_lock(%A_L3L1_3_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %4, %c2 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_3_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_3_cons_prod_lock_0, Release, 1)
      %10 = arith.addi %2, %c1 : index
      cf.br ^bb3(%10 : index)
    ^bb8:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "mv.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c2 = arith.constant 2 : index
      %c8192_i32 = arith.constant 8192 : i32
      %c1_i32 = arith.constant 1 : i32
      %c256 = arith.constant 256 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_4_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L3L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = index.casts %4 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %6, %A_L3L1_4_cons_buff_0, %B_L3L1_4_cons_buff_0, %C_L1L3_4_buff_0) : (i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>) -> ()
      aie.use_lock(%A_L3L1_4_cons_prod_lock_0, Release, 1)
      %7 = arith.addi %4, %c1 : index
      aie.use_lock(%A_L3L1_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = index.casts %7 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %8, %A_L3L1_4_cons_buff_1, %B_L3L1_4_cons_buff_0, %C_L1L3_4_buff_0) : (i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>) -> ()
      aie.use_lock(%A_L3L1_4_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %4, %c2 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_4_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_4_cons_prod_lock_0, Release, 1)
      %10 = arith.addi %2, %c1 : index
      cf.br ^bb3(%10 : index)
    ^bb8:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "mv.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c2 = arith.constant 2 : index
      %c8192_i32 = arith.constant 8192 : i32
      %c1_i32 = arith.constant 1 : i32
      %c256 = arith.constant 256 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_5_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L3L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = index.casts %4 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %6, %A_L3L1_5_cons_buff_0, %B_L3L1_5_cons_buff_0, %C_L1L3_5_buff_0) : (i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>) -> ()
      aie.use_lock(%A_L3L1_5_cons_prod_lock_0, Release, 1)
      %7 = arith.addi %4, %c1 : index
      aie.use_lock(%A_L3L1_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = index.casts %7 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %8, %A_L3L1_5_cons_buff_1, %B_L3L1_5_cons_buff_0, %C_L1L3_5_buff_0) : (i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>) -> ()
      aie.use_lock(%A_L3L1_5_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %4, %c2 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_5_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_5_cons_prod_lock_0, Release, 1)
      %10 = arith.addi %2, %c1 : index
      cf.br ^bb3(%10 : index)
    ^bb8:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "mv.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c2 = arith.constant 2 : index
      %c8192_i32 = arith.constant 8192 : i32
      %c1_i32 = arith.constant 1 : i32
      %c256 = arith.constant 256 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_6_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L3L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = index.casts %4 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %6, %A_L3L1_6_cons_buff_0, %B_L3L1_6_cons_buff_0, %C_L1L3_6_buff_0) : (i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>) -> ()
      aie.use_lock(%A_L3L1_6_cons_prod_lock_0, Release, 1)
      %7 = arith.addi %4, %c1 : index
      aie.use_lock(%A_L3L1_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = index.casts %7 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %8, %A_L3L1_6_cons_buff_1, %B_L3L1_6_cons_buff_0, %C_L1L3_6_buff_0) : (i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>) -> ()
      aie.use_lock(%A_L3L1_6_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %4, %c2 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_6_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_6_cons_prod_lock_0, Release, 1)
      %10 = arith.addi %2, %c1 : index
      cf.br ^bb3(%10 : index)
    ^bb8:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "mv.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c2 = arith.constant 2 : index
      %c8192_i32 = arith.constant 8192 : i32
      %c1_i32 = arith.constant 1 : i32
      %c256 = arith.constant 256 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c4294967295 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_7_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c256 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L3L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %6 = index.casts %4 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %6, %A_L3L1_7_cons_buff_0, %B_L3L1_7_cons_buff_0, %C_L1L3_7_buff_0) : (i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>) -> ()
      aie.use_lock(%A_L3L1_7_cons_prod_lock_0, Release, 1)
      %7 = arith.addi %4, %c1 : index
      aie.use_lock(%A_L3L1_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = index.casts %7 : index to i32
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %8, %A_L3L1_7_cons_buff_1, %B_L3L1_7_cons_buff_0, %C_L1L3_7_buff_0) : (i32, i32, i32, memref<8192xbf16>, memref<8192xbf16>, memref<256xbf16>) -> ()
      aie.use_lock(%A_L3L1_7_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %4, %c2 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_7_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_7_cons_prod_lock_0, Release, 1)
      %10 = arith.addi %2, %c1 : index
      cf.br ^bb3(%10 : index)
    ^bb8:  // pred: ^bb3
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "mv.o"}
    aie.runtime_sequence(%arg0: memref<16777216xbf16>, %arg1: memref<8192xbf16>, %arg2: memref<2048xbf16>) {
      %0 = aiex.dma_configure_task_for @A_L3L1_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<16777216xbf16>, 0, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L1_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 0, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 8192, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @A_L3L1_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<16777216xbf16>, 2097152, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @B_L3L1_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 0, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 8192, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @A_L3L1_2_shim_alloc {
        aie.dma_bd(%arg0 : memref<16777216xbf16>, 4194304, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L1_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 0, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 8192, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @A_L3L1_3_shim_alloc {
        aie.dma_bd(%arg0 : memref<16777216xbf16>, 6291456, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L1_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 0, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 8192, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @A_L3L1_4_shim_alloc {
        aie.dma_bd(%arg0 : memref<16777216xbf16>, 8388608, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @B_L3L1_4_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 0, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 8192, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @A_L3L1_5_shim_alloc {
        aie.dma_bd(%arg0 : memref<16777216xbf16>, 10485760, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @B_L3L1_5_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 0, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 8192, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @A_L3L1_6_shim_alloc {
        aie.dma_bd(%arg0 : memref<16777216xbf16>, 12582912, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @B_L3L1_6_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 0, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 8192, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @A_L3L1_7_shim_alloc {
        aie.dma_bd(%arg0 : memref<16777216xbf16>, 14680064, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @B_L3L1_7_shim_alloc {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 0, 8192, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 8192, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @C_L1L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @C_L1L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 256, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @C_L1L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 512, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @C_L1L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 768, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @C_L1L3_4_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1024, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @C_L1L3_5_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1280, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @C_L1L3_6_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1536, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @C_L1L3_7_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 1792, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%17)
      aiex.dma_await_task(%18)
      aiex.dma_await_task(%19)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%23)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%12)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%15)
    }
    aie.shim_dma_allocation @A_L3L1_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_0_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_0_cons_buff_1 : memref<8192xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_0_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%C_L1L3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_0_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @A_L3L1_1_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_1_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_1_cons_buff_1 : memref<8192xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_1_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%C_L1L3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_1_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @A_L3L1_2_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_2_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_2_cons_buff_1 : memref<8192xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_2_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%C_L1L3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_2_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @A_L3L1_3_shim_alloc(%shim_noc_tile_1_0, MM2S, 1)
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_3_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_3_cons_buff_1 : memref<8192xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_3_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%C_L1L3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_3_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @A_L3L1_4_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_4_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_4_cons_buff_1 : memref<8192xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_4_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%C_L1L3_4_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_4_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_4_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @A_L3L1_5_shim_alloc(%shim_noc_tile_2_0, MM2S, 1)
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_5_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_5_cons_buff_1 : memref<8192xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_5_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%C_L1L3_5_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_5_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_5_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @A_L3L1_6_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_6_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_6_cons_buff_1 : memref<8192xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_6_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%C_L1L3_6_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_6_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_6_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @A_L3L1_7_shim_alloc(%shim_noc_tile_3_0, MM2S, 1)
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_7_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_7_cons_buff_1 : memref<8192xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_7_cons_buff_0 : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%C_L1L3_7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_7_buff_0 : memref<256xbf16>, 0, 256) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_7_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @B_L3L1_0_shim_alloc(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @B_L3L1_1_shim_alloc(%shim_noc_tile_4_0, MM2S, 1)
    aie.shim_dma_allocation @B_L3L1_2_shim_alloc(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @B_L3L1_3_shim_alloc(%shim_noc_tile_5_0, MM2S, 1)
    aie.shim_dma_allocation @B_L3L1_4_shim_alloc(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @B_L3L1_5_shim_alloc(%shim_noc_tile_6_0, MM2S, 1)
    aie.shim_dma_allocation @B_L3L1_6_shim_alloc(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @B_L3L1_7_shim_alloc(%shim_noc_tile_7_0, MM2S, 1)
    aie.shim_dma_allocation @C_L1L3_0_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @C_L1L3_1_shim_alloc(%shim_noc_tile_0_0, S2MM, 1)
    aie.shim_dma_allocation @C_L1L3_2_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @C_L1L3_3_shim_alloc(%shim_noc_tile_1_0, S2MM, 1)
    aie.shim_dma_allocation @C_L1L3_4_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @C_L1L3_5_shim_alloc(%shim_noc_tile_2_0, S2MM, 1)
    aie.shim_dma_allocation @C_L1L3_6_shim_alloc(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @C_L1L3_7_shim_alloc(%shim_noc_tile_3_0, S2MM, 1)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_4_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_4_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_5_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_5_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_6_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_6_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_7_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_7_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<South : 7, North : 2>
      aie.connect<North : 1, South : 2>
      aie.connect<North : 3, South : 3>
      aie.connect<North : 0, East : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 2, North : 2>
      aie.connect<North : 1, South : 1>
      aie.connect<North : 3, South : 3>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 2, North : 0>
      aie.connect<East : 0, North : 5>
      aie.connect<East : 3, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 1, South : 3>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 5, North : 4>
      aie.connect<East : 0, DMA : 1>
      aie.connect<DMA : 0, South : 1>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<South : 4, DMA : 0>
      aie.connect<East : 3, DMA : 1>
      aie.connect<DMA : 0, East : 0>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<South : 7, North : 5>
      aie.connect<East : 2, North : 3>
      aie.connect<East : 1, North : 2>
      aie.connect<East : 0, North : 0>
      aie.connect<North : 0, South : 2>
      aie.connect<West : 3, South : 3>
      aie.connect<North : 2, East : 3>
      aie.connect<North : 1, East : 0>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 5, North : 5>
      aie.connect<South : 3, North : 3>
      aie.connect<South : 2, North : 2>
      aie.connect<South : 0, North : 0>
      aie.connect<North : 0, South : 0>
      aie.connect<North : 2, South : 2>
      aie.connect<North : 1, South : 1>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<South : 1, West : 0>
      aie.connect<South : 5, North : 3>
      aie.connect<East : 0, DMA : 0>
      aie.connect<South : 3, North : 2>
      aie.connect<South : 2, North : 4>
      aie.connect<South : 0, West : 3>
      aie.connect<East : 2, North : 5>
      aie.connect<East : 3, DMA : 1>
      aie.connect<East : 1, North : 0>
      aie.connect<North : 3, South : 0>
      aie.connect<DMA : 0, East : 0>
      aie.connect<North : 2, South : 2>
      aie.connect<North : 0, South : 1>
    }
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
      aie.connect<East : 0, DMA : 0>
      aie.connect<East : 3, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<South : 3, North : 5>
      aie.connect<South : 2, DMA : 0>
      aie.connect<East : 0, North : 1>
      aie.connect<South : 4, North : 2>
      aie.connect<South : 5, West : 0>
      aie.connect<East : 3, North : 4>
      aie.connect<South : 0, DMA : 1>
      aie.connect<East : 2, North : 3>
      aie.connect<North : 1, South : 3>
      aie.connect<DMA : 0, South : 2>
      aie.connect<North : 3, South : 0>
    }
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<South : 5, North : 5>
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 2, North : 2>
      aie.connect<South : 4, West : 3>
      aie.connect<South : 3, DMA : 1>
      aie.connect<West : 0, South : 1>
      aie.connect<DMA : 0, South : 3>
    }
    %switchbox_1_5 = aie.switchbox(%tile_1_5) {
      aie.connect<South : 5, West : 0>
      aie.connect<South : 2, DMA : 0>
      aie.connect<East : 3, West : 3>
      aie.connect<East : 2, DMA : 1>
      aie.connect<DMA : 0, East : 3>
    }
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<South : 7, West : 2>
      aie.connect<East : 2, North : 5>
      aie.connect<East : 1, West : 1>
      aie.connect<East : 0, West : 0>
      aie.connect<East : 3, North : 4>
      aie.connect<North : 3, South : 2>
      aie.connect<West : 3, South : 3>
      aie.connect<West : 0, East : 0>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_2_0 = aie.shim_mux(%shim_noc_tile_2_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %mem_tile_2_1 = aie.tile(2, 1)
    %switchbox_2_1 = aie.switchbox(%mem_tile_2_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 5, North : 5>
      aie.connect<South : 4, North : 4>
      aie.connect<North : 3, South : 3>
    }
    %tile_2_2 = aie.tile(2, 2)
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<South : 1, West : 0>
      aie.connect<South : 5, North : 5>
      aie.connect<South : 4, West : 2>
      aie.connect<East : 0, West : 3>
      aie.connect<East : 1, West : 1>
      aie.connect<East : 2, North : 0>
      aie.connect<West : 0, South : 3>
    }
    %tile_2_3 = aie.tile(2, 3)
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
      aie.connect<South : 5, West : 0>
      aie.connect<East : 1, West : 3>
      aie.connect<East : 2, North : 5>
      aie.connect<South : 0, West : 2>
      aie.connect<East : 3, North : 0>
    }
    %switchbox_3_0 = aie.switchbox(%shim_noc_tile_3_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 2, West : 0>
      aie.connect<East : 1, West : 3>
      aie.connect<East : 0, North : 1>
      aie.connect<East : 3, North : 5>
      aie.connect<West : 0, South : 2>
      aie.connect<North : 3, South : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_3_0 = aie.shim_mux(%shim_noc_tile_3_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %switchbox_4_0 = aie.switchbox(%shim_noc_tile_4_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 2, North : 5>
      aie.connect<East : 1, West : 0>
      aie.connect<East : 0, West : 3>
      aie.connect<East : 3, North : 0>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_4_0 = aie.shim_mux(%shim_noc_tile_4_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %tile_3_2 = aie.tile(3, 2)
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<East : 3, North : 0>
      aie.connect<South : 1, North : 3>
      aie.connect<South : 5, West : 0>
      aie.connect<East : 2, West : 1>
      aie.connect<East : 1, West : 2>
      aie.connect<East : 0, North : 4>
      aie.connect<North : 2, South : 3>
    }
    %tile_3_3 = aie.tile(3, 3)
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
      aie.connect<South : 0, West : 1>
      aie.connect<South : 3, West : 2>
      aie.connect<South : 4, West : 3>
      aie.connect<North : 3, South : 2>
    }
    %mem_tile_4_1 = aie.tile(4, 1)
    %switchbox_4_1 = aie.switchbox(%mem_tile_4_1) {
      aie.connect<South : 5, North : 5>
      aie.connect<South : 0, North : 0>
    }
    %tile_4_2 = aie.tile(4, 2)
    %switchbox_4_2 = aie.switchbox(%tile_4_2) {
      aie.connect<South : 5, West : 3>
      aie.connect<East : 1, West : 2>
      aie.connect<East : 3, West : 1>
      aie.connect<South : 0, West : 0>
    }
    %switchbox_5_0 = aie.switchbox(%shim_noc_tile_5_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 2, West : 0>
      aie.connect<East : 1, North : 5>
      aie.connect<East : 0, West : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_5_0 = aie.shim_mux(%shim_noc_tile_5_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %tile_2_4 = aie.tile(2, 4)
    %switchbox_2_4 = aie.switchbox(%tile_2_4) {
      aie.connect<South : 5, North : 5>
      aie.connect<South : 0, North : 3>
      aie.connect<North : 0, East : 3>
    }
    %tile_2_5 = aie.tile(2, 5)
    %switchbox_2_5 = aie.switchbox(%tile_2_5) {
      aie.connect<South : 5, West : 3>
      aie.connect<South : 3, West : 2>
      aie.connect<West : 3, South : 0>
    }
    %mem_tile_3_1 = aie.tile(3, 1)
    %switchbox_3_1 = aie.switchbox(%mem_tile_3_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 5, North : 5>
      aie.connect<North : 3, South : 3>
    }
    %switchbox_6_0 = aie.switchbox(%shim_noc_tile_6_0) {
      aie.connect<South : 3, West : 2>
      aie.connect<South : 7, West : 1>
      aie.connect<East : 3, West : 0>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_6_0 = aie.shim_mux(%shim_noc_tile_6_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %mem_tile_5_1 = aie.tile(5, 1)
    %switchbox_5_1 = aie.switchbox(%mem_tile_5_1) {
      aie.connect<South : 5, North : 5>
    }
    %tile_5_2 = aie.tile(5, 2)
    %switchbox_5_2 = aie.switchbox(%tile_5_2) {
      aie.connect<South : 5, West : 1>
      aie.connect<East : 3, West : 3>
    }
    %tile_6_2 = aie.tile(6, 2)
    %switchbox_6_2 = aie.switchbox(%tile_6_2) {
      aie.connect<East : 1, West : 3>
    }
    %switchbox_7_0 = aie.switchbox(%shim_noc_tile_7_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, West : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_7_0 = aie.shim_mux(%shim_noc_tile_7_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %mem_tile_7_1 = aie.tile(7, 1)
    %switchbox_7_1 = aie.switchbox(%mem_tile_7_1) {
      aie.connect<South : 0, North : 0>
    }
    %tile_7_2 = aie.tile(7, 2)
    %switchbox_7_2 = aie.switchbox(%tile_7_2) {
      aie.connect<South : 0, West : 1>
    }
    %tile_3_4 = aie.tile(3, 4)
    %switchbox_3_4 = aie.switchbox(%tile_3_4) {
      aie.connect<West : 3, South : 3>
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%tile_0_4 : Core, %switchbox_0_4 : Core)
    aie.wire(%tile_0_4 : DMA, %switchbox_0_4 : DMA)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
    aie.wire(%tile_0_5 : Core, %switchbox_0_5 : Core)
    aie.wire(%tile_0_5 : DMA, %switchbox_0_5 : DMA)
    aie.wire(%switchbox_0_4 : North, %switchbox_0_5 : South)
    aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%shim_noc_tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
    aie.wire(%switchbox_0_3 : East, %switchbox_1_3 : West)
    aie.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
    aie.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
    aie.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
    aie.wire(%switchbox_0_4 : East, %switchbox_1_4 : West)
    aie.wire(%tile_1_4 : Core, %switchbox_1_4 : Core)
    aie.wire(%tile_1_4 : DMA, %switchbox_1_4 : DMA)
    aie.wire(%switchbox_1_3 : North, %switchbox_1_4 : South)
    aie.wire(%switchbox_0_5 : East, %switchbox_1_5 : West)
    aie.wire(%tile_1_5 : Core, %switchbox_1_5 : Core)
    aie.wire(%tile_1_5 : DMA, %switchbox_1_5 : DMA)
    aie.wire(%switchbox_1_4 : North, %switchbox_1_5 : South)
    aie.wire(%switchbox_1_0 : East, %switchbox_2_0 : West)
    aie.wire(%shim_mux_2_0 : North, %switchbox_2_0 : South)
    aie.wire(%shim_noc_tile_2_0 : DMA, %shim_mux_2_0 : DMA)
    aie.wire(%switchbox_1_1 : East, %switchbox_2_1 : West)
    aie.wire(%mem_tile_2_1 : Core, %switchbox_2_1 : Core)
    aie.wire(%mem_tile_2_1 : DMA, %switchbox_2_1 : DMA)
    aie.wire(%switchbox_2_0 : North, %switchbox_2_1 : South)
    aie.wire(%switchbox_1_2 : East, %switchbox_2_2 : West)
    aie.wire(%tile_2_2 : Core, %switchbox_2_2 : Core)
    aie.wire(%tile_2_2 : DMA, %switchbox_2_2 : DMA)
    aie.wire(%switchbox_2_1 : North, %switchbox_2_2 : South)
    aie.wire(%switchbox_1_3 : East, %switchbox_2_3 : West)
    aie.wire(%tile_2_3 : Core, %switchbox_2_3 : Core)
    aie.wire(%tile_2_3 : DMA, %switchbox_2_3 : DMA)
    aie.wire(%switchbox_2_2 : North, %switchbox_2_3 : South)
    aie.wire(%switchbox_1_4 : East, %switchbox_2_4 : West)
    aie.wire(%tile_2_4 : Core, %switchbox_2_4 : Core)
    aie.wire(%tile_2_4 : DMA, %switchbox_2_4 : DMA)
    aie.wire(%switchbox_2_3 : North, %switchbox_2_4 : South)
    aie.wire(%switchbox_1_5 : East, %switchbox_2_5 : West)
    aie.wire(%tile_2_5 : Core, %switchbox_2_5 : Core)
    aie.wire(%tile_2_5 : DMA, %switchbox_2_5 : DMA)
    aie.wire(%switchbox_2_4 : North, %switchbox_2_5 : South)
    aie.wire(%switchbox_2_0 : East, %switchbox_3_0 : West)
    aie.wire(%shim_mux_3_0 : North, %switchbox_3_0 : South)
    aie.wire(%shim_noc_tile_3_0 : DMA, %shim_mux_3_0 : DMA)
    aie.wire(%switchbox_2_1 : East, %switchbox_3_1 : West)
    aie.wire(%mem_tile_3_1 : Core, %switchbox_3_1 : Core)
    aie.wire(%mem_tile_3_1 : DMA, %switchbox_3_1 : DMA)
    aie.wire(%switchbox_3_0 : North, %switchbox_3_1 : South)
    aie.wire(%switchbox_2_2 : East, %switchbox_3_2 : West)
    aie.wire(%tile_3_2 : Core, %switchbox_3_2 : Core)
    aie.wire(%tile_3_2 : DMA, %switchbox_3_2 : DMA)
    aie.wire(%switchbox_3_1 : North, %switchbox_3_2 : South)
    aie.wire(%switchbox_2_3 : East, %switchbox_3_3 : West)
    aie.wire(%tile_3_3 : Core, %switchbox_3_3 : Core)
    aie.wire(%tile_3_3 : DMA, %switchbox_3_3 : DMA)
    aie.wire(%switchbox_3_2 : North, %switchbox_3_3 : South)
    aie.wire(%switchbox_2_4 : East, %switchbox_3_4 : West)
    aie.wire(%tile_3_4 : Core, %switchbox_3_4 : Core)
    aie.wire(%tile_3_4 : DMA, %switchbox_3_4 : DMA)
    aie.wire(%switchbox_3_3 : North, %switchbox_3_4 : South)
    aie.wire(%switchbox_3_0 : East, %switchbox_4_0 : West)
    aie.wire(%shim_mux_4_0 : North, %switchbox_4_0 : South)
    aie.wire(%shim_noc_tile_4_0 : DMA, %shim_mux_4_0 : DMA)
    aie.wire(%switchbox_3_1 : East, %switchbox_4_1 : West)
    aie.wire(%mem_tile_4_1 : Core, %switchbox_4_1 : Core)
    aie.wire(%mem_tile_4_1 : DMA, %switchbox_4_1 : DMA)
    aie.wire(%switchbox_4_0 : North, %switchbox_4_1 : South)
    aie.wire(%switchbox_3_2 : East, %switchbox_4_2 : West)
    aie.wire(%tile_4_2 : Core, %switchbox_4_2 : Core)
    aie.wire(%tile_4_2 : DMA, %switchbox_4_2 : DMA)
    aie.wire(%switchbox_4_1 : North, %switchbox_4_2 : South)
    aie.wire(%switchbox_4_0 : East, %switchbox_5_0 : West)
    aie.wire(%shim_mux_5_0 : North, %switchbox_5_0 : South)
    aie.wire(%shim_noc_tile_5_0 : DMA, %shim_mux_5_0 : DMA)
    aie.wire(%switchbox_4_1 : East, %switchbox_5_1 : West)
    aie.wire(%mem_tile_5_1 : Core, %switchbox_5_1 : Core)
    aie.wire(%mem_tile_5_1 : DMA, %switchbox_5_1 : DMA)
    aie.wire(%switchbox_5_0 : North, %switchbox_5_1 : South)
    aie.wire(%switchbox_4_2 : East, %switchbox_5_2 : West)
    aie.wire(%tile_5_2 : Core, %switchbox_5_2 : Core)
    aie.wire(%tile_5_2 : DMA, %switchbox_5_2 : DMA)
    aie.wire(%switchbox_5_1 : North, %switchbox_5_2 : South)
    aie.wire(%switchbox_5_0 : East, %switchbox_6_0 : West)
    aie.wire(%shim_mux_6_0 : North, %switchbox_6_0 : South)
    aie.wire(%shim_noc_tile_6_0 : DMA, %shim_mux_6_0 : DMA)
    aie.wire(%switchbox_5_2 : East, %switchbox_6_2 : West)
    aie.wire(%tile_6_2 : Core, %switchbox_6_2 : Core)
    aie.wire(%tile_6_2 : DMA, %switchbox_6_2 : DMA)
    aie.wire(%switchbox_6_0 : East, %switchbox_7_0 : West)
    aie.wire(%shim_mux_7_0 : North, %switchbox_7_0 : South)
    aie.wire(%shim_noc_tile_7_0 : DMA, %shim_mux_7_0 : DMA)
    aie.wire(%mem_tile_7_1 : Core, %switchbox_7_1 : Core)
    aie.wire(%mem_tile_7_1 : DMA, %switchbox_7_1 : DMA)
    aie.wire(%switchbox_7_0 : North, %switchbox_7_1 : South)
    aie.wire(%switchbox_6_2 : East, %switchbox_7_2 : West)
    aie.wire(%tile_7_2 : Core, %switchbox_7_2 : Core)
    aie.wire(%tile_7_2 : DMA, %switchbox_7_2 : DMA)
    aie.wire(%switchbox_7_1 : North, %switchbox_7_2 : South)
  }
}

