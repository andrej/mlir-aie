module {
  aie.device(npu2_1col) @base {
    aie.runtime_sequence @configure() {
      aiex.npu.write32 {address = 258056 : ui32, value = 3221225538 : ui32}
      aiex.npu.write32 {address = 258100 : ui32, value = 3221225539 : ui32}
      aiex.npu.write32 {address = 258112 : ui32, value = 3221225540 : ui32}
      aiex.npu.write32 {address = 258048 : ui32, value = 3221225541 : ui32}
      aiex.npu.write32 {address = 258324 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 258640 : ui32, value = 455016755 : ui32}
      aiex.npu.write32 {address = 258324 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 258644 : ui32, value = 438239540 : ui32}
      aiex.npu.write32 {address = 258324 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 258648 : ui32, value = 253690165 : ui32}
      aiex.npu.write32 {address = 258304 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 258560 : ui32, value = 253690162 : ui32}
      aiex.npu.write32 {address = 1769520 : ui32, value = 3221225541 : ui32}
      aiex.npu.write32 {address = 1769496 : ui32, value = 3221225540 : ui32}
      aiex.npu.write32 {address = 1769760 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 1770112 : ui32, value = 455016757 : ui32}
      aiex.npu.write32 {address = 1769772 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 1770160 : ui32, value = 438239540 : ui32}
      aiex.npu.write32 {address = 2355212 : ui32, value = 3221225541 : ui32}
      aiex.npu.write32 {address = 2355480 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 2355808 : ui32, value = 455016757 : ui32}
      aiex.npu.maskwrite32 {address = 126976 : ui32, mask = 3072 : ui32, value = 1024 : ui32}
    }
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<2> (3)
      %1 = aie.amsel<3> (3)
      %2 = aie.amsel<4> (3)
      %3 = aie.amsel<5> (3)
      %4 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      %5 = aie.masterset(North : 1, %1)
      %6 = aie.masterset(North : 4, %2)
      %7 = aie.masterset(TileControl : 0, %3) {keep_pkt_header = true}
      aie.packet_rules(South : 3) {
        aie.rule(31, 27, %1)
        aie.rule(31, 26, %2)
        aie.rule(31, 15, %3)
      }
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<4> (3)
      %1 = aie.amsel<5> (3)
      %2 = aie.masterset(North : 1, %1)
      %3 = aie.masterset(TileControl : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(South : 1) {
        aie.rule(31, 27, %1)
      }
      aie.packet_rules(South : 4) {
        aie.rule(31, 26, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(TileControl : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(South : 1) {
        aie.rule(31, 27, %0)
      }
    }
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%shim_noc_tile_0_0, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%shim_noc_tile_0_0, MM2S, 0)
    aie.packet_flow(26) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(27) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
  }
  aie.device(npu2_1col) {
    memref.global "private" constant @blockwrite_data : memref<9xi32> = dense<[608, 0, 0, 0, 0, 0, 0, 0, 8]>
    memref.global "private" constant @blockwrite_data_0 : memref<135xi32> = dense<"0x4400E00907004400200400004400200800009807211284014040008000000000000000000000BA725028018B80F72C00E1000010FE37FFFF3F5B012000F02C00D48139D583DF00000000000000000000000018003010000000000000000000009888A01284014020008000000000000000000000F670502A008B9C2000F02C0004010070000000000000000000000000040100880100B8006018000000000000E1000078A5010000005B012000F02C00000000000000000000000000180008100000000000000000000000009867010084000068000000000000000000000000BA70000000020000E1FF983DFC0F04010010010018C500100000000000000000BA40000044000000E1FF18C900100000000000003681AD000000002000F02C00BA10B07800000000F0074480E30600004400C8000700BA1000B0E00100F02C00E1000010A8300100005B012000F02C002CFA0350C50300000000000000000000E1000078A5010000005B012000E0C823E1000078A5019821235B01200050C503E1000078A5010000005B012000F02C00E1000078A5010000005B012000F02C00E1000078A5010000005B012000F02C00E1000078A5010000005B012000F02C00E1000078A5010000005B012000F02C0018A0201418471E0118336414000000000000BA40000048000000210018C10010000000000000BA7EA501001000F02C002C9A012087FF84000020010018050210C4010000F8FF000000000000">
    memref.global "private" constant @blockwrite_data_1 : memref<5xi32> = dense<[269621272, 271056920, 0, 0, 404226048]>
    memref.global "private" constant @blockwrite_data_2 : memref<5xi32> = dense<[269490200, 271056920, 0, 0, 29622272]>
    memref.global "private" constant @blockwrite_data_3 : memref<42xi32> = dense<[452, 289013768, 133120, -21319680, -939519686, -1342177279, 1105526643, 1948319440, 1883242884, -174555135, 1033375741, 4080, 472776704, 441909248, -1241513920, 2945026, 119283352, 0, 0, 0, 271581208, 467583224, 0, 0, 333506712, 1346371972, 32769, 0, 472776704, 441909248, -1241513920, 2945026, 133183768, 133550360, 134082840, 133929240, 0, 1572864, 29626408, -524288, 0, 0]>
    memref.global "private" constant @blockwrite_data_4 : memref<81xi32> = dense<[28858, 512, 2033909665, 1091071456, -83382272, 1776316418, -91312128, 1848668162, -66146304, 1781559298, -57757696, 839913786, -1342176828, -1043006066, 9097649, 1806725122, -32591872, 1785753602, -24203264, 741371906, 9580545, 137392130, -15814655, 450519288, 445800696, 1715438052, -1042018116, 323211429, 800092162, -7950336, 779120642, -49893376, -874483710, -75059199, 279445572, -1990721529, 1618878048, 989986859, 536871030, 2945024, 102538904, 115089048, 0, 0, 1075314688, 553127984, 7120, 0, 2013266145, 421, 536959744, 2945024, 324788248, 404201720, 0, 0, -1697613638, 538176002, 2025519879, -905930096, -91873263, 645496876, -2144535542, -57792655, 132968728, 133206296, 133476632, 133746968, 553156652, -2144534650, -49667728, 570130476, 29686670, -524288, 73875940, 1105462528, 660604773, 297500696, 276594712, 280657944, 284950552]>
    memref.global "private" constant @blockwrite_data_5 : memref<6xi32> = dense<[4195328, 0, 0, 0, 0, 100941792]>
    memref.global "private" constant @blockwrite_data_6 : memref<6xi32> = dense<[134218752, 1074266112, 0, 0, 0, 235167715]>
    memref.global "private" constant @blockwrite_data_7 : memref<8xi32> = dense<[1024, 655360, 0, 0, 0, 0, 0, -2126381248]>
    memref.global "private" constant @blockwrite_data_8 : memref<8xi32> = dense<[-2147482624, 1703936, 0, 0, 0, 0, 0, -2126446783]>
    memref.global "private" constant @blockwrite_data_9 : memref<8xi32> = dense<[-2130705408, 25853952, 0, 0, 0, 0, 0, -2126315709]>
    memref.global "private" constant @blockwrite_data_10 : memref<8xi32> = dense<[1024, 26902528, 0, 0, 0, 0, 0, -2126250174]>
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<2> (3)
      %2 = aie.amsel<3> (3)
      %3 = aie.amsel<4> (3)
      %4 = aie.amsel<5> (3)
      %5 = aie.masterset(South : 2, %0)
      %6 = aie.masterset(South : 0, %1) {keep_pkt_header = true}
      %7 = aie.masterset(North : 1, %2)
      %8 = aie.masterset(North : 4, %3)
      %9 = aie.masterset(TileControl : 0, %4) {keep_pkt_header = true}
      aie.packet_rules(South : 3) {
        aie.rule(31, 27, %2)
        aie.rule(6, 2, %3)
        aie.rule(31, 15, %4)
      }
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %1)
      }
      aie.packet_rules(North : 2) {
        aie.rule(31, 2, %0)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.amsel<3> (0)
      %4 = aie.amsel<4> (3)
      %5 = aie.amsel<5> (3)
      %6 = aie.masterset(DMA : 0, %2)
      %7 = aie.masterset(DMA : 1, %3)
      %8 = aie.masterset(South : 2, %1)
      %9 = aie.masterset(North : 1, %5)
      %10 = aie.masterset(North : 5, %0)
      %11 = aie.masterset(TileControl : 0, %4) {keep_pkt_header = true}
      aie.packet_rules(South : 1) {
        aie.rule(31, 27, %5)
      }
      aie.packet_rules(South : 4) {
        aie.rule(31, 26, %4)
        aie.rule(31, 3, %2)
      }
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 2, %1)
      }
      aie.packet_rules(North : 0) {
        aie.rule(31, 1, %3)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<5> (3)
      %3 = aie.masterset(DMA : 0, %1)
      %4 = aie.masterset(South : 0, %0)
      %5 = aie.masterset(TileControl : 0, %2) {keep_pkt_header = true}
      aie.packet_rules(South : 1) {
        aie.rule(31, 27, %2)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 1, %0)
      }
      aie.packet_rules(South : 5) {
        aie.rule(31, 0, %1)
      }
    }
    %objFifo_in1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "objFifo_in1_cons_buff_0"} : memref<64x64xi8> 
    %objFifo_in1_cons_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "objFifo_in1_cons_buff_1"} : memref<64x64xi8> 
    %objFifo_out1_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "objFifo_out1_buff_0"} : memref<64x64xi8> 
    %objFifo_out1_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "objFifo_out1_buff_1"} : memref<64x64xi8> 
    %objFifo_in1_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "objFifo_in1_cons_prod_lock"}
    %objFifo_in1_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "objFifo_in1_cons_cons_lock"}
    %objFifo_out1_prod_lock = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "objFifo_out1_prod_lock"}
    %objFifo_out1_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "objFifo_out1_cons_lock"}
    aie.packet_flow(0) {
      aie.packet_source<%mem_tile_0_1, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 1>
    }
    aie.packet_flow(2) {
      aie.packet_source<%mem_tile_0_1, DMA : 1>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 0>
    }
    aie.packet_flow(3) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 0>
    }
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {elf_file = "/scratch/roesti/mlir-aie/test/npu-xrt/ctrl_packet_reconfig/aie_overlay.mlir.prj/main_core_0_2.elf"}
    aie.shim_dma_allocation @objFifo_in0(%shim_noc_tile_0_0, MM2S, 0)
    memref.global "private" constant @blockwrite_data_11 : memref<8xi32> = dense<[15, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_12 : memref<8xi32> = dense<[203, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_13 : memref<8xi32> = dense<[9, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_14 : memref<8xi32> = dense<[64, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_15 : memref<8xi32> = dense<[123, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_16 : memref<8xi32> = dense<[10, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_17 : memref<8xi32> = dense<[12, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_18 : memref<8xi32> = dense<[1024, 0, 1075314688, 16777216, -1006632945, 33554432, 0, 33554432]>
    memref.global "private" constant @blockwrite_data_19 : memref<8xi32> = dense<[1024, 0, 0, 16777216, -1006632945, 33554432, 0, 33554432]>
    aie.runtime_sequence @run(%arg0: memref<64x64xi8>, %arg1: memref<64x64xi8>, %arg2: memref<?xi32>) {
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219536 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 2219544 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 2219520 : ui32, mask = 2 : ui32, value = 2 : ui32}
      aiex.npu.maskwrite32 {address = 2219528 : ui32, mask = 2 : ui32, value = 2 : ui32}
      %0 = memref.get_global @blockwrite_data : memref<9xi32>
      %1 = memref.get_global @blockwrite_data_11 : memref<8xi32>
      aiex.npu.blockwrite(%1) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %2 = memref.get_global @blockwrite_data_0 : memref<135xi32>
      %3 = memref.get_global @blockwrite_data_12 : memref<8xi32>
      aiex.npu.blockwrite(%3) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 60 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %4 = memref.get_global @blockwrite_data_1 : memref<5xi32>
      %5 = memref.get_global @blockwrite_data_13 : memref<8xi32>
      aiex.npu.blockwrite(%5) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 872 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %6 = memref.get_global @blockwrite_data_2 : memref<5xi32>
      %7 = memref.get_global @blockwrite_data_13 : memref<8xi32>
      aiex.npu.blockwrite(%7) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 908 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %8 = memref.get_global @blockwrite_data_3 : memref<42xi32>
      %9 = memref.get_global @blockwrite_data_14 : memref<8xi32>
      aiex.npu.blockwrite(%9) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 944 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %10 = memref.get_global @blockwrite_data_4 : memref<81xi32>
      %11 = memref.get_global @blockwrite_data_15 : memref<8xi32>
      aiex.npu.blockwrite(%11) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 1200 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
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
      aiex.npu.write32 {address = 2224128 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 2224144 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224160 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 2224176 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1835008 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 1835024 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1835040 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 1835056 : ui32, value = 0 : ui32}
      %12 = memref.get_global @blockwrite_data_5 : memref<6xi32>
      %13 = memref.get_global @blockwrite_data_16 : memref<8xi32>
      aiex.npu.blockwrite(%13) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 1692 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %14 = memref.get_global @blockwrite_data_6 : memref<6xi32>
      %15 = memref.get_global @blockwrite_data_16 : memref<8xi32>
      aiex.npu.blockwrite(%15) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 1732 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.write32 {address = 2219524 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2219520 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 2219540 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 2219536 : ui32, value = 1 : ui32}
      %16 = memref.get_global @blockwrite_data_7 : memref<8xi32>
      %17 = memref.get_global @blockwrite_data_17 : memref<8xi32>
      aiex.npu.blockwrite(%17) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 1772 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %18 = memref.get_global @blockwrite_data_8 : memref<8xi32>
      %19 = memref.get_global @blockwrite_data_17 : memref<8xi32>
      aiex.npu.blockwrite(%19) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 1820 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %20 = memref.get_global @blockwrite_data_9 : memref<8xi32>
      %21 = memref.get_global @blockwrite_data_17 : memref<8xi32>
      aiex.npu.blockwrite(%21) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 1868 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      %22 = memref.get_global @blockwrite_data_10 : memref<8xi32>
      %23 = memref.get_global @blockwrite_data_17 : memref<8xi32>
      aiex.npu.blockwrite(%23) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 1916 : i32}
      aiex.npu.maskwrite32 {address = 119312 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119316 : ui32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.write32 {address = 1705476 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1705472 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 1705524 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 1705520 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 1705532 : ui32, value = 24 : ui32}
      aiex.npu.write32 {address = 1705528 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 1705484 : ui32, value = 25 : ui32}
      aiex.npu.write32 {address = 1705480 : ui32, value = 1 : ui32}
      aiex.npu.write32 {address = 258064 : ui32, value = 3221225608 : ui32}
      aiex.npu.write32 {address = 258056 : ui32, value = 3221225538 : ui32}
      aiex.npu.write32 {address = 258100 : ui32, value = 3221225539 : ui32}
      aiex.npu.write32 {address = 258112 : ui32, value = 3221225540 : ui32}
      aiex.npu.write32 {address = 258048 : ui32, value = 3221225541 : ui32}
      aiex.npu.write32 {address = 258324 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 258640 : ui32, value = 455016755 : ui32}
      aiex.npu.write32 {address = 258324 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 258644 : ui32, value = 33947956 : ui32}
      aiex.npu.write32 {address = 258324 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 258648 : ui32, value = 253690165 : ui32}
      aiex.npu.write32 {address = 258304 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 258560 : ui32, value = 253690162 : ui32}
      aiex.npu.write32 {address = 258368 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 258816 : ui32, value = 35586304 : ui32}
      aiex.npu.write32 {address = 1769472 : ui32, value = 3221225610 : ui32}
      aiex.npu.write32 {address = 1769476 : ui32, value = 3221225611 : ui32}
      aiex.npu.write32 {address = 1769508 : ui32, value = 3221225481 : ui32}
      aiex.npu.write32 {address = 1769520 : ui32, value = 3221225541 : ui32}
      aiex.npu.write32 {address = 1769536 : ui32, value = 3221225480 : ui32}
      aiex.npu.write32 {address = 1769496 : ui32, value = 3221225540 : ui32}
      aiex.npu.write32 {address = 1769760 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 1770112 : ui32, value = 455016757 : ui32}
      aiex.npu.write32 {address = 1769772 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 1770160 : ui32, value = 438239540 : ui32}
      aiex.npu.write32 {address = 1769772 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 1770164 : ui32, value = 52363522 : ui32}
      aiex.npu.write32 {address = 1769732 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 1770000 : ui32, value = 35586305 : ui32}
      aiex.npu.write32 {address = 1769780 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 1770192 : ui32, value = 18809091 : ui32}
      aiex.npu.write32 {address = 1769728 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 1769984 : ui32, value = 2031872 : ui32}
      aiex.npu.write32 {address = 2355204 : ui32, value = 3221225609 : ui32}
      aiex.npu.write32 {address = 2355220 : ui32, value = 3221225480 : ui32}
      aiex.npu.write32 {address = 2355212 : ui32, value = 3221225541 : ui32}
      aiex.npu.write32 {address = 2355480 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 2355808 : ui32, value = 455016757 : ui32}
      aiex.npu.write32 {address = 2355460 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 2355728 : ui32, value = 18809088 : ui32}
      aiex.npu.write32 {address = 2355496 : ui32, value = 3221225472 : ui32}
      aiex.npu.write32 {address = 2355872 : ui32, value = 2031873 : ui32}
      aiex.npu.maskwrite32 {address = 126976 : ui32, mask = 3072 : ui32, value = 1024 : ui32}
      aiex.npu.maskwrite32 {address = 126980 : ui32, mask = 48 : ui32, value = 16 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 1 : ui32}
      %24 = memref.get_global @blockwrite_data_18 : memref<8xi32>
      aiex.npu.blockwrite(%24) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119316 : ui32, value = 0 : ui32}
      %25 = memref.get_global @blockwrite_data_19 : memref<8xi32>
      aiex.npu.blockwrite(%25) {address = 118816 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.maskwrite32 {address = 119296 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119300 : ui32, value = 2147483649 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %objFifo_in0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "objFifo_in0_cons_buff_0"} : memref<64x64xi8> 
      %objFifo_in0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "objFifo_in0_cons_buff_1"} : memref<64x64xi8> 
      %objFifo_out0_buff_0 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "objFifo_out0_buff_0"} : memref<64x64xi8> 
      %objFifo_out0_buff_1 = aie.buffer(%mem_tile_0_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "objFifo_out0_buff_1"} : memref<64x64xi8> 
      %objFifo_in0_cons_prod_lock = aie.lock(%mem_tile_0_1, 0) {init = 1 : i32, sym_name = "objFifo_in0_cons_prod_lock"}
      %objFifo_in0_cons_cons_lock = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_cons_lock"}
      %objFifo_out0_prod_lock = aie.lock(%mem_tile_0_1, 2) {init = 1 : i32, sym_name = "objFifo_out0_prod_lock"}
      %objFifo_out0_cons_lock = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "objFifo_out0_cons_lock"}
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<64x64xi8>) {bd_id = 0 : i32, next_bd_id = 0 : i32}
        aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<64x64xi8>) {bd_id = 1 : i32, next_bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      }]
      %2 = aie.dma(MM2S, 1) [{
        aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out0_buff_0 : memref<64x64xi8>) {bd_id = 24 : i32, next_bd_id = 24 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
      }]
      %3 = aie.dma(S2MM, 1) [{
        aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out0_buff_0 : memref<64x64xi8>) {bd_id = 25 : i32, next_bd_id = 25 : i32}
        aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
      }]
      aie.end
    }
    aie.shim_dma_allocation @objFifo_out0(%shim_noc_tile_0_0, S2MM, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in1_cons_buff_0 : memref<64x64xi8>) {bd_id = 0 : i32, next_bd_id = 0 : i32}
        aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out1_buff_0 : memref<64x64xi8>) {bd_id = 1 : i32, next_bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
      }]
      aie.end
    }
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%shim_noc_tile_0_0, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%shim_noc_tile_0_0, MM2S, 0)
    aie.packet_flow(26) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(27) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
  }
}
