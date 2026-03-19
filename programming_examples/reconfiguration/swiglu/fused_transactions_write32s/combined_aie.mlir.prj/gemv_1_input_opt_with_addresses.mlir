module attributes {llvm.target_triple = "aie2p"} {
  llvm.mlir.global external @A_L3L1_0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @A_L3L1_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @A_L3L1_1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @A_L3L1_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @A_L3L1_2_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @A_L3L1_2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @A_L3L1_3_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @A_L3L1_3_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @A_L3L1_4_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @A_L3L1_4_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @A_L3L1_5_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @A_L3L1_5_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @A_L3L1_6_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @A_L3L1_6_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @A_L3L1_7_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @A_L3L1_7_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @B_L3L1_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @B_L3L1_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @B_L3L1_2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @B_L3L1_3_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @B_L3L1_4_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @B_L3L1_5_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @B_L3L1_6_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @B_L3L1_7_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x bf16>
  llvm.mlir.global external @C_L1L3_0_buff_0() {addr_space = 0 : i32} : !llvm.array<1024 x bf16>
  llvm.mlir.global external @C_L1L3_1_buff_0() {addr_space = 0 : i32} : !llvm.array<1024 x bf16>
  llvm.mlir.global external @C_L1L3_2_buff_0() {addr_space = 0 : i32} : !llvm.array<1024 x bf16>
  llvm.mlir.global external @C_L1L3_3_buff_0() {addr_space = 0 : i32} : !llvm.array<1024 x bf16>
  llvm.mlir.global external @C_L1L3_4_buff_0() {addr_space = 0 : i32} : !llvm.array<1024 x bf16>
  llvm.mlir.global external @C_L1L3_5_buff_0() {addr_space = 0 : i32} : !llvm.array<1024 x bf16>
  llvm.mlir.global external @C_L1L3_6_buff_0() {addr_space = 0 : i32} : !llvm.array<1024 x bf16>
  llvm.mlir.global external @C_L1L3_7_buff_0() {addr_space = 0 : i32} : !llvm.array<1024 x bf16>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.event(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @matvec_vectorized_bf16_bf16(i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @core_1_5() {
    %0 = llvm.mlir.addressof @A_L3L1_7_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @A_L3L1_7_cons_buff_0 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_L3L1_7_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @C_L1L3_7_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(50 : i32) : i32
    %5 = llvm.mlir.constant(53 : i32) : i32
    %6 = llvm.mlir.constant(48 : i32) : i32
    %7 = llvm.mlir.constant(49 : i32) : i32
    %8 = llvm.mlir.constant(52 : i32) : i32
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(-1 : i32) : i32
    %11 = llvm.mlir.constant(2 : index) : i64
    %12 = llvm.mlir.constant(2048 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(1024 : index) : i64
    %15 = llvm.mlir.constant(4294967295 : index) : i64
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(9223372036854775807 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%16 : i64)
  ^bb1(%19: i64):  // 2 preds: ^bb0, ^bb7
    %20 = llvm.icmp "slt" %19, %17 : i64
    llvm.cond_br %20, ^bb2(%16 : i64), ^bb8
  ^bb2(%21: i64):  // 2 preds: ^bb1, ^bb6
    %22 = llvm.icmp "slt" %21, %15 : i64
    llvm.cond_br %22, ^bb3, ^bb7
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2p.acquire(%9, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%8, %10) : (i32, i32) -> ()
    llvm.br ^bb4(%16 : i64)
  ^bb4(%23: i64):  // 2 preds: ^bb3, ^bb5
    %24 = llvm.icmp "slt" %23, %14 : i64
    llvm.cond_br %24, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    %25 = llvm.trunc %23 : i64 to i32
    %26 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x bf16>
    %27 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    %28 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    llvm.call @matvec_vectorized_bf16_bf16(%13, %12, %25, %28, %27, %26) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%6, %13) : (i32, i32) -> ()
    %29 = llvm.add %23, %18 : i64
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    %30 = llvm.trunc %29 : i64 to i32
    %31 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    llvm.call @matvec_vectorized_bf16_bf16(%13, %12, %30, %31, %27, %26) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%6, %13) : (i32, i32) -> ()
    %32 = llvm.add %23, %11 : i64
    llvm.br ^bb4(%32 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2p.release(%5, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%4, %13) : (i32, i32) -> ()
    %33 = llvm.add %21, %18 : i64
    llvm.br ^bb2(%33 : i64)
  ^bb7:  // pred: ^bb2
    %34 = llvm.add %19, %18 : i64
    llvm.br ^bb1(%34 : i64)
  ^bb8:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_1_4() {
    %0 = llvm.mlir.addressof @A_L3L1_6_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @A_L3L1_6_cons_buff_0 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_L3L1_6_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @C_L1L3_6_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(50 : i32) : i32
    %5 = llvm.mlir.constant(53 : i32) : i32
    %6 = llvm.mlir.constant(48 : i32) : i32
    %7 = llvm.mlir.constant(49 : i32) : i32
    %8 = llvm.mlir.constant(52 : i32) : i32
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(-1 : i32) : i32
    %11 = llvm.mlir.constant(2 : index) : i64
    %12 = llvm.mlir.constant(2048 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(1024 : index) : i64
    %15 = llvm.mlir.constant(4294967295 : index) : i64
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(9223372036854775807 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%16 : i64)
  ^bb1(%19: i64):  // 2 preds: ^bb0, ^bb7
    %20 = llvm.icmp "slt" %19, %17 : i64
    llvm.cond_br %20, ^bb2(%16 : i64), ^bb8
  ^bb2(%21: i64):  // 2 preds: ^bb1, ^bb6
    %22 = llvm.icmp "slt" %21, %15 : i64
    llvm.cond_br %22, ^bb3, ^bb7
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2p.acquire(%9, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%8, %10) : (i32, i32) -> ()
    llvm.br ^bb4(%16 : i64)
  ^bb4(%23: i64):  // 2 preds: ^bb3, ^bb5
    %24 = llvm.icmp "slt" %23, %14 : i64
    llvm.cond_br %24, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    %25 = llvm.trunc %23 : i64 to i32
    %26 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x bf16>
    %27 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    %28 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    llvm.call @matvec_vectorized_bf16_bf16(%13, %12, %25, %28, %27, %26) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%6, %13) : (i32, i32) -> ()
    %29 = llvm.add %23, %18 : i64
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    %30 = llvm.trunc %29 : i64 to i32
    %31 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    llvm.call @matvec_vectorized_bf16_bf16(%13, %12, %30, %31, %27, %26) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%6, %13) : (i32, i32) -> ()
    %32 = llvm.add %23, %11 : i64
    llvm.br ^bb4(%32 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2p.release(%5, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%4, %13) : (i32, i32) -> ()
    %33 = llvm.add %21, %18 : i64
    llvm.br ^bb2(%33 : i64)
  ^bb7:  // pred: ^bb2
    %34 = llvm.add %19, %18 : i64
    llvm.br ^bb1(%34 : i64)
  ^bb8:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_1_3() {
    %0 = llvm.mlir.addressof @A_L3L1_5_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @A_L3L1_5_cons_buff_0 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_L3L1_5_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @C_L1L3_5_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(50 : i32) : i32
    %5 = llvm.mlir.constant(53 : i32) : i32
    %6 = llvm.mlir.constant(48 : i32) : i32
    %7 = llvm.mlir.constant(49 : i32) : i32
    %8 = llvm.mlir.constant(52 : i32) : i32
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(-1 : i32) : i32
    %11 = llvm.mlir.constant(2 : index) : i64
    %12 = llvm.mlir.constant(2048 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(1024 : index) : i64
    %15 = llvm.mlir.constant(4294967295 : index) : i64
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(9223372036854775807 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%16 : i64)
  ^bb1(%19: i64):  // 2 preds: ^bb0, ^bb7
    %20 = llvm.icmp "slt" %19, %17 : i64
    llvm.cond_br %20, ^bb2(%16 : i64), ^bb8
  ^bb2(%21: i64):  // 2 preds: ^bb1, ^bb6
    %22 = llvm.icmp "slt" %21, %15 : i64
    llvm.cond_br %22, ^bb3, ^bb7
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2p.acquire(%9, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%8, %10) : (i32, i32) -> ()
    llvm.br ^bb4(%16 : i64)
  ^bb4(%23: i64):  // 2 preds: ^bb3, ^bb5
    %24 = llvm.icmp "slt" %23, %14 : i64
    llvm.cond_br %24, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    %25 = llvm.trunc %23 : i64 to i32
    %26 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x bf16>
    %27 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    %28 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    llvm.call @matvec_vectorized_bf16_bf16(%13, %12, %25, %28, %27, %26) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%6, %13) : (i32, i32) -> ()
    %29 = llvm.add %23, %18 : i64
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    %30 = llvm.trunc %29 : i64 to i32
    %31 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    llvm.call @matvec_vectorized_bf16_bf16(%13, %12, %30, %31, %27, %26) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%6, %13) : (i32, i32) -> ()
    %32 = llvm.add %23, %11 : i64
    llvm.br ^bb4(%32 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2p.release(%5, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%4, %13) : (i32, i32) -> ()
    %33 = llvm.add %21, %18 : i64
    llvm.br ^bb2(%33 : i64)
  ^bb7:  // pred: ^bb2
    %34 = llvm.add %19, %18 : i64
    llvm.br ^bb1(%34 : i64)
  ^bb8:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_1_2() {
    %0 = llvm.mlir.addressof @A_L3L1_4_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @A_L3L1_4_cons_buff_0 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_L3L1_4_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @C_L1L3_4_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(50 : i32) : i32
    %5 = llvm.mlir.constant(53 : i32) : i32
    %6 = llvm.mlir.constant(48 : i32) : i32
    %7 = llvm.mlir.constant(49 : i32) : i32
    %8 = llvm.mlir.constant(52 : i32) : i32
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(-1 : i32) : i32
    %11 = llvm.mlir.constant(2 : index) : i64
    %12 = llvm.mlir.constant(2048 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(1024 : index) : i64
    %15 = llvm.mlir.constant(4294967295 : index) : i64
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(9223372036854775807 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%16 : i64)
  ^bb1(%19: i64):  // 2 preds: ^bb0, ^bb7
    %20 = llvm.icmp "slt" %19, %17 : i64
    llvm.cond_br %20, ^bb2(%16 : i64), ^bb8
  ^bb2(%21: i64):  // 2 preds: ^bb1, ^bb6
    %22 = llvm.icmp "slt" %21, %15 : i64
    llvm.cond_br %22, ^bb3, ^bb7
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2p.acquire(%9, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%8, %10) : (i32, i32) -> ()
    llvm.br ^bb4(%16 : i64)
  ^bb4(%23: i64):  // 2 preds: ^bb3, ^bb5
    %24 = llvm.icmp "slt" %23, %14 : i64
    llvm.cond_br %24, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    %25 = llvm.trunc %23 : i64 to i32
    %26 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x bf16>
    %27 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    %28 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    llvm.call @matvec_vectorized_bf16_bf16(%13, %12, %25, %28, %27, %26) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%6, %13) : (i32, i32) -> ()
    %29 = llvm.add %23, %18 : i64
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    %30 = llvm.trunc %29 : i64 to i32
    %31 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    llvm.call @matvec_vectorized_bf16_bf16(%13, %12, %30, %31, %27, %26) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%6, %13) : (i32, i32) -> ()
    %32 = llvm.add %23, %11 : i64
    llvm.br ^bb4(%32 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2p.release(%5, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%4, %13) : (i32, i32) -> ()
    %33 = llvm.add %21, %18 : i64
    llvm.br ^bb2(%33 : i64)
  ^bb7:  // pred: ^bb2
    %34 = llvm.add %19, %18 : i64
    llvm.br ^bb1(%34 : i64)
  ^bb8:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_0_5() {
    %0 = llvm.mlir.addressof @A_L3L1_3_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @A_L3L1_3_cons_buff_0 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_L3L1_3_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @C_L1L3_3_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(50 : i32) : i32
    %5 = llvm.mlir.constant(53 : i32) : i32
    %6 = llvm.mlir.constant(48 : i32) : i32
    %7 = llvm.mlir.constant(49 : i32) : i32
    %8 = llvm.mlir.constant(52 : i32) : i32
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(-1 : i32) : i32
    %11 = llvm.mlir.constant(2 : index) : i64
    %12 = llvm.mlir.constant(2048 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(1024 : index) : i64
    %15 = llvm.mlir.constant(4294967295 : index) : i64
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(9223372036854775807 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%16 : i64)
  ^bb1(%19: i64):  // 2 preds: ^bb0, ^bb7
    %20 = llvm.icmp "slt" %19, %17 : i64
    llvm.cond_br %20, ^bb2(%16 : i64), ^bb8
  ^bb2(%21: i64):  // 2 preds: ^bb1, ^bb6
    %22 = llvm.icmp "slt" %21, %15 : i64
    llvm.cond_br %22, ^bb3, ^bb7
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2p.acquire(%9, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%8, %10) : (i32, i32) -> ()
    llvm.br ^bb4(%16 : i64)
  ^bb4(%23: i64):  // 2 preds: ^bb3, ^bb5
    %24 = llvm.icmp "slt" %23, %14 : i64
    llvm.cond_br %24, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    %25 = llvm.trunc %23 : i64 to i32
    %26 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x bf16>
    %27 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    %28 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    llvm.call @matvec_vectorized_bf16_bf16(%13, %12, %25, %28, %27, %26) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%6, %13) : (i32, i32) -> ()
    %29 = llvm.add %23, %18 : i64
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    %30 = llvm.trunc %29 : i64 to i32
    %31 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    llvm.call @matvec_vectorized_bf16_bf16(%13, %12, %30, %31, %27, %26) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%6, %13) : (i32, i32) -> ()
    %32 = llvm.add %23, %11 : i64
    llvm.br ^bb4(%32 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2p.release(%5, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%4, %13) : (i32, i32) -> ()
    %33 = llvm.add %21, %18 : i64
    llvm.br ^bb2(%33 : i64)
  ^bb7:  // pred: ^bb2
    %34 = llvm.add %19, %18 : i64
    llvm.br ^bb1(%34 : i64)
  ^bb8:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_0_4() {
    %0 = llvm.mlir.addressof @A_L3L1_2_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @A_L3L1_2_cons_buff_0 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_L3L1_2_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @C_L1L3_2_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(50 : i32) : i32
    %5 = llvm.mlir.constant(53 : i32) : i32
    %6 = llvm.mlir.constant(48 : i32) : i32
    %7 = llvm.mlir.constant(49 : i32) : i32
    %8 = llvm.mlir.constant(52 : i32) : i32
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(-1 : i32) : i32
    %11 = llvm.mlir.constant(2 : index) : i64
    %12 = llvm.mlir.constant(2048 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(1024 : index) : i64
    %15 = llvm.mlir.constant(4294967295 : index) : i64
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(9223372036854775807 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%16 : i64)
  ^bb1(%19: i64):  // 2 preds: ^bb0, ^bb7
    %20 = llvm.icmp "slt" %19, %17 : i64
    llvm.cond_br %20, ^bb2(%16 : i64), ^bb8
  ^bb2(%21: i64):  // 2 preds: ^bb1, ^bb6
    %22 = llvm.icmp "slt" %21, %15 : i64
    llvm.cond_br %22, ^bb3, ^bb7
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2p.acquire(%9, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%8, %10) : (i32, i32) -> ()
    llvm.br ^bb4(%16 : i64)
  ^bb4(%23: i64):  // 2 preds: ^bb3, ^bb5
    %24 = llvm.icmp "slt" %23, %14 : i64
    llvm.cond_br %24, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    %25 = llvm.trunc %23 : i64 to i32
    %26 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x bf16>
    %27 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    %28 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    llvm.call @matvec_vectorized_bf16_bf16(%13, %12, %25, %28, %27, %26) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%6, %13) : (i32, i32) -> ()
    %29 = llvm.add %23, %18 : i64
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    %30 = llvm.trunc %29 : i64 to i32
    %31 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    llvm.call @matvec_vectorized_bf16_bf16(%13, %12, %30, %31, %27, %26) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%6, %13) : (i32, i32) -> ()
    %32 = llvm.add %23, %11 : i64
    llvm.br ^bb4(%32 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2p.release(%5, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%4, %13) : (i32, i32) -> ()
    %33 = llvm.add %21, %18 : i64
    llvm.br ^bb2(%33 : i64)
  ^bb7:  // pred: ^bb2
    %34 = llvm.add %19, %18 : i64
    llvm.br ^bb1(%34 : i64)
  ^bb8:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_0_3() {
    %0 = llvm.mlir.addressof @A_L3L1_1_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @A_L3L1_1_cons_buff_0 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_L3L1_1_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @C_L1L3_1_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(50 : i32) : i32
    %5 = llvm.mlir.constant(53 : i32) : i32
    %6 = llvm.mlir.constant(48 : i32) : i32
    %7 = llvm.mlir.constant(49 : i32) : i32
    %8 = llvm.mlir.constant(52 : i32) : i32
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(-1 : i32) : i32
    %11 = llvm.mlir.constant(2 : index) : i64
    %12 = llvm.mlir.constant(2048 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(1024 : index) : i64
    %15 = llvm.mlir.constant(4294967295 : index) : i64
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(9223372036854775807 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%16 : i64)
  ^bb1(%19: i64):  // 2 preds: ^bb0, ^bb7
    %20 = llvm.icmp "slt" %19, %17 : i64
    llvm.cond_br %20, ^bb2(%16 : i64), ^bb8
  ^bb2(%21: i64):  // 2 preds: ^bb1, ^bb6
    %22 = llvm.icmp "slt" %21, %15 : i64
    llvm.cond_br %22, ^bb3, ^bb7
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2p.acquire(%9, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%8, %10) : (i32, i32) -> ()
    llvm.br ^bb4(%16 : i64)
  ^bb4(%23: i64):  // 2 preds: ^bb3, ^bb5
    %24 = llvm.icmp "slt" %23, %14 : i64
    llvm.cond_br %24, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    %25 = llvm.trunc %23 : i64 to i32
    %26 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x bf16>
    %27 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    %28 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    llvm.call @matvec_vectorized_bf16_bf16(%13, %12, %25, %28, %27, %26) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%6, %13) : (i32, i32) -> ()
    %29 = llvm.add %23, %18 : i64
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    %30 = llvm.trunc %29 : i64 to i32
    %31 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    llvm.call @matvec_vectorized_bf16_bf16(%13, %12, %30, %31, %27, %26) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%6, %13) : (i32, i32) -> ()
    %32 = llvm.add %23, %11 : i64
    llvm.br ^bb4(%32 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2p.release(%5, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%4, %13) : (i32, i32) -> ()
    %33 = llvm.add %21, %18 : i64
    llvm.br ^bb2(%33 : i64)
  ^bb7:  // pred: ^bb2
    %34 = llvm.add %19, %18 : i64
    llvm.br ^bb1(%34 : i64)
  ^bb8:  // pred: ^bb1
    llvm.return
  }
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @A_L3L1_0_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @A_L3L1_0_cons_buff_0 : !llvm.ptr
    %2 = llvm.mlir.addressof @B_L3L1_0_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @C_L1L3_0_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(50 : i32) : i32
    %5 = llvm.mlir.constant(53 : i32) : i32
    %6 = llvm.mlir.constant(48 : i32) : i32
    %7 = llvm.mlir.constant(49 : i32) : i32
    %8 = llvm.mlir.constant(52 : i32) : i32
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(-1 : i32) : i32
    %11 = llvm.mlir.constant(2 : index) : i64
    %12 = llvm.mlir.constant(2048 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(1024 : index) : i64
    %15 = llvm.mlir.constant(4294967295 : index) : i64
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(9223372036854775807 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%16 : i64)
  ^bb1(%19: i64):  // 2 preds: ^bb0, ^bb7
    %20 = llvm.icmp "slt" %19, %17 : i64
    llvm.cond_br %20, ^bb2(%16 : i64), ^bb8
  ^bb2(%21: i64):  // 2 preds: ^bb1, ^bb6
    %22 = llvm.icmp "slt" %21, %15 : i64
    llvm.cond_br %22, ^bb3, ^bb7
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2p.acquire(%9, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%8, %10) : (i32, i32) -> ()
    llvm.br ^bb4(%16 : i64)
  ^bb4(%23: i64):  // 2 preds: ^bb3, ^bb5
    %24 = llvm.icmp "slt" %23, %14 : i64
    llvm.cond_br %24, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    %25 = llvm.trunc %23 : i64 to i32
    %26 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1024 x bf16>
    %27 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    %28 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    llvm.call @matvec_vectorized_bf16_bf16(%13, %12, %25, %28, %27, %26) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%6, %13) : (i32, i32) -> ()
    %29 = llvm.add %23, %18 : i64
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    %30 = llvm.trunc %29 : i64 to i32
    %31 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x bf16>
    llvm.call @matvec_vectorized_bf16_bf16(%13, %12, %30, %31, %27, %26) : (i32, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2p.release(%6, %13) : (i32, i32) -> ()
    %32 = llvm.add %23, %11 : i64
    llvm.br ^bb4(%32 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2p.release(%5, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%4, %13) : (i32, i32) -> ()
    %33 = llvm.add %21, %18 : i64
    llvm.br ^bb2(%33 : i64)
  ^bb7:  // pred: ^bb2
    %34 = llvm.add %19, %18 : i64
    llvm.br ^bb1(%34 : i64)
  ^bb8:  // pred: ^bb1
    llvm.return
  }
}

