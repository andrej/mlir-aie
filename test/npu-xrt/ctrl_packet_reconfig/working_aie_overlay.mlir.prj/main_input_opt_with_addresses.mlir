module attributes {llvm.target_triple = "aie2p"} {
  llvm.mlir.global external @objFifo_out0_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.mlir.global external @objFifo_out0_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.mlir.global external @objFifo_in0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.mlir.global external @objFifo_in0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.mlir.global external @objFifo_out1_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.mlir.global external @objFifo_out1_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.mlir.global external @objFifo_in1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.mlir.global external @objFifo_in1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x i8>>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.event(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @objFifo_out1_buff_0 : !llvm.ptr
    %1 = llvm.mlir.addressof @objFifo_in1_cons_buff_0 : !llvm.ptr
    %2 = llvm.mlir.constant(51 : i32) : i32
    %3 = llvm.mlir.constant(48 : i32) : i32
    %4 = llvm.mlir.constant(50 : i32) : i32
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(-1 : i32) : i32
    %7 = llvm.mlir.constant(0 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(12 : i8) : i8
    %10 = llvm.mlir.constant(64 : index) : i64
    %11 = llvm.mlir.constant(49 : i32) : i32
    llvm.call @llvm.aie2p.acquire(%11, %6) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%4, %6) : (i32, i32) -> ()
    llvm.br ^bb1(%7 : i64)
  ^bb1(%12: i64):  // 2 preds: ^bb0, ^bb4
    %13 = llvm.icmp "slt" %12, %10 : i64
    llvm.cond_br %13, ^bb2(%7 : i64), ^bb5
  ^bb2(%14: i64):  // 2 preds: ^bb1, ^bb3
    %15 = llvm.icmp "slt" %14, %10 : i64
    llvm.cond_br %15, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %16 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<64 x i8>>
    %17 = llvm.mul %12, %10 overflow<nsw, nuw> : i64
    %18 = llvm.add %17, %14 overflow<nsw, nuw> : i64
    %19 = llvm.getelementptr inbounds|nuw %16[%18] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %20 = llvm.load %19 : !llvm.ptr -> i8
    %21 = llvm.add %20, %9 : i8
    %22 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<64 x i8>>
    %23 = llvm.getelementptr inbounds|nuw %22[%18] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %21, %23 : i8, !llvm.ptr
    %24 = llvm.add %14, %8 : i64
    llvm.br ^bb2(%24 : i64)
  ^bb4:  // pred: ^bb2
    %25 = llvm.add %12, %8 : i64
    llvm.br ^bb1(%25 : i64)
  ^bb5:  // pred: ^bb1
    llvm.call @llvm.aie2p.release(%3, %5) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%2, %5) : (i32, i32) -> ()
    llvm.return
  }
}

