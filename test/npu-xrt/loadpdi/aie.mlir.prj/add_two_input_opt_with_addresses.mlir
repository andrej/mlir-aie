module attributes {llvm.target_triple = "aie2p"} {
  llvm.mlir.global external @objfifo_in_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<128 x i32>
  llvm.mlir.global external @objfifo_out_buff_0() {addr_space = 0 : i32} : !llvm.array<128 x i32>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.event(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @objfifo_out_buff_0 : !llvm.ptr
    %1 = llvm.mlir.addressof @objfifo_in_cons_buff_0 : !llvm.ptr
    %2 = llvm.mlir.constant(51 : i32) : i32
    %3 = llvm.mlir.constant(48 : i32) : i32
    %4 = llvm.mlir.constant(50 : i32) : i32
    %5 = llvm.mlir.constant(49 : i32) : i32
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(-1 : i32) : i32
    %8 = llvm.mlir.constant(0 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.constant(2 : i32) : i32
    %11 = llvm.mlir.constant(128 : index) : i64
    %12 = llvm.mlir.constant(16777214 : index) : i64
    llvm.br ^bb1(%8 : i64)
  ^bb1(%13: i64):  // 2 preds: ^bb0, ^bb5
    %14 = llvm.icmp "slt" %13, %12 : i64
    llvm.cond_br %14, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2p.acquire(%5, %7) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%4, %7) : (i32, i32) -> ()
    llvm.br ^bb3(%8 : i64)
  ^bb3(%15: i64):  // 2 preds: ^bb2, ^bb4
    %16 = llvm.icmp "slt" %15, %11 : i64
    llvm.cond_br %16, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %17 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<128 x i32>
    %18 = llvm.getelementptr inbounds|nuw %17[%15] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %19 = llvm.load %18 : !llvm.ptr -> i32
    %20 = llvm.add %19, %10 : i32
    %21 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<128 x i32>
    %22 = llvm.getelementptr inbounds|nuw %21[%15] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %20, %22 : i32, !llvm.ptr
    %23 = llvm.add %15, %9 : i64
    llvm.br ^bb3(%23 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2p.release(%3, %6) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%2, %6) : (i32, i32) -> ()
    %24 = llvm.add %13, %9 : i64
    llvm.br ^bb1(%24 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}

