module attributes {llvm.target_triple = "aie2p"} {
  llvm.mlir.global external @constant_buffer() {addr_space = 0 : i32} : !llvm.array<8 x i32>
  llvm.mlir.global external @objFifo_out1_buff_1() {addr_space = 0 : i32} : !llvm.array<8 x i32>
  llvm.mlir.global external @objFifo_out1_buff_0() {addr_space = 0 : i32} : !llvm.array<8 x i32>
  llvm.mlir.global external @objFifo_in1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<8 x i32>
  llvm.mlir.global external @objFifo_in1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<8 x i32>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.event(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global private @myData(dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi32>) {addr_space = 0 : i32} : !llvm.array<8 x i32>
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @objFifo_out1_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @objFifo_in1_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @objFifo_out1_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @constant_buffer : !llvm.ptr
    %4 = llvm.mlir.addressof @objFifo_in1_cons_buff_0 : !llvm.ptr
    %5 = llvm.mlir.constant(51 : i32) : i32
    %6 = llvm.mlir.constant(48 : i32) : i32
    %7 = llvm.mlir.constant(50 : i32) : i32
    %8 = llvm.mlir.constant(49 : i32) : i32
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.mlir.constant(-1 : i32) : i32
    %11 = llvm.mlir.constant(8 : index) : i64
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.mlir.constant(2 : index) : i64
    llvm.br ^bb1(%12 : i64)
  ^bb1(%15: i64):  // 2 preds: ^bb0, ^bb8
    %16 = llvm.icmp "slt" %15, %11 : i64
    llvm.cond_br %16, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2p.acquire(%8, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    llvm.br ^bb3(%12 : i64)
  ^bb3(%17: i64):  // 2 preds: ^bb2, ^bb4
    %18 = llvm.icmp "slt" %17, %11 : i64
    llvm.cond_br %18, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %19 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>
    %20 = llvm.getelementptr inbounds|nuw %19[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %21 = llvm.load %20 : !llvm.ptr -> i32
    %22 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>
    %23 = llvm.getelementptr inbounds|nuw %22[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %24 = llvm.load %23 : !llvm.ptr -> i32
    %25 = llvm.add %21, %24 : i32
    %26 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>
    %27 = llvm.getelementptr inbounds|nuw %26[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %25, %27 : i32, !llvm.ptr
    %28 = llvm.add %17, %13 : i64
    llvm.br ^bb3(%28 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2p.release(%6, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%5, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%8, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%7, %10) : (i32, i32) -> ()
    llvm.br ^bb6(%12 : i64)
  ^bb6(%29: i64):  // 2 preds: ^bb5, ^bb7
    %30 = llvm.icmp "slt" %29, %11 : i64
    llvm.cond_br %30, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %31 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>
    %32 = llvm.getelementptr inbounds|nuw %31[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %33 = llvm.load %32 : !llvm.ptr -> i32
    %34 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>
    %35 = llvm.getelementptr inbounds|nuw %34[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %36 = llvm.load %35 : !llvm.ptr -> i32
    %37 = llvm.add %33, %36 : i32
    %38 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>
    %39 = llvm.getelementptr inbounds|nuw %38[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %37, %39 : i32, !llvm.ptr
    %40 = llvm.add %29, %13 : i64
    llvm.br ^bb6(%40 : i64)
  ^bb8:  // pred: ^bb6
    llvm.call @llvm.aie2p.release(%6, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%5, %9) : (i32, i32) -> ()
    %41 = llvm.add %15, %14 : i64
    llvm.br ^bb1(%41 : i64)
  ^bb9:  // pred: ^bb1
    llvm.return
  }
}

