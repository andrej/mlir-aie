; ModuleID = '/scratch/roesti/mlir-aie/programming_examples/reconfiguration/swiglu/fused_transactions_write32s/combined_aie.mlir.prj/gemv_1_input.llpeanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@A_L3L1_0_cons_buff_1 = external global [2048 x bfloat]
@A_L3L1_0_cons_buff_0 = external global [2048 x bfloat]
@A_L3L1_1_cons_buff_1 = external global [2048 x bfloat]
@A_L3L1_1_cons_buff_0 = external global [2048 x bfloat]
@A_L3L1_2_cons_buff_1 = external global [2048 x bfloat]
@A_L3L1_2_cons_buff_0 = external global [2048 x bfloat]
@A_L3L1_3_cons_buff_1 = external global [2048 x bfloat]
@A_L3L1_3_cons_buff_0 = external global [2048 x bfloat]
@A_L3L1_4_cons_buff_1 = external global [2048 x bfloat]
@A_L3L1_4_cons_buff_0 = external global [2048 x bfloat]
@A_L3L1_5_cons_buff_1 = external global [2048 x bfloat]
@A_L3L1_5_cons_buff_0 = external global [2048 x bfloat]
@A_L3L1_6_cons_buff_1 = external global [2048 x bfloat]
@A_L3L1_6_cons_buff_0 = external global [2048 x bfloat]
@A_L3L1_7_cons_buff_1 = external global [2048 x bfloat]
@A_L3L1_7_cons_buff_0 = external global [2048 x bfloat]
@B_L3L1_0_cons_buff_0 = external global [2048 x bfloat]
@B_L3L1_1_cons_buff_0 = external global [2048 x bfloat]
@B_L3L1_2_cons_buff_0 = external global [2048 x bfloat]
@B_L3L1_3_cons_buff_0 = external global [2048 x bfloat]
@B_L3L1_4_cons_buff_0 = external global [2048 x bfloat]
@B_L3L1_5_cons_buff_0 = external global [2048 x bfloat]
@B_L3L1_6_cons_buff_0 = external global [2048 x bfloat]
@B_L3L1_7_cons_buff_0 = external global [2048 x bfloat]
@C_L1L3_0_buff_0 = external global [1024 x bfloat]
@C_L1L3_1_buff_0 = external global [1024 x bfloat]
@C_L1L3_2_buff_0 = external global [1024 x bfloat]
@C_L1L3_3_buff_0 = external global [1024 x bfloat]
@C_L1L3_4_buff_0 = external global [1024 x bfloat]
@C_L1L3_5_buff_0 = external global [1024 x bfloat]
@C_L1L3_6_buff_0 = external global [1024 x bfloat]
@C_L1L3_7_buff_0 = external global [1024 x bfloat]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @matvec_vectorized_bf16_bf16(i32, i32, i32, ptr, ptr, ptr) local_unnamed_addr

define void @core_1_5() local_unnamed_addr {
  br label %.preheader

.preheader:                                       ; preds = %0, %22
  %1 = phi i64 [ 0, %0 ], [ %23, %22 ]
  br label %2

2:                                                ; preds = %.preheader, %19
  %3 = phi i64 [ 0, %.preheader ], [ %20, %19 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %4

4:                                                ; preds = %4, %2
  %5 = phi i64 [ 0, %2 ], [ %17, %4 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %6 = trunc nuw i64 %5 to i32
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %6, ptr nonnull @A_L3L1_7_cons_buff_0, ptr nonnull @B_L3L1_7_cons_buff_0, ptr nonnull @C_L1L3_7_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %7 = or disjoint i32 %6, 1
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %7, ptr nonnull @A_L3L1_7_cons_buff_1, ptr nonnull @B_L3L1_7_cons_buff_0, ptr nonnull @C_L1L3_7_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %8 = trunc i64 %5 to i32
  %9 = or disjoint i32 %8, 2
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %9, ptr nonnull @A_L3L1_7_cons_buff_0, ptr nonnull @B_L3L1_7_cons_buff_0, ptr nonnull @C_L1L3_7_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %10 = or disjoint i32 %8, 3
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %10, ptr nonnull @A_L3L1_7_cons_buff_1, ptr nonnull @B_L3L1_7_cons_buff_0, ptr nonnull @C_L1L3_7_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %11 = trunc i64 %5 to i32
  %12 = or disjoint i32 %11, 4
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %12, ptr nonnull @A_L3L1_7_cons_buff_0, ptr nonnull @B_L3L1_7_cons_buff_0, ptr nonnull @C_L1L3_7_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %13 = or disjoint i32 %11, 5
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %13, ptr nonnull @A_L3L1_7_cons_buff_1, ptr nonnull @B_L3L1_7_cons_buff_0, ptr nonnull @C_L1L3_7_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %14 = or disjoint i64 %5, 6
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %15 = trunc nuw i64 %14 to i32
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %15, ptr nonnull @A_L3L1_7_cons_buff_0, ptr nonnull @B_L3L1_7_cons_buff_0, ptr nonnull @C_L1L3_7_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %16 = or disjoint i32 %15, 1
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %16, ptr nonnull @A_L3L1_7_cons_buff_1, ptr nonnull @B_L3L1_7_cons_buff_0, ptr nonnull @C_L1L3_7_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %17 = add nuw nsw i64 %5, 8
  %18 = icmp samesign ult i64 %14, 1022
  br i1 %18, label %4, label %19

19:                                               ; preds = %4
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %20 = add nuw nsw i64 %3, 1
  %21 = icmp samesign ult i64 %3, 4294967294
  br i1 %21, label %2, label %22

22:                                               ; preds = %19
  %23 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %23, 9223372036854775807
  br i1 %.not, label %24, label %.preheader

24:                                               ; preds = %22
  ret void
}

define void @core_1_4() local_unnamed_addr {
  br label %.preheader

.preheader:                                       ; preds = %0, %22
  %1 = phi i64 [ 0, %0 ], [ %23, %22 ]
  br label %2

2:                                                ; preds = %.preheader, %19
  %3 = phi i64 [ 0, %.preheader ], [ %20, %19 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %4

4:                                                ; preds = %4, %2
  %5 = phi i64 [ 0, %2 ], [ %17, %4 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %6 = trunc nuw i64 %5 to i32
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %6, ptr nonnull @A_L3L1_6_cons_buff_0, ptr nonnull @B_L3L1_6_cons_buff_0, ptr nonnull @C_L1L3_6_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %7 = or disjoint i32 %6, 1
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %7, ptr nonnull @A_L3L1_6_cons_buff_1, ptr nonnull @B_L3L1_6_cons_buff_0, ptr nonnull @C_L1L3_6_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %8 = trunc i64 %5 to i32
  %9 = or disjoint i32 %8, 2
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %9, ptr nonnull @A_L3L1_6_cons_buff_0, ptr nonnull @B_L3L1_6_cons_buff_0, ptr nonnull @C_L1L3_6_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %10 = or disjoint i32 %8, 3
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %10, ptr nonnull @A_L3L1_6_cons_buff_1, ptr nonnull @B_L3L1_6_cons_buff_0, ptr nonnull @C_L1L3_6_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %11 = trunc i64 %5 to i32
  %12 = or disjoint i32 %11, 4
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %12, ptr nonnull @A_L3L1_6_cons_buff_0, ptr nonnull @B_L3L1_6_cons_buff_0, ptr nonnull @C_L1L3_6_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %13 = or disjoint i32 %11, 5
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %13, ptr nonnull @A_L3L1_6_cons_buff_1, ptr nonnull @B_L3L1_6_cons_buff_0, ptr nonnull @C_L1L3_6_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %14 = or disjoint i64 %5, 6
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %15 = trunc nuw i64 %14 to i32
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %15, ptr nonnull @A_L3L1_6_cons_buff_0, ptr nonnull @B_L3L1_6_cons_buff_0, ptr nonnull @C_L1L3_6_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %16 = or disjoint i32 %15, 1
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %16, ptr nonnull @A_L3L1_6_cons_buff_1, ptr nonnull @B_L3L1_6_cons_buff_0, ptr nonnull @C_L1L3_6_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %17 = add nuw nsw i64 %5, 8
  %18 = icmp samesign ult i64 %14, 1022
  br i1 %18, label %4, label %19

19:                                               ; preds = %4
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %20 = add nuw nsw i64 %3, 1
  %21 = icmp samesign ult i64 %3, 4294967294
  br i1 %21, label %2, label %22

22:                                               ; preds = %19
  %23 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %23, 9223372036854775807
  br i1 %.not, label %24, label %.preheader

24:                                               ; preds = %22
  ret void
}

define void @core_1_3() local_unnamed_addr {
  br label %.preheader

.preheader:                                       ; preds = %0, %22
  %1 = phi i64 [ 0, %0 ], [ %23, %22 ]
  br label %2

2:                                                ; preds = %.preheader, %19
  %3 = phi i64 [ 0, %.preheader ], [ %20, %19 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %4

4:                                                ; preds = %4, %2
  %5 = phi i64 [ 0, %2 ], [ %17, %4 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %6 = trunc nuw i64 %5 to i32
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %6, ptr nonnull @A_L3L1_5_cons_buff_0, ptr nonnull @B_L3L1_5_cons_buff_0, ptr nonnull @C_L1L3_5_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %7 = or disjoint i32 %6, 1
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %7, ptr nonnull @A_L3L1_5_cons_buff_1, ptr nonnull @B_L3L1_5_cons_buff_0, ptr nonnull @C_L1L3_5_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %8 = trunc i64 %5 to i32
  %9 = or disjoint i32 %8, 2
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %9, ptr nonnull @A_L3L1_5_cons_buff_0, ptr nonnull @B_L3L1_5_cons_buff_0, ptr nonnull @C_L1L3_5_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %10 = or disjoint i32 %8, 3
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %10, ptr nonnull @A_L3L1_5_cons_buff_1, ptr nonnull @B_L3L1_5_cons_buff_0, ptr nonnull @C_L1L3_5_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %11 = trunc i64 %5 to i32
  %12 = or disjoint i32 %11, 4
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %12, ptr nonnull @A_L3L1_5_cons_buff_0, ptr nonnull @B_L3L1_5_cons_buff_0, ptr nonnull @C_L1L3_5_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %13 = or disjoint i32 %11, 5
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %13, ptr nonnull @A_L3L1_5_cons_buff_1, ptr nonnull @B_L3L1_5_cons_buff_0, ptr nonnull @C_L1L3_5_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %14 = or disjoint i64 %5, 6
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %15 = trunc nuw i64 %14 to i32
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %15, ptr nonnull @A_L3L1_5_cons_buff_0, ptr nonnull @B_L3L1_5_cons_buff_0, ptr nonnull @C_L1L3_5_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %16 = or disjoint i32 %15, 1
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %16, ptr nonnull @A_L3L1_5_cons_buff_1, ptr nonnull @B_L3L1_5_cons_buff_0, ptr nonnull @C_L1L3_5_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %17 = add nuw nsw i64 %5, 8
  %18 = icmp samesign ult i64 %14, 1022
  br i1 %18, label %4, label %19

19:                                               ; preds = %4
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %20 = add nuw nsw i64 %3, 1
  %21 = icmp samesign ult i64 %3, 4294967294
  br i1 %21, label %2, label %22

22:                                               ; preds = %19
  %23 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %23, 9223372036854775807
  br i1 %.not, label %24, label %.preheader

24:                                               ; preds = %22
  ret void
}

define void @core_1_2() local_unnamed_addr {
  br label %.preheader

.preheader:                                       ; preds = %0, %22
  %1 = phi i64 [ 0, %0 ], [ %23, %22 ]
  br label %2

2:                                                ; preds = %.preheader, %19
  %3 = phi i64 [ 0, %.preheader ], [ %20, %19 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %4

4:                                                ; preds = %4, %2
  %5 = phi i64 [ 0, %2 ], [ %17, %4 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %6 = trunc nuw i64 %5 to i32
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %6, ptr nonnull @A_L3L1_4_cons_buff_0, ptr nonnull @B_L3L1_4_cons_buff_0, ptr nonnull @C_L1L3_4_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %7 = or disjoint i32 %6, 1
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %7, ptr nonnull @A_L3L1_4_cons_buff_1, ptr nonnull @B_L3L1_4_cons_buff_0, ptr nonnull @C_L1L3_4_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %8 = trunc i64 %5 to i32
  %9 = or disjoint i32 %8, 2
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %9, ptr nonnull @A_L3L1_4_cons_buff_0, ptr nonnull @B_L3L1_4_cons_buff_0, ptr nonnull @C_L1L3_4_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %10 = or disjoint i32 %8, 3
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %10, ptr nonnull @A_L3L1_4_cons_buff_1, ptr nonnull @B_L3L1_4_cons_buff_0, ptr nonnull @C_L1L3_4_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %11 = trunc i64 %5 to i32
  %12 = or disjoint i32 %11, 4
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %12, ptr nonnull @A_L3L1_4_cons_buff_0, ptr nonnull @B_L3L1_4_cons_buff_0, ptr nonnull @C_L1L3_4_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %13 = or disjoint i32 %11, 5
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %13, ptr nonnull @A_L3L1_4_cons_buff_1, ptr nonnull @B_L3L1_4_cons_buff_0, ptr nonnull @C_L1L3_4_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %14 = or disjoint i64 %5, 6
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %15 = trunc nuw i64 %14 to i32
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %15, ptr nonnull @A_L3L1_4_cons_buff_0, ptr nonnull @B_L3L1_4_cons_buff_0, ptr nonnull @C_L1L3_4_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %16 = or disjoint i32 %15, 1
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %16, ptr nonnull @A_L3L1_4_cons_buff_1, ptr nonnull @B_L3L1_4_cons_buff_0, ptr nonnull @C_L1L3_4_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %17 = add nuw nsw i64 %5, 8
  %18 = icmp samesign ult i64 %14, 1022
  br i1 %18, label %4, label %19

19:                                               ; preds = %4
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %20 = add nuw nsw i64 %3, 1
  %21 = icmp samesign ult i64 %3, 4294967294
  br i1 %21, label %2, label %22

22:                                               ; preds = %19
  %23 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %23, 9223372036854775807
  br i1 %.not, label %24, label %.preheader

24:                                               ; preds = %22
  ret void
}

define void @core_0_5() local_unnamed_addr {
  br label %.preheader

.preheader:                                       ; preds = %0, %22
  %1 = phi i64 [ 0, %0 ], [ %23, %22 ]
  br label %2

2:                                                ; preds = %.preheader, %19
  %3 = phi i64 [ 0, %.preheader ], [ %20, %19 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %4

4:                                                ; preds = %4, %2
  %5 = phi i64 [ 0, %2 ], [ %17, %4 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %6 = trunc nuw i64 %5 to i32
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %6, ptr nonnull @A_L3L1_3_cons_buff_0, ptr nonnull @B_L3L1_3_cons_buff_0, ptr nonnull @C_L1L3_3_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %7 = or disjoint i32 %6, 1
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %7, ptr nonnull @A_L3L1_3_cons_buff_1, ptr nonnull @B_L3L1_3_cons_buff_0, ptr nonnull @C_L1L3_3_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %8 = trunc i64 %5 to i32
  %9 = or disjoint i32 %8, 2
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %9, ptr nonnull @A_L3L1_3_cons_buff_0, ptr nonnull @B_L3L1_3_cons_buff_0, ptr nonnull @C_L1L3_3_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %10 = or disjoint i32 %8, 3
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %10, ptr nonnull @A_L3L1_3_cons_buff_1, ptr nonnull @B_L3L1_3_cons_buff_0, ptr nonnull @C_L1L3_3_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %11 = trunc i64 %5 to i32
  %12 = or disjoint i32 %11, 4
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %12, ptr nonnull @A_L3L1_3_cons_buff_0, ptr nonnull @B_L3L1_3_cons_buff_0, ptr nonnull @C_L1L3_3_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %13 = or disjoint i32 %11, 5
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %13, ptr nonnull @A_L3L1_3_cons_buff_1, ptr nonnull @B_L3L1_3_cons_buff_0, ptr nonnull @C_L1L3_3_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %14 = or disjoint i64 %5, 6
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %15 = trunc nuw i64 %14 to i32
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %15, ptr nonnull @A_L3L1_3_cons_buff_0, ptr nonnull @B_L3L1_3_cons_buff_0, ptr nonnull @C_L1L3_3_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %16 = or disjoint i32 %15, 1
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %16, ptr nonnull @A_L3L1_3_cons_buff_1, ptr nonnull @B_L3L1_3_cons_buff_0, ptr nonnull @C_L1L3_3_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %17 = add nuw nsw i64 %5, 8
  %18 = icmp samesign ult i64 %14, 1022
  br i1 %18, label %4, label %19

19:                                               ; preds = %4
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %20 = add nuw nsw i64 %3, 1
  %21 = icmp samesign ult i64 %3, 4294967294
  br i1 %21, label %2, label %22

22:                                               ; preds = %19
  %23 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %23, 9223372036854775807
  br i1 %.not, label %24, label %.preheader

24:                                               ; preds = %22
  ret void
}

define void @core_0_4() local_unnamed_addr {
  br label %.preheader

.preheader:                                       ; preds = %0, %22
  %1 = phi i64 [ 0, %0 ], [ %23, %22 ]
  br label %2

2:                                                ; preds = %.preheader, %19
  %3 = phi i64 [ 0, %.preheader ], [ %20, %19 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %4

4:                                                ; preds = %4, %2
  %5 = phi i64 [ 0, %2 ], [ %17, %4 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %6 = trunc nuw i64 %5 to i32
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %6, ptr nonnull @A_L3L1_2_cons_buff_0, ptr nonnull @B_L3L1_2_cons_buff_0, ptr nonnull @C_L1L3_2_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %7 = or disjoint i32 %6, 1
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %7, ptr nonnull @A_L3L1_2_cons_buff_1, ptr nonnull @B_L3L1_2_cons_buff_0, ptr nonnull @C_L1L3_2_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %8 = trunc i64 %5 to i32
  %9 = or disjoint i32 %8, 2
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %9, ptr nonnull @A_L3L1_2_cons_buff_0, ptr nonnull @B_L3L1_2_cons_buff_0, ptr nonnull @C_L1L3_2_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %10 = or disjoint i32 %8, 3
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %10, ptr nonnull @A_L3L1_2_cons_buff_1, ptr nonnull @B_L3L1_2_cons_buff_0, ptr nonnull @C_L1L3_2_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %11 = trunc i64 %5 to i32
  %12 = or disjoint i32 %11, 4
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %12, ptr nonnull @A_L3L1_2_cons_buff_0, ptr nonnull @B_L3L1_2_cons_buff_0, ptr nonnull @C_L1L3_2_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %13 = or disjoint i32 %11, 5
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %13, ptr nonnull @A_L3L1_2_cons_buff_1, ptr nonnull @B_L3L1_2_cons_buff_0, ptr nonnull @C_L1L3_2_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %14 = or disjoint i64 %5, 6
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %15 = trunc nuw i64 %14 to i32
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %15, ptr nonnull @A_L3L1_2_cons_buff_0, ptr nonnull @B_L3L1_2_cons_buff_0, ptr nonnull @C_L1L3_2_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %16 = or disjoint i32 %15, 1
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %16, ptr nonnull @A_L3L1_2_cons_buff_1, ptr nonnull @B_L3L1_2_cons_buff_0, ptr nonnull @C_L1L3_2_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %17 = add nuw nsw i64 %5, 8
  %18 = icmp samesign ult i64 %14, 1022
  br i1 %18, label %4, label %19

19:                                               ; preds = %4
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %20 = add nuw nsw i64 %3, 1
  %21 = icmp samesign ult i64 %3, 4294967294
  br i1 %21, label %2, label %22

22:                                               ; preds = %19
  %23 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %23, 9223372036854775807
  br i1 %.not, label %24, label %.preheader

24:                                               ; preds = %22
  ret void
}

define void @core_0_3() local_unnamed_addr {
  br label %.preheader

.preheader:                                       ; preds = %0, %22
  %1 = phi i64 [ 0, %0 ], [ %23, %22 ]
  br label %2

2:                                                ; preds = %.preheader, %19
  %3 = phi i64 [ 0, %.preheader ], [ %20, %19 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %4

4:                                                ; preds = %4, %2
  %5 = phi i64 [ 0, %2 ], [ %17, %4 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %6 = trunc nuw i64 %5 to i32
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %6, ptr nonnull @A_L3L1_1_cons_buff_0, ptr nonnull @B_L3L1_1_cons_buff_0, ptr nonnull @C_L1L3_1_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %7 = or disjoint i32 %6, 1
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %7, ptr nonnull @A_L3L1_1_cons_buff_1, ptr nonnull @B_L3L1_1_cons_buff_0, ptr nonnull @C_L1L3_1_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %8 = trunc i64 %5 to i32
  %9 = or disjoint i32 %8, 2
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %9, ptr nonnull @A_L3L1_1_cons_buff_0, ptr nonnull @B_L3L1_1_cons_buff_0, ptr nonnull @C_L1L3_1_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %10 = or disjoint i32 %8, 3
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %10, ptr nonnull @A_L3L1_1_cons_buff_1, ptr nonnull @B_L3L1_1_cons_buff_0, ptr nonnull @C_L1L3_1_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %11 = trunc i64 %5 to i32
  %12 = or disjoint i32 %11, 4
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %12, ptr nonnull @A_L3L1_1_cons_buff_0, ptr nonnull @B_L3L1_1_cons_buff_0, ptr nonnull @C_L1L3_1_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %13 = or disjoint i32 %11, 5
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %13, ptr nonnull @A_L3L1_1_cons_buff_1, ptr nonnull @B_L3L1_1_cons_buff_0, ptr nonnull @C_L1L3_1_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %14 = or disjoint i64 %5, 6
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %15 = trunc nuw i64 %14 to i32
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %15, ptr nonnull @A_L3L1_1_cons_buff_0, ptr nonnull @B_L3L1_1_cons_buff_0, ptr nonnull @C_L1L3_1_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %16 = or disjoint i32 %15, 1
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %16, ptr nonnull @A_L3L1_1_cons_buff_1, ptr nonnull @B_L3L1_1_cons_buff_0, ptr nonnull @C_L1L3_1_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %17 = add nuw nsw i64 %5, 8
  %18 = icmp samesign ult i64 %14, 1022
  br i1 %18, label %4, label %19

19:                                               ; preds = %4
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %20 = add nuw nsw i64 %3, 1
  %21 = icmp samesign ult i64 %3, 4294967294
  br i1 %21, label %2, label %22

22:                                               ; preds = %19
  %23 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %23, 9223372036854775807
  br i1 %.not, label %24, label %.preheader

24:                                               ; preds = %22
  ret void
}

define void @core_0_2() local_unnamed_addr {
  br label %.preheader

.preheader:                                       ; preds = %0, %22
  %1 = phi i64 [ 0, %0 ], [ %23, %22 ]
  br label %2

2:                                                ; preds = %.preheader, %19
  %3 = phi i64 [ 0, %.preheader ], [ %20, %19 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %4

4:                                                ; preds = %4, %2
  %5 = phi i64 [ 0, %2 ], [ %17, %4 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %6 = trunc nuw i64 %5 to i32
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %6, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %7 = or disjoint i32 %6, 1
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %7, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %8 = trunc i64 %5 to i32
  %9 = or disjoint i32 %8, 2
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %9, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %10 = or disjoint i32 %8, 3
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %10, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %11 = trunc i64 %5 to i32
  %12 = or disjoint i32 %11, 4
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %12, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %13 = or disjoint i32 %11, 5
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %13, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %14 = or disjoint i64 %5, 6
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %15 = trunc nuw i64 %14 to i32
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %15, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %16 = or disjoint i32 %15, 1
  tail call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %16, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %17 = add nuw nsw i64 %5, 8
  %18 = icmp samesign ult i64 %14, 1022
  br i1 %18, label %4, label %19

19:                                               ; preds = %4
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %20 = add nuw nsw i64 %3, 1
  %21 = icmp samesign ult i64 %3, 4294967294
  br i1 %21, label %2, label %22

22:                                               ; preds = %19
  %23 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %23, 9223372036854775807
  br i1 %.not, label %24, label %.preheader

24:                                               ; preds = %22
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
