; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
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

declare void @debug_i32(i32)

; Unknown intrinsic
declare void @llvm.aie2p.event(i32)

; Unknown intrinsic
declare void @llvm.aie2p.put.ms(i32, i32)

; Unknown intrinsic
declare { i32, i32 } @llvm.aie2p.get.ss()

; Unknown intrinsic
declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32)

; Unknown intrinsic
declare <16 x i32> @llvm.aie2p.scd.read.vec(i32)

; Unknown intrinsic
declare void @llvm.aie2p.acquire(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.release(i32, i32)

declare void @matvec_vectorized_bf16_bf16(i32, i32, i32, ptr, ptr, ptr)

define void @core_1_5() {
  br label %1

1:                                                ; preds = %18, %0
  %2 = phi i64 [ %19, %18 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %20

4:                                                ; preds = %16, %1
  %5 = phi i64 [ %17, %16 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4294967295
  br i1 %6, label %7, label %18

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ %15, %11 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 1024
  br i1 %10, label %11, label %16

11:                                               ; preds = %8
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %12 = trunc i64 %9 to i32
  call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %12, ptr @A_L3L1_7_cons_buff_0, ptr @B_L3L1_7_cons_buff_0, ptr @C_L1L3_7_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %13 = add i64 %9, 1
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = trunc i64 %13 to i32
  call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %14, ptr @A_L3L1_7_cons_buff_1, ptr @B_L3L1_7_cons_buff_0, ptr @C_L1L3_7_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %15 = add i64 %9, 2
  br label %8

16:                                               ; preds = %8
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %17 = add i64 %5, 1
  br label %4

18:                                               ; preds = %4
  %19 = add i64 %2, 1
  br label %1

20:                                               ; preds = %1
  ret void
}

define void @core_1_4() {
  br label %1

1:                                                ; preds = %18, %0
  %2 = phi i64 [ %19, %18 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %20

4:                                                ; preds = %16, %1
  %5 = phi i64 [ %17, %16 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4294967295
  br i1 %6, label %7, label %18

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ %15, %11 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 1024
  br i1 %10, label %11, label %16

11:                                               ; preds = %8
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %12 = trunc i64 %9 to i32
  call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %12, ptr @A_L3L1_6_cons_buff_0, ptr @B_L3L1_6_cons_buff_0, ptr @C_L1L3_6_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %13 = add i64 %9, 1
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = trunc i64 %13 to i32
  call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %14, ptr @A_L3L1_6_cons_buff_1, ptr @B_L3L1_6_cons_buff_0, ptr @C_L1L3_6_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %15 = add i64 %9, 2
  br label %8

16:                                               ; preds = %8
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %17 = add i64 %5, 1
  br label %4

18:                                               ; preds = %4
  %19 = add i64 %2, 1
  br label %1

20:                                               ; preds = %1
  ret void
}

define void @core_1_3() {
  br label %1

1:                                                ; preds = %18, %0
  %2 = phi i64 [ %19, %18 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %20

4:                                                ; preds = %16, %1
  %5 = phi i64 [ %17, %16 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4294967295
  br i1 %6, label %7, label %18

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ %15, %11 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 1024
  br i1 %10, label %11, label %16

11:                                               ; preds = %8
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %12 = trunc i64 %9 to i32
  call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %12, ptr @A_L3L1_5_cons_buff_0, ptr @B_L3L1_5_cons_buff_0, ptr @C_L1L3_5_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %13 = add i64 %9, 1
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = trunc i64 %13 to i32
  call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %14, ptr @A_L3L1_5_cons_buff_1, ptr @B_L3L1_5_cons_buff_0, ptr @C_L1L3_5_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %15 = add i64 %9, 2
  br label %8

16:                                               ; preds = %8
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %17 = add i64 %5, 1
  br label %4

18:                                               ; preds = %4
  %19 = add i64 %2, 1
  br label %1

20:                                               ; preds = %1
  ret void
}

define void @core_1_2() {
  br label %1

1:                                                ; preds = %18, %0
  %2 = phi i64 [ %19, %18 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %20

4:                                                ; preds = %16, %1
  %5 = phi i64 [ %17, %16 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4294967295
  br i1 %6, label %7, label %18

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ %15, %11 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 1024
  br i1 %10, label %11, label %16

11:                                               ; preds = %8
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %12 = trunc i64 %9 to i32
  call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %12, ptr @A_L3L1_4_cons_buff_0, ptr @B_L3L1_4_cons_buff_0, ptr @C_L1L3_4_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %13 = add i64 %9, 1
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = trunc i64 %13 to i32
  call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %14, ptr @A_L3L1_4_cons_buff_1, ptr @B_L3L1_4_cons_buff_0, ptr @C_L1L3_4_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %15 = add i64 %9, 2
  br label %8

16:                                               ; preds = %8
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %17 = add i64 %5, 1
  br label %4

18:                                               ; preds = %4
  %19 = add i64 %2, 1
  br label %1

20:                                               ; preds = %1
  ret void
}

define void @core_0_5() {
  br label %1

1:                                                ; preds = %18, %0
  %2 = phi i64 [ %19, %18 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %20

4:                                                ; preds = %16, %1
  %5 = phi i64 [ %17, %16 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4294967295
  br i1 %6, label %7, label %18

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ %15, %11 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 1024
  br i1 %10, label %11, label %16

11:                                               ; preds = %8
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %12 = trunc i64 %9 to i32
  call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %12, ptr @A_L3L1_3_cons_buff_0, ptr @B_L3L1_3_cons_buff_0, ptr @C_L1L3_3_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %13 = add i64 %9, 1
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = trunc i64 %13 to i32
  call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %14, ptr @A_L3L1_3_cons_buff_1, ptr @B_L3L1_3_cons_buff_0, ptr @C_L1L3_3_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %15 = add i64 %9, 2
  br label %8

16:                                               ; preds = %8
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %17 = add i64 %5, 1
  br label %4

18:                                               ; preds = %4
  %19 = add i64 %2, 1
  br label %1

20:                                               ; preds = %1
  ret void
}

define void @core_0_4() {
  br label %1

1:                                                ; preds = %18, %0
  %2 = phi i64 [ %19, %18 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %20

4:                                                ; preds = %16, %1
  %5 = phi i64 [ %17, %16 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4294967295
  br i1 %6, label %7, label %18

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ %15, %11 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 1024
  br i1 %10, label %11, label %16

11:                                               ; preds = %8
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %12 = trunc i64 %9 to i32
  call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %12, ptr @A_L3L1_2_cons_buff_0, ptr @B_L3L1_2_cons_buff_0, ptr @C_L1L3_2_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %13 = add i64 %9, 1
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = trunc i64 %13 to i32
  call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %14, ptr @A_L3L1_2_cons_buff_1, ptr @B_L3L1_2_cons_buff_0, ptr @C_L1L3_2_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %15 = add i64 %9, 2
  br label %8

16:                                               ; preds = %8
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %17 = add i64 %5, 1
  br label %4

18:                                               ; preds = %4
  %19 = add i64 %2, 1
  br label %1

20:                                               ; preds = %1
  ret void
}

define void @core_0_3() {
  br label %1

1:                                                ; preds = %18, %0
  %2 = phi i64 [ %19, %18 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %20

4:                                                ; preds = %16, %1
  %5 = phi i64 [ %17, %16 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4294967295
  br i1 %6, label %7, label %18

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ %15, %11 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 1024
  br i1 %10, label %11, label %16

11:                                               ; preds = %8
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %12 = trunc i64 %9 to i32
  call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %12, ptr @A_L3L1_1_cons_buff_0, ptr @B_L3L1_1_cons_buff_0, ptr @C_L1L3_1_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %13 = add i64 %9, 1
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = trunc i64 %13 to i32
  call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %14, ptr @A_L3L1_1_cons_buff_1, ptr @B_L3L1_1_cons_buff_0, ptr @C_L1L3_1_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %15 = add i64 %9, 2
  br label %8

16:                                               ; preds = %8
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %17 = add i64 %5, 1
  br label %4

18:                                               ; preds = %4
  %19 = add i64 %2, 1
  br label %1

20:                                               ; preds = %1
  ret void
}

define void @core_0_2() {
  br label %1

1:                                                ; preds = %18, %0
  %2 = phi i64 [ %19, %18 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %20

4:                                                ; preds = %16, %1
  %5 = phi i64 [ %17, %16 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4294967295
  br i1 %6, label %7, label %18

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ %15, %11 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 1024
  br i1 %10, label %11, label %16

11:                                               ; preds = %8
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %12 = trunc i64 %9 to i32
  call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %12, ptr @A_L3L1_0_cons_buff_0, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %13 = add i64 %9, 1
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = trunc i64 %13 to i32
  call void @matvec_vectorized_bf16_bf16(i32 1, i32 2048, i32 %14, ptr @A_L3L1_0_cons_buff_1, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %15 = add i64 %9, 2
  br label %8

16:                                               ; preds = %8
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %17 = add i64 %5, 1
  br label %4

18:                                               ; preds = %4
  %19 = add i64 %2, 1
  br label %1

20:                                               ; preds = %1
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
