; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@objFifo_out0_buff_1 = external global [64 x [64 x i8]]
@objFifo_out0_buff_0 = external global [64 x [64 x i8]]
@objFifo_in0_cons_buff_1 = external global [64 x [64 x i8]]
@objFifo_in0_cons_buff_0 = external global [64 x [64 x i8]]
@objFifo_out1_buff_1 = external global [64 x [64 x i8]]
@objFifo_out1_buff_0 = external global [64 x [64 x i8]]
@objFifo_in1_cons_buff_1 = external global [64 x [64 x i8]]
@objFifo_in1_cons_buff_0 = external global [64 x [64 x i8]]

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

define void @core_0_2() {
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  br label %1

1:                                                ; preds = %15, %0
  %2 = phi i64 [ %16, %15 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 64
  br i1 %3, label %4, label %17

4:                                                ; preds = %7, %1
  %5 = phi i64 [ %14, %7 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 64
  br i1 %6, label %7, label %15

7:                                                ; preds = %4
  %8 = mul nuw nsw i64 %2, 64
  %9 = add nuw nsw i64 %8, %5
  %10 = getelementptr inbounds i8, ptr @objFifo_in1_cons_buff_0, i64 %9
  %11 = load i8, ptr %10, align 1
  %12 = add i8 %11, 12
  %13 = getelementptr inbounds i8, ptr @objFifo_out1_buff_0, i64 %9
  store i8 %12, ptr %13, align 1
  %14 = add i64 %5, 1
  br label %4

15:                                               ; preds = %4
  %16 = add i64 %2, 1
  br label %1

17:                                               ; preds = %1
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 51, i32 1)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
