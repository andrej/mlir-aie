; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@constant_buffer = external global [8 x i32]
@objFifo_out1_buff_1 = external global [8 x i32]
@objFifo_out1_buff_0 = external global [8 x i32]
@objFifo_in1_cons_buff_1 = external global [8 x i32]
@objFifo_in1_cons_buff_0 = external global [8 x i32]
@myData = private global [8 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8]

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
  br label %1

1:                                                ; preds = %28, %0
  %2 = phi i64 [ %29, %28 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 8
  br i1 %3, label %4, label %30

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %15, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 8
  br i1 %7, label %8, label %16

8:                                                ; preds = %5
  %9 = getelementptr inbounds nuw i32, ptr @objFifo_in1_cons_buff_0, i64 %6
  %10 = load i32, ptr %9, align 4
  %11 = getelementptr inbounds nuw i32, ptr @constant_buffer, i64 %6
  %12 = load i32, ptr %11, align 4
  %13 = add i32 %10, %12
  %14 = getelementptr inbounds nuw i32, ptr @objFifo_out1_buff_0, i64 %6
  store i32 %13, ptr %14, align 4
  %15 = add i64 %6, 1
  br label %5

16:                                               ; preds = %5
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 51, i32 1)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  br label %17

17:                                               ; preds = %20, %16
  %18 = phi i64 [ %27, %20 ], [ 0, %16 ]
  %19 = icmp slt i64 %18, 8
  br i1 %19, label %20, label %28

20:                                               ; preds = %17
  %21 = getelementptr inbounds nuw i32, ptr @objFifo_in1_cons_buff_1, i64 %18
  %22 = load i32, ptr %21, align 4
  %23 = getelementptr inbounds nuw i32, ptr @constant_buffer, i64 %18
  %24 = load i32, ptr %23, align 4
  %25 = add i32 %22, %24
  %26 = getelementptr inbounds nuw i32, ptr @objFifo_out1_buff_1, i64 %18
  store i32 %25, ptr %26, align 4
  %27 = add i64 %18, 1
  br label %17

28:                                               ; preds = %17
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %29 = add i64 %2, 2
  br label %1

30:                                               ; preds = %1
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
