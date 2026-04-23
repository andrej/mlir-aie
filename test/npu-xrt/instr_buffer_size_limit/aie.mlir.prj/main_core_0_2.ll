; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@objFifo_in0_cons_buff_1 = external global [16 x i32]
@objFifo_in0_cons_buff_0 = external global [16 x i32]
@objFifo_in1_cons_buff_1 = external global [8 x i32]
@objFifo_in1_cons_buff_0 = external global [8 x i32]
@objFifo_out1_buff_1 = external global [8 x i32]
@objFifo_out1_buff_0 = external global [8 x i32]
@objFifo_out0_buff_1 = external global [16 x i32]
@objFifo_out0_buff_0 = external global [16 x i32]

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

; Unknown intrinsic
declare void @llvm.aie2p.set.ctrl.reg(i32, i32)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %24, %0
  %2 = phi i64 [ %25, %24 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 8
  br i1 %3, label %4, label %26

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %13, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 8
  br i1 %7, label %8, label %14

8:                                                ; preds = %5
  %9 = getelementptr inbounds nuw i32, ptr @objFifo_in1_cons_buff_0, i64 %6
  %10 = load i32, ptr %9, align 4
  %11 = add i32 %10, 2
  %12 = getelementptr inbounds nuw i32, ptr @objFifo_out1_buff_0, i64 %6
  store i32 %11, ptr %12, align 4
  %13 = add i64 %6, 1
  br label %5

14:                                               ; preds = %5
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 51, i32 1)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  br label %15

15:                                               ; preds = %18, %14
  %16 = phi i64 [ %23, %18 ], [ 0, %14 ]
  %17 = icmp slt i64 %16, 8
  br i1 %17, label %18, label %24

18:                                               ; preds = %15
  %19 = getelementptr inbounds nuw i32, ptr @objFifo_in1_cons_buff_1, i64 %16
  %20 = load i32, ptr %19, align 4
  %21 = add i32 %20, 2
  %22 = getelementptr inbounds nuw i32, ptr @objFifo_out1_buff_1, i64 %16
  store i32 %21, ptr %22, align 4
  %23 = add i64 %16, 1
  br label %15

24:                                               ; preds = %15
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %25 = add i64 %2, 2
  br label %1

26:                                               ; preds = %1
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
