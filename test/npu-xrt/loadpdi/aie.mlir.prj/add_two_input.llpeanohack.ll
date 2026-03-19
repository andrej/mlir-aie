; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@objfifo_in_cons_buff_0 = external global [128 x i32]
@objfifo_out_buff_0 = external global [128 x i32]

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

1:                                                ; preds = %14, %0
  %2 = phi i64 [ %15, %14 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 16777214
  br i1 %3, label %4, label %16

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %13, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 128
  br i1 %7, label %8, label %14

8:                                                ; preds = %5
  %9 = getelementptr inbounds i32, ptr @objfifo_in_cons_buff_0, i64 %6
  %10 = load i32, ptr %9
  %11 = add i32 %10, 2
  %12 = getelementptr inbounds i32, ptr @objfifo_out_buff_0, i64 %6
  store i32 %11, ptr %12
  %13 = add i64 %6, 1
  br label %5

14:                                               ; preds = %5
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %15 = add i64 %2, 1
  br label %1

16:                                               ; preds = %1
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
