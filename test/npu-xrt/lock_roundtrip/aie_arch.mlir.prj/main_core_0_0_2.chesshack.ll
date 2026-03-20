; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@buf_out = external global [16 x i32]
@buf_in = external global [16 x i32]

declare void @debug_i32(i32)

; Unknown intrinsic
declare void @llvm.aie2.event(i32)

; Unknown intrinsic
declare void @llvm.aie2.put.ms(i32, i32)

; Unknown intrinsic
declare { i32, i32 } @llvm.aie2.get.ss()

; Unknown intrinsic
declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

; Unknown intrinsic
declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

; Unknown intrinsic
declare void @llvm.aie2.acquire(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2.release(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2.set.ctrl.reg(i32, i32)

define void @core_0_2() {
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %8, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 16
  br i1 %3, label %4, label %9

4:                                                ; preds = %1
  %5 = getelementptr inbounds i32, ptr @buf_in, i64 %2
  %6 = load i32, ptr %5, align 4
  %7 = getelementptr inbounds i32, ptr @buf_out, i64 %2
  store i32 %6, ptr %7, align 4
  %8 = add i64 %2, 1
  br label %1

9:                                                ; preds = %1
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
