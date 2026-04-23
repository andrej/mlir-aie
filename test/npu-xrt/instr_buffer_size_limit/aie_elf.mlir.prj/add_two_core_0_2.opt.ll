; ModuleID = 'aie_elf.mlir.prj/add_two_core_0_2.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@objfifo_in_cons_buff_0 = external local_unnamed_addr global [128 x i32]
@objfifo_out_buff_0 = external local_unnamed_addr global [128 x i32]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

; Function Attrs: nofree norecurse nosync nounwind
define void @core_0_2() local_unnamed_addr #1 {
  br label %1

1:                                                ; preds = %0, %30
  %2 = phi i64 [ 0, %0 ], [ %31, %30 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  br label %3

3:                                                ; preds = %3, %1
  %4 = phi i64 [ 0, %1 ], [ %28, %3 ]
  %5 = trunc nuw i64 %4 to i20
  %6 = getelementptr inbounds nuw i32, ptr @objfifo_in_cons_buff_0, i20 %5
  %7 = load i32, ptr %6, align 4
  %8 = add i32 %7, 2
  %9 = getelementptr inbounds nuw i32, ptr @objfifo_out_buff_0, i20 %5
  store i32 %8, ptr %9, align 4
  %10 = trunc i64 %4 to i20
  %11 = or disjoint i20 %10, 1
  %12 = getelementptr inbounds nuw i32, ptr @objfifo_in_cons_buff_0, i20 %11
  %13 = load i32, ptr %12, align 4
  %14 = add i32 %13, 2
  %15 = getelementptr inbounds nuw i32, ptr @objfifo_out_buff_0, i20 %11
  store i32 %14, ptr %15, align 4
  %16 = trunc i64 %4 to i20
  %17 = or disjoint i20 %16, 2
  %18 = getelementptr inbounds nuw i32, ptr @objfifo_in_cons_buff_0, i20 %17
  %19 = load i32, ptr %18, align 4
  %20 = add i32 %19, 2
  %21 = getelementptr inbounds nuw i32, ptr @objfifo_out_buff_0, i20 %17
  store i32 %20, ptr %21, align 4
  %22 = or disjoint i64 %4, 3
  %23 = trunc nuw i64 %22 to i20
  %24 = getelementptr inbounds nuw i32, ptr @objfifo_in_cons_buff_0, i20 %23
  %25 = load i32, ptr %24, align 4
  %26 = add i32 %25, 2
  %27 = getelementptr inbounds nuw i32, ptr @objfifo_out_buff_0, i20 %23
  store i32 %26, ptr %27, align 4
  %28 = add nuw nsw i64 %4, 4
  %29 = icmp samesign ult i64 %22, 127
  br i1 %29, label %3, label %30

30:                                               ; preds = %3
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %31 = add nuw nsw i64 %2, 1
  %32 = icmp samesign ult i64 %2, 16777213
  br i1 %32, label %1, label %33

33:                                               ; preds = %30
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #1 = { nofree norecurse nosync nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
