; ModuleID = '/scratch/roesti/mlir-aie/test/npu-xrt/reconfigure_loadpdi/aie.mlir.prj/add_three_input.llpeanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@objfifo_in_cons_buff_0 = external local_unnamed_addr global [4 x i32]
@objfifo_out_buff_0 = external local_unnamed_addr global [4 x i32]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

; Function Attrs: nofree norecurse nosync nounwind
define void @core_0_2() local_unnamed_addr #1 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %20, %1 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %3 = load i32, ptr @objfifo_in_cons_buff_0, align 4
  %4 = add i32 %3, 3
  store i32 %4, ptr @objfifo_out_buff_0, align 4
  %5 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objfifo_in_cons_buff_0, i20 4), align 4
  %6 = add i32 %5, 3
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @objfifo_out_buff_0, i20 4), align 4
  %7 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objfifo_in_cons_buff_0, i20 8), align 4
  %8 = add i32 %7, 3
  store i32 %8, ptr getelementptr inbounds nuw (i8, ptr @objfifo_out_buff_0, i20 8), align 4
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objfifo_in_cons_buff_0, i20 12), align 4
  %10 = add i32 %9, 3
  store i32 %10, ptr getelementptr inbounds nuw (i8, ptr @objfifo_out_buff_0, i20 12), align 4
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %11 = or disjoint i64 %2, 1
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %12 = load i32, ptr @objfifo_in_cons_buff_0, align 4
  %13 = add i32 %12, 3
  store i32 %13, ptr @objfifo_out_buff_0, align 4
  %14 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objfifo_in_cons_buff_0, i20 4), align 4
  %15 = add i32 %14, 3
  store i32 %15, ptr getelementptr inbounds nuw (i8, ptr @objfifo_out_buff_0, i20 4), align 4
  %16 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objfifo_in_cons_buff_0, i20 8), align 4
  %17 = add i32 %16, 3
  store i32 %17, ptr getelementptr inbounds nuw (i8, ptr @objfifo_out_buff_0, i20 8), align 4
  %18 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objfifo_in_cons_buff_0, i20 12), align 4
  %19 = add i32 %18, 3
  store i32 %19, ptr getelementptr inbounds nuw (i8, ptr @objfifo_out_buff_0, i20 12), align 4
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %20 = add nuw nsw i64 %2, 2
  %21 = icmp samesign ult i64 %11, 16777213
  br i1 %21, label %1, label %22

22:                                               ; preds = %1
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #1 = { nofree norecurse nosync nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
