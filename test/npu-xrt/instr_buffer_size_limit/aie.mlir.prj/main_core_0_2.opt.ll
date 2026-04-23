; ModuleID = 'aie.mlir.prj/main_core_0_2.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@objFifo_in1_cons_buff_1 = external local_unnamed_addr global [8 x i32]
@objFifo_in1_cons_buff_0 = external local_unnamed_addr global [8 x i32]
@objFifo_out1_buff_1 = external local_unnamed_addr global [8 x i32]
@objFifo_out1_buff_0 = external local_unnamed_addr global [8 x i32]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

; Function Attrs: nofree norecurse nosync nounwind
define void @core_0_2() local_unnamed_addr #1 {
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %1 = load i32, ptr @objFifo_in1_cons_buff_0, align 4
  %2 = add i32 %1, 2
  store i32 %2, ptr @objFifo_out1_buff_0, align 4
  %3 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 4), align 4
  %4 = add i32 %3, 2
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 4), align 4
  %5 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 8), align 4
  %6 = add i32 %5, 2
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 8), align 4
  %7 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 12), align 4
  %8 = add i32 %7, 2
  store i32 %8, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 12), align 4
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 16), align 4
  %10 = add i32 %9, 2
  store i32 %10, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 16), align 4
  %11 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 20), align 4
  %12 = add i32 %11, 2
  store i32 %12, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 20), align 4
  %13 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 24), align 4
  %14 = add i32 %13, 2
  store i32 %14, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 24), align 4
  %15 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 28), align 4
  %16 = add i32 %15, 2
  store i32 %16, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 28), align 4
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %17 = load i32, ptr @objFifo_in1_cons_buff_1, align 4
  %18 = add i32 %17, 2
  store i32 %18, ptr @objFifo_out1_buff_1, align 4
  %19 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 4), align 4
  %20 = add i32 %19, 2
  store i32 %20, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 4), align 4
  %21 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 8), align 4
  %22 = add i32 %21, 2
  store i32 %22, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 8), align 4
  %23 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 12), align 4
  %24 = add i32 %23, 2
  store i32 %24, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 12), align 4
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 16), align 4
  %26 = add i32 %25, 2
  store i32 %26, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 16), align 4
  %27 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 20), align 4
  %28 = add i32 %27, 2
  store i32 %28, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 20), align 4
  %29 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 24), align 4
  %30 = add i32 %29, 2
  store i32 %30, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 24), align 4
  %31 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 28), align 4
  %32 = add i32 %31, 2
  store i32 %32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 28), align 4
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %33 = load i32, ptr @objFifo_in1_cons_buff_0, align 4
  %34 = add i32 %33, 2
  store i32 %34, ptr @objFifo_out1_buff_0, align 4
  %35 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 4), align 4
  %36 = add i32 %35, 2
  store i32 %36, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 4), align 4
  %37 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 8), align 4
  %38 = add i32 %37, 2
  store i32 %38, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 8), align 4
  %39 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 12), align 4
  %40 = add i32 %39, 2
  store i32 %40, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 12), align 4
  %41 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 16), align 4
  %42 = add i32 %41, 2
  store i32 %42, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 16), align 4
  %43 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 20), align 4
  %44 = add i32 %43, 2
  store i32 %44, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 20), align 4
  %45 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 24), align 4
  %46 = add i32 %45, 2
  store i32 %46, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 24), align 4
  %47 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 28), align 4
  %48 = add i32 %47, 2
  store i32 %48, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 28), align 4
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %49 = load i32, ptr @objFifo_in1_cons_buff_1, align 4
  %50 = add i32 %49, 2
  store i32 %50, ptr @objFifo_out1_buff_1, align 4
  %51 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 4), align 4
  %52 = add i32 %51, 2
  store i32 %52, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 4), align 4
  %53 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 8), align 4
  %54 = add i32 %53, 2
  store i32 %54, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 8), align 4
  %55 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 12), align 4
  %56 = add i32 %55, 2
  store i32 %56, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 12), align 4
  %57 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 16), align 4
  %58 = add i32 %57, 2
  store i32 %58, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 16), align 4
  %59 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 20), align 4
  %60 = add i32 %59, 2
  store i32 %60, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 20), align 4
  %61 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 24), align 4
  %62 = add i32 %61, 2
  store i32 %62, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 24), align 4
  %63 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 28), align 4
  %64 = add i32 %63, 2
  store i32 %64, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 28), align 4
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %65 = load i32, ptr @objFifo_in1_cons_buff_0, align 4
  %66 = add i32 %65, 2
  store i32 %66, ptr @objFifo_out1_buff_0, align 4
  %67 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 4), align 4
  %68 = add i32 %67, 2
  store i32 %68, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 4), align 4
  %69 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 8), align 4
  %70 = add i32 %69, 2
  store i32 %70, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 8), align 4
  %71 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 12), align 4
  %72 = add i32 %71, 2
  store i32 %72, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 12), align 4
  %73 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 16), align 4
  %74 = add i32 %73, 2
  store i32 %74, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 16), align 4
  %75 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 20), align 4
  %76 = add i32 %75, 2
  store i32 %76, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 20), align 4
  %77 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 24), align 4
  %78 = add i32 %77, 2
  store i32 %78, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 24), align 4
  %79 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 28), align 4
  %80 = add i32 %79, 2
  store i32 %80, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 28), align 4
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %81 = load i32, ptr @objFifo_in1_cons_buff_1, align 4
  %82 = add i32 %81, 2
  store i32 %82, ptr @objFifo_out1_buff_1, align 4
  %83 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 4), align 4
  %84 = add i32 %83, 2
  store i32 %84, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 4), align 4
  %85 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 8), align 4
  %86 = add i32 %85, 2
  store i32 %86, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 8), align 4
  %87 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 12), align 4
  %88 = add i32 %87, 2
  store i32 %88, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 12), align 4
  %89 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 16), align 4
  %90 = add i32 %89, 2
  store i32 %90, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 16), align 4
  %91 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 20), align 4
  %92 = add i32 %91, 2
  store i32 %92, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 20), align 4
  %93 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 24), align 4
  %94 = add i32 %93, 2
  store i32 %94, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 24), align 4
  %95 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 28), align 4
  %96 = add i32 %95, 2
  store i32 %96, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 28), align 4
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %97 = load i32, ptr @objFifo_in1_cons_buff_0, align 4
  %98 = add i32 %97, 2
  store i32 %98, ptr @objFifo_out1_buff_0, align 4
  %99 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 4), align 4
  %100 = add i32 %99, 2
  store i32 %100, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 4), align 4
  %101 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 8), align 4
  %102 = add i32 %101, 2
  store i32 %102, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 8), align 4
  %103 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 12), align 4
  %104 = add i32 %103, 2
  store i32 %104, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 12), align 4
  %105 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 16), align 4
  %106 = add i32 %105, 2
  store i32 %106, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 16), align 4
  %107 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 20), align 4
  %108 = add i32 %107, 2
  store i32 %108, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 20), align 4
  %109 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 24), align 4
  %110 = add i32 %109, 2
  store i32 %110, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 24), align 4
  %111 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_0, i20 28), align 4
  %112 = add i32 %111, 2
  store i32 %112, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_0, i20 28), align 4
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %113 = load i32, ptr @objFifo_in1_cons_buff_1, align 4
  %114 = add i32 %113, 2
  store i32 %114, ptr @objFifo_out1_buff_1, align 4
  %115 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 4), align 4
  %116 = add i32 %115, 2
  store i32 %116, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 4), align 4
  %117 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 8), align 4
  %118 = add i32 %117, 2
  store i32 %118, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 8), align 4
  %119 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 12), align 4
  %120 = add i32 %119, 2
  store i32 %120, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 12), align 4
  %121 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 16), align 4
  %122 = add i32 %121, 2
  store i32 %122, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 16), align 4
  %123 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 20), align 4
  %124 = add i32 %123, 2
  store i32 %124, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 20), align 4
  %125 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 24), align 4
  %126 = add i32 %125, 2
  store i32 %126, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 24), align 4
  %127 = load i32, ptr getelementptr inbounds nuw (i8, ptr @objFifo_in1_cons_buff_1, i20 28), align 4
  %128 = add i32 %127, 2
  store i32 %128, ptr getelementptr inbounds nuw (i8, ptr @objFifo_out1_buff_1, i20 28), align 4
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #1 = { nofree norecurse nosync nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
