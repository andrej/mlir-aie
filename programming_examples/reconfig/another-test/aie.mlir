module {

  aie.device(npu2) @empty{

  }

  aie.device(npu2) @main {
    aiex.runtime_sequence @rt(%inout : memref<1048576xbf16>) {
      %c1 = aiex.configure @mm
      aiex.run %c1 -> @sequence(%inout, %inout, %inout) : (memref<1048576xbf16>, memref<1048576xbf16>, memref<1048576xbf16>)

      // Reset module
      aiex.npu.write32 {row = 0 : i32, col = 0 : i32, address = 0x00060010 : ui32, value = 7 : ui32}
      aiex.npu.write32 {row = 1 : i32, col = 0 : i32, address = 0x00060010 : ui32, value = 7 : ui32}
      aiex.npu.write32 {row = 2 : i32, col = 0 : i32, address = 0x00060010 : ui32, value = 7 : ui32}

      // Reset S2MM_0
      aiex.npu.write32 {row = 0 : i32, col = 0 : i32, address = 0x0001DE00 : ui32, value = 1 : ui32}
      aiex.npu.write32 {row = 1 : i32, col = 0 : i32, address = 0x0001DE00 : ui32, value = 1 : ui32}
      aiex.npu.write32 {row = 2 : i32, col = 0 : i32, address = 0x0001DE00 : ui32, value = 1 : ui32}

      // Reset S2MM_1
      aiex.npu.write32 {row = 0 : i32, col = 0 : i32, address = 0x0001DE08 : ui32, value = 1 : ui32}
      aiex.npu.write32 {row = 1 : i32, col = 0 : i32, address = 0x0001DE08 : ui32, value = 1 : ui32}
      aiex.npu.write32 {row = 2 : i32, col = 0 : i32, address = 0x0001DE08 : ui32, value = 1 : ui32}

      // Reset MM2S_0
      aiex.npu.write32 {row = 0 : i32, col = 0 : i32, address = 0x0001DE10 : ui32, value = 1 : ui32}
      aiex.npu.write32 {row = 1 : i32, col = 0 : i32, address = 0x0001DE10 : ui32, value = 1 : ui32}
      aiex.npu.write32 {row = 2 : i32, col = 0 : i32, address = 0x0001DE10 : ui32, value = 1 : ui32}

      // Reset MM2S_1
      aiex.npu.write32 {row = 0 : i32, col = 0 : i32, address = 0x0001DE18 : ui32, value = 1 : ui32}
      aiex.npu.write32 {row = 1 : i32, col = 0 : i32, address = 0x0001DE18 : ui32, value = 1 : ui32}
      aiex.npu.write32 {row = 2 : i32, col = 0 : i32, address = 0x0001DE18 : ui32, value = 1 : ui32}

      aiex.npu.write32 { col = 0 : i32, row = 0 : i32, address = 0x0001D200 : ui32, value = 3 : ui32}
      aiex.npu.write32 { col = 0 : i32, row = 0 : i32, address = 0x00060010 : ui32, value = 0 : ui32 }
      aiex.npu.write32 { col = 0 : i32, row = 0 : i32, address = 0x00060010 : ui32, value = 1 : ui32 }
      aiex.npu.write32 { col = 0 : i32, row = 0 : i32, address = 0x0001D200 : ui32, value = 0 : ui32}

      aiex.npu.write32 { col = 0 : i32, row = 0 : i32, address = 0x0003F040 : ui32, value = 0 : ui32}

      //%c2 = aiex.configure @rms_norm
      //aiex.run %c2 -> @sequence(%inout, %inout) : (memref<1048576xbf16>, memref<1048576xbf16>)
    }
  }

  aie.device(npu2) @rms_norm {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    aie.objectfifo @in_4(%shim_noc_tile_2_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @out_6(%tile_1_4, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @in_7(%shim_noc_tile_1_0, {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @out_0(%tile_0_2, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @out_4(%tile_1_2, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @in_0(%shim_noc_tile_2_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @in_5(%shim_noc_tile_3_0, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @out_7(%tile_1_5, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @out_5(%tile_1_3, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @in_1(%shim_noc_tile_0_0, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @in_6(%shim_noc_tile_3_0, {%tile_1_4}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @out_1(%tile_0_3, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @in_2(%shim_noc_tile_0_0, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @out_3(%tile_0_5, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @out_2(%tile_0_4, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @in_3(%shim_noc_tile_1_0, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    func.func private @rms_norm(memref<1024xbf16>, memref<1024xbf16>, i32)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c128 step %c1_1 {
          %0 = aie.objectfifo.acquire @in_0(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %2 = aie.objectfifo.acquire @out_0(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @rms_norm(%1, %3, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
          aie.objectfifo.release @in_0(Consume, 1)
          aie.objectfifo.release @out_0(Produce, 1)
        }
      }
      aie.end
    } {link_with = "rms_norm.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c128 step %c1_1 {
          %0 = aie.objectfifo.acquire @in_1(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %2 = aie.objectfifo.acquire @out_1(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @rms_norm(%1, %3, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
          aie.objectfifo.release @in_1(Consume, 1)
          aie.objectfifo.release @out_1(Produce, 1)
        }
      }
      aie.end
    } {link_with = "rms_norm.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c128 step %c1_1 {
          %0 = aie.objectfifo.acquire @in_2(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %2 = aie.objectfifo.acquire @out_2(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @rms_norm(%1, %3, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
          aie.objectfifo.release @in_2(Consume, 1)
          aie.objectfifo.release @out_2(Produce, 1)
        }
      }
      aie.end
    } {link_with = "rms_norm.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c128 step %c1_1 {
          %0 = aie.objectfifo.acquire @in_3(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %2 = aie.objectfifo.acquire @out_3(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @rms_norm(%1, %3, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
          aie.objectfifo.release @in_3(Consume, 1)
          aie.objectfifo.release @out_3(Produce, 1)
        }
      }
      aie.end
    } {link_with = "rms_norm.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c128 step %c1_1 {
          %0 = aie.objectfifo.acquire @in_4(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %2 = aie.objectfifo.acquire @out_4(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @rms_norm(%1, %3, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
          aie.objectfifo.release @in_4(Consume, 1)
          aie.objectfifo.release @out_4(Produce, 1)
        }
      }
      aie.end
    } {link_with = "rms_norm.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c128 step %c1_1 {
          %0 = aie.objectfifo.acquire @in_5(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %2 = aie.objectfifo.acquire @out_5(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @rms_norm(%1, %3, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
          aie.objectfifo.release @in_5(Consume, 1)
          aie.objectfifo.release @out_5(Produce, 1)
        }
      }
      aie.end
    } {link_with = "rms_norm.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c128 step %c1_1 {
          %0 = aie.objectfifo.acquire @in_6(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %2 = aie.objectfifo.acquire @out_6(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @rms_norm(%1, %3, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
          aie.objectfifo.release @in_6(Consume, 1)
          aie.objectfifo.release @out_6(Produce, 1)
        }
      }
      aie.end
    } {link_with = "rms_norm.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c128 step %c1_1 {
          %0 = aie.objectfifo.acquire @in_7(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %2 = aie.objectfifo.acquire @out_7(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @rms_norm(%1, %3, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
          aie.objectfifo.release @in_7(Consume, 1)
          aie.objectfifo.release @out_7(Produce, 1)
        }
      }
      aie.end
    } {link_with = "rms_norm.o"}
    aiex.runtime_sequence @sequence(%arg0: memref<1048576xbf16>, %arg1: memref<1048576xbf16>) {
      %0 = aiex.dma_configure_task_for @in_0 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @in_1 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @in_2 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @in_3 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @in_4 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @in_5 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @in_6 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @in_7 {
        aie.dma_bd(%arg0 : memref<1048576xbf16>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @out_0 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 0, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%8)
      aiex.dma_await_task(%8)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      %9 = aiex.dma_configure_task_for @out_1 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 131072, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      aiex.dma_await_task(%9)
      %10 = aiex.dma_configure_task_for @out_2 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 262144, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      aiex.dma_await_task(%10)
      %11 = aiex.dma_configure_task_for @out_3 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 393216, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      aiex.dma_await_task(%11)
      %12 = aiex.dma_configure_task_for @out_4 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 524288, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      aiex.dma_await_task(%12)
      %13 = aiex.dma_configure_task_for @out_5 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 655360, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      aiex.dma_await_task(%13)
      %14 = aiex.dma_configure_task_for @out_6 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 786432, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      aiex.dma_await_task(%14)
      %15 = aiex.dma_configure_task_for @out_7 {
        aie.dma_bd(%arg1 : memref<1048576xbf16>, 917504, 131072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 131072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      aiex.dma_await_task(%15)
    }
  }
  aie.device(npu2) @mm {
    func.func private @zero_bf16(memref<32x32xbf16>)
    func.func private @matmul_bf16_bf16(memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_4 = aie.tile(3, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    aie.objectfifo @A_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @A_L3L2_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @A_L3L2_2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @A_L3L2_3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @A_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%tile_0_2, %tile_1_2, %tile_2_2, %tile_3_2}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @A_L2L1_1(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%tile_0_3, %tile_1_3, %tile_2_3, %tile_3_3}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @A_L2L1_2(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%tile_0_4, %tile_1_4, %tile_2_4, %tile_3_4}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @A_L2L1_3(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%tile_0_5, %tile_1_5, %tile_2_5, %tile_3_5}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo.link [@A_L3L2_0] -> [@A_L2L1_0]([] [])
    aie.objectfifo.link [@A_L3L2_1] -> [@A_L2L1_1]([] [])
    aie.objectfifo.link [@A_L3L2_2] -> [@A_L2L1_2]([] [])
    aie.objectfifo.link [@A_L3L2_3] -> [@A_L2L1_3]([] [])
    aie.objectfifo @B_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @B_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 4, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>], {%tile_0_2, %tile_0_3, %tile_0_4, %tile_0_5}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo.link [@B_L3L2_0] -> [@B_L2L1_0]([] [])
    aie.objectfifo @B_L3L2_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @B_L2L1_1(%mem_tile_1_1 dimensionsToStream [<size = 4, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>], {%tile_1_2, %tile_1_3, %tile_1_4, %tile_1_5}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo.link [@B_L3L2_1] -> [@B_L2L1_1]([] [])
    aie.objectfifo @B_L3L2_2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @B_L2L1_2(%mem_tile_2_1 dimensionsToStream [<size = 4, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>], {%tile_2_2, %tile_2_3, %tile_2_4, %tile_2_5}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo.link [@B_L3L2_2] -> [@B_L2L1_2]([] [])
    aie.objectfifo @B_L3L2_3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @B_L2L1_3(%mem_tile_3_1 dimensionsToStream [<size = 4, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>], {%tile_3_2, %tile_3_3, %tile_3_4, %tile_3_5}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo.link [@B_L3L2_3] -> [@B_L2L1_3]([] [])
    aie.objectfifo @C_L1L2_0_0(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @C_L1L2_0_1(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @C_L1L2_0_2(%tile_0_4, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @C_L1L2_0_3(%tile_0_5, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @C_L2L3_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>> 
    aie.objectfifo.link [@C_L1L2_0_0, @C_L1L2_0_1, @C_L1L2_0_2, @C_L1L2_0_3] -> [@C_L2L3_0]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C_L1L2_1_0(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @C_L1L2_1_1(%tile_1_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @C_L1L2_1_2(%tile_1_4, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @C_L1L2_1_3(%tile_1_5, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @C_L2L3_1(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>> 
    aie.objectfifo.link [@C_L1L2_1_0, @C_L1L2_1_1, @C_L1L2_1_2, @C_L1L2_1_3] -> [@C_L2L3_1]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C_L1L2_2_0(%tile_2_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @C_L1L2_2_1(%tile_2_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @C_L1L2_2_2(%tile_2_4, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @C_L1L2_2_3(%tile_2_5, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @C_L2L3_2(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>> 
    aie.objectfifo.link [@C_L1L2_2_0, @C_L1L2_2_1, @C_L1L2_2_2, @C_L1L2_2_3] -> [@C_L2L3_2]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C_L1L2_3_0(%tile_3_2, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @C_L1L2_3_1(%tile_3_3, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @C_L1L2_3_2(%tile_3_4, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @C_L1L2_3_3(%tile_3_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @C_L2L3_3(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>> 
    aie.objectfifo.link [@C_L1L2_3_0, @C_L1L2_3_1, @C_L1L2_3_2, @C_L1L2_3_3] -> [@C_L2L3_3]([0, 1024, 2048, 3072] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    aiex.runtime_sequence @sequence(%arg0: memref<1048576xbf16>, %arg1: memref<1048576xbf16>, %arg2: memref<1048576xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][2, 8, 128, 32][131072, 128, 1024, 1]) {id = 0 : i64, metadata = @C_L2L3_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][8, 32, 32, 32][0, 32, 1024, 1]) {id = 1 : i64, metadata = @A_L3L2_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 2 : i64, metadata = @B_L3L2_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 131072][8, 32, 32, 32][0, 32, 1024, 1]) {id = 3 : i64, metadata = @A_L3L2_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 4 : i64, metadata = @B_L3L2_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 32][2, 8, 128, 32][131072, 128, 1024, 1]) {id = 0 : i64, metadata = @C_L2L3_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 32768][8, 32, 32, 32][0, 32, 1024, 1]) {id = 1 : i64, metadata = @A_L3L2_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 32][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 2 : i64, metadata = @B_L3L2_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 163840][8, 32, 32, 32][0, 32, 1024, 1]) {id = 3 : i64, metadata = @A_L3L2_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 32][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 4 : i64, metadata = @B_L3L2_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 64][2, 8, 128, 32][131072, 128, 1024, 1]) {id = 0 : i64, metadata = @C_L2L3_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 65536][8, 32, 32, 32][0, 32, 1024, 1]) {id = 1 : i64, metadata = @A_L3L2_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 2 : i64, metadata = @B_L3L2_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 196608][8, 32, 32, 32][0, 32, 1024, 1]) {id = 3 : i64, metadata = @A_L3L2_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 4 : i64, metadata = @B_L3L2_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 96][2, 8, 128, 32][131072, 128, 1024, 1]) {id = 0 : i64, metadata = @C_L2L3_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 98304][8, 32, 32, 32][0, 32, 1024, 1]) {id = 1 : i64, metadata = @A_L3L2_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 96][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 2 : i64, metadata = @B_L3L2_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 229376][8, 32, 32, 32][0, 32, 1024, 1]) {id = 3 : i64, metadata = @A_L3L2_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 96][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 4 : i64, metadata = @B_L3L2_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 262144][2, 8, 128, 32][131072, 128, 1024, 1]) {id = 8 : i64, metadata = @C_L2L3_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 262144][8, 32, 32, 32][0, 32, 1024, 1]) {id = 9 : i64, metadata = @A_L3L2_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 10 : i64, metadata = @B_L3L2_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 393216][8, 32, 32, 32][0, 32, 1024, 1]) {id = 11 : i64, metadata = @A_L3L2_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 12 : i64, metadata = @B_L3L2_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 262176][2, 8, 128, 32][131072, 128, 1024, 1]) {id = 8 : i64, metadata = @C_L2L3_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 294912][8, 32, 32, 32][0, 32, 1024, 1]) {id = 9 : i64, metadata = @A_L3L2_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 32][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 10 : i64, metadata = @B_L3L2_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 425984][8, 32, 32, 32][0, 32, 1024, 1]) {id = 11 : i64, metadata = @A_L3L2_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 32][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 12 : i64, metadata = @B_L3L2_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 262208][2, 8, 128, 32][131072, 128, 1024, 1]) {id = 8 : i64, metadata = @C_L2L3_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 327680][8, 32, 32, 32][0, 32, 1024, 1]) {id = 9 : i64, metadata = @A_L3L2_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 10 : i64, metadata = @B_L3L2_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 458752][8, 32, 32, 32][0, 32, 1024, 1]) {id = 11 : i64, metadata = @A_L3L2_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 12 : i64, metadata = @B_L3L2_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 262240][2, 8, 128, 32][131072, 128, 1024, 1]) {id = 8 : i64, metadata = @C_L2L3_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 360448][8, 32, 32, 32][0, 32, 1024, 1]) {id = 9 : i64, metadata = @A_L3L2_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 96][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 10 : i64, metadata = @B_L3L2_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 491520][8, 32, 32, 32][0, 32, 1024, 1]) {id = 11 : i64, metadata = @A_L3L2_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 96][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 12 : i64, metadata = @B_L3L2_3} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @C_L2L3_0}
      aiex.npu.dma_wait {symbol = @C_L2L3_1}
      aiex.npu.dma_wait {symbol = @C_L2L3_2}
      aiex.npu.dma_wait {symbol = @C_L2L3_3}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 524288][2, 8, 128, 32][131072, 128, 1024, 1]) {id = 0 : i64, metadata = @C_L2L3_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 524288][8, 32, 32, 32][0, 32, 1024, 1]) {id = 1 : i64, metadata = @A_L3L2_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 2 : i64, metadata = @B_L3L2_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 655360][8, 32, 32, 32][0, 32, 1024, 1]) {id = 3 : i64, metadata = @A_L3L2_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 4 : i64, metadata = @B_L3L2_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 524320][2, 8, 128, 32][131072, 128, 1024, 1]) {id = 0 : i64, metadata = @C_L2L3_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 557056][8, 32, 32, 32][0, 32, 1024, 1]) {id = 1 : i64, metadata = @A_L3L2_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 32][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 2 : i64, metadata = @B_L3L2_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 688128][8, 32, 32, 32][0, 32, 1024, 1]) {id = 3 : i64, metadata = @A_L3L2_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 32][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 4 : i64, metadata = @B_L3L2_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 524352][2, 8, 128, 32][131072, 128, 1024, 1]) {id = 0 : i64, metadata = @C_L2L3_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 589824][8, 32, 32, 32][0, 32, 1024, 1]) {id = 1 : i64, metadata = @A_L3L2_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 2 : i64, metadata = @B_L3L2_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 720896][8, 32, 32, 32][0, 32, 1024, 1]) {id = 3 : i64, metadata = @A_L3L2_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 4 : i64, metadata = @B_L3L2_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 524384][2, 8, 128, 32][131072, 128, 1024, 1]) {id = 0 : i64, metadata = @C_L2L3_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 622592][8, 32, 32, 32][0, 32, 1024, 1]) {id = 1 : i64, metadata = @A_L3L2_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 96][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 2 : i64, metadata = @B_L3L2_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 753664][8, 32, 32, 32][0, 32, 1024, 1]) {id = 3 : i64, metadata = @A_L3L2_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 96][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 4 : i64, metadata = @B_L3L2_3} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @C_L2L3_0}
      aiex.npu.dma_wait {symbol = @C_L2L3_1}
      aiex.npu.dma_wait {symbol = @C_L2L3_2}
      aiex.npu.dma_wait {symbol = @C_L2L3_3}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 786432][2, 8, 128, 32][131072, 128, 1024, 1]) {id = 8 : i64, metadata = @C_L2L3_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 786432][8, 32, 32, 32][0, 32, 1024, 1]) {id = 9 : i64, metadata = @A_L3L2_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 10 : i64, metadata = @B_L3L2_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 917504][8, 32, 32, 32][0, 32, 1024, 1]) {id = 11 : i64, metadata = @A_L3L2_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 12 : i64, metadata = @B_L3L2_0} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 786464][2, 8, 128, 32][131072, 128, 1024, 1]) {id = 8 : i64, metadata = @C_L2L3_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 819200][8, 32, 32, 32][0, 32, 1024, 1]) {id = 9 : i64, metadata = @A_L3L2_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 32][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 10 : i64, metadata = @B_L3L2_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 950272][8, 32, 32, 32][0, 32, 1024, 1]) {id = 11 : i64, metadata = @A_L3L2_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 32][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 12 : i64, metadata = @B_L3L2_1} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 786496][2, 8, 128, 32][131072, 128, 1024, 1]) {id = 8 : i64, metadata = @C_L2L3_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 851968][8, 32, 32, 32][0, 32, 1024, 1]) {id = 9 : i64, metadata = @A_L3L2_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 10 : i64, metadata = @B_L3L2_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 983040][8, 32, 32, 32][0, 32, 1024, 1]) {id = 11 : i64, metadata = @A_L3L2_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 12 : i64, metadata = @B_L3L2_2} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 786528][2, 8, 128, 32][131072, 128, 1024, 1]) {id = 8 : i64, metadata = @C_L2L3_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 884736][8, 32, 32, 32][0, 32, 1024, 1]) {id = 9 : i64, metadata = @A_L3L2_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 96][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 10 : i64, metadata = @B_L3L2_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1015808][8, 32, 32, 32][0, 32, 1024, 1]) {id = 11 : i64, metadata = @A_L3L2_3} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 96][8, 32, 32, 32][128, 32768, 1024, 1]) {id = 12 : i64, metadata = @B_L3L2_3} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @C_L2L3_0}
      aiex.npu.dma_wait {symbol = @C_L2L3_1}
      aiex.npu.dma_wait {symbol = @C_L2L3_2}
      aiex.npu.dma_wait {symbol = @C_L2L3_3}
      aiex.npu.dma_wait {symbol = @C_L2L3_0}
      aiex.npu.dma_wait {symbol = @C_L2L3_1}
      aiex.npu.dma_wait {symbol = @C_L2L3_2}
      aiex.npu.dma_wait {symbol = @C_L2L3_3}
    }
  }
}