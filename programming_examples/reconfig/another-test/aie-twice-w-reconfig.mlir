module {
  aie.device(npu2) @empty{
  }

  aie.device(npu2) @main {
    aiex.runtime_sequence @rt(%inout : memref<1048576xbf16>) {
      %c1 = aiex.configure @mm
      aiex.run %c1 -> @sequence(%inout, %inout, %inout) : (memref<1048576xbf16>, memref<1048576xbf16>, memref<1048576xbf16>)

      aiex.npu.patch_marker { id = "loadpdi" }
      aiex.npu.load_pdi { id = 0x01 : ui16, size = 1 : ui32, address = 2 : ui64 }
      %c2 = aiex.configure @mm

      aiex.run %c1 -> @sequence(%inout, %inout, %inout) : (memref<1048576xbf16>, memref<1048576xbf16>, memref<1048576xbf16>)
    }
  }

  aie.device(npu2) @mm {
    func.func private @zero_bf16(memref<32x32xbf16>)
    func.func private @matmul_bf16_bf16(memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @inA(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @memA(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo.link [@inA] -> [@memA]([] [])
    aie.objectfifo @inB(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @memB(%mem_tile_0_1 dimensionsToStream [<size = 4, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo.link [@inB] -> [@memB]([] [])
    aie.objectfifo @memC(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @outC(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo.link [@memC] -> [@outC]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c1024 step %c1_1 {
          %0 = aie.objectfifo.acquire @memC(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @zero_bf16(%1) : (memref<32x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @memA(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            %4 = aie.objectfifo.acquire @memB(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
            aie.objectfifo.release @memA(Consume, 1)
            aie.objectfifo.release @memB(Consume, 1)
          }
          aie.objectfifo.release @memC(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o", stack_size = 3328 : i32}
    aiex.runtime_sequence @sequence(%arg0: memref<1048576xbf16>, %arg1: memref<1048576xbf16>, %arg2: memref<1048576xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][2, 32, 32, 32][32768, 32, 1024, 1]) {id = 0 : i64, metadata = @outC} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 32, 32, 32][0, 32, 1024, 1]) {id = 1 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 2 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 32768][32, 32, 32, 32][0, 32, 1024, 1]) {id = 3 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 4 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 65536][2, 32, 32, 32][32768, 32, 1024, 1]) {id = 8 : i64, metadata = @outC} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 65536][32, 32, 32, 32][0, 32, 1024, 1]) {id = 9 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 10 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 98304][32, 32, 32, 32][0, 32, 1024, 1]) {id = 11 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 12 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 131072][2, 32, 32, 32][32768, 32, 1024, 1]) {id = 0 : i64, metadata = @outC} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 131072][32, 32, 32, 32][0, 32, 1024, 1]) {id = 1 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 2 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 163840][32, 32, 32, 32][0, 32, 1024, 1]) {id = 3 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 4 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 196608][2, 32, 32, 32][32768, 32, 1024, 1]) {id = 8 : i64, metadata = @outC} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 196608][32, 32, 32, 32][0, 32, 1024, 1]) {id = 9 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 10 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 229376][32, 32, 32, 32][0, 32, 1024, 1]) {id = 11 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 12 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 262144][2, 32, 32, 32][32768, 32, 1024, 1]) {id = 0 : i64, metadata = @outC} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 262144][32, 32, 32, 32][0, 32, 1024, 1]) {id = 1 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 2 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 294912][32, 32, 32, 32][0, 32, 1024, 1]) {id = 3 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 4 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 327680][2, 32, 32, 32][32768, 32, 1024, 1]) {id = 8 : i64, metadata = @outC} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 327680][32, 32, 32, 32][0, 32, 1024, 1]) {id = 9 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 10 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 360448][32, 32, 32, 32][0, 32, 1024, 1]) {id = 11 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 12 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 393216][2, 32, 32, 32][32768, 32, 1024, 1]) {id = 0 : i64, metadata = @outC} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 393216][32, 32, 32, 32][0, 32, 1024, 1]) {id = 1 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 2 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 425984][32, 32, 32, 32][0, 32, 1024, 1]) {id = 3 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 4 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 458752][2, 32, 32, 32][32768, 32, 1024, 1]) {id = 8 : i64, metadata = @outC} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 458752][32, 32, 32, 32][0, 32, 1024, 1]) {id = 9 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 10 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 491520][32, 32, 32, 32][0, 32, 1024, 1]) {id = 11 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 12 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 524288][2, 32, 32, 32][32768, 32, 1024, 1]) {id = 0 : i64, metadata = @outC} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 524288][32, 32, 32, 32][0, 32, 1024, 1]) {id = 1 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 2 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 557056][32, 32, 32, 32][0, 32, 1024, 1]) {id = 3 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 4 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 589824][2, 32, 32, 32][32768, 32, 1024, 1]) {id = 8 : i64, metadata = @outC} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 589824][32, 32, 32, 32][0, 32, 1024, 1]) {id = 9 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 10 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 622592][32, 32, 32, 32][0, 32, 1024, 1]) {id = 11 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 12 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 655360][2, 32, 32, 32][32768, 32, 1024, 1]) {id = 0 : i64, metadata = @outC} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 655360][32, 32, 32, 32][0, 32, 1024, 1]) {id = 1 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 2 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 688128][32, 32, 32, 32][0, 32, 1024, 1]) {id = 3 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 4 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 720896][2, 32, 32, 32][32768, 32, 1024, 1]) {id = 8 : i64, metadata = @outC} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 720896][32, 32, 32, 32][0, 32, 1024, 1]) {id = 9 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 10 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 753664][32, 32, 32, 32][0, 32, 1024, 1]) {id = 11 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 12 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 786432][2, 32, 32, 32][32768, 32, 1024, 1]) {id = 0 : i64, metadata = @outC} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 786432][32, 32, 32, 32][0, 32, 1024, 1]) {id = 1 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 2 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 819200][32, 32, 32, 32][0, 32, 1024, 1]) {id = 3 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 4 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 851968][2, 32, 32, 32][32768, 32, 1024, 1]) {id = 8 : i64, metadata = @outC} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 851968][32, 32, 32, 32][0, 32, 1024, 1]) {id = 9 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 10 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 884736][32, 32, 32, 32][0, 32, 1024, 1]) {id = 11 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 12 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 917504][2, 32, 32, 32][32768, 32, 1024, 1]) {id = 0 : i64, metadata = @outC} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 917504][32, 32, 32, 32][0, 32, 1024, 1]) {id = 1 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 2 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 950272][32, 32, 32, 32][0, 32, 1024, 1]) {id = 3 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 4 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 983040][2, 32, 32, 32][32768, 32, 1024, 1]) {id = 8 : i64, metadata = @outC} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 983040][32, 32, 32, 32][0, 32, 1024, 1]) {id = 9 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 10 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1015808][32, 32, 32, 32][0, 32, 1024, 1]) {id = 11 : i64, metadata = @inA} : memref<1048576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 32, 32, 32][32, 32768, 1024, 1]) {id = 12 : i64, metadata = @inB} : memref<1048576xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_wait {symbol = @outC}
    }
  }

}


