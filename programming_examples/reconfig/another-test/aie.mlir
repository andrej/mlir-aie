module {
  aie.device(npu2) @empty{

  }

  aie.device(npu2) @main {
    aiex.runtime_sequence @rt(%inout : memref<1048576xbf16>) {
      %c1 = aiex.configure @mm
      aiex.run %c1 -> @sequence(%inout, %inout, %inout) : (memref<1048576xbf16>, memref<1048576xbf16>, memref<1048576xbf16>)


      aiex.npu.patch_marker { id = "loadpdi" }
      aiex.npu.load_pdi { id = 0x01 : ui16, size = 1 : ui32, address = 2 : ui64 }

      %c2 = aiex.configure @rms_norm
      //%c2 = aiex.configure @mm

      aiex.run %c2 -> @sequence(%inout, %inout) : (memref<1048576xbf16>, memref<1048576xbf16>)
      //aiex.run %c1 -> @sequence(%inout, %inout, %inout) : (memref<1048576xbf16>, memref<1048576xbf16>, memref<1048576xbf16>)
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

}


