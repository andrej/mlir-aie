module {
  aie.device(npu2_1col) @empty {
  }

  aie.device(npu2_1col) @add_two {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
  
    aie.objectfifo @objFifo_in0(%t00, {%t01}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @objFifo_in1(%t01, {%t02}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@objFifo_in0] -> [@objFifo_in1] ([] [])

    aie.objectfifo @objFifo_out1(%t02, {%t01}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @objFifo_out0(%t01, {%t00}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@objFifo_out1] -> [@objFifo_out0] ([] [])
  
    aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      %c2_32 = arith.constant 2 : i32

      %c_intmax = arith.constant 0xFFFFFF : index

      scf.for %niter = %c0 to %c_intmax step %c1 {
        scf.for %i = %c0 to %c512 step %c1 {
          %subview0 = aie.objectfifo.acquire @objFifo_in1(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
          %elem0 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
          %subview1 = aie.objectfifo.acquire @objFifo_out1(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
          %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
          scf.for %arg3 = %c0 to %c8 step %c1 {
              %0 = memref.load %elem0[%arg3] : memref<8xi32>
              %1 = arith.addi %0, %c2_32 : i32
              memref.store %1, %elem1[%arg3] : memref<8xi32>
          }
          aie.objectfifo.release @objFifo_in1(Consume, 1)
          aie.objectfifo.release @objFifo_out1(Produce, 1)
        }
      }
      aie.end
    }

    aiex.runtime_sequence @rt(%a : memref<1024x1024xi32>) {
      %t_in = aiex.dma_configure_task_for @objFifo_in0 {
        aie.dma_bd(%a : memref<1024x1024xi32>, 0, 1048576)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @objFifo_out0 {
        aie.dma_bd(%a: memref<1024x1024xi32>, 0, 1048576)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_in)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
    }
  }

  aie.device(npu2_1col) @subtract_three {
    %t00 = aie.tile(0, 0)
    %t03 = aie.tile(0, 3)

    aie.objectfifo @in(%t00, {%t03}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out(%t03, {%t00}, 2 : i32) : !aie.objectfifo<memref<32xi32>>

    aie.core(%t03) {
      %c_0 = arith.constant 0 : index
      %c_1 = arith.constant 1 : index
      %c_2 = arith.constant 2 : index
      %c_32 = arith.constant 32 : index
      %c_64 = arith.constant 64 : index
      %c_128 = arith.constant 128 : index
      %c_intmax = arith.constant 0xFFFFFF : index
      %c_3_i32 = arith.constant 3 : i32
      scf.for %run = %c_0 to %c_intmax step %c_1 {
        scf.for %j = %c_0 to %c_128 step %c_1 {
          %subview0 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<32xi32>>
          %in_elem = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
          %subview1 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<32xi32>>
          %out_elem = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
          scf.for %i = %c_0 to %c_32 step %c_1 {
              %in_val = memref.load %in_elem[%i] : memref<32xi32>
              %out_val = arith.subi %in_val, %c_3_i32 : i32
              memref.store %out_val, %out_elem[%i] : memref<32xi32>
          }
          aie.objectfifo.release @in(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
      }
      aie.end
    }

    aiex.runtime_sequence @rt(%a : memref<1024x1024xi32>) {
      %t_in = aiex.dma_configure_task_for @in {
        aie.dma_bd(%a : memref<1024x1024xi32>, 0, 1048576)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @out {
        aie.dma_bd(%a: memref<1024x1024xi32>, 0, 1048576)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_in)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
    }

  }

  aie.device(npu2_1col) @main {
    aiex.runtime_sequence @rt (%a: memref<1024x1024xi32>) {
      %c1 = aiex.configure @add_two
      aiex.run %c1 -> @rt (%a) : (memref<1024x1024xi32>)

      // // Reset module
      // aiex.npu.write32 {row = 0 : i32, col = 0 : i32, address = 0x00060010 : ui32, value = 7 : ui32}
      // aiex.npu.write32 {row = 1 : i32, col = 0 : i32, address = 0x00060010 : ui32, value = 7 : ui32}
      // aiex.npu.write32 {row = 2 : i32, col = 0 : i32, address = 0x00060010 : ui32, value = 7 : ui32}

      // // Reset S2MM_0
      // //aiex.npu.write32 {row = 0 : i32, col = 0 : i32, address = 0x0001DE00 : ui32, value = 1 : ui32}
      // aiex.npu.write32 {row = 1 : i32, col = 0 : i32, address = 0x0001DE00 : ui32, value = 1 : ui32}
      // aiex.npu.write32 {row = 2 : i32, col = 0 : i32, address = 0x0001DE00 : ui32, value = 1 : ui32}

      // // Reset S2MM_1
      // //aiex.npu.write32 {row = 0 : i32, col = 0 : i32, address = 0x0001DE08 : ui32, value = 1 : ui32}
      // aiex.npu.write32 {row = 1 : i32, col = 0 : i32, address = 0x0001DE08 : ui32, value = 1 : ui32}
      // aiex.npu.write32 {row = 2 : i32, col = 0 : i32, address = 0x0001DE08 : ui32, value = 1 : ui32}

      // // Reset MM2S_0
      // //aiex.npu.write32 {row = 0 : i32, col = 0 : i32, address = 0x0001DE10 : ui32, value = 1 : ui32}
      // aiex.npu.write32 {row = 1 : i32, col = 0 : i32, address = 0x0001DE10 : ui32, value = 1 : ui32}
      // aiex.npu.write32 {row = 2 : i32, col = 0 : i32, address = 0x0001DE10 : ui32, value = 1 : ui32}

      // // Reset MM2S_1
      // //aiex.npu.write32 {row = 0 : i32, col = 0 : i32, address = 0x0001DE18 : ui32, value = 1 : ui32}
      // aiex.npu.write32 {row = 1 : i32, col = 0 : i32, address = 0x0001DE18 : ui32, value = 1 : ui32}
      // aiex.npu.write32 {row = 2 : i32, col = 0 : i32, address = 0x0001DE18 : ui32, value = 1 : ui32}

      // aiex.npu.write32 { col = 0 : i32, row = 0 : i32, address = 0x0001D200 : ui32, value = 3 : ui32}
      // aiex.npu.write32 { col = 0 : i32, row = 0 : i32, address = 0x00060010 : ui32, value = 0 : ui32 }
      // aiex.npu.write32 { col = 0 : i32, row = 0 : i32, address = 0x00060010 : ui32, value = 1 : ui32 }
      // aiex.npu.write32 { col = 0 : i32, row = 0 : i32, address = 0x0001D200 : ui32, value = 0 : ui32}

      // aiex.npu.write32 { col = 0 : i32, row = 0 : i32, address = 0x0003F040 : ui32, value = 0 : ui32}

      // reset ---

      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219536 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219544 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219520 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219528 : ui32, mask = 2 : ui32, value = 0 : ui32}
      //%0 = memref.get_global @blockwrite_data : memref<200xi32>
      //aiex.npu.blockwrite(%0) {address = 2228224 : ui32} : memref<200xi32>
      aiex.npu.maskwrite32 {address = 2219536 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219544 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219520 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2219528 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 2 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224128 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224144 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224160 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224176 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224192 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224208 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224224 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224240 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224256 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224272 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224288 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224304 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224320 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224336 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224352 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224368 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 81952 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 81968 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1835040 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1835056 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224160 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224176 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224128 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224144 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1835008 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1835024 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 81920 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 81936 : ui32, value = 0 : ui32}
      //%1 = memref.get_global @blockwrite_data_0 : memref<6xi32>
      //aiex.npu.blockwrite(%1) {address = 2215936 : ui32} : memref<6xi32>
      //%2 = memref.get_global @blockwrite_data_1 : memref<6xi32>
      //aiex.npu.blockwrite(%2) {address = 2216000 : ui32} : memref<6xi32>
      //%3 = memref.get_global @blockwrite_data_2 : memref<6xi32>
      //aiex.npu.blockwrite(%3) {address = 2216032 : ui32} : memref<6xi32>
      //%4 = memref.get_global @blockwrite_data_3 : memref<6xi32>
      //aiex.npu.blockwrite(%4) {address = 2215968 : ui32} : memref<6xi32>
      aiex.npu.write32 {address = 2219524 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2219520 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2219540 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2219536 : ui32, value = 0 : ui32}
      //%5 = memref.get_global @blockwrite_data_4 : memref<8xi32>
      //aiex.npu.blockwrite(%5) {address = 1703936 : ui32} : memref<8xi32>
      //%6 = memref.get_global @blockwrite_data_5 : memref<8xi32>
      //aiex.npu.blockwrite(%6) {address = 1704000 : ui32} : memref<8xi32>
      //%7 = memref.get_global @blockwrite_data_6 : memref<8xi32>
      //aiex.npu.blockwrite(%7) {address = 1704704 : ui32} : memref<8xi32>
      //%8 = memref.get_global @blockwrite_data_7 : memref<8xi32>
      //aiex.npu.blockwrite(%8) {address = 1704768 : ui32} : memref<8xi32>
      //%9 = memref.get_global @blockwrite_data_8 : memref<8xi32>
      //aiex.npu.blockwrite(%9) {address = 1704800 : ui32} : memref<8xi32>
      //%10 = memref.get_global @blockwrite_data_9 : memref<8xi32>
      //aiex.npu.blockwrite(%10) {address = 1704736 : ui32} : memref<8xi32>
      //%11 = memref.get_global @blockwrite_data_10 : memref<8xi32>
      //aiex.npu.blockwrite(%11) {address = 1704032 : ui32} : memref<8xi32>
      //%12 = memref.get_global @blockwrite_data_11 : memref<8xi32>
      //aiex.npu.blockwrite(%12) {address = 1703968 : ui32} : memref<8xi32>
      aiex.npu.write32 {address = 1705476 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1705472 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1705524 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1705520 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1705484 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1705480 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1705532 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1705528 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258112 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258324 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258064 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258368 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258056 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258304 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258560 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769472 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769772 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769520 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769728 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769476 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769780 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769508 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 1769732 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2355204 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2355480 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2355220 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2355460 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 126976 : ui32, mask = 3072 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 126980 : ui32, mask = 48 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 0 : ui32}
      // reste ---

      %c2 = aiex.configure @subtract_three
      aiex.run %c2 -> @rt (%a) : (memref<1024x1024xi32>)
    }
  }
}
