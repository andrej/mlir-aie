// A memtile receives %in and broadcasts it to two cores. Each core scales it
// into its own output, which returns to the host.
module {
  aie.device(npu1_1col) {
    func.func private @scale(memref<16xi32>, memref<16xi32>)
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core_a = aie.tile(0, 2)
    %core_b = aie.tile(0, 3)
    aie.objectfifo @in(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in_l1(%mem, {%core_a, %core_b}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@in_l1]([] [0])
    aie.objectfifo @out_a(%core_a, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_b(%core_b, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %body_a = aie.core(%core_a) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %x = aie.objectfifo.acquire @in_l1(Consume, 1) : memref<16xi32>
        %y = aie.objectfifo.acquire @out_a(Produce, 1) : memref<16xi32>
        func.call @scale(%x, %y) : (memref<16xi32>, memref<16xi32>) -> ()
        aie.objectfifo.release @in_l1(Consume, 1)
        aie.objectfifo.release @out_a(Produce, 1)
      }
      aie.end
    }
    %body_b = aie.core(%core_b) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %x = aie.objectfifo.acquire @in_l1(Consume, 1) : memref<16xi32>
        %y = aie.objectfifo.acquire @out_b(Produce, 1) : memref<16xi32>
        func.call @scale(%x, %y) : (memref<16xi32>, memref<16xi32>) -> ()
        aie.objectfifo.release @in_l1(Consume, 1)
        aie.objectfifo.release @out_b(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%in_host: memref<16xi32>, %out_host: memref<32xi32>) {
      %t0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%in_host : memref<16xi32> offset = 0 len = 16)
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @out_a {
        aie.dma_bd(%out_host : memref<32xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @out_b {
        aie.dma_bd(%out_host : memref<32xi32> offset = 16 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t1)
      aiex.dma_await_task(%t2)
    }
  }
}
