Found EMBEDDED_METADATA section, size = 1351 bytes
Extracted metadata, full content:
<?xml version="1.0" encoding="utf-8"?>
<project>
  <platform>
    <device>
      <core>
        <kernel name="MLIR_AIE" language="c" type="dpu">
          <extended-data subtype="1" functional="0" dpu_kernel_id="0x901"/>
          <arg name="opcode" addressQualifier="0" id="0" size="0x8" offset="0x00" hostOffset="0x0" hostSize="0x8" type="uint64_t"/>
          <arg name="instr" addressQualifier="1" id="1" size="0x8" offset="0x8" hostOffset="0x0" hostSize="0x8" type="char *"/>
          <arg name="ninstr" addressQualifier="0" id="2" size="0x4" offset="0x10" hostOffset="0x0" hostSize="0x4" type="uint32_t"/>
          <arg name="bo0" addressQualifier="1" id="3" size="0x8" offset="0x14" hostOffset="0x0" hostSize="0x8" type="void*"/>
          <arg name="bo1" addressQualifier="1" id="4" size="0x8" offset="0x1c" hostOffset="0x0" hostSize="0x8" type="void*"/>
          <arg name="bo2" addressQualifier="1" id="5" size="0x8" offset="0x24" hostOffset="0x0" hostSize="0x8" type="void*"/>
          <arg name="bo3" addressQualifier="1" id="6" size="0x8" offset="0x2c" hostOffset="0x0" hostSize="0x8" type="void*"/>
          <arg name="bo4" addressQualifier="1" id="7" size="0x8" offset="0x34" hostOffset="0x0" hostSize="0x8" type="void*"/>
          <instance name="MLIRAIE"/>
        </kernel>
      </core>
    </device>
  </platform>
</project>

Successfully parsed transaction binary from insts.bin
Transaction module contains: 3 write32, 3 blockwrite, 0 pushqueue ops
Detected max column: 0 -> device type: npu1_1col
Processed 3 write32 ops from transaction
Processing blockwrite at address 0x0x200480, isBDAddress=0
Processing blockwrite at address 0x0x01d000, isBDAddress=1
  BD address recognized: tile(0,0) BD#0 reg#0
  Found getGlobalOp: config_blockwrite_data_1
  Found globalOp
Extracted BD from transaction: tile(0,0) BD#0 channel=-1 length=64
Processing blockwrite at address 0x0x01d020, isBDAddress=1
  BD address recognized: tile(0,0) BD#1 reg#0
  Found getGlobalOp: config_blockwrite_data_2
  Found globalOp
Extracted BD from transaction: tile(0,0) BD#1 channel=-1 length=64
Processed 3 blockwrite ops from transaction
Warning: Emitting 2 BDs with inferred channel MM2S_0 for tile(0,0)
Warning: Emitting 4 BDs with inferred channel S2MM_0 for tile(0,2)
module {
  aie.device(npu1_1col) @xclbin_device {
    %ext_buf_0_0_1 = aie.external_buffer {sym_name = "ext_buf_0_0_1"} : memref<64xi32>
    %ext_buf_0_0_0 = aie.external_buffer {sym_name = "ext_buf_0_0_0"} : memref<64xi32>
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_dma_0_0 = aie.shim_dma(%shim_noc_tile_0_0) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 3 preds: ^bb0, ^bb1, ^bb2
      aie.dma_bd(%ext_buf_0_0_0 : memref<64xi32>, 0, 64)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%ext_buf_0_0_1 : memref<64xi32>, 0, 64)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    %tile_0_2 = aie.tile(0, 2)
    %bd_buf_0_2_3 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_3"} : memref<1xi32> 
    %bd_buf_0_2_2 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_2"} : memref<1xi32> 
    %bd_buf_0_2_1 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_1"} : memref<1xi32> 
    %bd_buf_0_2_0 = aie.buffer(%tile_0_2) {sym_name = "bd_buf_0_2_0"} : memref<1xi32> 
    %lock_0_2 = aie.lock(%tile_0_2, 3) {init = 0 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 1) {init = 0 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 0) {init = 2 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 5 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4
      aie.dma_bd(%bd_buf_0_2_0 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb2:  // no predecessors
      aie.dma_bd(%bd_buf_0_2_1 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb3:  // no predecessors
      aie.dma_bd(%bd_buf_0_2_2 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb4:  // no predecessors
      aie.dma_bd(%bd_buf_0_2_3 : memref<1xi32>, 0, 0)
      aie.next_bd ^bb1
    ^bb5:  // pred: ^bb0
      aie.end
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    aie.runtime_sequence @configure() {
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2098336 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2228224 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2229120 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2229152 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2229184 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2229360 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 2 : ui32, value = 2 : ui32}
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
      aiex.npu.write32 {address = 2224128 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224144 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224160 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2224176 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2219524 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 2219540 : ui32, value = 0 : ui32}
      aiex.npu.write32 {address = 258560 : ui32, value = 0 : ui32}
      aiex.npu.maskwrite32 {address = 126976 : ui32, mask = 3072 : ui32, value = 1024 : ui32}
      aiex.npu.maskwrite32 {address = 126980 : ui32, mask = 48 : ui32, value = 16 : ui32}
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aie.end
    }
  }
}
