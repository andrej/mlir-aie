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
<unknown>:0: error: 'aie.connect' op targets same destination South: 0 as another connect operation
<unknown>:0: note: see current operation: "aie.connect"() <{dest_bundle = 3 : i32, dest_channel = 0 : i32, source_bundle = 3 : i32, source_channel = 6 : i32}> : () -> ()
