//===- aie_row.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: %aiecc %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% -o test.elf -- %S/test.cpp
// RUN: %run_on_board ./test.elf

// CHECK: AIE2 ISS
// CHECK: test start.
// CHECK: PASS!

module @test04_shared_memory {
  aie.device(xcve2802) {
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    aie.buffer @a(%tile23)  : memref<256xi32>
    %buf13_0 = aie.get_buffer @a : memref<256xi32>
    aie.buffer @b(%tile23)  : memref<256xi32>
    %buf13_1 = aie.get_buffer @b : memref<256xi32>
    aie.buffer @c(%tile33)  : memref<256xi32>
    %buf23_0 = aie.get_buffer @c : memref<256xi32>

    aie.lock @test_lock(%tile23, 2)  // test lock
    %lock13_2 = aie.get_lock @test_lock

    aie.lock @input_write_lock(%tile23, 3) { init = 1 : i32 } // input buffer lock
    %lock13_3 = aie.get_lock @input_write_lock
    aie.lock @input_read_lock(%tile23, 4)  // input buffer lock
    %lock13_4 = aie.get_lock @input_read_lock
    aie.lock @hidden_write_lock(%tile23, 5) { init = 1 : i32 } // interbuffer lock
    %lock13_5 = aie.get_lock @hidden_write_lock
    aie.lock @hidden_read_lock(%tile23, 6)  // interbuffer lock
    %lock13_6 = aie.get_lock @hidden_read_lock
    aie.lock @output_write_lock(%tile33, 7) { init = 1 : i32 } // output buffer lock
    %lock23_7 = aie.get_lock @output_write_lock
    aie.lock @output_read_lock(%tile33, 8)  // output buffer lock
    %lock23_8 = aie.get_lock @output_read_lock

    %core13 = aie.core(%tile23) {
      %c1_ul0 = arith.constant 1 : i32
      aie.use_lock(@input_read_lock, AcquireGreaterEqual, %c1_ul0) // acquire input for read(e.g. input ping)
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(@hidden_write_lock, AcquireGreaterEqual, %c1_ul1) // acquire input for write
      %idx1 = arith.constant 3 : index
      %val1 = memref.load %buf13_0[%idx1] : memref<256xi32>
      %2    = arith.addi %val1, %val1 : i32
      %3 = arith.addi %2, %val1 : i32
      %4 = arith.addi %3, %val1 : i32
      %5 = arith.addi %4, %val1 : i32
      %idx2 = arith.constant 5 : index
      memref.store %5, %buf13_1[%idx2] : memref<256xi32>
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(@input_write_lock, Release, %c1_ul2) // release input for write
      %c1_ul3 = arith.constant 1 : i32
      aie.use_lock(@hidden_read_lock, Release, %c1_ul3) // release output for read
      aie.end
    }

    %core23 = aie.core(%tile33) {
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(@hidden_read_lock, AcquireGreaterEqual, %c1_ul4) // acquire input for read(e.g. input ping)
      %c1_ul5 = arith.constant 1 : i32
      aie.use_lock(@output_write_lock, AcquireGreaterEqual, %c1_ul5) // acquire output for write
      %idx1 = arith.constant 5 : index
      %val1 = memref.load %buf13_1[%idx1] : memref<256xi32>
      %2    = arith.addi %val1, %val1 : i32
      %3 = arith.addi %2, %val1 : i32
      %4 = arith.addi %3, %val1 : i32
      %5 = arith.addi %4, %val1 : i32
      %idx2 = arith.constant 5 : index
      memref.store %5, %buf23_0[%idx2] : memref<256xi32>
      %c1_ul6 = arith.constant 1 : i32
      aie.use_lock(@hidden_write_lock, Release, %c1_ul6) // release input for write
      %c1_ul7 = arith.constant 1 : i32
      aie.use_lock(@output_read_lock, Release, %c1_ul7) // release output for read
      aie.end
    }
  }
}
