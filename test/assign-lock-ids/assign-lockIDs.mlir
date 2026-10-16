//===- assign-lockIDs.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-lock-ids --split-input-file %s | FileCheck %s
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_2:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_3:.*]] = aie.tile(3, 4)
// CHECK:           aie.lock @l22_0(%[[VAL_0]], 0)
// CHECK:           aie.lock @l22_2(%[[VAL_0]], 2)
// CHECK:           aie.lock @l22_1(%[[VAL_0]], 1)
// CHECK:           aie.lock @l23_0(%[[VAL_1]], 0)
// CHECK:           aie.lock @l23_1(%[[VAL_1]], 1)
// CHECK:           aie.lock @l23_4(%[[VAL_1]], 4)
// CHECK:           aie.lock @l23_2(%[[VAL_1]], 2)
// CHECK:           aie.lock @l23_3(%[[VAL_1]], 3)
// CHECK:           aie.lock @l23_5(%[[VAL_1]], 5)
// CHECK:           aie.lock @l23_6(%[[VAL_1]], 6)
// CHECK:           aie.lock @l23_7(%[[VAL_1]], 7)
// CHECK:           aie.lock @l23_10(%[[VAL_1]], 10)
// CHECK:           aie.lock @l23_11(%[[VAL_1]], 11)
// CHECK:           aie.lock @l23_8(%[[VAL_1]], 8)
// CHECK:           aie.lock @l23_9(%[[VAL_1]], 9)
// CHECK:           aie.lock @l23_12(%[[VAL_1]], 12)
// CHECK:           aie.lock @l23_13(%[[VAL_1]], 13)
// CHECK:           aie.lock @l23_14(%[[VAL_1]], 14)
// CHECK:           aie.lock @l23_15(%[[VAL_1]], 15)
// CHECK:           aie.lock @l33_0(%[[VAL_2]], 0)
// CHECK:           aie.lock @l33_1(%[[VAL_2]], 1)
// CHECK:           aie.lock @l33_9(%[[VAL_2]], 9)
// CHECK:           aie.lock @l33_2(%[[VAL_2]], 2)
// CHECK:           aie.lock @l34_0(%[[VAL_3]], 0)
// CHECK:           aie.lock @l34_1(%[[VAL_3]], 1)
// CHECK:           aie.lock @l34_2(%[[VAL_3]], 2)
// CHECK:           aie.lock @l34_3(%[[VAL_3]], 3)
// CHECK:           %[[VAL_31:.*]] = aie.tile(6, 0)
// CHECK:           aie.lock @l60(%[[VAL_31]], 0)

module @test_assign_lockIDs {
 aie.device(xcvc1902) {
  %t22 = aie.tile(2, 2)
  %t23 = aie.tile(2, 3)
  %t33 = aie.tile(3, 3)
  %t34 = aie.tile(3, 4)

  aie.lock @l22_0(%t22, 0)
  %l22_0 = aie.get_lock @l22_0
  aie.lock @l22_2(%t22, 2)
  %l22_2 = aie.get_lock @l22_2
  aie.lock @l22_1(%t22)
  %l22_1 = aie.get_lock @l22_1

  aie.lock @l23_0(%t23)
  %l23_0 = aie.get_lock @l23_0
  aie.lock @l23_1(%t23)
  %l23_1 = aie.get_lock @l23_1
  aie.lock @l23_4(%t23, 4)
  %l23_4 = aie.get_lock @l23_4
  aie.lock @l23_2(%t23)
  %l23_2 = aie.get_lock @l23_2
  aie.lock @l23_3(%t23)
  %l23_3 = aie.get_lock @l23_3
  aie.lock @l23_5(%t23)
  %l23_5 = aie.get_lock @l23_5
  aie.lock @l23_6(%t23)
  %l23_6 = aie.get_lock @l23_6
  aie.lock @l23_7(%t23)
  %l23_7 = aie.get_lock @l23_7
  aie.lock @l23_10(%t23)
  %l23_10 = aie.get_lock @l23_10
  aie.lock @l23_11(%t23)
  %l23_11 = aie.get_lock @l23_11
  aie.lock @l23_8(%t23, 8)
  %l23_8 = aie.get_lock @l23_8
  aie.lock @l23_9(%t23, 9)
  %l23_9 = aie.get_lock @l23_9
  aie.lock @l23_12(%t23)
  %l23_12 = aie.get_lock @l23_12
  aie.lock @l23_13(%t23)
  %l23_13 = aie.get_lock @l23_13
  aie.lock @l23_14(%t23)
  %l23_14 = aie.get_lock @l23_14
  aie.lock @l23_15(%t23)
  %l23_15 = aie.get_lock @l23_15

  aie.lock @l33_0(%t33, 0)
  %l33_0 = aie.get_lock @l33_0
  aie.lock @l33_1(%t33)
  %l33_1 = aie.get_lock @l33_1
  aie.lock @l33_9(%t33, 9)
  %l33_9 = aie.get_lock @l33_9
  aie.lock @l33_2(%t33)
  %l33_2 = aie.get_lock @l33_2

  aie.lock @l34_0(%t34)
  %l34_0 = aie.get_lock @l34_0
  aie.lock @l34_1(%t34)
  %l34_1 = aie.get_lock @l34_1
  aie.lock @l34_2(%t34)
  %l34_2 = aie.get_lock @l34_2
  aie.lock @l34_3(%t34)
  %l34_3 = aie.get_lock @l34_3

  %t60 = aie.tile(6, 0)
  aie.lock @l60(%t60)
  %l60 = aie.get_lock @l60
 }
}

// -----

module @memTileTest {
  aie.device(xcve2802) {

    // Memory tiles on xcve have 64 locks.
    %tmemtile = aie.tile(1,1)
    aie.lock @l0(%tmemtile, 1)
    %l0 = aie.get_lock @l0
    aie.lock @l1(%tmemtile, 0)
    %l1 = aie.get_lock @l1
    aie.lock @l2(%tmemtile)
    %l2 = aie.get_lock @l2
    aie.lock @l3(%tmemtile)
    %l3 = aie.get_lock @l3
    aie.lock @l4(%tmemtile)
    %l4 = aie.get_lock @l4
    aie.lock @l5(%tmemtile)
    %l5 = aie.get_lock @l5
    aie.lock @l6(%tmemtile)
    %l6 = aie.get_lock @l6
    aie.lock @l7(%tmemtile)
    %l7 = aie.get_lock @l7
    aie.lock @l8(%tmemtile)
    %l8 = aie.get_lock @l8
    aie.lock @l9(%tmemtile)
    %l9 = aie.get_lock @l9
    aie.lock @l10(%tmemtile)
    %l10 = aie.get_lock @l10
    aie.lock @l11(%tmemtile)
    %l11 = aie.get_lock @l11
    aie.lock @l12(%tmemtile)
    %l12 = aie.get_lock @l12
    aie.lock @l13(%tmemtile)
    %l13 = aie.get_lock @l13
    aie.lock @l14(%tmemtile,33)
    %l14 = aie.get_lock @l14
    aie.lock @l15(%tmemtile)
    %l15 = aie.get_lock @l15
    aie.lock @l16(%tmemtile)
    %l16 = aie.get_lock @l16
    aie.lock @l17(%tmemtile)
    %l17 = aie.get_lock @l17
    aie.lock @l18(%tmemtile)
    %l18 = aie.get_lock @l18
    aie.lock @l19(%tmemtile,2)
    %l19 = aie.get_lock @l19
  }
}


// CHECK-LABEL: memTileTest
// CHECK-COUNT-20: aie.lock
// CHECK-NOT: aie.lock
