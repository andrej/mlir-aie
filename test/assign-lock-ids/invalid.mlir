// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-lock-ids %s  -split-input-file -verify-diagnostics

aie.device(xcve2802) {
  // expected-note @below {{because only 16 locks available in this tile}}
  %tMemTile = aie.tile(4,4)
  aie.lock @l0(%tMemTile)
  %l0 = aie.get_lock @l0
  aie.lock @l1(%tMemTile)
  %l1 = aie.get_lock @l1
  aie.lock @l2(%tMemTile)
  %l2 = aie.get_lock @l2
  aie.lock @l3(%tMemTile)
  %l3 = aie.get_lock @l3
  aie.lock @l4(%tMemTile)
  %l4 = aie.get_lock @l4
  aie.lock @l5(%tMemTile)
  %l5 = aie.get_lock @l5
  aie.lock @l6(%tMemTile)
  %l6 = aie.get_lock @l6
  aie.lock @l7(%tMemTile)
  %l7 = aie.get_lock @l7
  aie.lock @l8(%tMemTile)
  %l8 = aie.get_lock @l8
  aie.lock @l9(%tMemTile)
  %l9 = aie.get_lock @l9
  aie.lock @l10(%tMemTile)
  %l10 = aie.get_lock @l10
  aie.lock @l11(%tMemTile)
  %l11 = aie.get_lock @l11
  aie.lock @l12(%tMemTile)
  %l12 = aie.get_lock @l12
  aie.lock @l13(%tMemTile)
  %l13 = aie.get_lock @l13
  aie.lock @l14(%tMemTile)
  %l14 = aie.get_lock @l14
  aie.lock @l15(%tMemTile)
  %l15 = aie.get_lock @l15
  // expected-error @below {{not allocated a lock}}
  aie.lock @l16(%tMemTile)
  %l16 = aie.get_lock @l16
  aie.lock @l17(%tMemTile)
  %l17 = aie.get_lock @l17
  aie.lock @l18(%tMemTile)
  %l18 = aie.get_lock @l18
  aie.lock @l19(%tMemTile)
  %l19 = aie.get_lock @l19
}

// -----

aie.device(xcve2802) {
  // expected-note @below {{because only 16 locks available in this tile}}
  %tMemTile = aie.tile(4,4)
  aie.lock @l0(%tMemTile)
  %l0 = aie.get_lock @l0
  aie.lock @l1(%tMemTile, 1)
  %l1 = aie.get_lock @l1
  aie.lock @l2(%tMemTile)
  %l2 = aie.get_lock @l2
  aie.lock @l3(%tMemTile)
  %l3 = aie.get_lock @l3
  aie.lock @l4(%tMemTile)
  %l4 = aie.get_lock @l4
  aie.lock @l5(%tMemTile)
  %l5 = aie.get_lock @l5
  aie.lock @l6(%tMemTile)
  %l6 = aie.get_lock @l6
  aie.lock @l7(%tMemTile, 2)
  %l7 = aie.get_lock @l7
  aie.lock @l8(%tMemTile)
  %l8 = aie.get_lock @l8
  aie.lock @l9(%tMemTile)
  %l9 = aie.get_lock @l9
  aie.lock @l10(%tMemTile, 15)
  %l10 = aie.get_lock @l10
  aie.lock @l11(%tMemTile)
  %l11 = aie.get_lock @l11
  aie.lock @l12(%tMemTile)
  %l12 = aie.get_lock @l12
  aie.lock @l13(%tMemTile)
  %l13 = aie.get_lock @l13
  aie.lock @l14(%tMemTile)
  %l14 = aie.get_lock @l14
  // expected-error @below {{not allocated a lock}}
  aie.lock @l15(%tMemTile)
  %l15 = aie.get_lock @l15
  aie.lock @l16(%tMemTile)
  %l16 = aie.get_lock @l16
  aie.lock @l17(%tMemTile, 3)
  %l17 = aie.get_lock @l17
  aie.lock @l18(%tMemTile)
  %l18 = aie.get_lock @l18
  aie.lock @l19(%tMemTile)
  %l19 = aie.get_lock @l19
}

// -----

aie.device(xcve2802) {
 //expected-note @below {{tile has lock ops assigned to same lock}}
  %t22 = aie.tile(2,2)
  aie.lock @l0(%t22, 7)
  %l0 = aie.get_lock @l0
  // expected-error @below {{is assigned to the same lock (7) as another op}}
  aie.lock @l1(%t22, 7)
  %l1 = aie.get_lock @l1
}
