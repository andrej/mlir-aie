//===- duplicate_sym_name.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The device symbol table rejects duplicate names across AIE symbol kinds.

// RUN: not aie-opt --split-input-file %s 2>&1 | FileCheck %s

// CHECK: error{{.*}}redefinition of symbol named 'dup_buf'
// CHECK: note{{.*}}see existing symbol definition here
aie.device(npu1) {
  %t = aie.tile(0, 2)
  aie.buffer @dup_buf(%t)  : memref<4xi32>
  %b0 = aie.get_buffer @dup_buf : memref<4xi32>
  aie.buffer @dup_buf(%t)  : memref<4xi32>
  %b1 = aie.get_buffer @dup_buf : memref<4xi32>
}

// -----

// CHECK: error{{.*}}redefinition of symbol named 'dup_lock'
aie.device(npu1) {
  %t = aie.tile(0, 2)
  aie.lock @dup_lock(%t, 0)
  %l0 = aie.get_lock @dup_lock
  aie.lock @dup_lock(%t, 1)
  %l1 = aie.get_lock @dup_lock
}

// -----

// A buffer and a lock collide just as two buffers do.
// CHECK: error{{.*}}redefinition of symbol named 'shared'
aie.device(npu1) {
  %t = aie.tile(0, 2)
  aie.buffer @shared(%t)  : memref<4xi32>
  aie.lock @shared(%t, 0)
}

// -----

// CHECK: error{{.*}}redefinition of symbol named 'shared_with_fifo'
aie.device(npu1) {
  %shim = aie.tile(0, 0)
  %core = aie.tile(0, 2)
  aie.objectfifo @shared_with_fifo(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
  aie.buffer @shared_with_fifo(%core)  : memref<4xi32>
}
