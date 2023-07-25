//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "test_library.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <xaiengine.h>

#include "aie_inc.cpp"
#include "memory_allocator.h"

int main(int argc, char *argv[]) {
  ext_mem_model_t buf;
  aie_libxaie_ctx_t *_xaie;
  int *mem_ptr;
  
  // Boilerplate setup code
  _xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);
  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_initialize_locks(_xaie);

  // External buffer setup
  mem_ptr = mlir_aie_mem_alloc(buf, 256);
#if defined(__AIESIM__)
  mlir_aie_external_set_addr_extbuf(
      (u64)(buf.physicalAddr));
#else
  mlir_aie_external_set_addr_extbuf((u64)mem_ptr_in);
#endif
  mlir_aie_configure_shimdma_30(_xaie);

  // Write value to buffer
  *mem_ptr = 35;
  mlir_aie_sync_mem_dev(buf);

  // Start and run cores
  mlir_aie_start_cores(_xaie);

  // Wait for completion, check results
  assert(XAIE_OK == mlir_aie_acquire_lock34_0(_xaie, 1, 1000));
  sleep(1);
  mlir_aie_sync_mem_cpu(buf);
  printf("buf[0] = %d\n", mem_ptr[0]);
  //assert(70 == mem_ptr[0]);

  // Teardown
  mlir_aie_deinit_libxaie(_xaie);
  return 0;
}
