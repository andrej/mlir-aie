# (c) Copyright 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

srcdir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

XRT_FLAGS := -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib
XRT_LIBS := -lxrt_coreutil

TEST_UTILS_FLAGS := -I/scratch/roesti/autonomous-g/mlir-aie/build/runtime_lib/x86_64/test_lib/include \
                    -L/scratch/roesti/autonomous-g/mlir-aie/build/runtime_lib/x86_64/test_lib/lib
TEST_UTILS_LIBS := -ltest_utils

AIECC := aiecc.py
AIECC_FLAGS := --no-aiesim --no-xchesscc --no-xbridge \
               --aie-generate-xclbin --aie-generate-npu-insts --aie-generate-pdi \
               --no-compile-host \
               --xclbin-name=aie.xclbin \
               --npu-insts-name=insts.bin \
               --npu-insts-offsets-name=insts_offsets.json

ARTIFACTS := aie.xclbin insts.bin insts_offsets.json add_two.pdi test.exe

.PHONY: all run clean

all: $(ARTIFACTS)

aie.xclbin insts.bin insts_offsets.json add_two.pdi &: $(srcdir)aie.mlir
	$(AIECC) $(AIECC_FLAGS) $<

test.exe: $(srcdir)test.cpp
	clang++ $< -o $@ -std=c++17 -Wall $(XRT_FLAGS) -lrt -lstdc++ $(XRT_LIBS) $(TEST_UTILS_FLAGS) $(TEST_UTILS_LIBS)

run: all
	./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin --offsets insts_offsets.json --pdi add_two.pdi -v 1

clean:
	rm -rf $(ARTIFACTS) aie.mlir.prj
