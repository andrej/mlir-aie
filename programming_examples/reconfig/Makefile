##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##


# Dependencies
# --------------------------------------------------------------------------

# Where is Vitis/aietools for kernel/AIE compilation?
AIETOOLS_DIR?=$(shell realpath $(dir $(shell which xchesscc))/../)

# Where is Peano for kernel/AIE compilation?
AIEOPT_DIR?=$(shell realpath $(dir $(shell which aie-opt))/..)
PEANOWRAP2_FLAGS=-O2 -std=c++20 --target=aie2-none-unknown-elf \
	-Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
	-Wno-missing-template-arg-list-after-template-kw \
	-DNDEBUG -I ${AIEOPT_DIR}/include 

# Where is the XRT library for our runtime host code?
XRT_INC_DIR=/opt/xilinx/xrt/include
XRT_LIB_DIR=/opt/xilinx/xrt/lib


# Targets
# --------------------------------------------------------------------------

.PHONY: all
all: build/empty.xclbin build/config_insts.bin build/run_insts.bin build/test

# Intermediate lowering of our NPU design.
build/design.mlir: design.py
	python3 $< ${FLAGS} > $@

# Complete binary for loading onto the NPU.
# The following target generates both build/final.xclbin AND build/insts.txt
build/empty.xclbin: aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
	            --device-name=empty \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				$(<:%=../%)

build/config_insts.bin: aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
	            --device-name=subtract_three \
				--aie-generate-txn --txn-name=txns.mlir ${<:%=../%}
	aie-translate --aie-device-name=subtract_three --aie-npu-to-binary build/txns.mlir -o $@

build/run_insts.bin: aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
	            --device-name=subtract_three \
				--aie-generate-npu-insts --npu-insts-name=${@F} $(<:%=../%)

build/combined_insts.bin: aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				--device-name=subtract_three \
				--aie-generate-txn --txn-name=txns.mlir ${<:%=../%}
	aie-opt --pass-pipeline="builtin.module(aie.device(aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu,aie-dma-to-npu,aie-lower-set-lock))" \
		build/txns.mlir -o build/txns_dma_lowered.mlir
	aie-opt --aie-materialize-runtime-sequence build/txns_dma_lowered.mlir -o build/txns_inlined.mlir
	aie-translate --aie-device-name main --aie-npu-to-binary --aie-sequence-name main_rt build/txns_inlined.mlir -o $@

# Host-side code (runs on the CPU) that loads and starts our design onto the NPU using the XRT (Xilinx runtime) library.
build/test: test.cpp 
	mkdir -p ${@D}
	g++ -std=c++20 -I${XRT_INC_DIR} -L${XRT_LIB_DIR} ${FLAGS} -o $@ $^ -lxrt_coreutil

# Convenience to run the host-side code above.
.PHONY: run
run: build/test build/empty.xclbin build/config_insts.bin build/run_insts.bin
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	./$< build/config_insts.bin build/run_insts.bin

# Remove all build artifacts.
.PHONY: clean
clean: 
	rm -rf build