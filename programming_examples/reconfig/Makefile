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

INPUT_MLIR?=another-test/aie.mlir

.PHONY: all
all: build/empty.xclbin build/config_subtract_three_insts.bin build/run_subtract_three_insts.bin build/test

build/%.o: another-test/%.o
	mkdir -p ${@D}
	cp $< $@

# Intermediate lowering of our NPU design.
build/design.mlir: design.py
	python3 $< ${FLAGS} > $@

# Complete binary for loading onto the NPU.
# The following target generates both build/final.xclbin AND build/insts.txt
build/%.xclbin: ${INPUT_MLIR}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v  --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
	            --device-name ${*F} \
				--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				$(<:%=../%)

build/combined_xclbin.xclbin: ${INPUT_MLIR}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
	            --xclbin-kernel-name=mm --xclbin-kernel-id=0x901 --xclbin-instance-name=mm_inst \
				--device-name mm \
				--no-aiesim --aie-generate-xclbin --no-compile-host \
				--xclbin-name=intermediate_mm.xclbin \
				${<:%=../%}
	cd ${@D} && aiecc.py -v --no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
	            --xclbin-kernel-name=rms_norm --xclbin-kernel-id=0x902 --xclbin-instance-name=rms_norm_inst \
				--device-name rms_norm \
				--no-aiesim --aie-generate-xclbin --no-compile-host \
				--xclbin-input=intermediate_mm.xclbin --xclbin-name=${@F} \
				${<:%=../%}
	

build/config_%_insts.mlir: ${INPUT_MLIR}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v  --no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				--aie-generate-txn --txn-name=${@F} ${<:%=../%}

build/config_%_insts.bin: build/config_%_insts.mlir
	aie-translate --aie-device-name=${*F} --aie-sequence-name=configure --aie-npu-to-binary $< -o $@

build/run_%_insts.mlir: ${INPUT_MLIR}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v  --no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
	            --aie-generate-txn --txn-name=txns.mlir \
				--aie-generate-npu-insts --sequence-name=rt --npu-insts-name=${@F} $(<:%=../%)
	aie-opt --pass-pipeline="builtin.module(aie.device(aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu,aie-dma-to-npu,aie-lower-set-lock))" \
		build/${INPUT_MLIR}.prj/input_with_addresses.mlir -o $@

build/run_%_insts.bin: build/run_%_insts.mlir
	mkdir -p ${@D}
	aie-translate --aie-npu-to-binary --aie-device-name=${*F} --aie-sequence-name=rt $< -o $@

build/npu_insts_main_rt.bin: ${INPUT_MLIR}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v  --no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
	            --aie-generate-txn --txn-name=txns.mlir \
				--aie-generate-npu-insts ${<:%=../%}


VERBOSE?=0

# Host-side code (runs on the CPU) that loads and starts our design onto the NPU using the XRT (Xilinx runtime) library.
build/test: test.cpp 
	mkdir -p ${@D}
	g++-13 -std=c++23 -g -DVERBOSE=${VERBOSE} -I${XRT_INC_DIR} -L${XRT_LIB_DIR} ${FLAGS} -o $@ $^ -lxrt_coreutil

build/test-runlist: test.cpp
	mkdir -p ${@D}
	g++-13 -std=c++23 -g -DVERBOSE=${VERBOSE} -DUSE_RUNLIST=1 -I${XRT_INC_DIR} -L${XRT_LIB_DIR} ${FLAGS} -o $@ $^ -lxrt_coreutil


# Remove all build artifacts.
.PHONY: clean
clean: 
	rm -rf build