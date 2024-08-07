##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

# ---

# The following environment variables that point to the Xilinx runtime (XRT)
# should be set up by an environment setup script already.
XILINX_XRT?=/opt/xilinx/xrt
XILINX_VITIS?=$(shell realpath $(dir $(shell which vitis))/../)

# ---

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

XILINX_XRT_INCLUDE?=${XILINX_XRT}/include
XILINX_XRT_LIB?=${XILINX_XRT}/lib

XRT_FLAGS=-I${XILINX_XRT_INCLUDE} -L${XILINX_XRT_LIB}
XRT_LIBS=-lxrt_coreutil

CXX=g++-13 -ggdb 

iterations?=9

host_target?=build/test
xclbin_targets?=\
	build/final_do_nothing.xclbin \programming_examples/basic/opsync_benchmarks/aie2_config.mlir
all: ${host_target} ${xclbin_targets}

build/insts_%.txt: build/final_%.xclbin

build/final_%.xclbin: aie2_%.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=${@F:final_%.xclbin=insts_%.txt} ${<:%=../%}

${host_target}: ${srcdir}/test.cpp
	mkdir -p ${@D}
	${CXX} ${XRT_FLAGS} -o $@ $< ${XRT_LIBS} 

.PHONY: run
run: ${host_target} ${xclbin_targets}
	./${host_target} 0 ${iterations}
	./${host_target} 1 ${iterations}
	./${host_target} 2 ${iterations}
	./${host_target} 3 ${iterations}

.PHONY: run_do_nothing
run_do_nothing: ${host_target} build/final_do_nothing.xclbin
	./${host_target} 0 ${iterations}

.PHONY: run_config
run_config: ${host_target} build/final_config.xclbin
	./${host_target} 1 ${iterations}

.PHONY: run_config_start
run_config_start: ${host_target} build/final_config_start.xclbin
	./${host_target} 2 ${iterations}

.PHONY: run_config_start_sync
run_config_start_sync: ${host_target} build/final_config_start_sync.xclbin
	./${host_target} 3 ${iterations} 

.PHONY: clean
clean:
	-rm -r build
