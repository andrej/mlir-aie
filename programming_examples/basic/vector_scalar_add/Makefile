##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

all: build/aie.elf

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = vector_scalar_add

aie_py_src=${targetname}.py
use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/aie.elf: build/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc --generate-full-elf --no-compile-host \
		--no-xchesscc --no-xbridge --dynamic-objFifos ${<F}
		
${targetname}.exe: ${srcdir}/test.cpp
	clang++ -o $@ $< -std=c++17 \
		-I/opt/xilinx/xrt/include \
		-I${srcdir}/../../../runtime_lib/test_lib \
		-L/opt/xilinx/xrt/lib -lxrt_core -lxrt_coreutil

run: ${targetname}.exe build/aie.elf
	${powershell} ./$< -e build/aie.elf

clean:
	rm -rf build _build ${targetname}.exe
