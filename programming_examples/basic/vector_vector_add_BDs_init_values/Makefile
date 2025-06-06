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

targetname = vector_vector_add
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
col = 0

aie_py_src=${targetname}.py

all: build/final.xclbin

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} ${col} > $@

build/final.xclbin: build/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin ${<F}

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

# Changing variables when we target VCK5000
ACDC_AIE = $(dir $(shell which aie-opt))/..

vck5000: devicename=xcvc1902
vck5000: col=6

vck5000: build/aie.mlir
	aiecc.py \
    --link_against_hsa --host-target=x86_64-amd-linux-gnu build/aie.mlir \
		-I${srcdir}/../../../install/runtime_lib/x86_64-hsa/test_lib/include \
		${srcdir}/test_vck5000.cpp \
		${srcdir}/../../../install/runtime_lib/x86_64-hsa/test_lib/src/test_library.cpp \
		-Wl,--whole-archive -Wl,--no-whole-archive -lstdc++ -ldl -lelf -o test.elf

run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
