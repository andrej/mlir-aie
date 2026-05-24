##===- Makefile -----------------------------------------------------------===##
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# vector_scalar_mul for AIE2PS (xcve3858, VEK385).
# Supports both Chess and Peano, int32/int16, scalar/vectorized.
# Usage:
#   make                                  - int32, scalar, Peano
#   make DTYPE=i16                        - int16, scalar, Peano
#   make VECTORIZED=1                     - int32, vectorized (UPS+SRS), Peano
#   make DTYPE=i16 VECTORIZED=1           - int16, vectorized (UPS+SRS), Peano
#   make CHESS=true                       - int32, vectorized, Chess (aie_api)
#   make CHESS=true DTYPE=i16             - int16, vectorized, Chess (aie_api)
#   make clean                            - remove build/

SRCDIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
KERNEL_DIR := $(SRCDIR)/../../../aie_kernels/aie2

CHESS ?= false
DTYPE ?= i32
VECTORIZED ?= 0
PEANO_INSTALL_DIR ?= $(shell echo $$PEANO_INSTALL_DIR)
AIETOOLS_DIR ?= $(shell realpath $(dir $(shell which xchesscc 2>/dev/null))/../ 2>/dev/null)

ifeq ($(DTYPE), i16)
  DTYPE_FLAG = -DDTYPE=DTYPE_I16
  BIT_WIDTH_FLAG = -DBIT_WIDTH=16
  HOST_DTYPE_FLAG = -DDTYPE_I16
else
  DTYPE_FLAG = -DDTYPE=DTYPE_I32
  BIT_WIDTH_FLAG = -DBIT_WIDTH=32
  HOST_DTYPE_FLAG =
endif

# EDF SDK for cross-compilation
EDF_SDK_PATH ?= $(HOME)/amd-edf-sdk
EDF_SYSROOT := $(EDF_SDK_PATH)/sysroots/cortexa72-cortexa53-amd-linux
EDF_CXX := $(EDF_SDK_PATH)/sysroots/x86_64-amdedfsdk-linux/usr/bin/aarch64-amd-linux/aarch64-amd-linux-g++

.PHONY: all elf host clean

all: build/aie_control_full.elf build/host.exe

build/aie.mlir: $(SRCDIR)/design.py
	mkdir -p ${@D}
	python3 $< --dtype $(DTYPE) > $@

build/scale.o:
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	xchesscc_wrapper aie2ps -I $(AIETOOLS_DIR)/include \
		-I $(SRCDIR)/../../../aie_kernels \
		$(BIT_WIDTH_FLAG) -c $(KERNEL_DIR)/scale.cc -o $@
else
	$(PEANO_INSTALL_DIR)/bin/clang++ -O2 -std=c++20 --target=aie2ps-none-unknown-elf \
		-DNDEBUG $(DTYPE_FLAG) -DVECTORIZED=$(VECTORIZED) \
		-c $(SRCDIR)/scale_peano.cc -o $@
endif

elf: build/aie_control_full.elf

build/aie_control_full.elf: build/aie.mlir build/scale.o
ifeq ($(CHESS), true)
	cd ${@D} && aiecc --xchesscc --xbridge --no-compile-host \
		--generate-full-elf --full-elf-name=${@F} aie.mlir
else
	cd ${@D} && aiecc --no-xchesscc --no-xbridge --no-compile-host \
		--generate-full-elf --full-elf-name=${@F} aie.mlir
endif

host: build/host.exe

build/host.exe: $(SRCDIR)/test.cpp
	mkdir -p ${@D}
	$(EDF_CXX) --sysroot=$(EDF_SYSROOT) -std=c++17 -O0 -g \
		$(HOST_DTYPE_FLAG) \
		-I$(EDF_SYSROOT)/usr/include -L$(EDF_SYSROOT)/usr/lib \
		-lxrt_coreutil -lstdc++ -lpthread -lrt \
		-o $@ $<

clean:
	rm -rf build/
