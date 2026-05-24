##===- Makefile -----------------------------------------------------------===##
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# passthrough_kernel for AIE2PS (xcve3858, VEK385).
# Kernel is plain C — compiles with both Chess and Peano.
# Usage:
#   make                  - build with Peano (default)
#   make CHESS=true       - build with Chess (xchesscc)
#   make clean            - remove build/

SRCDIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

CHESS ?= false
PEANO_INSTALL_DIR ?= $(shell echo $$PEANO_INSTALL_DIR)
AIETOOLS_DIR ?= $(shell realpath $(dir $(shell which xchesscc 2>/dev/null))/../ 2>/dev/null)

# EDF SDK for cross-compilation
EDF_SDK_PATH ?= $(HOME)/amd-edf-sdk
EDF_SYSROOT := $(EDF_SDK_PATH)/sysroots/cortexa72-cortexa53-amd-linux
EDF_CXX := $(EDF_SDK_PATH)/sysroots/x86_64-amdedfsdk-linux/usr/bin/aarch64-amd-linux/aarch64-amd-linux-g++

.PHONY: all elf host clean

all: build/aie_control_full.elf build/host.exe

build/passthrough.o: $(SRCDIR)/passthrough.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	xchesscc_wrapper aie2ps -c $< -o $@
else
	$(PEANO_INSTALL_DIR)/bin/clang++ -O2 -std=c++20 --target=aie2ps-none-unknown-elf \
		-DNDEBUG -c $< -o $@
endif

elf: build/aie_control_full.elf

build/aie_control_full.elf: $(SRCDIR)/aie.mlir build/passthrough.o
	mkdir -p ${@D}
	cp $(SRCDIR)/aie.mlir ${@D}/aie.mlir
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
		-I$(EDF_SYSROOT)/usr/include -L$(EDF_SYSROOT)/usr/lib \
		-lxrt_coreutil -lstdc++ -lpthread -lrt \
		-o $@ $<

clean:
	rm -rf build/
