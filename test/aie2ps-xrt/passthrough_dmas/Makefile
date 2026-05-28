##===- Makefile -----------------------------------------------------------===##
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# DMA-only test (no core kernel) — no compiler selection needed.
# Usage:
#   make                  - build
#   make clean            - remove build/

SRCDIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# EDF SDK for cross-compilation
EDF_SDK_PATH ?= $(HOME)/amd-edf-sdk
EDF_SYSROOT := $(EDF_SDK_PATH)/sysroots/cortexa72-cortexa53-amd-linux
EDF_CXX := $(EDF_SDK_PATH)/sysroots/x86_64-amdedfsdk-linux/usr/bin/aarch64-amd-linux/aarch64-amd-linux-g++

.PHONY: all elf host clean

all: build/aie_control_full.elf build/host.exe

elf: build/aie_control_full.elf

build/aie_control_full.elf: $(SRCDIR)/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc --no-compile-host \
		--generate-full-elf --full-elf-name=${@F} $<

host: build/host.exe

build/host.exe: $(SRCDIR)/test.cpp
	mkdir -p ${@D}
	$(EDF_CXX) --sysroot=$(EDF_SYSROOT) -std=c++17 -O0 -g \
		-I$(EDF_SYSROOT)/usr/include -L$(EDF_SYSROOT)/usr/lib \
		-lxrt_coreutil -lstdc++ -lpthread -lrt \
		-o $@ $<

clean:
	rm -rf build/
