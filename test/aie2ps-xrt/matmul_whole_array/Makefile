##===- Makefile -----------------------------------------------------------===##
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# Whole-array matmul for AIE2PS (xcve3858, VEK385).
# 36 columns × 4 core rows = 144 cores. Peano only.
# Usage:
#   make                                - i16×i16→i32, 512×512×1152
#   make DTYPE=bf16                     - bf16×bf16→f32
#   make M=256 K=256 N=1152             - custom dimensions
#   make clean                          - remove build/

SRCDIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PEANO_INSTALL_DIR ?= $(shell echo $$PEANO_INSTALL_DIR)

# Matrix dimensions
M ?= 512
K ?= 512
N ?= 1152
m ?= 32
k ?= 32
n ?= 32
N_AIE_COLS ?= 36

# Data type
DTYPE ?= i16
ifeq ($(DTYPE), bf16)
  DTYPE_IN = bf16
  DTYPE_OUT = f32
  KERNEL_DTYPE_FLAG = -DDTYPE_BF16
  HOST_DTYPE_FLAG = -DDTYPE_BF16
else
  DTYPE_IN = i16
  DTYPE_OUT = i32
  KERNEL_DTYPE_FLAG =
  HOST_DTYPE_FLAG =
endif

# EDF SDK for cross-compilation
EDF_SDK_PATH ?= $(HOME)/amd-edf-sdk
EDF_SYSROOT := $(EDF_SDK_PATH)/sysroots/cortexa72-cortexa53-amd-linux
EDF_CXX := $(EDF_SDK_PATH)/sysroots/x86_64-amdedfsdk-linux/usr/bin/aarch64-amd-linux/aarch64-amd-linux-g++

.PHONY: all elf host clean

all: build/aie_control_full.elf build/host.exe

build/aie.mlir: $(SRCDIR)/design_scalar.py
	mkdir -p ${@D}
	python3 $< -M $(M) -K $(K) -N $(N) -m $(m) -k $(k) -n $(n) \
		--n-aie-cols $(N_AIE_COLS) \
		--dtype_in $(DTYPE_IN) --dtype_out $(DTYPE_OUT) > $@

build/mm_$(m)x$(k)x$(n).o: $(SRCDIR)/mm_peano.cc
	mkdir -p ${@D}
	$(PEANO_INSTALL_DIR)/bin/clang++ -O2 -std=c++20 --target=aie2ps-none-unknown-elf \
		-DNDEBUG -DDIM_M=$(m) -DDIM_K=$(k) -DDIM_N=$(n) \
		$(KERNEL_DTYPE_FLAG) -c $< -o $@

elf: build/aie_control_full.elf

build/aie_control_full.elf: build/aie.mlir build/mm_$(m)x$(k)x$(n).o
	cd ${@D} && aiecc --no-xchesscc --no-xbridge --no-compile-host \
		--generate-full-elf --full-elf-name=${@F} aie.mlir

host: build/host.exe

build/host.exe: $(SRCDIR)/test.cpp
	mkdir -p ${@D}
	$(EDF_CXX) --sysroot=$(EDF_SYSROOT) -std=c++17 -O2 -g \
		-DMAT_M=$(M) -DMAT_K=$(K) -DMAT_N=$(N) \
		$(HOST_DTYPE_FLAG) \
		-I$(EDF_SYSROOT)/usr/include -L$(EDF_SYSROOT)/usr/lib \
		-lxrt_coreutil -lstdc++ -lpthread -lrt \
		-o $@ $<

clean:
	rm -rf build/
