##===- Makefile -----------------------------------------------------------===##
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# event_trace for AIE2PS (xcve3858, VEK385).
# Usage:
#   make              - build ELF + host.exe into build/
#   make parse_trace  - parse build/trace.bin -> trace.json + trace_timeline.png
#   make clean        - remove generated files

SRCDIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
KERNEL_DIR := $(SRCDIR)/../../../aie_kernels/aie2
TRACE_UTILS := $(SRCDIR)/../../../python/utils/trace

# EDF SDK for cross-compilation
EDF_SDK_PATH ?= $(HOME)/amd-edf-sdk
EDF_SYSROOT := $(EDF_SDK_PATH)/sysroots/cortexa72-cortexa53-amd-linux
EDF_CXX := $(EDF_SDK_PATH)/sysroots/x86_64-amdedfsdk-linux/usr/bin/aarch64-amd-linux/aarch64-amd-linux-g++

.PHONY: all elf host host_debug parse_trace clean_trace clean

all: build/aie_control_full.elf build/host.exe

build/aie.mlir: $(SRCDIR)/design.py
	mkdir -p ${@D}
	python3 $< > $@

build/scale.o: $(KERNEL_DIR)/scale.cc
	mkdir -p ${@D}
	xchesscc_wrapper aie2ps -I $(SRCDIR)/../../../aie_kernels -c $< -o $@

elf: build/aie_control_full.elf

build/aie_control_full.elf: build/aie.mlir build/scale.o
	cd ${@D} && aiecc --xchesscc --xbridge --no-compile-host \
		--generate-full-elf --full-elf-name=${@F} aie.mlir

host: build/host.exe

build/host.exe: $(SRCDIR)/test.cpp
	mkdir -p ${@D}
	$(EDF_CXX) --sysroot=$(EDF_SYSROOT) -std=c++17 -O0 -g \
		-I$(EDF_SYSROOT)/usr/include -L$(EDF_SYSROOT)/usr/lib \
		-lxrt_coreutil -lstdc++ -lpthread -lrt \
		-o $@ $<

host_debug: build/host_debug.exe

build/host_debug.exe: $(SRCDIR)/test_debug.cpp
	mkdir -p ${@D}
	$(EDF_CXX) --sysroot=$(EDF_SYSROOT) -std=c++17 -O0 -g \
		-I$(EDF_SYSROOT)/usr/include -L$(EDF_SYSROOT)/usr/lib \
		-lxrt_coreutil -lstdc++ -lpthread -lrt \
		-o $@ $<

parse_trace: build/trace.bin
	@echo "Converting trace.bin to hex text..."
	python3 -c "import struct; data=open('build/trace.bin','rb').read(); \
	  words=struct.unpack('<'+'I'*(len(data)//4),data); \
	  [print(f'{w:08x}') for w in words]" > build/trace.txt
	@echo "Parsing trace output..."
	python3 $(TRACE_UTILS)/parse.py \
	  --input build/trace.txt \
	  --mlir build/aie.mlir.prj/input_with_addresses.mlir \
	  --output build/trace.json || true
	@if [ -f build/trace.json ]; then \
		python3 $(TRACE_UTILS)/get_trace_summary.py \
		  --input build/trace.json || true; \
		echo ""; \
		echo "Generating trace visualization..."; \
		python3 $(SRCDIR)/visualize_trace.py \
		  -i build/trace.json \
		  -o build/trace_timeline.png \
		  -t "AIE2PS Event Trace" || true; \
	fi

clean_trace:
	rm -f build/trace.bin build/trace.txt build/trace.json build/trace_timeline.png

clean: clean_trace
	rm -rf build/
