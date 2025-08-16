#!/bin/bash

set -e

MODE=2
ITERS=100

rm -f run_${MODE}.txt

make build/mm.o
make build/rms_norm.o

if [[ $MODE -eq 0 ]] then
    # To get individual xclbins runtime:
    make build/test
    make build/mm.xclbin
    make build/rms_norm.xclbin
    make build/npu_insts_main_rt.bin
    for i in $(seq 0 $ITERS); do
        ./build/test ./build/mm.xclbin ./build/npu_insts_mm_sequence.bin ./build/rms_norm.xclbin ./build/npu_insts_rms_norm_sequence.bin | tee -a run_${MODE}.txt
    done
fi

# runlist

if [[ $MODE -eq 1 ]] then
    make build/test-runlist
    make build/combined_xclbin.xclbin
    make build/npu_insts_main_rt.bin
    for i in $(seq 0 $ITERS); do
        ./build/test-runlist ./build/combined_xclbin.xclbin:mm ./build/npu_insts_mm_sequence.bin ./build/combined_xclbin.xclbin:rms_norm ./build/npu_insts_rms_norm_sequence.bin | tee -a run_${MODE}.txt
    done
fi

# To get empty xclbin with individual configure and run instruction invocations:

if [[ $MODE -eq 2 ]] then
    # To get inlined txns runtime:
    make build/test
    make build/empty.xclbin
    make build/npu_insts_main_rt.bin
    for i in $(seq $ITERS); do
        ./build/test ./build/empty.xclbin ./build/npu_insts_main_rt.bin:./build/npu_insts_patch_map_main_rt.txt | tee -a run_${MODE}.txt
    done
fi

