#!/bin/bash

set -ex

MODES="0 1 2"
ITERS=10

KERNEL_1=add_two
KERNEL_2=subtract_three

for MODE in $MODES; do

    rm -f run_${MODE}.txt

    make build/mm.o
    make build/rms_norm.o

    if [[ $MODE -eq 0 ]] then
        # To get individual xclbins runtime:
        make build/test
        make build/${KERNEL_1}.xclbin
        make build/${KERNEL_2}.xclbin
        make build/npu_insts_main_sequence.bin
        for i in $(seq 0 $ITERS); do
            ./build/test ./build/${KERNEL_1}.xclbin ./build/npu_insts_${KERNEL_1}_sequence.bin ./build/${KERNEL_1}.xclbin ./build/npu_insts_${KERNEL_1}_sequence.bin ./build/${KERNEL_2}.xclbin ./build/npu_insts_${KERNEL_2}_sequence.bin | tee -a run_${MODE}.txt
        done
    fi

    # runlist

    if [[ $MODE -eq 1 ]] then
        make build/test-runlist
        make build/combined_xclbin.xclbin
        make build/npu_insts_main_sequence.bin
        for i in $(seq 0 $ITERS); do
            ./build/test-runlist ./build/combined_xclbin.xclbin:${KERNEL_1} ./build/npu_insts_${KERNEL_1}_sequence.bin ./build/combined_xclbin.xclbin:${KERNEL_1} ./build/npu_insts_${KERNEL_1}_sequence.bin ./build/combined_xclbin.xclbin:${KERNEL_2} ./build/npu_insts_${KERNEL_2}_sequence.bin | tee -a run_${MODE}.txt
        done
    fi

    # To get empty xclbin with individual configure and run instruction invocations:

    if [[ $MODE -eq 2 ]] then
        # To get inlined txns runtime:
        make build/test
        make build/empty.xclbin
        make build/npu_insts_main_sequence.bin
        for i in $(seq $ITERS); do
            ./build/test ./build/empty.xclbin ./build/npu_insts_main_sequence.bin:./build/npu_insts_patch_map_main_sequence.txt | tee -a run_${MODE}.txt
        done
    fi

done

./eval.py
