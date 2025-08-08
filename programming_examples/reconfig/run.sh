#!/bin/bash

set -e

MODE=3
ITERS=20

if [[ $MODE -eq 0 ]] then
    # To get individual xclbins runtime:
    make build/test
    make build/add_two.xclbin
    make build/subtract_three.xclbin
    make build/npu_insts_main_rt.bin
    for i in $(seq 0 $ITERS); do
        ./build/test ./build/add_two.xclbin ./build/npu_insts_add_two_rt.bin ./build/subtract_three.xclbin ./build/npu_insts_subtract_three_rt.bin | tee -a run_${MODE}.txt
    done
fi


# To get empty xclbin with individual configure and run instruction invocations:

if [[ $MODE -eq 1 ]] then
    # To get inlined txns runtime:
    make build/test
    make build/empty.xclbin
    make build/npu_insts_main_rt.bin
    for i in $(seq $ITERS); do
        ./build/test ./build/empty.xclbin ./build/npu_insts_main_rt.bin | tee -a run_${MODE}.txt
    done
fi


# Individual configs and run insts with empty xclbin

if [[ $MODE -eq 2 ]] then
    make build/test
    make build/empty.xclbin
    make build/config_add_two_insts.bin
    make build/run_add_two_insts.bin
    make build/config_subtract_three_insts.bin
    make build/run_subtract_three_insts.bin
    for i in $(seq $ITERS); do
        ./build/test ./build/empty.xclbin ./build/config_add_two_insts.bin ./build/empty.xclbin ./build/run_add_two_insts.bin ./build/empty.xclbin ./build/config_subtract_three_insts.bin ./build/empty.xclbin ./build/run_subtract_three_insts.bin | tee -a run_${MODE}.txt
    done
fi

# runlist

if [[ $MODE -eq 3 ]] then
    make build/test-runlist
    make build/combined_xclbin.xclbin
    make build/npu_insts_main_rt.bin
    for i in $(seq 0 $ITERS); do
        ./build/test-runlist ./build/combined_xclbin.xclbin:add_two ./build/npu_insts_add_two_rt.bin ./build/combined_xclbin.xclbin:subtract_three ./build/npu_insts_subtract_three_rt.bin | tee -a run_${MODE}.txt
    done
fi