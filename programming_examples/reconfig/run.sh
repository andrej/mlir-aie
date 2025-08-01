#!/bin/bash

MODE=1
ITERS=100

if [[ $MODE -eq 0 ]] then
    # To get individual xclbins runtime:
    make build/test
    make build/add_two.xclbin
    make build/subtract_three.xclbin
    make build/npu_insts_add_two_rt.bin
    make build/npu_insts_subtract_three_rt.bin
    for i in $(seq 0 $ITERS); do
        ./build/test ./build/add_two.xclbin ./build/npu_insts_add_two_rt.bin ./build/subtract_three.xclbin ./build/npu_insts_subtract_three_rt.bin
    done
fi


# To get empty xclbin with individual configure and run instruction invocations:

if [[ $MODE -eq 1 ]] then
    # To get inlined txns runtime:
    make build/test
    make build/empty.xclbin
    make build/npu_insts_main_rt.bin
    for i in $(seq $ITERS); do
        ./build/test ./build/empty.xclbin ./build/npu_insts_main_rt.bin
    done
fi