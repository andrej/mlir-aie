#!/bin/bash

set -e

MODE=2
ITERS=100

mv run_${MODE}.txt run_${MODE}.txt.old

if [[ $MODE -eq 0 ]] then
    export INPUT_MLIR=another-test/aie-once.mlir
elif [[ $MODE -eq 1 ]] then
    export INPUT_MLIR=another-test/aie-twice-wo-reconfig.mlir
elif [[ $MODE -eq 2 ]] then
    export INPUT_MLIR=another-test/aie-twice-w-reconfig.mlir
fi

make clean
make build/mm.o
make build/test
make build/empty.xclbin
make build/npu_insts_main_rt.bin
for i in $(seq $ITERS); do
    ./build/test ./build/empty.xclbin ./build/npu_insts_main_rt.bin:./build/npu_insts_patch_map_main_rt.txt | tee -a run_${MODE}.txt
done
