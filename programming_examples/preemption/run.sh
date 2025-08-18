#!/bin/bash

set -ex

STEMS="aie-without-preemption aie-with-preemption"
ITERS=100

make clean
make build/test

for STEM in $STEMS; do

    make build/$STEM.xclbin

    sudo /opt/xilinx/xrt/bin/xrt-smi configure --advanced --force-preemption --type layer --action enable
    sudo /opt/xilinx/xrt/bin/xrt-smi configure --advanced --force-preemption --type frame --action enable

    echo "" > results-$STEM.txt
    for i in $(seq $ITERS); do
        ./build/test -k MLIR_AIE -x build/$STEM.xclbin -i build/$STEM.elf | tee -a results-$STEM.txt
        xrt-smi examine --advanced --report preemption | tee -a results-$STEM.txt
    done

    sudo /opt/xilinx/xrt/bin/xrt-smi configure --advanced --force-preemption --type layer --action disable
    sudo /opt/xilinx/xrt/bin/xrt-smi configure --advanced --force-preemption --type frame --action disable

done

python3 ./eval.py