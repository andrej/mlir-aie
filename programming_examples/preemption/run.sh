#!/bin/bash

set -ex

STEMS="aie-without-preemption aie-with-preemption"
ITERS=100

make clean
make build/test

for STEM in $STEMS; do
    for FORCE_PREEMPT in "enable" "disable"; do

        make build/$STEM.xclbin

        sudo /opt/xilinx/xrt/bin/xrt-smi configure --advanced --force-preemption --type layer --action $FORCE_PREEMPT
        sudo /opt/xilinx/xrt/bin/xrt-smi configure --advanced --force-preemption --type frame --action $FORCE_PREEMPT

        echo "" > results-$STEM-$FORCE_PREEMPT.txt
        for i in $(seq $ITERS); do
            ./build/test -k MLIR_AIE -x build/$STEM.xclbin -i build/$STEM.elf | tee -a results-$STEM-$FORCE_PREEMPT.txt
            xrt-smi examine --advanced --report preemption | tee -a results-$STEM-$FORCE_PREEMPT.txt
        done

        sudo /opt/xilinx/xrt/bin/xrt-smi configure --advanced --force-preemption --type layer --action disable
        sudo /opt/xilinx/xrt/bin/xrt-smi configure --advanced --force-preemption --type frame --action disable

    done
done

python3 ./eval.py