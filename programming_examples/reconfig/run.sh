#!/bin/bash

set -e

MODE=1
ITERS=100

rm -f run_${MODE}.txt
#mv run_${MODE}.txt run_${MODE}.txt.old

if [[ $MODE -eq 0 ]] then
    export INPUT_MLIR=another-test/aie-twice-wo-preemption.mlir
elif [[ $MODE -eq 1 ]] then
    export INPUT_MLIR=another-test/aie-twice-w-preemption.mlir
fi

make clean
make build/mm.o
make build/test
make build/empty.xclbin
make build/npu_insts_main_rt.bin
sudo /opt/xilinx/xrt/bin/xrt-smi configure --advanced --force-preemption --type layer --action disable
sudo /opt/xilinx/xrt/bin/xrt-smi configure --advanced --force-preemption --type frame --action disable
sudo /opt/xilinx/xrt/bin/xrt-smi configure --advanced --force-preemption --type layer --action enable
sudo /opt/xilinx/xrt/bin/xrt-smi configure --advanced --force-preemption --type frame --action enable
/scratch/roesti/aiebu/build/Debug/opt/xilinx/aiebu/bin/aiebu-asm -t aie2txn -c build/npu_insts_main_rt.bin -o build/npu_insts_main_rt.elf
for i in $(seq $ITERS); do
    ./build/test ./build/empty.xclbin ./build/npu_insts_main_rt.elf | tee -a run_${MODE}.txt
done
xrt-smi examine --advanced --report preemption
