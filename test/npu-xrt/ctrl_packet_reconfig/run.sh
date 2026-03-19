#!/bin/bash
cp ./aie.mlir aie_arch.mlir

sed 's/NPUDEVICE/npu2_1col/g' -i aie_arch.mlir

aie-opt -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" aie_arch.mlir -o aie_overlay.mlir
aiecc.py -v --device-name=base --aie-generate-xclbin --xclbin-name=aie.xclbin aie_overlay.mlir
aiecc.py -v --device-name=main --aie-generate-ctrlpkt --ctrlpkt-name=ctrlpkt.bin --aie-generate-npu-insts --npu-insts-name=aie_run_seq.bin aie_overlay.mlir

clang ./test.cpp -o test.exe -std=c++17 -Wall -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil -lrt -lstdc++ -I/scratch/roesti/mlir-aie/install/runtime_lib/x86_64/test_lib/include -L/scratch/roesti/mlir-aie/install/runtime_lib/x86_64/test_lib/lib -ltest_utils

./test.exe