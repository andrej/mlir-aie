#!/bin/bash
cp ./aie.mlir aie_arch.mlir
sed 's/NPUDEVICE/npu2_1col/g' -i aie_arch.mlir
aiecc.py -v --no-aiesim --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential --xclbin-name=aie.xclbin --npu-insts-name=insts.bin ./aie_arch.mlir
clang ./test.cpp -o test.exe -std=c++17 -Wall -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil -lrt -lstdc++ -I/scratch/roesti/mlir-aie/install/runtime_lib/x86_64/test_lib/include -L/scratch/roesti/mlir-aie/install/runtime_lib/x86_64/test_lib/lib -ltest_utils
./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin
./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin