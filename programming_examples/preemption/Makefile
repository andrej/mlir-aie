mlir_aie_root=$(shell dirname $(shell which aiecc.py))/../
xrt_flags=-I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil
test_utils_flags=-I${mlir_aie_root}/runtime_lib/x86_64/test_lib/include/ -L${mlir_aie_root}/runtime_lib/x86_64/test_lib/lib/ -ltest_utils

build/%.xclbin build/%.elf: %.mlir
	mkdir -p ${@D}
	aiecc.py --tmpdir ${@D} --aie-generate-xclbin --xclbin-name=build/$*.xclbin --aie-generate-elf --elf-name=build/$*.elf --no-compile-host --no-xchesscc --no-xbridge $^

build/test: test.cpp
	mkdir -p ${@D}
	clang $^ -o $@ -std=c++17 -Wall -lrt -lstdc++  ${xrt_flags} ${test_utils_flags}

.PHONY: clean
clean:
	rm -rf build
	rm -rf core_*.elf

.PHONY: clean_results
clean_results:
	rm -rf results_*.txt