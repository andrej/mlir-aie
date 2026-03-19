
aie.elf: aie.mlir
	aiecc.py -v --generate-full-elf --no-xchesscc --no-xbridge --dynamic-objFifos aie.mlir

test: test_elf.cpp
	clang -o test test_elf.cpp -std=c++17 -lstdc++ -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_core -lxrt_coreutil

.PHONY: run
run: test aie.elf
	./test 

.PHONY: clean
clean:
	rm -rf test aie.mlir.prj aie.elf
