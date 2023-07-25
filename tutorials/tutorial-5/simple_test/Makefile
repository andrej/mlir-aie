.PHONY: sim clean

AIE_RUNTIME_LIB ?= $(shell realpath $(dir $(shell which aie-opt))/../runtime_lib)

sim: build/test 

build/test: test.cpp aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -j4 --xchesscc --xbridge --aiesim ../$(word 2,$^) -I$(AIE_RUNTIME_LIB)/x86_64/test_lib/include -L$(AIE_RUNTIME_LIB)/x86_64/test_lib/lib -ltest_lib -D__AIESIM__ ../$<

clean:
	rm -rf build
