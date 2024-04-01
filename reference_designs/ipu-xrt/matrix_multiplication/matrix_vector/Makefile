##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

subdir=matrix_vector
targetname=matrixVectorMultiplication
kernels=mv

M?=288
K?=288
N=1

cores?=1

aieargs=-M ${M} -K ${K} -N ${N} --cores ${cores}

mlir_target?=build/aie_${M}x${K}_${cores}c.mlir
xclbin_target?=build/final_${M}x${K}_${cores}c.xclbin
insts_target?=build/insts_${M}x${K}_${cores}.txt

include ../makefile-common