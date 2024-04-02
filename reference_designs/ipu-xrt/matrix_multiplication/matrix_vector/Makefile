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
trace?=0

target_suffix?=_${M}x${K}_c${cores}_t${trace}

include ../makefile-common
aieargs+= --cores ${cores}