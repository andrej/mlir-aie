#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.
# (c) Copyright 2024 Advanced Micro Devices Inc.

llvm_map_components_to_libnames(llvm_libs support)

add_llvm_tool(aie-translate aie-translate.cpp)
llvm_update_compile_flags(aie-translate)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LIBS
  ${dialect_libs}
  ${translation_libs}
  AIE
  AIETransforms
  AIEX
  AIEXTransforms
  AIEXUtils
  AIETargets
  MLIRAIEVecDialect
  MLIRAIEVecAIE1Dialect
  MLIRXLLVMToLLVMIRTranslation
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTargetAIEVecCpp
  MLIRTargetLLVMIRExport
  MLIRTargetLLVMIRImport
)

target_link_libraries(aie-translate PRIVATE ${LIBS})

# For xclbin decompilation (decode_cdo_binary)
if(TARGET bootgen-lib)
  find_package(OpenSSL)
  if(OpenSSL_FOUND)
    target_link_libraries(aie-translate PRIVATE bootgen-lib OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(aie-translate PRIVATE HAVE_BOOTGEN=1)
  else()
    target_link_libraries(aie-translate PRIVATE bootgen-lib)
    target_compile_definitions(aie-translate PRIVATE HAVE_BOOTGEN=1)
  endif()
else()
  message(WARNING "bootgen library not available - xclbin CDO decoding will be unavailable")
endif()
