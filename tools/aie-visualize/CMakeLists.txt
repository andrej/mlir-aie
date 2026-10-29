#
# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_executable(aie-visualize aie-visualize.cpp)

# The viewer opens the drawings aie-visualize writes: aie-visualize.html?svg=...
configure_file(${PROJECT_SOURCE_DIR}/lib/Visualize/viewer.html
  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/aie-visualize.html COPYONLY)

target_include_directories(aie-visualize PUBLIC ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
llvm_update_compile_flags(aie-visualize)

llvm_map_components_to_libnames(llvm_libs support)
target_link_libraries(aie-visualize ${llvm_libs})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# configure_file(configure.h.in configure.h)
# target_include_directories(aie-visualize PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(aie-visualize
  ${dialect_libs}
  MLIRParser
  MLIRSCFToControlFlow
  MLIRAffineToStandard
  MLIRAIEVecDialect
  MLIRAIEVecToLLVM
  MLIRAIEVecTransforms
  MLIRBuiltinToLLVMIRTranslation
  MLIRXLLVMToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  AIE
  AIETransforms
  AIETargets
  AIEVisualize
  AIEX
  AIEXTransforms
  MLIRAIEVecDialect
  MLIRAIEVecAIE1Dialect
  MLIRXLLVMDialect)

install(TARGETS aie-visualize
  EXPORT AIE-VISUALIZE
  RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
  COMPONENT aie-visualize)
install(FILES ${PROJECT_SOURCE_DIR}/lib/Visualize/viewer.html
  DESTINATION bin
  RENAME aie-visualize.html
  COMPONENT aie-visualize)
