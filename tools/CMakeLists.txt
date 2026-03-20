#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# Only build bootgen if OpenSSL is available (MOVED BEFORE aie-translate)
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
  add_subdirectory(bootgen)
else()
  message(WARNING "OpenSSL not found - skipping full bootgen build. Building cdo_driver_mlir_aie only.")
  # Still build cdo_driver_mlir_aie even without OpenSSL, as it's needed by runtime_lib
  set(BOOTGEN_SOURCE ${PROJECT_SOURCE_DIR}/third_party/bootgen)
  add_library(cdo_driver_mlir_aie STATIC ${BOOTGEN_SOURCE}/cdo-driver/cdo_driver.c)
  target_include_directories(
    cdo_driver_mlir_aie PUBLIC $<BUILD_INTERFACE:${BOOTGEN_SOURCE}/cdo-driver>
    $<INSTALL_INTERFACE:include/cdo_driver>)
  if (NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(cdo_driver_mlir_aie PRIVATE -Wno-cast-qual -Wno-sign-compare)
  endif()
  install(
    TARGETS cdo_driver_mlir_aie
    DESTINATION lib
    EXPORT cdo_driver_mlir_aie)
  install(EXPORT cdo_driver_mlir_aie DESTINATION lib/cmake/aie)
endif()

add_subdirectory(aie-opt)
if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
  add_subdirectory(aie-reset)
endif()
add_subdirectory(aie-lsp-server)
add_subdirectory(aie-translate)
add_subdirectory(aie-visualize)
add_subdirectory(aiecc)
add_subdirectory(chess-clang)
