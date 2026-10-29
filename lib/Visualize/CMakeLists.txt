#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set(VIEWER_TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/viewer.html)
set(VIEWER_INC ${CMAKE_CURRENT_BINARY_DIR}/ViewerTemplate.inc)
add_custom_command(
  OUTPUT ${VIEWER_INC}
  COMMAND ${CMAKE_COMMAND} -DINPUT=${VIEWER_TEMPLATE} -DOUTPUT=${VIEWER_INC}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/EmbedFile.cmake
  DEPENDS ${VIEWER_TEMPLATE} ${CMAKE_CURRENT_SOURCE_DIR}/EmbedFile.cmake
  COMMENT "Embedding the aie-visualize viewer")
add_custom_target(AIEVisualizeViewerTemplate DEPENDS ${VIEWER_INC})

add_mlir_library(AIEVisualize
  BuildGraph.cpp
  DesignGraph.cpp
  OrthogonalRouter.cpp
  RenderHtml.cpp
  RenderSvg.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/aie/Visualize

  DEPENDS
  AIEVisualizeViewerTemplate
  MLIRAIEAttrDefsIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEIncGen
  MLIRAIEInterfacesIncGen
  MLIRAIEOpsIncGen
  MLIRAIETypesIncGen
  MLIRAIEXIncGen

  LINK_LIBS PUBLIC
  AIE
  AIETransforms
  AIEX
  MLIRTransforms
  MLIRPass
)
target_include_directories(obj.AIEVisualize PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# libavoid (Adaptagrams, LGPL-2.1) routes the edges of the must-hit and free
# routings. It is optional and linked dynamically; without it, only the
# physical routing is available.
set(AIE_VISUALIZE_LIBAVOID "" CACHE PATH
    "Install prefix of libavoid, for aie-visualize's must-hit and free routing")
if(AIE_VISUALIZE_LIBAVOID)
  find_library(AIE_LIBAVOID_LIBRARY avoid
    PATHS ${AIE_VISUALIZE_LIBAVOID}/lib NO_DEFAULT_PATH REQUIRED)
  target_compile_definitions(obj.AIEVisualize PRIVATE AIE_VISUALIZE_HAS_LIBAVOID)
  target_include_directories(obj.AIEVisualize PRIVATE
    ${AIE_VISUALIZE_LIBAVOID}/include)
  target_link_libraries(AIEVisualize PUBLIC ${AIE_LIBAVOID_LIBRARY})
endif()
