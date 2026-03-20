#!/usr/bin/env bash
# Simple script to run the xclbin2mlir roundtrip tests
# Usage: ./run_tests.sh

set -e

# Setup environment
source /opt/xilinx/xrt/setup.sh > /dev/null 2>&1
source /workspace/buildenv/bin/activate
export PEANO_INSTALL_DIR="$(pip show llvm-aie 2>/dev/null | grep ^Location: | awk '{print $2}')/llvm-aie"
source /workspace/mlir-aie/utils/env_setup.sh /workspace/mlir-aie/install ${PEANO_INSTALL_DIR} > /dev/null 2>&1

# FileCheck location
FILECHECK=/workspace/mlir-aie/my_install/mlir/bin/FileCheck

# Test directory
TESTDIR=$(dirname "$0")

echo "Running xclbin2mlir roundtrip tests..."
echo ""

# Test 1: add_blockwrite_raw
echo "Testing add_blockwrite_raw.mlir..."
if aie-translate --xclbin-to-mlir "$TESTDIR/../../npu-xrt/add_blockwrite/aie.xclbin" 2>&1 | \
   $FILECHECK "$TESTDIR/add_blockwrite_raw.mlir"; then
    echo "✓ add_blockwrite_raw.mlir PASSED"
else
    echo "✗ add_blockwrite_raw.mlir FAILED"
    exit 1
fi

echo ""

# Test 2: ctrl_packet_reconfig_raw
echo "Testing ctrl_packet_reconfig_raw.mlir..."
if aie-translate --xclbin-to-mlir "$TESTDIR/../../npu-xrt/ctrl_packet_reconfig/aie.xclbin" 2>&1 | \
   $FILECHECK "$TESTDIR/ctrl_packet_reconfig_raw.mlir"; then
    echo "✓ ctrl_packet_reconfig_raw.mlir PASSED"
else
    echo "✗ ctrl_packet_reconfig_raw.mlir FAILED"
    exit 1
fi

echo ""
echo "All tests PASSED!"
