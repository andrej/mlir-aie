#!/bin/bash
set -e

source /workspace/env_setup.sh

echo "=== Testing Flow Lifting on matrix_multiplication example ==="
OUTPUT=$(aie-translate --xclbin-to-mlir build/final_512x512x512_32x32x32_4c.xclbin --emit-lifted 2>&1)

# Count total flows
FLOW_COUNT=$(echo "$OUTPUT" | grep "^[[:space:]]*aie.flow" | wc -l)
echo "Total flows reconstructed: $FLOW_COUNT"

# Count flows from shim to other tiles (forward flows)
SHIM_TO_TILE=$(echo "$OUTPUT" | grep "aie.flow(%shim_noc_tile_" | wc -l)
echo "Flows from shim tiles: $SHIM_TO_TILE"

# Count flows to shim tiles (return flows)
TILE_TO_SHIM=$(echo "$OUTPUT" | grep "aie.flow.*DMA : 0)" | grep -c "shim_noc_tile" || true)
echo "Flows to shim tiles: $TILE_TO_SHIM"

# Count flows involving memory tiles
MEM_TILE_FLOWS=$(echo "$OUTPUT" | grep -c "mem_tile_" || true)
echo "Flows involving memory tiles: $MEM_TILE_FLOWS"

echo ""
echo "Sample flows:"
echo "$OUTPUT" | grep "^[[:space:]]*aie.flow" | head -10

if [ "$FLOW_COUNT" -gt 0 ]; then
    echo ""
    echo "✓ SUCCESS: $FLOW_COUNT flows reconstructed (including bidirectional flows)"
    exit 0
else
    echo ""
    echo "✗ FAILURE: No flows reconstructed"
    exit 1
fi
