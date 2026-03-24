#!/bin/bash
set -e

source /workspace/env_setup.sh

echo "=== Testing Flow Lifting on add_blockwrite example ==="
OUTPUT=$(aie-translate --xclbin-to-mlir aie.xclbin --emit-lifted 2>&1)

# Count total flows
FLOW_COUNT=$(echo "$OUTPUT" | grep "^[[:space:]]*aie.flow" | wc -l)
echo "Total flows reconstructed: $FLOW_COUNT"

# Check for forward flow (shim → tile)
FORWARD_FLOW=$(echo "$OUTPUT" | grep -c "aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)" || true)
echo "Forward flow (shim→tile): $FORWARD_FLOW"

# Check for return flow (tile → shim)
RETURN_FLOW=$(echo "$OUTPUT" | grep -c "aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)" || true)
echo "Return flow (tile→shim): $RETURN_FLOW"

# Verify expectations
if [ "$FLOW_COUNT" -eq 2 ] && [ "$FORWARD_FLOW" -eq 1 ] && [ "$RETURN_FLOW" -eq 1 ]; then
    echo "✓ SUCCESS: Both forward and return flows are correctly reconstructed!"
    exit 0
else
    echo "✗ FAILURE: Expected 2 flows (1 forward, 1 return), got $FLOW_COUNT total ($FORWARD_FLOW forward, $RETURN_FLOW return)"
    exit 1
fi
