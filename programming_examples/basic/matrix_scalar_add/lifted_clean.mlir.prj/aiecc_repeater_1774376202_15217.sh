#!/bin/bash
set -e
# Repeater script for: routing
echo "Original MLIR Diagnostics:"
cat << 'DIAGNOSTICS_EOF'
'aie.connect' op targets same destination DMA: 0 as another connect operation
DIAGNOSTICS_EOF
echo ""

MLIR_FILE='lifted_clean.mlir.prj/aiecc_failure_1774376202_15217.mlir'
PASS_PIPELINE='builtin.module(aie.device(aie-create-pathfinder-flows))'
aie-opt --mlir-print-ir-after-all --mlir-disable-threading --pass-pipeline="$PASS_PIPELINE" "$MLIR_FILE"
