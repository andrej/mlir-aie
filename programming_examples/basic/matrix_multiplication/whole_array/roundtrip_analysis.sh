#!/bin/bash
source /workspace/env_setup.sh

echo "=== ROUNDTRIP VERIFICATION REPORT ==="
echo ""
echo "Test File: build/final_512x512x512_32x32x32_4c.xclbin"
echo "Decompiled Output: decompiled_clean.mlir"
echo ""

echo "--- High-Level Constructs Lifted ---"
echo "aie.flow operations: $(grep -c "aie.flow" decompiled_clean.mlir)"
echo "aie.switchbox operations: $(grep -c "aie.switchbox" decompiled_clean.mlir)"
echo "aie.shim_mux operations: $(grep -c "aie.shim_mux" decompiled_clean.mlir)"
echo "aie.mem operations: $(grep -c "aie.mem(" decompiled_clean.mlir)"
echo "aie.memtile_dma operations: $(grep -c "aie.memtile_dma" decompiled_clean.mlir)"
echo "aie.buffer operations: $(grep -c "aie.buffer" decompiled_clean.mlir)"
echo "aie.lock operations: $(grep -c "aie.lock" decompiled_clean.mlir)"
echo "aie.dma_bd operations: $(grep -c "aie.dma_bd" decompiled_clean.mlir)"
echo ""

echo "--- Raw Register Writes (Should be 0 for full lifting) ---"
echo "aiex.npu.write32: $(grep -c "aiex.npu.write32" decompiled_clean.mlir)"
echo "aiex.npu.maskwrite32: $(grep -c "aiex.npu.maskwrite32" decompiled_clean.mlir)"
echo "Total raw writes: $(grep -cE "aiex.npu.write32|aiex.npu.maskwrite32" decompiled_clean.mlir)"
echo ""

echo "--- Lock Operations in DMA BDs ---"
echo "aie.use_lock in DMA context: $(grep -B2 -A2 "aie.dma_bd" decompiled_clean.mlir | grep -c "aie.use_lock")"
echo ""

echo "--- DMA BD Chain Analysis ---"
echo "aie.next_bd operations: $(grep -c "aie.next_bd" decompiled_clean.mlir)"
echo "aie.end operations: $(grep -c "aie.end" decompiled_clean.mlir)"
echo ""

echo "--- MLIR Validation ---"
aie-opt decompiled_clean.mlir -o /dev/null 2>&1 && echo "✓ MLIR validates successfully" || echo "✗ MLIR validation failed"
echo ""

echo "--- File Sizes ---"
echo "Original xclbin: $(ls -lh build/final_512x512x512_32x32x32_4c.xclbin | awk '{print $5}')"
echo "Decompiled MLIR: $(ls -lh decompiled_clean.mlir | awk '{print $5}')"
echo ""
