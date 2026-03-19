#!/bin/bash

# Script to repeatedly run the cpp_multi_device_sequence.mlir test
# to reproduce spurious failures

set +e  # Don't exit on error - we want to catch failures

# Source the required environment
source ~/setup_buildenv.sh

# Change to build directory
cd /scratch/roesti/mlir-aie/build || { echo "Failed to cd to build directory"; exit 1; }

# Configuration
MAX_ITERATIONS=100
ITERATION=0
FAILURES=0
LOG_DIR="test_loop_logs"

# Create log directory
mkdir -p "$LOG_DIR"

echo "Starting test loop - will run up to $MAX_ITERATIONS iterations"
echo "Logs will be saved to $LOG_DIR/"
echo "Press Ctrl+C to stop early"
echo ""

while [ $ITERATION -lt $MAX_ITERATIONS ]; do
    ITERATION=$((ITERATION + 1))
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/iteration_${ITERATION}_${TIMESTAMP}.log"

    echo "[$TIMESTAMP] Running iteration $ITERATION/$MAX_ITERATIONS..."

    # Run the test and capture output
    LIT_FILTER="aiecc/cpp_multi_device_sequence.mlir" ninja check-aie > "$LOG_FILE" 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        FAILURES=$((FAILURES + 1))
        echo "  ❌ FAILURE detected! (Exit code: $EXIT_CODE)"
        echo "  Log saved to: $LOG_FILE"
        echo ""
        echo "=== Failure Summary ==="
        echo "Failed at iteration: $ITERATION"
        echo "Total failures so far: $FAILURES"
        echo "Failure log: $LOG_FILE"
        echo ""
        echo "=== Last 50 lines of failure log ==="
        tail -n 50 "$LOG_FILE"
        echo ""
        echo "=== End of failure log excerpt ==="
        echo ""

        # Ask if we should continue
        read -p "Continue running tests? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            break
        fi
    else
        echo "  ✓ Passed"
        # Remove passing logs to save space
        rm "$LOG_FILE"
    fi
done

echo ""
echo "=== Test Loop Complete ==="
echo "Total iterations: $ITERATION"
echo "Total failures: $FAILURES"
if [ $FAILURES -gt 0 ]; then
    echo "Failure rate: $(awk "BEGIN {printf \"%.2f%%\", ($FAILURES/$ITERATION)*100}")"
    echo "Failure logs saved in: $LOG_DIR/"
else
    echo "No failures detected!"
fi
