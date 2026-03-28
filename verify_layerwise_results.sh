#!/bin/bash
# Verify layerwise compression calibration generation and run final test

set -e

CALIB_DIR="/tmp/am_calibrations_ultra_dense"

echo "====================================================================="
echo "🔍 Verifying Multi-Ratio Calibration Files"
echo "====================================================================="

# Check R2.0 files
echo ""
echo "R2.0 files:"
ls -lh "$CALIB_DIR"/*R2.0*.pkl 2>/dev/null | wc -l | xargs echo "  Count:"
ls "$CALIB_DIR"/*R2.0*.pkl 2>/dev/null | head -3

# Check R3.0 files
echo ""
echo "R3.0 files:"
ls -lh "$CALIB_DIR"/*R3.0*.pkl 2>/dev/null | wc -l | xargs echo "  Count:"
ls "$CALIB_DIR"/*R3.0*.pkl 2>/dev/null | head -3

# Check R5.0 files
echo ""
echo "R5.0 files:"
ls -lh "$CALIB_DIR"/*R5.0*.pkl 2>/dev/null | wc -l | xargs echo "  Count:"
ls "$CALIB_DIR"/*R5.0*.pkl 2>/dev/null | head -3

# Total count
echo ""
echo "====================================================================="
TOTAL_NEW=$(ls "$CALIB_DIR"/*R[235].0*.pkl 2>/dev/null | wc -l)
echo "Total new calibration files: $TOTAL_NEW"
echo "Expected: 21 (3 ratios × 7 lengths)"

if [ "$TOTAL_NEW" -eq 21 ]; then
    echo "✅ All calibration files generated successfully!"
    echo ""
    echo "====================================================================="
    echo "🚀 Running Layerwise Compression Benchmark"
    echo "====================================================================="

    python3 benchmark_layerwise_compression.py \
        --calibration-dir "$CALIB_DIR" \
        --memory-budget 2.0 \
        --num-generate 100 \
        2>&1 | tee /tmp/layerwise_final_results.log

    echo ""
    echo "====================================================================="
    echo "✅ Layerwise Compression Test Complete!"
    echo "====================================================================="
    echo "Results saved to: /tmp/layerwise_final_results.log"

else
    echo "⚠️  Only $TOTAL_NEW files found (expected 21)"
    echo "Please wait for calibration generation to complete"
fi

echo "====================================================================="
