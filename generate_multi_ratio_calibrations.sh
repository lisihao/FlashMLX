#!/bin/bash
# Batch generate calibration files for multiple compression ratios
# Ratios: 2.0x, 3.0x, 5.0x (for layerwise compression)

set -e

MODEL_PATH="/Volumes/toshiba/models/qwen3-8b-mlx"
OUTPUT_DIR="/tmp/am_calibrations_ultra_dense"
LENGTHS="512,768,1024,1536,2048,2500,3000"

echo "====================================================================="
echo "🔬 Multi-Ratio Calibration Generation"
echo "====================================================================="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Lengths: $LENGTHS"
echo "Ratios: 2.0x, 3.0x, 5.0x"
echo ""
echo "Estimated time: 2-3 hours"
echo "====================================================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Ratio 2.0x
echo ""
echo "====================================================================="
echo "Step 1/3: Generating R2.0 calibrations..."
echo "====================================================================="
python3 calibrate_am_multi_length.py \
    --model-path "$MODEL_PATH" \
    --ratio 2.0 \
    --lengths "$LENGTHS" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee /tmp/calibrate_r2.0.log

# Ratio 3.0x
echo ""
echo "====================================================================="
echo "Step 2/3: Generating R3.0 calibrations..."
echo "====================================================================="
python3 calibrate_am_multi_length.py \
    --model-path "$MODEL_PATH" \
    --ratio 3.0 \
    --lengths "$LENGTHS" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee /tmp/calibrate_r3.0.log

# Ratio 5.0x
echo ""
echo "====================================================================="
echo "Step 3/3: Generating R5.0 calibrations..."
echo "====================================================================="
python3 calibrate_am_multi_length.py \
    --model-path "$MODEL_PATH" \
    --ratio 5.0 \
    --lengths "$LENGTHS" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee /tmp/calibrate_r5.0.log

# Summary
echo ""
echo "====================================================================="
echo "✅ Multi-Ratio Calibration Complete!"
echo "====================================================================="
echo "Generated calibration files:"
ls -lh "$OUTPUT_DIR" | grep -E "R(2|3|5)\.0" | head -20
echo ""
echo "Total files:"
ls "$OUTPUT_DIR"/*.pkl | wc -l
echo ""
echo "Next step: Re-run benchmark_layerwise_compression.py"
echo "====================================================================="
