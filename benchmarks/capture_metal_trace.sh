#!/bin/bash
# Capture Metal System Trace for FlashMLX profiling
set -e

MODEL_PATH="$HOME/models/qwen3.5-2b-opus-distilled"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE_DIR="$SCRIPT_DIR/../profiling_data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRACE_FILE="$TRACE_DIR/metal_trace_${TIMESTAMP}.gputrace"

mkdir -p "$TRACE_DIR"

echo "========================================================"
echo "Metal System Trace - FlashMLX Profiling"
echo "========================================================"
echo ""
echo "Configuration:"
echo "  Model:       $MODEL_PATH"
echo "  Trace file:  $TRACE_FILE"
echo ""
echo "Starting trace capture..."
echo "This will take ~5-10 seconds"
echo ""

# Activate venv and run benchmark with Metal trace
cd "$SCRIPT_DIR/.."

xcrun xctrace record \
  --template "Metal System Trace" \
  --output "$TRACE_FILE" \
  --time-limit 10000ms \
  --launch -- \
  ./venv/bin/python -c "
import mlx.core as mx
from mlx_lm import load, stream_generate

print('Loading model...')
model, tokenizer = load('$MODEL_PATH')

# Short prompt for quick trace
prompt = 'The quick brown fox ' * 100  # ~500 tokens

print('Generating tokens...')
count = 0
for response in stream_generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=50,  # Short run for trace
):
    count += 1
    if count == 1:
        print(f'First token: {response}')

print(f'Generated {count} tokens')
"

echo ""
echo "========================================================"
echo "Trace captured successfully!"
echo ""
echo "Trace file: $TRACE_FILE"
echo ""
echo "To view:"
echo "  open '$TRACE_FILE'"
echo ""
echo "Or analyze with xctrace:"
echo "  xctrace export --input '$TRACE_FILE' --output trace_export"
echo "========================================================"
