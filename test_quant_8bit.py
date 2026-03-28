#!/usr/bin/env python3
"""Test 8-bit quantization instead of 4-bit"""

import sys
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
sys.path.insert(0, str(Path(__file__).parent / "mlx-lm-source"))
from mlx_lm.models.triple_layer_cache import TripleLayerKVCache

PROMPT = """You are investigating a database connection pool exhaustion issue. The application shows 500 errors with "Connection pool exhausted" messages. What is the root cause?"""

model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
model, tokenizer = load(model_path)

tokens = tokenizer.encode(PROMPT)
y = mx.array([tokens])

# Test 8-bit quantization
cache = TripleLayerKVCache(
    recent_size=512,
    warm_size=1536,
    layer_idx=0,
    quant_bits=8,  # 8-bit instead of 4-bit
    enable_warm_quant=True,
    enable_cold_am=False
)

# Prefill
logits = model(y[:, :-1], cache=[cache] * len(model.model.layers))
mx.eval(logits)

# Generate
y = mx.array([[tokens[-1]]])
generated = []
for i in range(50):
    logits = model(y, cache=[cache] * len(model.model.layers))
    y = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(y)
    generated.append(y[0, 0].item())
    if y[0, 0].item() == tokenizer.eos_token_id:
        break

output = tokenizer.decode(generated)
print("8-bit quantization output:")
print(output[:200])
print("\n" + "="*80)

# Check for repetition
if "What is" in output and output.count("What is") > 3:
    print("❌ STILL REPEATING with 8-bit!")
else:
    print("✅ No repetition with 8-bit")
