#!/usr/bin/env python3
"""Test ORIGINAL code scored_pq at 32K."""

import sys
sys.path.insert(0, "mlx-lm-source")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_PATH = "/Volumes/toshiba/models/qwen3-8b-mlx"
CALIBRATION = "calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl"

print("Testing ORIGINAL code (no fix) - scored_pq at 32K")
print("="*60)

model, tokenizer = load(MODEL_PATH)

text = "The quick brown fox jumps over the lazy dog. " * 4000
tokens_list = tokenizer.encode(text)[:32768]
tokens = mx.array([tokens_list])
print(f"Context: {len(tokens_list)} tokens\n")

mx.clear_cache()
mx.reset_peak_memory()

try:
    print("Creating scored_pq cache...")
    cache = make_prompt_cache(model,
        kv_cache="scored_pq",
        kv_calibration=CALIBRATION,
        kv_flat_quant="q8_0",
    )

    print("Running prefill...")
    logits = model(tokens, cache=cache)
    mx.eval(logits)

    pp_peak = mx.get_peak_memory() / (1024**2)
    print(f"✅ Prefill SUCCESS: {pp_peak:.0f} MB")

    print("Generating tokens...")
    for _ in range(5):
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        logits = model(next_token, cache=cache)
        mx.eval(logits)

    tg_peak = mx.get_peak_memory() / (1024**2)
    print(f"✅ Generation SUCCESS: {tg_peak:.0f} MB")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nConclusion:")
print("If OOM → Original code has OOM at 32K (not my fix's fault)")
print("If SUCCESS → My fix broke something")
