#!/usr/bin/env python3
"""调试 beta 应用：打印 shapes 和值"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.hybrid_cache import HybridKVCache

def debug_beta_application():
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    calibration_file = "calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl"

    prompt = "The capital of France is"

    print("=" * 70)
    print("🔍 Beta Application Debug")
    print("=" * 70)

    # Load model
    model, tokenizer = load(model_path)
    tokens = mx.array(tokenizer.encode(prompt))
    print(f"\nPrompt tokens: {tokens.shape[0]}")

    # Create hybrid cache
    cache = [
        HybridKVCache(
            compression_ratio=2.0,
            calibration_file=calibration_file,
            layer_idx=i
        )
        for i in range(len(model.model.layers))
    ]

    # Prefill
    print("\n1️⃣ Prefill (uncompressed)...")
    logits = model(tokens[None, :], cache=cache)

    # Check cache state before compression
    print(f"\nCache[0] before compression:")
    print(f"  keys shape: {cache[0].keys.shape if cache[0].keys is not None else None}")
    print(f"  values shape: {cache[0].values.shape if cache[0].values is not None else None}")
    print(f"  state: {cache[0].state}")

    # Compress
    print("\n2️⃣ Compressing...")
    for i, c in enumerate(cache):
        before, after = c.compress()
        if i == 0:
            print(f"\nCache[0] after compression:")
            print(f"  keys shape: {c.keys.shape}")
            print(f"  values shape: {c.values.shape}")
            print(f"  beta shape: {c.beta.shape if c.beta is not None else None}")
            print(f"  state: {c.state}")
            print(f"  compressed: {before} → {after}")

    # Generate first token
    print("\n3️⃣ Generating first token (with compression)...")

    # Manually inject debug into model
    # We'll check shapes during the first generation step

    # Patch the attention function to print shapes
    original_forward = model.model.layers[0].self_attn.forward

    def debug_forward(x, mask=None, cache=None):
        print(f"\n[Layer 0 Attention Debug]")
        print(f"  Input x shape: {x.shape}")
        if cache is not None:
            if hasattr(cache, 'keys') and cache.keys is not None:
                print(f"  Cache keys shape: {cache.keys.shape}")
            if hasattr(cache, 'beta') and cache.beta is not None:
                print(f"  Cache beta shape: {cache.beta.shape}")
                print(f"  Cache beta min/max: {mx.min(cache.beta).item():.4f} / {mx.max(cache.beta).item():.4f}")
        return original_forward(x, mask=mask, cache=cache)

    model.model.layers[0].self_attn.forward = debug_forward

    # Generate one token
    token = mx.argmax(logits[0, -1]).item()
    print(f"\nFirst generated token: {tokenizer.decode([token])}")

    # Generate second token (this will call debug_forward)
    logits = model(mx.array([[token]]), cache=cache)
    token2 = mx.argmax(logits[0, -1]).item()
    print(f"Second generated token: {tokenizer.decode([token2])}")

    print("\n" + "=" * 70)
    print("✓ Debug complete")
    print("=" * 70)

if __name__ == "__main__":
    debug_beta_application()
