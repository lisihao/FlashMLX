#!/usr/bin/env python3
"""诊断 HybridKVCache 的 beta"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.hybrid_cache import HybridKVCache

def diagnose_hybrid():
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    calibration_file = "calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl"
    prompt = "The capital of France is"

    print("=" * 70)
    print("🔍 诊断 HybridKVCache Beta")
    print("=" * 70)

    model, tokenizer = load(model_path)
    tokens = mx.array(tokenizer.encode(prompt))

    # 创建 HybridKVCache
    hybrid_cache = [
        HybridKVCache(
            compression_ratio=2.0,
            calibration_file=calibration_file,
            layer_idx=i
        )
        for i in range(len(model.model.layers))
    ]

    print(f"\n1️⃣ 检查 Layer 0 beta:")
    print(f"   beta shape: {hybrid_cache[0].beta.shape if hybrid_cache[0].beta is not None else None}")
    if hybrid_cache[0].beta is not None:
        print(f"   beta min/max: {mx.min(hybrid_cache[0].beta).item():.4f} / {mx.max(hybrid_cache[0].beta).item():.4f}")
        print(f"   beta mean: {mx.mean(hybrid_cache[0].beta).item():.4f}")

    # Prefill
    print(f"\n2️⃣ Prefill:")
    logits = model(tokens[None, :], cache=hybrid_cache)

    # Compress
    print(f"\n3️⃣ Compress:")
    for i, c in enumerate(hybrid_cache):
        before, after = c.compress()
        if i == 0:
            print(f"   Layer 0: {before} → {after} tokens")

    # 检查压缩后的 beta
    print(f"\n4️⃣ 检查压缩后的 beta:")
    beta = hybrid_cache[0].get_beta()
    if beta is not None:
        print(f"   beta shape: {beta.shape}")
        print(f"   beta min/max: {mx.min(beta).item():.4f} / {mx.max(beta).item():.4f}")
        print(f"   beta mean: {mx.mean(beta).item():.4f}")
        print(f"   beta first 10: {beta[:10].tolist()}")
    else:
        print(f"   ❌ beta is None!")

    # 生成多个 tokens
    print(f"\n5️⃣ 生成 10 个 tokens:")
    for i in range(10):
        token_id = mx.argmax(logits[0, -1]).item()
        decoded = tokenizer.decode([token_id])
        print(f"   Token {i+1}: {repr(decoded)} (ID: {token_id})")

        # Generate next token
        logits = model(mx.array([[token_id]]), cache=hybrid_cache)

    # 检查 cache 是否正确传递给 scaled_dot_product_attention
    print(f"\n6️⃣ 检查 cache.get_beta() 是否可调用:")
    print(f"   callable(hybrid_cache[0].get_beta): {callable(getattr(hybrid_cache[0], 'get_beta', None))}")

    print("\n" + "=" * 70)
    print("✓ 诊断完成")
    print("=" * 70)

if __name__ == "__main__":
    diagnose_hybrid()
