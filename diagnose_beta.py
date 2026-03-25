#!/usr/bin/env python3
"""诊断：检查 beta 是否被应用"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache

def diagnose():
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    prompt = "The capital of France is"

    print("=" * 70)
    print("🔍 诊断 Beta 应用")
    print("=" * 70)

    model, tokenizer = load(model_path)
    tokens = mx.array(tokenizer.encode(prompt))

    # 使用官方 cache 创建方式
    prompt_cache = cache.make_prompt_cache(model)

    print(f"\n1️⃣ 检查 cache 类型:")
    print(f"   Cache[0] type: {type(prompt_cache[0])}")
    print(f"   Has beta? {hasattr(prompt_cache[0], 'beta')}")
    if hasattr(prompt_cache[0], 'beta'):
        print(f"   beta value: {prompt_cache[0].beta}")

    # Prefill
    print(f"\n2️⃣ Prefill:")
    logits = model(tokens[None, :], cache=prompt_cache)

    print(f"   Cache[0] after prefill:")
    print(f"   Has beta? {hasattr(prompt_cache[0], 'beta')}")
    if hasattr(prompt_cache[0], 'beta'):
        beta_val = prompt_cache[0].beta
        print(f"   beta: {beta_val}")
        if beta_val is not None:
            print(f"   beta shape: {beta_val.shape}")
            print(f"   beta min/max: {mx.min(beta_val).item():.4f} / {mx.max(beta_val).item():.4f}")

    # 生成几个 token
    print(f"\n3️⃣ Generate 5 tokens:")
    for i in range(5):
        token_id = mx.argmax(logits[0, -1]).item()
        decoded = tokenizer.decode([token_id])
        print(f"   Token {i+1}: {repr(decoded)} (ID: {token_id})")
        logits = model(mx.array([[token_id]]), cache=prompt_cache)

    print("\n" + "=" * 70)
    print("✓ 诊断完成")
    print("=" * 70)

if __name__ == "__main__":
    diagnose()
