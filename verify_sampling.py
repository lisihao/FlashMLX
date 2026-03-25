#!/usr/bin/env python3
"""验证采样方式：greedy vs temperature"""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
from mlx_lm import load
import numpy as np

def sample_token(logits, temperature=0.0):
    """
    Sample next token.
    temperature=0.0 → greedy (argmax)
    temperature>0.0 → sampling
    """
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1).item()
    else:
        # Temperature sampling
        logits = logits / temperature
        probs = mx.softmax(logits, axis=-1)
        # Sample from distribution
        probs_np = np.array(probs)
        token = np.random.choice(len(probs_np), p=probs_np)
        return token

def test_sampling():
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"

    prompt = "The capital of France is"

    print("=" * 70)
    print("🔍 对比 Greedy vs Temperature Sampling")
    print("=" * 70)

    model, tokenizer = load(model_path)
    tokens = mx.array(tokenizer.encode(prompt))

    for temp in [0.0, 0.7, 1.0]:
        print(f"\n📝 Temperature = {temp}")

        # Fresh cache
        cache = [None] * len(model.model.layers)
        logits = model(tokens[None, :], cache=cache)

        # Generate 10 tokens
        response = []
        for i in range(10):
            token = sample_token(logits[0, -1], temperature=temp)
            response.append(token)
            logits = model(mx.array([[token]]), cache=cache)

        output = tokenizer.decode(response)
        print(f"   Output: {output}")

    print("\n" + "=" * 70)
    print("✓ 测试完成")
    print("=" * 70)

if __name__ == "__main__":
    test_sampling()
